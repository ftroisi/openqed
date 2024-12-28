# Copyright 2024 Francesco Troisi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"This module contains the multi-communicator functions, to handle the parallelization strategies."

from openqed.io.input_file import InputFile
try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False

ROOT = 0

class Multicommunicator():
    """
    This class handles the distribution of the mpi tasks over all the parallelization strategies.

    Attributes:
        `mpi_w_cavity`: number of tasks assigned to the cavity frequencies parallelization strategy.
        `mpi_w_probe`: number of tasks assigned to the probe frequencies parallelization strategy.
        `world_comm`: the world communicator from mpi4py if available, None otherwise.
        `cartesian_comm`: the cartesian communicator from mpi4py if available, None otherwise.
    """
    def __init__(self, input_file: InputFile) -> None:
        """Initialize the multicommunicator
        Args:
            input_file: The input file object containing the parsed input file. This constructor
                will look for the following keys (all default to 1 (serial)):
            - `w_cavity_parallel`: number of tasks for the cavity frequencies
                parallelization strategy.
            - `w_probe_parallel`: number of tasks for the probe frequencies parallelization strategy

        Raises:
            ValueError: If the requested number of tasks for any strategy is less than 1.
            ValueError: If the product of the two parallelization strategies is not equal to the
                total number of tasks.
        """
        # Get data from the input file
        self.mpi_w_cavity: int = input_file.input["w_cavity_parallel"]
        self.mpi_w_probe: int = input_file.input["w_probe_parallel"]
        self.world_comm = None
        # Sanity check: the product of the two parallelization strategies must be equal to the
        # total number of tasks
        total_available_tasks: int = MPI.COMM_WORLD.Get_size() if MPI_AVAILABLE else 1
        if self.mpi_w_cavity < 1:
            raise ValueError(f"Requested {self.mpi_w_cavity} tasks for the cavity frequencies,"+
                            "but at least 1 task is needed")
        if self.mpi_w_probe < 1:
            raise ValueError(f"Requested {self.mpi_w_probe} tasks for the probe frequencies,"+
                            "but at least 1 task is needed")
        if self.mpi_w_cavity * self.mpi_w_probe != total_available_tasks:
            w_c_message: str = \
                f"Requested {self.mpi_w_cavity} tasks for cavity frequencies parallelization\n"
            w_p_message: str = \
                f"Requested {self.mpi_w_probe} tasks for probe frequencies parallelization\n"
            raise ValueError(f"Inconsistent number of MPI tasks!\n {w_c_message} {w_p_message}" +
                            f"Total available number of tasks: {total_available_tasks}\n" +
                            "The product of the number of tasks assigned to each parallelization " +
                            "strategy must be equal to the total number of tasks")
        if MPI_AVAILABLE:
            # Define the groups
            self.world_comm = MPI.COMM_WORLD
            self.cartesian_comm = self.world_comm.Create_cart(
                dims=[self.mpi_w_cavity, self.mpi_w_probe], periods=[False,False], reorder=False)

    @property
    def multicomm_topology(self) -> tuple[list[int], list[int], list[int]]:
        """Get the topology of the multicommunicator

        Returns:
            tuple[list[int], list[int], list[int]]: the topology of the multicommunicator. The first
                list contains the number of tasks in each dimension, the second list contains
                information about the periodicity of the topology, the third list contains the
                coordinates of all tasks
        """
        return self.cartesian_comm.Get_topo() if self.world_comm is not None else ([1], [0], [0])

    @property
    def task_cartesian_coordinates(self) -> list[int]:
        """Get cartesian coordinates of this processor in the multicommunicator

        Returns:
            list[int]: The list of coordinates of this processor in the multicommunicator
        """
        rank = self.world_comm.Get_rank() if self.world_comm is not None else 0
        return self.cartesian_comm.Get_coords(rank) if self.cartesian_comm is not None else [0]

    def get_tasks_in_dimension(self, dimension: int, ranks: list[int]) -> list[int]:
        """This method computes the tasks in a certain dimension. For instance, if we create a 2D
        topology with 2 tasks in the first dimension and 2 tasks in the second dimension, such that
        the topology looks like:
        - 0: [0, 0]
        - 1: [0, 1]
        - 2: [1, 0]
        - 3: [1, 1]\n
        the calling `get_tasks_in_dimension(0, [0])` means "get all tasks in dimension 0, that have
        rank 0 in dimension 1". This will return `[0, 2]`.
        Similarly `get_tasks_in_dimension(1, [0]) = [0, 1]`
        
        Args:
            dimension: the dimension to consider in which to retrieve the tasks
            ranks: the ranks of the tasks in all other dimension

        Returns:
            list[int]: the global rank of the tasks in the specified dimension
        """
        # First, get the task in world comm
        world_size: int = self.world_comm.Get_size() if self.world_comm is not None else 1
        if world_size == 1:
            return [0]
        # Sanity check
        if dimension < 0 or dimension > len(self.multicomm_topology[0]):
            raise ValueError(f"Invalid dimension: {dimension}")
        if len(ranks) != len(self.multicomm_topology[0]) - 1:
            raise ValueError("Invalid number of ranks for the other dimensions. Expected " +
                             f"{len(self.multicomm_topology[0])}, got {len(ranks)}")
        # Get the tasks in the specified dimension
        tasks_in_dimension: list[int] = []
        for rk in range(world_size):
            coords = self.cartesian_comm.Get_coords(rk)
            coords = coords[:dimension] + coords[dimension+1:]
            if coords == ranks:
                tasks_in_dimension.append(rk)
        return tasks_in_dimension

    def group_name_to_tolopogy_dimension(self, name: str) -> int:
        """This method converts the name of the group to the corresponding dimension in the
        topology.

        Args:
            name (str): the name of the group. Possible values are:
            - `w_cavity`
            - `w_probe`

        Returns:
            int: the dimension of the topology

        Raises:
            ValueError: if the name is not valid
        """
        if name == "w_cavity":
            return 0
        if name == "w_probe":
            return 1
        raise ValueError(f"Invalid group name: {name}")
