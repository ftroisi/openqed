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

"This module contains functions for MPI communication"

from typing import Literal
from .multicommunicator import Multicommunicator
try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False

ROOT = 0

class MpiGroup():
    """
    This class is a wrapper for MPI groups.

    Attributes:
        `multi_comm`: the multicommunicator that handles all parallelization strategies. See
            :class:`openqed.mpi.multicommunicator.Multicommunicator` for details.
        `world_comm`: the world communicator from mpi4py if available, None otherwise.
        `group`: the group of tasks in the multicommunicator.
        `comm`: the communicator for this group.
        `rank`: the rank of the current task in the communicator.
        `size`: the size of the communicator.
    """
    def __init__(self,
                multicommunicator: Multicommunicator,
                *,
                name: str) -> None:
        """Initialize the MpiGroup
        
        Args:
            multicommunicator: The multicommunicator that handles all parallelization strategies.
                See :class:`openqed.mpi.multicommunicator.Multicommunicator` for details.
            name: The name of the group. This name must be one of the parallelization strategies
                in the multicommunicator.
        """
        self.multi_comm: Multicommunicator = multicommunicator
        self.group = None
        self.comm = None
        self.rank = 0
        self.size = 1
        if MPI_AVAILABLE:
            world_comm = multicommunicator.world_comm
            # Get the dimension of this group in the multicommunicator
            dimension = multicommunicator.group_name_to_tolopogy_dimension(name)
            # Then get the coordinates of this task in the topology
            coords: list[int] = multicommunicator.task_cartesian_coordinates
            coords = coords[:dimension] + coords[dimension + 1:]
            # Get all the tasks in this group
            tasks_in_group: list[int] = multicommunicator.get_tasks_in_dimension(dimension, coords)
            # Finally create the group
            self.group = world_comm.group.Incl(tasks_in_group)  # type: ignore
            # Create a new communicator for the group
            self.comm = world_comm.Create(self.group)  # type: ignore
            self.comm.Set_name(name)
            # Get rank and size of the new communicator
            self.rank: int = self.comm.Get_rank()
            self.size: int = self.comm.Get_size()

    @property
    def world_rank(self) -> int:
        """Returns the rank of the current task in the world communicator"""
        if self.multi_comm.world_comm is not None:
            return self.multi_comm.world_comm.Get_rank()
        return 0

    @property
    def world_size(self) -> int:
        """Returns the size of the world communicator"""
        if self.multi_comm.world_comm is not None:
            return self.multi_comm.world_comm.Get_size()
        return 1

    @property
    def am_i_root(self) -> bool:
        """Returns True if the current task is the root of the group"""
        return self.comm.Get_rank() == ROOT if self.comm is not None else True

    @property
    def am_i_world_root(self) -> bool:
        """Returns True if the current task is the root of the world communicator"""
        if self.multi_comm.world_comm is not None:
            return self.multi_comm.world_comm.Get_rank() == ROOT
        return True

    @property
    def group_rank(self) -> int:
        """Returns the rank of this group in the multicommunicator"""
        comm_name = self.comm.Get_name() if self.comm is not None else "world"
        mc_dimension = self.multi_comm.group_name_to_tolopogy_dimension(comm_name)
        return self.multi_comm.task_cartesian_coordinates[mc_dimension]

    def all_reduce(self, buffer, *, op: Literal["sum", "prod", "max", "min"] = "sum"):
        """Performs an all reduce operation

        Args:
            buffer: the buffer to reduce
            op: the operation to perform. Can be "sum", "prod" (product), "max", "min"

        Returns:
            The reduced buffer

        Raises:
            ValueError: If the operation is not supported
        """
        # Perform the operation
        if MPI_AVAILABLE and self.comm is not None:
            # First, determine which operation to perform
            if op == "sum":
                mpi_op = MPI.SUM
            elif op == "prod":
                mpi_op = MPI.PROD
            elif op == "max":
                mpi_op = MPI.MAX
            elif op == "min":
                mpi_op = MPI.MIN
            else:
                raise ValueError(f"Unknown MPI operation {op}")
            self.comm.Allreduce(MPI.IN_PLACE, buffer, op=mpi_op)
        return buffer
