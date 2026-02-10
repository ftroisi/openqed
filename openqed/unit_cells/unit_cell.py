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

"""This module contains the base abstract UnitCell class"""

from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt
from openqed.grid.grid import GridType, Grid

class UnitCell(ABC):
    """This class is the abstract UnitCell class. All modules in the `unit_cells` folder"""
    def __init__(self, dimension: int) -> None:
        """Initialize the UnitCell class

        Args:
            dimension: The dimension of the unit cell.
        """
        if dimension not in [1, 2, 3]:
            raise ValueError("Unit cells are only supported in 1, 2, or 3 dimensions")
        self.dimension = dimension

    @abstractmethod
    def get_lattice_vectors(self, space: GridType) -> npt.NDArray[np.float64]:
        """This method computes the lattice vectors of the unit cell in k space.

        Args:
            space: The space in which the lattice vectors are computed.

        Returns:
            The lattice vectors of the unit cell in the desired space.
        """
        pass

    def fit_lattice_vectors_to_grid(self, grid: Grid) -> npt.NDArray[np.float64]:
        """Fit the lattice vectors to stay within the grid limits.

        This method scales the unit cell lattice vectors so that their magnitude equals
        the radius of the largest circle inscribed in the grid. This ensures that lattice
        vectors remain within grid boundaries even after rotation, which is essential for
        operations like constructing Moiré superlattices or computing twisted bilayer
        structures.

        The scaling preserves the direction and relative proportions of all lattice vectors
        while ensuring they fit commensurate with the computational grid.

        Args:
            grid: The computational grid (momentum or real space) to which the lattice
                vectors should be fitted. The grid must be 2-dimensional.

        Returns:
            Scaled lattice vectors as an array of shape (n_vectors, 2), where each vector
            has magnitude equal to the inscribed circle radius of the grid. The returned
            vectors are in the same space (momentum/real) as the input grid.

        Note:
            This method scales all lattice vectors uniformly by the ratio of the inscribed
            circle radius to the magnitude of the first lattice vector. For rotation-invariant
            operations, this ensures that any rotated lattice vector remains inside the grid.
        """
        radius = grid.get_max_inscribed_circle_radius()
        lattice_vectors = self.get_lattice_vectors(space=grid.grid_type)
        scaling_factor = radius / np.sqrt(np.sum(lattice_vectors[0] ** 2))
        return lattice_vectors * scaling_factor

