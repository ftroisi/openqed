# Copyright 2025 Francesco Troisi
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

"""This module contains the ExagonalUnitCell class"""

import numpy as np
import numpy.typing as npt
from openqed.grid.grid import GridType
from openqed.unit_cells.unit_cell import UnitCell

class HexagonalUnitCell(UnitCell):
    """
    This class represents the HexagonalUnitCell. A hexagonal unit cell is one of the possible 2D
    Bravais lattices. It is characterized by a single lattice parameter (the distance between two atoms),
    and it has a 120° angle between the lattice vectors.
    """
    def __init__(self, lattice_parameter: np.float64) -> None:
        """Initialize the HexagonalUnitCell class

        Args:
            lattice_parameter: The lattice parameter in real space. Its value must be in atomic units.
        """
        super().__init__(dimension=2)
        if lattice_parameter < 0 or np.isclose(lattice_parameter, 0):
            raise ValueError("The lattice parameter must be positive and non-zero")
        self.a = lattice_parameter

    @property
    def lattice_parameter(self) -> np.float64:
        """The lattice parameter."""
        return self.a

    def get_lattice_vectors(self, space: GridType) -> npt.NDArray[np.float64]:
        """This method computes the hexagonal cell in k space of a 2D material.

        Returns:
            The lattice vectors of hexagonal cell in k space
        """
        real_space_lattice = [
            [self.a, 0],
            [self.a / 2, self.a * np.sqrt(3) / 2]
        ]
        if space == GridType.REAL_SPACE:
            return np.array(real_space_lattice, dtype=np.float64)
        return np.array(2 * np.pi * np.linalg.inv(real_space_lattice).T, dtype=np.float64)
