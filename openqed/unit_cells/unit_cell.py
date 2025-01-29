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
from openqed.grid.grid import GridType

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
