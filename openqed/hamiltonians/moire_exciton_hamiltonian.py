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

"""This module contains the Moiré Hamiltonian class"""

import numpy as np
from numpy._typing import NDArray
import numpy.typing as npt

from openqed.grid.grid import Grid
from openqed.grid.real_space_grid import RealSpaceGrid
from openqed.hamiltonians.matter_hamiltonian import MatterHamiltonian
from openqed.hamiltonians.terms.hamiltonian_term import HamiltonianTerm

class MoireExcitonHamiltonian(MatterHamiltonian):
    """
    This class is the Moiré Exciton Hamiltonian class. A Moiré exciton is a quasi particle that
    arises in a multilayered strucutre.
    """
    def __init__(self,
                terms: list[str],
                 *,
                grid: Grid
            ) -> None:
        """Initialize the Moiré Exciton Hamiltonain

        Args:
            name: The name of the Hamiltonian.
            terms: a list of strings representing the terms of the Hamiltonian.
            grid: The grid object to use. The Moiré Exciton Hamiltonian will be represented on this grid.
            electron_effective_mass: The effective mass of the electrons in the bilayer structure.
            hole_effective_mass: The effective mass of the holes in the bilayer structure.
            thicknesses: The thicknesses of the layers in the bilayer structure.
            dielectric_constants: The dielectric constants of the layers in the bilayer structure.
        """
        # TODO: Add RealSpaceGrid support
        if isinstance(grid, RealSpaceGrid):
            raise NotImplementedError("RealSpaceGrid is not supported yet")
        # First, initialize the base Hamiltonian class
        super().__init__("Moiré Exciton", terms, grid=grid)
        # Initialize the Hamiltonian terms
        self.term_to_instance: dict[str, HamiltonianTerm] = {}

    def get_hamiltonian(self, **kwargs) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]] | np.ndarray[tuple[int, ...], np.dtype[np.complex128]]:
        # 0. Create a matrix of dimension h_mat + 1 (for the GS)
        # 1. Call get matter hamiltonian method
        raise NotImplementedError("This method is not implemented yet")