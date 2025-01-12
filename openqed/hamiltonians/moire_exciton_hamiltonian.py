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
            grid: The grid object to use. Note that the Moiré Exciton Hamiltonian will be represented
                on this grid.
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
        for term in terms:
            if term == "free_exciton":
                if electron_effective_mass is None or hole_effective_mass is None:
                    raise ValueError("The effective masses for electrons and holes must be defined")

    def get_matter_hamiltonian(self, **kwargs) -> npt.NDArray[np.float64 | np.complex128]:
        """This method generates the uncoupled matter hamiltonian for this class. It is the term that
        describes the matter system without the radiation field
        """
        h_mat_size: int = 1 + np.size(self.grid.flat_grid(), 0) * self.n_exc * n_exc_types
        h_mat = np.zeros((h_mat_size, h_mat_size), dtype=np.complex128)
        # CASE 1: the user is providing the free exciton hamiltonian and the moire hamiltonian
        if h_free is not None and 'free_exc' in self.terms:
            # check that the dimensions are consistent
            if np.shape(h_free)[0] != h_mat_size:
                raise ValueError("The free exciton hamiltonian has the wrong dimensions. " +
                                 f"Detected: {np.shape(h_free)}; expected: {h_mat_size}")
            h_mat += np.diag(h_free)
        if h_moire is not None and 'moire' in self.hamiltonian_terms:
            if np.shape(h_moire) != (h_mat_size, h_mat_size):
                raise ValueError("The Moire hamiltonian has the wrong dimensions. Detected:" +
                                 f"{np.shape(h_moire)}; expected: {(h_mat_size, h_mat_size)}")
            h_mat += h_moire
        if not spl.ishermitian(h_mat):
            raise ValueError("H_mat is not Hermitian")
        if np.linalg.norm(h_mat) > 1e-5:
            return h_mat
        # Reset matrix to avoid numerical noise
        h_mat = np.zeros((h_mat_size, h_mat_size), dtype=np.complex128)
        # CASE 2: the user is providing the wannier eigenvalues and eigenstates and the theta index
        if th_idx is not None and wannier_eigenval is not None and wannier_eigenvec is not None:
            if (len(wannier_eigenval) != n_exc_types or len(wannier_eigenvec) != n_exc_types):
                raise ValueError("The number of exciton types is not consistent with the " +
                                 "number of excitons in the dictionary")
            # Build the free exciton hamiltonian
            if 'free_exc' in self.hamiltonian_terms:
                h_mat += np.diag(self.build_free_exc_h(wannier_eigenval=wannier_eigenval))
            # Build the Moire hamiltonian
            if 'moire' in self.hamiltonian_terms:
                h_mat += self.build_moire_h(wannier_eigenvec=wannier_eigenvec, th_idx=th_idx)
            if not spl.ishermitian(h_mat):
                raise ValueError("H_mat is not Hermitian")
            return h_mat
        raise ValueError("You must either pass the free exciton H and the moire H, " +
                         "or the wannier eigenvalues and eigenstates and the theta index")

    def get_hamiltonian(self, **kwargs) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]] | np.ndarray[tuple[int, ...], np.dtype[np.complex128]]:
        # 0. Create a matrix of dimension h_mat + 1 (for the GS)
        # 1. Call get matter hamiltonian method
        raise NotImplementedError("This method is not implemented yet")