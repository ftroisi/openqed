# Copyright 2026 Francesco Troisi
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

from typing import Literal, Dict

import numpy as np
import numpy.typing as npt

from openqed.grid.grid import Grid
from openqed.grid.real_space_grid import RealSpaceGrid
from openqed.hamiltonians.matter_hamiltonian import MatterHamiltonian
from openqed.hamiltonians.terms.hamiltonian_term import HamiltonianTerm
from openqed.hamiltonians.terms.moire_potential_term import MoirePotentialTerm

from openqed.hamiltonians.types.bilayer_excitons import BilayerExcitons, BilayerStructure, excitons_key_to_structure_key

class MoireExcitonHamiltonian(MatterHamiltonian):
    """
    This class is the Moiré Exciton Hamiltonian class. A Moiré exciton is a quasi particle that
    arises in a multilayered structure.

    References:
        [1] Tunable Phases of Moiré Excitons in van der Waals Heterostructures:
            https://pubs.acs.org/doi/10.1021/acs.nanolett.0c03019
    """
    def __init__(self,
                terms: list[str],
                 *,
                grid: Grid,
                wannier_states: BilayerExcitons,
            ) -> None:
        """Initialize the Moiré Exciton Hamiltonian

        Args:
            name: The name of the Hamiltonian.
            terms: a list of strings representing the terms of the Hamiltonian.
            exciton: The exciton for which the analytical screened potential is computed. It is a string
                with the format `li_lj`, where `i` and `j` are numbers and the whole string is a key of
                `:class:BilayerExcitons`. For instance, specifying `l1_l1` will compute the analytical
                screened potential for the intralayer exciton of the first layer
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
        # Save the wannier states
        self.wannier_states = wannier_states
        if len(wannier_states) != 1:
            raise ValueError("Only one exciton type is supported")
        # Initialize the Hamiltonian terms
        self._moire_potential_term: Dict[str, MoirePotentialTerm] | None = None

    @property
    def moire_potential_term(self) -> Dict[str, MoirePotentialTerm]:
        """"The Moiré potential term."""
        if self._moire_potential_term is None:
            raise ValueError("The Moiré potential term must be set before being used")
        return self._moire_potential_term

    @moire_potential_term.setter
    def moire_potential_term(self, moire_potential_term: Dict[str, MoirePotentialTerm]) -> None:
        self._moire_potential_term = moire_potential_term

    def get_form_factor(self,
                        psi: npt.NDArray[np.complex128],
                        q_idx: int) -> tuple[np.complex128, np.complex128]:
        """
        This method computes for the form factor appearing in Eq. 16 of the Supporting Info of [1].
        The idea is to shift the vector psi (which is known on the k-k') base
        by the vector q (a vector on the kpt grid).

        Args:
            psi: the wavefunction of the exciton
            q_idx: the index of the q vector for which the form factor is computed

        Returns:
            A tuple where each element is the form factor at the request k point:
            [F(q) = sum_k conj(psi(k))*psi(k+q), F^*(q) = sum_k psi(k)*conj(psi(k+q))]
        """
        flat_k_grid = self.grid.flat_grid()
        size: int = np.size(flat_k_grid, 0)
        form_factor: np.complex128 = np.complex128(0)
        form_factor_conj: np.complex128 = np.complex128(0)
        # Define the k + q grid
        flat_k_p_q_grid = flat_k_grid[:, :] + flat_k_grid[q_idx, :]
        # Loop over the k grid to compute the form factor
        for i in range(size):
            if np.abs(psi[i]) < 1e-9:
                continue
            # Find the indexes for which the k+q grid matches the original k grid
            idx = self.grid.get_point_index(flat_k_p_q_grid[i])
            if np.size(idx) == 1:
                # Compute the form factor
                form_factor += np.conj(psi[i]) * psi[idx]
                form_factor_conj += psi[i] * np.conj(psi[idx])
        return form_factor, form_factor_conj

    def get_moire_hamiltonian(self) -> npt.NDArray[np.complex128]:
        # 0. Create a matrix with the dimension of the grid
        flat_k_grid = self.grid.flat_grid()
        k_grid_size = np.size(flat_k_grid, 0)
        hamiltonian: npt.NDArray[np.complex128] =\
            np.zeros((k_grid_size, k_grid_size), dtype=np.complex128)
        # 1. First, identify for what layers the Moiré potential should be computed
        exciton_type = list(self.wannier_states.keys())[0]
        layer1 = excitons_key_to_structure_key(exciton_type.split("_")[0])
        layer2 = excitons_key_to_structure_key(exciton_type.split("_")[1])
        moire_potential_term_layer1 = self.moire_potential_term.get(layer1)
        moire_potential_term_layer2 = self.moire_potential_term.get(layer2)
        if (moire_potential_term_layer1 is None or moire_potential_term_layer2 is None):
            raise ValueError("The Moiré potential term must be set for both layers before being used")
        # 2. Compute the Moire potential and determine the q vectors
        moire_potential_layer1 = moire_potential_term_layer1.get_hamiltonian_term()
        moire_potential_layer2 = moire_potential_term_layer2.get_hamiltonian_term()
        moire_indexes = np.where(np.abs(moire_potential_layer1) > 1e-12)[0]
        n_points = int(int(moire_potential_term_layer1.moire_symmetry[1]) / 2)
        if np.size(moire_indexes) != n_points:
            raise ValueError("The number of points in the Moiré potential is not correct")
        # 3. Compute the Hamiltonian
        for moire_index in moire_indexes:
            # 3.1 Compute the form factor for the exciton
            form_factor, form_factor_conj = self.get_form_factor(
                self.wannier_states[exciton_type], moire_index)
            # 3.2 Compute the Moire potential
            moire_potential = (moire_potential_layer1[moire_index] * form_factor -
                               moire_potential_layer2[moire_index] * form_factor_conj)
            # 3.3 Find the index of the first point
            diagonal_index = self.grid.get_point_index(flat_k_grid[moire_index])
            hamiltonian += (np.diag(moire_potential, k=diagonal_index) +
                            np.diag(np.conj(moire_potential), k=-diagonal_index))
        return hamiltonian

    def get_hamiltonian(self) -> npt.NDArray[np.complex128]:
        # 0. Create a matrix with the dimension of the grid
        flat_k_grid = self.grid.flat_grid()
        k_grid_size = np.size(flat_k_grid, 0)
        hamiltonian: npt.NDArray[np.complex128] =\
            np.zeros((k_grid_size, k_grid_size), dtype=np.complex128)
        # 1. First, identify for what layers the Moiré potential should be computed
        exciton_type = list(self.wannier_states.keys())[0]
        layer1 = excitons_key_to_structure_key(exciton_type.split("_")[0])
        layer2 = excitons_key_to_structure_key(exciton_type.split("_")[1])
        moire_potential_term_layer1 = self.moire_potential_term.get(layer1)
        moire_potential_term_layer2 = self.moire_potential_term.get(layer2)
        if (moire_potential_term_layer1 is None or moire_potential_term_layer2 is None):
            raise ValueError("The Moiré potential term must be set for both layers before being used")
        # 2. Compute the Moiré potential and determine the q vectors
        moire_potential_layer1 = moire_potential_term_layer1.get_hamiltonian_term()
        moire_potential_layer2 = moire_potential_term_layer2.get_hamiltonian_term()
        moire_indexes = np.where(np.abs(moire_potential_layer1) > 1e-12)[0]
        n_points = int(int(moire_potential_term_layer1.moire_symmetry[1]) / 2)
        if np.size(moire_indexes) != n_points:
            raise ValueError("The number of points in the Moiré potential is not correct")
        # 3. Compute the Hamiltonian
        for _, moire_index in enumerate(moire_indexes):
            # 3.1 Compute the form factor for the exciton
            form_factor, form_factor_conj = self.get_form_factor(
                self.wannier_states[exciton_type], moire_index)
            # 3.2 Compute the Moiré potential
            moire_potential = (moire_potential_layer1[moire_index] * form_factor -
                               moire_potential_layer2[moire_index] * form_factor_conj)
            # 3. Compute the Hamiltonian
            # Find the index of the first point
            diagonal_index = self.grid.get_point_index(flat_k_grid[moire_index])
            hamiltonian += (np.diag(moire_potential, k=diagonal_index) +
                            np.diag(np.conj(moire_potential), k=-diagonal_index))
        return hamiltonian
