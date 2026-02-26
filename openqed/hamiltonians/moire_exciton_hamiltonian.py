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

from typing import List, Literal, Dict
from enum import Enum
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from openqed.grid.grid import Grid
from openqed.grid.real_space_grid import RealSpaceGrid
from openqed.hamiltonians.matter_hamiltonian import MatterHamiltonian, MatterHamiltonianFormalism
from openqed.hamiltonians.terms.hamiltonian_term import HamiltonianTerm
from openqed.hamiltonians.terms.kinetic_term import KineticTerm
from openqed.hamiltonians.terms.free_exciton_term import FreeExcitonTerm
from openqed.hamiltonians.terms.moire_potential_term import MoirePotentialTerm, MoirePotentialParameters
from openqed.hamiltonians.types import ModePolarization

from openqed.hamiltonians.types.bilayer_excitons import BilayerExcitons, BilayerStructure, structure_key_to_exciton_key

class MoireExcitonHamiltonianTerms(Enum):
    FREE_EXCITON = "free_exciton"
    MOIRE_POTENTIAL = "moire_potential"

@dataclass
class MoireExcitonsMomentumMatrixElements:
    """This class contains the momentum matrix elements for the Moiré excitons."""
    x: BilayerExcitons | None
    y: BilayerExcitons | None
    z: BilayerExcitons | None

class MoireExcitonHamiltonian(MatterHamiltonian):
    """
    This class is the Moiré Exciton Hamiltonian class. A Moiré exciton is a quasi particle that
    arises in a multilayered structure.

    References:
        [1] Tunable Phases of Moiré Excitons in van der Waals Heterostructures
            https://pubs.acs.org/doi/10.1021/acs.nanolett.0c03019
        [2] Cavity-QED-controlled two-dimensional Moiré excitons without twisting
            https://www.nature.com/articles/s41467-025-67570-2
    """
    def __init__(self,
                 terms: list[str],
                 *,
                 grid: Grid,
                 wannier_states: BilayerExcitons,
                 wannier_binding_energies: BilayerExcitons,
                 band_gap: BilayerExcitons,
                 moire_matrix_elements: MoireExcitonsMomentumMatrixElements,
                 electron_effective_mass: BilayerStructure | None = None,
                 hole_effective_mass: BilayerStructure | None = None,
                 moire_parameters: MoirePotentialParameters,
                 ) -> None:
        """Initialize the Moiré Exciton Hamiltonian

        Args:
            terms: a list of strings representing the terms of the Hamiltonian.
            grid: The grid object to use. The Moiré Exciton Hamiltonian will be represented on this grid.
            wannier_states: A dictionary mapping exciton types (e.g., 'l1_l1', 'l1_l2') to their Wannier states.
            wannier_binding_energies: A dictionary mapping exciton types (e.g., 'l1_l1', 'l1_l2') to their Wannier
                binding energies (eigenvalues of the Wannier equation).
            electron_effective_mass: The effective mass of the electrons in the bilayer structure.
            hole_effective_mass: The effective mass of the holes in the bilayer structure.
        """
        # TODO: Add RealSpaceGrid support
        if isinstance(grid, RealSpaceGrid):
            raise NotImplementedError("RealSpaceGrid is not supported yet")
        # First, initialize the base Hamiltonian class
        super().__init__("Moiré Exciton", terms, mode=MatterHamiltonianFormalism.MANY_BODY, grid=grid)
        # Save the wannier states
        self.wannier_states = wannier_states
        if len(wannier_states) != 1:
            raise ValueError("Only one exciton type is supported")
        self.exciton: str = next(k for k, v in wannier_states.items() if v is not None)
        if self.wannier_states.get(self.exciton, np.array([])).ndim != 2:
            raise ValueError("The Wannier states must be 2D arrays with shape (n_k_points, n_exciton_states)")
        self.number_of_excitonic_states = wannier_states.get(self.exciton, np.array([])).shape[1]
        # Save the moire matrix elements
        self.matrix_elements = moire_matrix_elements
        # Initialize the Hamiltonian terms
        self.terms_to_instances: OrderedDict[MoireExcitonHamiltonianTerms, HamiltonianTerm] = OrderedDict()
        for term in [MoireExcitonHamiltonianTerms(t) for t in terms]:
            if term == MoireExcitonHamiltonianTerms.FREE_EXCITON:
                kinetic_term = KineticTerm(self, mass=self._get_total_mass(
                            electron_effective_mass, hole_effective_mass).get(self.exciton, np.float64(0.0)))
                self.terms_to_instances[term] = FreeExcitonTerm(self,
                                                                kinetic_term=kinetic_term,
                                                                wannier_binding_energies=wannier_binding_energies,
                                                                band_gap=band_gap)
            elif term == MoireExcitonHamiltonianTerms.MOIRE_POTENTIAL:
                self.terms_to_instances[term] = MoirePotentialTerm(self,
                                                                   wannier_eigenvectors=wannier_states,
                                                                   wannier_eigenvalues=wannier_binding_energies,
                                                                   moire_parameters=moire_parameters)

    @staticmethod
    def _get_total_mass(
            electron_effective_mass: BilayerStructure,
            hole_effective_mass: BilayerStructure
    ) -> BilayerExcitons:
        """Get the total mass of the electron and hole for a given exciton.

        Args:
            electron_effective_mass: The effective mass of electrons in the bilayer structure.
                Must contain at least 'layer1' and/or 'layer2' keys.
            hole_effective_mass: The effective mass of holes in the bilayer structure.
                Must have the same keys as electron_effective_mass.

        Returns:
            A dictionary mapping exciton types (e.g., 'l1_l1', 'l1_l2') to their reduced masses.

        Raises:
            ValueError: If the two structures don't have matching keys.

        References:
            https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.53.1391
        """
        if electron_effective_mass.keys() != hole_effective_mass.keys():
            raise ValueError("The effective masses for electrons and holes must be defined for the same layers")

        # Build the total masses for all possible excitons
        exciton_total_mass: BilayerExcitons = {}
        for elec_key in electron_effective_mass.keys():
            l1 = structure_key_to_exciton_key(elec_key)
            for hole_key in hole_effective_mass.keys():
                l2 = structure_key_to_exciton_key(hole_key)
                exciton_total_mass[f"{l1}_{l2}"] = electron_effective_mass[elec_key] + hole_effective_mass[hole_key]

        return exciton_total_mass

    def _get_many_body_hamiltonian(self) -> npt.NDArray[np.complex128]:
        """
        Build the Moiré Excitons Hamiltonian as a single ``(N, N)`` matrix.

        See Eq. 14 and 15 of the supplementary information of Ref. [1].

        Returns:
            The Hamiltonian that describes the Moiré excitons
        """
        # First compute mesh size
        hamiltonian_size = self.get_hamiltonian_size()
        hamiltonian: npt.NDArray[np.complex128] = np.zeros((hamiltonian_size, hamiltonian_size), dtype=np.complex128)
        # Then, get the different terms
        for term, instance in self.terms_to_instances.items():
            if term == MoireExcitonHamiltonianTerms.FREE_EXCITON:
                hamiltonian += np.diag(instance.get_hamiltonian_term())
            elif term == MoireExcitonHamiltonianTerms.MOIRE_POTENTIAL:
                hamiltonian += instance.get_hamiltonian_term()

        return hamiltonian

    def get_hamiltonian_size(self) -> int:
        """This method returns the size of the Moiré excitons Hamiltonian matrix"""
        mesh_size = np.size(self.grid.flat_grid(), 0)
        number_of_excitonic_states = self.number_of_excitonic_states
        hamiltonian_size = mesh_size * number_of_excitonic_states
        return hamiltonian_size

    def get_interaction_hamiltonian(self,
                                    light_momentum: npt.NDArray[np.float64],
                                    light_polarization: ModePolarization
        ) -> Dict[Literal['x', 'y', 'z'], npt.NDArray[np.float64 | np.complex128]]:
        if light_momentum.ndim != 1 or light_momentum.shape[0] != self.grid.dimensions:
            raise ValueError("The light momentum must be a 1D array with the dimensions of the grid")
        # Compute some grid parameters
        kgr_size: int = np.size(self.grid.flat_grid(), 0)
        size: int = self.get_hamiltonian_size()
        spacing = self.grid.get_spacing()
        # Interaction Hamiltonian dimensions ('1 + ' is for the polaritonic GS):
        # (1 + number_of_excitonic_states * grid_size, 1 + number_of_excitonic_states * grid_size)
        hamiltonian = dict()
        dipole_0n: Dict[str, List[np.complex128]] = dict()
        for pol in light_polarization.value:
            hamiltonian[pol] = np.zeros((1 + size, 1 + size), dtype=np.complex128)
            dipole_0n[pol] = []

        # Calculating Matrix Elements Envelope Functions \Psi_0 -> \Psi_n
        # the only non-zero matrix elements are the ones at Q=0

        # Loop over the number of excitons we want to include
        for i in range(self.number_of_excitonic_states):
            # Evaluate f for transitions from GS to n: see eq 7 and 8 of supporting info of
            # https://pubs.acs.org/doi/full/10.1021/acs.nanolett.9b00183
            # => Sum on all k point for a specific exciton: (1/n_kpt) * sum_k(wfn(k, exc))
            f_0 = (1 / kgr_size) * np.sum(self.wannier_states.get(self.exciton)[:, i])
            for pol in light_polarization.value:
                dipole_0n[pol].append(getattr(self.matrix_elements, pol).get(self.exciton) * f_0)

        # Find the index of the photon momentum in the grid
        q0_idx = self.grid.get_point_index(light_momentum) + 1
        # Polaritonic ground state to n-th polariton transition
        for pol in light_polarization.value:
            hamiltonian[pol][0, q0_idx::kgr_size] = np.array(dipole_0n[pol][:]).conj()
            hamiltonian[pol][q0_idx::kgr_size, 0] = np.array(dipole_0n[pol][:])
        # Now account for moire-like effect of finite q photon
        if not np.isclose(np.linalg.norm(light_momentum), 0):
            # ! Note well, this will only work for the first layer
            form_factor_vv, form_factor_cc = \
                self.get_form_factor(self.wannier_states[self.exciton][:, 0], [q0_idx])
            # See equation 9 of the supporting info of the paper
            # https://pubs.acs.org/doi/full/10.1021/acs.nanolett.9b00183
            finite_q_me: npt.NDArray[np.complex128] = \
                np.abs(light_momentum) * (form_factor_cc[0] + form_factor_vv[0]) / self.eff_mass[self.exciton]
            for k in range(kgr_size):
                # Define the Q + q_ph grid and get the index of the corresponding element.
                Q_plus_q = self.grid.flat_grid()[k, :] + light_momentum
                idx = np.where(np.linalg.norm(self.grid.flat_grid() - Q_plus_q, axis=1) < spacing)[0]
                if len(idx) == 0:
                    continue
                if len(idx) > 1:
                    raise ValueError("The index of the Q + q grid is not unique")
                # +1 because the first element is the GS
                i: int = 1 + k
                j: int = 1 + idx[0]
                me: npt.NDArray[np.complex128] = np.conj(finite_q_me) if i > j else finite_q_me
                for pol in light_polarization.value:
                    hamiltonian[pol][i, j] = me[0 if pol == 'x' else 1]

        return hamiltonian
