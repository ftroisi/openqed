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

"""This module contains the Mott-Wannier Hamiltonian class"""

from typing import get_type_hints, cast, Dict, Literal
from enum import Enum
from collections import OrderedDict

import numpy as np
import numpy.typing as npt

from openqed.grid.k_space_grid import KSpaceGrid
from openqed.hamiltonians.matter_hamiltonian import MatterHamiltonian, MatterHamiltonianFormalism
from openqed.hamiltonians.terms.hamiltonian_term import HamiltonianTerm
from openqed.hamiltonians.terms.kinetic_term import KineticTerm
from openqed.hamiltonians.terms.effective_2d_coulomb import Effective2DCoulombPotential
from openqed.hamiltonians.types.bilayer_excitons import (
    BilayerExcitons, BilayerStructure, structure_key_to_exciton_key)

class MottWannierHamiltonianTerms(Enum):
    KINETIC = "kinetic_term"
    EFFECTIVE_2D_COULOMB = "effective_2d_coulomb"

class MottWannierHamiltonian(MatterHamiltonian):
    """
    This class is the Mott-Wannier Hamiltonian class. It inherits from the base Hamiltonian class.
    """
    def __init__(self,
                 mode: MatterHamiltonianFormalism,
                 terms: list[str],
                 exciton: str,
                 *,
                 grid: KSpaceGrid,
                 electron_effective_mass: BilayerStructure | None = None,
                 hole_effective_mass: BilayerStructure | None = None,
                 thicknesses: BilayerStructure | None = None,
                 dielectric_constants: BilayerStructure | None = None
                 ) -> None:
        """Initialize the Mott-Wannier Hamiltonian

        Args:
            mode: The construction mode of the Hamiltonian. Must be one of ``"many_body"`` or ``"tight_binding"``. In
                ``"many_body"`` mode :meth:`get_hamiltonian` returns the full ``(N, N)`` matrix over all k-points. In
                ``"tight_binding"`` mode it returns a stacked array of shape ``(n_k, n_bands, n_bands)``,
                 one small matrix per k-point.
            terms: a list of strings representing the terms of the Hamiltonian.
            exciton: The exciton for which the analytical screened potential is computed. It is a string
                with the format `li_lj`, where `i` and `j` are numbers and the whole string is a key of
                `:class:BilayerExcitons`. For instance, specifying `l1_l1` will compute the analytical
                screened potential for the intralayer exciton of the first layer
            grid: The grid object to use. Note that the Mott-Wannier Hamiltonian will be represented
                on this grid. TODO: Add RealSpaceGrid support
            electron_effective_mass: The effective mass of the electrons in the bilayer structure.
            hole_effective_mass: The effective mass of the holes in the bilayer structure.
            thicknesses: The thicknesses of the layers in the bilayer structure. Their values must be in atomic units.
            dielectric_constants: The dielectric constants of the layers in the bilayer structure.
        """
        # First, initialize the base Hamiltonian class
        super().__init__("Mott-Wannier", terms, mode=mode, grid=grid)
        self.term_to_instance: "OrderedDict[MottWannierHamiltonianTerms, HamiltonianTerm]" = OrderedDict()
        self.exciton = exciton
        self.effective_masses = self._get_effective_mass(
            electron_effective_mass=electron_effective_mass,
            hole_effective_mass=hole_effective_mass)
        # Initialize the Hamiltonian terms
        for term in [MottWannierHamiltonianTerms(t) for t in terms]:
            if term == MottWannierHamiltonianTerms.KINETIC:
                self.term_to_instance[term] = \
                    KineticTerm(self, mass=self.effective_masses.get(self.exciton, np.float64(0.0)))
            elif term == MottWannierHamiltonianTerms.EFFECTIVE_2D_COULOMB:
                self.term_to_instance[term] = \
                    Effective2DCoulombPotential(self,
                                                exciton=exciton,
                                                thicknesses=thicknesses,
                                                dielectric_constants=dielectric_constants)

    @property
    def exciton(self) -> str:
        """
        The exciton for which the screened potential is computed analytically. It is a string with the
        format `li_lj`, where `i` and `j` are numbers, and the whole string is a key of
        `:class:BilayerExcitons`. For instance, specifying `l1_l1` will compute the analytical screened
        potential for the intralayer exciton of the first layer
        """
        return self._exciton

    @exciton.setter
    def exciton(self, exciton: str) -> None:
        if exciton not in get_type_hints(BilayerExcitons):
            raise ValueError(
                f"Unknown exciton type: {exciton}. Supported types: {get_type_hints(BilayerExcitons)}")
        # Update the Hamiltonian terms that depend on the exciton.
        for term, instance in self.term_to_instance.items():
            if term == MottWannierHamiltonianTerms.KINETIC:
                # Update the mass of the kinetic term
                cast(KineticTerm, instance).mass = self.effective_masses.get(exciton, 0)
            elif term == MottWannierHamiltonianTerms.EFFECTIVE_2D_COULOMB:
                # Update the exciton for the effective 2D Coulomb potential
                cast(Effective2DCoulombPotential, instance).exciton = exciton
        self._exciton = exciton

    @staticmethod
    def _get_effective_mass(
            electron_effective_mass: BilayerStructure,
            hole_effective_mass: BilayerStructure
    ) -> BilayerExcitons:
        """Get the effective mass of the electron and hole for a given exciton.

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

        # Build the effective masses for all possible excitons
        exciton_effective_mass: BilayerExcitons = {}
        for elec_key in electron_effective_mass.keys():
            l1 = structure_key_to_exciton_key(elec_key)
            for hole_key in hole_effective_mass.keys():
                l2 = structure_key_to_exciton_key(hole_key)
                exciton_effective_mass[f"{l1}_{l2}"] = \
                    (1 / electron_effective_mass[elec_key] + 1 / hole_effective_mass[hole_key]) ** (-1)

        return exciton_effective_mass

    def _get_many_body_hamiltonian(self) -> npt.NDArray[np.complex128]:
        """Build the Mott-Wannier Hamiltonian as a single ``(N, N)`` matrix.

        See equation (13) of the supplementary information of
        https://pubs.acs.org/doi/10.1021/acs.nanolett.0c03019?fig=fig4&ref=pdf

        Returns:
            The Hamiltonian that describes the exciton wavefunction for the chosen exciton
        """
        # First compute mesh size
        mesh_size = np.size(self.grid.flat_grid(), 0)
        hamiltonian: npt.NDArray[np.float64] = np.zeros((mesh_size, mesh_size), dtype=np.float64)
        # Then, get the different terms
        for term, instance in self.term_to_instance.items():
            if term == MottWannierHamiltonianTerms.KINETIC:
                hamiltonian += np.diag(instance.get_hamiltonian_term())
            elif term == MottWannierHamiltonianTerms.EFFECTIVE_2D_COULOMB:
                hamiltonian += instance.get_hamiltonian_term()

        return hamiltonian

    def get_hamiltonian_size(self) -> int:
        """This method returns the size of the MottWannier Hamiltonian matrix"""
        return np.size(self.grid.flat_grid(), 0)

    def get_interaction_hamiltonian(self,
                                    light_momentum: npt.NDArray[np.float64],
                                    light_polarization: Literal['x', 'y', 'z', 'xy', 'xz', 'yz', '+', '-']
        ) -> Dict[Literal['x', 'y', 'z'], npt.NDArray[np.float64 | np.complex128]]:
        raise NotImplementedError()
