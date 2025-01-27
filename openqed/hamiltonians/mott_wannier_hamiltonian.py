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

"""This module contains the Mott-Wannier Hamiltonian class"""

from typing import get_args, cast
import numpy as np
import numpy.typing as npt
import scipy.sparse.linalg as ssl

from openqed.grid.k_space_grid import KSpaceGrid
from openqed.hamiltonians.matter_hamiltonian import MatterHamiltonian
from openqed.hamiltonians.terms.hamiltonian_term import HamiltonianTerm
from openqed.hamiltonians.terms.kinetic_term import KineticTerm
from openqed.hamiltonians.terms.effective_2d_coulomb import Effective2DCoulombPotential
from openqed.hamiltonians.types.bilayer_excitons import (
    BilayerExcitons, BilayerStructure, structure_key_to_exciton_key)

class MottWannierHamiltonian(MatterHamiltonian):
    """
    This class is the Mott-Wannier Hamiltonian class. It inherits from the base Hamiltonian class.
    """
    def __init__(self,
                terms: list[str],
                exciton: str,
                 *,
                grid: KSpaceGrid,
                electron_effective_mass: BilayerStructure | None = None,
                hole_effective_mass: BilayerStructure | None = None,
                thicknesses: BilayerStructure | None = None,
                dielectric_constants: BilayerStructure | None = None
            ) -> None:
        """Initialize the Mott-Wannier Hamiltonain

        Args:
            name: The name of the Hamiltonian.
            terms: a list of strings representing the terms of the Hamiltonian.
            exciton: The exciton for which the analytical screened potential is computed. It is a string
                with the format `li_lj`, where `i` and `j` are numbers and the whole string is a key of
                `:class:BilayerExcitons`. For instance, specifying `l1_l1` will compute the analytical
                screened potential for the intralayer exciton of the first layer
            grid: The grid object to use. Note that the Mott-Wannier Hamiltonian will be represented
                on this grid. TODO: Add RealSpaceGrid support
            electron_effective_mass: The effective mass of the electrons in the bilayer structure.
            hole_effective_mass: The effective mass of the holes in the bilayer structure.
            thicknesses: The thicknesses of the layers in the bilayer structure.
            dielectric_constants: The dielectric constants of the layers in the bilayer structure.
        """
        # First, initialize the base Hamiltonian class
        super().__init__("Mott-Wannier", terms, grid=grid)
        self.exciton = exciton
        self.effective_masses = self._get_effective_mass(
            electron_effective_mass=electron_effective_mass,
            hole_effective_mass=hole_effective_mass)
        # Initialize the Hamiltonian terms
        self.term_to_instance: dict[str, HamiltonianTerm] = {}
        for term in terms:
            if term == "kinetic_term":
                self.term_to_instance[term] = \
                    KineticTerm(self, mass=self.effective_masses.get(exciton, 0))
            elif term == "effective_2d_coulomb":
                if thicknesses is None or dielectric_constants is None:
                    raise ValueError("The thicknesses and dielectric constants must be defined")
                self.term_to_instance[term] = \
                    Effective2DCoulombPotential(self,
                                                exciton=exciton,
                                                thicknesses=thicknesses,
                                                dielectric_constants=dielectric_constants)

    @property
    def exciton(self) -> str:
        """
        The exciton for which the analytical screened potential is computed. It is a string with the
        format `li_lj`, where `i` and `j` are numbers and the whole string is a key of
        `:class:BilayerExcitons`. For instance, specifying `l1_l1` will compute the analytical screened
        potential for the intralayer exciton of the first layer
        """
        return self._exciton

    @exciton.setter
    def exciton(self, exciton: str) -> None:
        if exciton not in get_args(BilayerExcitons):
            raise ValueError("Unknown exciton type")
        self._exciton = exciton

    def _get_effective_mass(self,
                            electron_effective_mass: BilayerStructure | None = None,
                            hole_effective_mass: BilayerStructure | None = None
        ) -> BilayerExcitons:
        """Get the effective mass of the electron and hole for a given exciton"""
        # First, check that the effective masses are defined correctly
        if electron_effective_mass is None or hole_effective_mass is None:
            raise ValueError("The effective masses for electrons and holes must be defined")
        if electron_effective_mass.keys() != hole_effective_mass.keys():
            raise ValueError(
                "The effective masses for electrons and holes must be defined for the same layers")
        # https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.53.1391
        # Then, build the effective masses for all possible excitons
        exciton_effective_mass: BilayerExcitons = {}
        for elec_key in electron_effective_mass.keys():
            l1 = structure_key_to_exciton_key(elec_key)
            for hole_key in hole_effective_mass.keys():
                l2 = structure_key_to_exciton_key(hole_key)
                key = f"{l1}_{l2}"
                exciton_effective_mass[key] = \
                    (1/electron_effective_mass[elec_key] + 1/hole_effective_mass[hole_key])**(-1)
        return exciton_effective_mass

    def get_hamiltonian(self) -> npt.NDArray[np.float64]:
        """This method builds the Mott-Wannier Hamiltonian for a couple of layers.
        See equation (13) of the supplementary information of
        https://pubs.acs.org/doi/10.1021/acs.nanolett.0c03019?fig=fig4&ref=pdf

        Args:
            exciton (str): the exciton for which the Hamiltonian is computed. Its value can be
                any key of the BilayerExcitons class

        Returns:
            The Hamiltonians that describe the exciton wavefunction for the chosen exciton
        """
        # First compute mesh size
        mesh_size = np.size(self.grid.flat_grid(), 0)
        hamiltonian: npt.NDArray[np.float64] = np.zeros((mesh_size, mesh_size), dtype=np.float64)
        # Then, get the different terms
        for term in self.terms:
            if term == "kinetic_term":
                # Update the mass of the kinetic term
                cast(KineticTerm, self.term_to_instance[term]).mass =\
                    self.effective_masses.get(self.exciton, 0)
                # Add the kinetic term to the Hamiltonian
                hamiltonian += np.diag(self.term_to_instance[term].get_hamiltonian_term())
            elif term == "effective_2d_coulomb":
                # Update the exciton for the effective 2D Coulomb potential
                cast(Effective2DCoulombPotential, self.term_to_instance[term]).exciton = self.exciton
                # Add the effective 2D Coulomb potential to the Hamiltonian
                hamiltonian += self.term_to_instance[term].get_hamiltonian_term()
        return hamiltonian

    def diagonalize_hamiltonian(self,
                                hamiltonian,
                                **kwargs
        ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Diagonalize the Hamiltonian

        Args:
            hamiltonian: A NxN numpy array representing hermitian matrix to diagonalize
            num_eigenvalues: The number of eigenvalues and eigenvectors to find.
                Defaults to None, meaning that all eigenvalues will be found (exact diagonalization)

        Returns:
            A tuple containing the eigenvalues and eigenvectors of the Hamiltonian
        """
        num_eigenvalues: int = kwargs.get('num_eigenvalues', None)
        if num_eigenvalues is None:
            eigenval, eigenvec = np.linalg.eigh(hamiltonian)
        else:
            eigenval, eigenvec = ssl.eigsh(
                hamiltonian,
                k=num_eigenvalues,
                which="SM",
                tol=1e-7, # type: ignore
                return_eigenvectors=True
            )
        return eigenval, eigenvec

    def get_matter_hamiltonian(self, **kwargs):
        raise NotImplementedError()

    def get_interaction_hamiltonian(self, **kwargs):
        raise NotImplementedError()
