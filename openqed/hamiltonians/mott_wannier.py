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

"""This model contains the Mott-Wannier Hamiltonian class"""

import numpy as np
import numpy.typing as npt
import scipy.sparse.linalg as ssl

from openqed.grid.k_space_grid import KSpaceGrid
from openqed.hamiltonians.hamiltonian import Hamiltonian
from openqed.hamiltonians.terms.hamiltonian_term import HamiltonianTerm
from openqed.hamiltonians.terms.free_exciton import FreeExciton
from openqed.hamiltonians.terms.effective_2d_coulomb import Effective2DCoulombPotential
from openqed.hamiltonians.types import BilayerStructure

class MottWannier(Hamiltonian):
    """
    This class is the Mott-Wannier Hamiltonian class. It inherits from the base Hamiltonian class.

    Attributes:
        `name`: The name of the Hamiltonian.
        `terms`: a list of strings representing the terms of the Hamiltonian.
    """
    def __init__(self,
                terms: list[str],
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
        """
        # First, initialize the base Hamiltonian class
        super().__init__("Mott-Wannier", terms, grid=grid)
        # Initialize the Hamiltonian terms
        self.term_to_instance: dict[str, HamiltonianTerm] = {}
        for term in terms:
            if term == "free_exciton":
                if electron_effective_mass is None or hole_effective_mass is None:
                    raise ValueError("The effective masses for electrons and holes must be defined")
                self.term_to_instance[term] = FreeExciton(
                    self,
                    electron_effective_mass=electron_effective_mass,
                    hole_effective_mass=hole_effective_mass)
            elif term == "effective_2d_coulomb":
                if thicknesses is None or dielectric_constants is None:
                    raise ValueError("The thicknesses and dielectric constants must be defined")
                self.term_to_instance[term] = Effective2DCoulombPotential(
                    self, thicknesses=thicknesses, dielectric_constants=dielectric_constants)

    def get_hamiltonian(self, layer: str) -> npt.NDArray[np.float64]:
        """This method builds the Mott-Wannier Hamiltonian for a couple of layers.
        See equation (13) of the supplementary information of
        https://pubs.acs.org/doi/10.1021/acs.nanolett.0c03019?fig=fig4&ref=pdf

        Args:
            layer (str): the layer for which the Hamiltonian is computed. Its value can be
                any key of the Excitons class

        Returns:
            h_ll: a matrix, corresponding to the Hamiltonians that describe the exciton
                wavefunction for the chosen layers
        """
        # First compute mesh size
        mesh_size = np.size(self.grid.flat_grid(), 1)
        hamiltonian: npt.NDArray[np.float64] = np.zeros((mesh_size, mesh_size), dtype=np.float64)
        # Then, get the different terms
        for term in self.terms:
            if term == "free_exciton":
                hamiltonian += np.diag(
                    self.term_to_instance[term].get_hamiltonian_term(layer=layer))
            elif term == "effective_2d_coulomb":
                hamiltonian += self.term_to_instance[term].get_hamiltonian_term(layer=layer)
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
