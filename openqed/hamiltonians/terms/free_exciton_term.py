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

"""This module contains the FreeExcitonTerm class"""

import numpy as np
import numpy.typing as npt

from openqed.hamiltonians.terms.hamiltonian_term import HamiltonianTerm
from openqed.hamiltonians.terms.kinetic_term import KineticTerm
from openqed.hamiltonians import MatterHamiltonian
from openqed.grid.real_space_grid import RealSpaceGrid

from openqed.hamiltonians.types.bilayer_excitons import BilayerExcitons

class FreeExcitonTerm(HamiltonianTerm):
    """
    This class represents the FreeExcitonTerm.
    It is defined as in Eq. 14 of the Supporting Info of [1].

    References:
        [1] Tunable Phases of Moiré Excitons in van der Waals Heterostructures:
            https://pubs.acs.org/doi/10.1021/acs.nanolett.0c03019
    """
    def __init__(self,
                 hamiltonian: MatterHamiltonian,
                 *,
                 kinetic_term: KineticTerm,
                 wannier_binding_energies: BilayerExcitons,
                 band_gap: BilayerExcitons
        ) -> None:
        """Initialize the FreeExcitonTerm class

        Args:
            hamiltonian: The Hamiltonian object to which the term belongs
            kinetic_term: The kinetic term of the Hamiltonian
            wannier_binding_energies: A dictionary containing the Mott-Wannier binding energies for the different
                excitons. The values are 1D arrays containing the Mott-Wannier binding energies for the different
                excitonic states. For instance, the first values will refer to the 1s state, then 2s, ...
            band_gap: A dictionary containing the band gap for the different excitons.

        Raises:
            ValueError: If the keys of the wannier_binding_energies and band_gap are not the same
            ValueError: If the energy gap is negative
        """
        super().__init__(hamiltonian)
        if isinstance(hamiltonian.grid, RealSpaceGrid):
            raise NotImplementedError("RealSpaceGrid is not supported yet")
        # Save the mass of the particle
        self.kinetic_term = kinetic_term
        # Ensure that the keys of the wannier_binding_energies and band_gap are the same
        if wannier_binding_energies.keys() != wannier_binding_energies.keys():
            raise ValueError("The keys of the wannier_binding_energies and band_gap must be the same")
        self.wannier_binding_energies: BilayerExcitons = wannier_binding_energies
        self.band_gap: BilayerExcitons = band_gap
        self.exciton = next(k for k, v in wannier_binding_energies.items() if v is not None)
        if self.band_gap[self.exciton] < 0:
            raise ValueError("The energy gap must be positive")

    def get_hamiltonian_term(self) -> npt.NDArray[np.float64]:
        """
        This method generates the FreeExciton term. The formula is the following:
        :math:`\\frac{\\h_{bar}^2 * k^2}{2 * m} + E_{gap} + E_{B}`

        Returns:
            The FreeExciton term as a 1D array
        """
        # Create the array to return
        hamiltonian = np.zeros(self.hamiltonian.get_hamiltonian_size(), dtype=np.float64)
        # First, add the energy gap and the excitonic bound energy
        hamiltonian += \
            (self.band_gap[self.exciton] + self.wannier_binding_energies[self.exciton]) * np.ones_like(hamiltonian)
        # Then, sum the energy gap and the excitonic bound energy
        hamiltonian += self.kinetic_term.get_hamiltonian_term()
        return hamiltonian
