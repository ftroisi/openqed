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

"""This module contains the FreeExcitonTerm class"""

from typing import cast
import numpy as np
import numpy.typing as npt

from openqed.hamiltonians.terms.hamiltonian_term import HamiltonianTerm
from openqed.hamiltonians.terms.kinetic_term import KineticTerm
from openqed.hamiltonians.hamiltonian import Hamiltonian
from openqed.grid.real_space_grid import RealSpaceGrid

class FreeExcitonTerm(HamiltonianTerm):
    """
    This class represents the FreeExcitonTerm.
    It is defined as in Eq. 14 of the Supporting Info of [1].

    References:
        [1] Tunable Phases of Moiré Excitons in van der Waals Heterostructures:
            https://pubs.acs.org/doi/10.1021/acs.nanolett.0c03019
    """
    def __init__(self,
                 hamiltonian: Hamiltonian,
                 *,
                 kinetic_term: KineticTerm,
                 energy_gap: np.float64,
                 excitonic_binding_energy: np.float64) -> None:
        """Initialize the FreeExcitonTerm class

        Args:
            hamiltonian: The Hamiltonian object to which the term belongs
            kinetic_term: The kinetic term of the Hamiltonian
            energy_gap: The energy gap of the system. Its value should be in atomic units and must be
                positive
            excitonic_binding_energy: The energy of the excitonic bound state (e.g. 1s, 2s, ...). 
                Its value should be in atomic units. This value is going to be summed to the energy gap,
                thus, at least for the lowest states (i.e. 1s), it should be negative
        """
        super().__init__(hamiltonian)
        if isinstance(hamiltonian.grid, RealSpaceGrid):
            raise NotImplementedError("RealSpaceGrid is not supported yet")
        # Save the mass of the particle
        self.kinetic_term = kinetic_term
        self.energy_gap = energy_gap
        if self.energy_gap < 0:
            raise ValueError("The energy gap must be positive")
        self._excitonic_binding_energy = excitonic_binding_energy

    @property
    def excitonic_binding_energy(self) -> np.float64:
        """The excitonic binding energy."""
        return self._excitonic_binding_energy

    @excitonic_binding_energy.setter
    def mass(self, excitonic_binding_energy: np.float64) -> None:
        self._excitonic_binding_energy: np.float64 = excitonic_binding_energy

    def get_hamiltonian_term(self) -> npt.NDArray[np.float64]:
        """
        This method generates the FreeExciton term. The formula is the following:
        :math:`\\frac{\\hbar^2 * k^2}{2 * m} + E_{gap} + E_{B}`

        Returns:
            The FreeExciton term as a 1D array
        """
        # First, get the kinetic term
        kinetic_term = self.kinetic_term.get_hamiltonian_term()
        # Then, sum the energy gap and the excitonic bound energy
        return cast(npt.NDArray[np.float64],
                    kinetic_term + self.energy_gap + self.excitonic_binding_energy)
