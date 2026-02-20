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

"""This module contains the CavityModesEnergyTerm class, which defines the uncouples light Hamiltonian"""

import numpy as np
import numpy.typing as npt

from openqed.hamiltonians.terms.hamiltonian_term import HamiltonianTerm
from openqed.hamiltonians.light_matter.light_matter_hamiltonian import LightMatterHamiltonian

class CavityModesEnergyTerm(HamiltonianTerm):
    """
    This class defines the uncouples light Hamiltonian, which is the sum of the energies of the cavity modes.
    It is defined as:
    .. math:: H_{ph} = \\sum_{\\q} \\h_{bar} \\omega_{\\alpha} a_{\\alpha}^{\\dagger} a_{\\alpha}

    To build this Hamiltonian term, one should notice that we can exploit a repetition pattern
    In a two mode example with 3 Fock states we would have:
    0    0    0    0    0    0    0    0    0
    0    w1   0    0    0    0    0    0    0
    0    0   2w1   0    0    0    0    0    0
    0    0    0    w2   0    0    0    0    0
    0    0    0    0  w1+w2  0    0    0    0
    0    0    0    0    0  2w1+w2 0    0    0
    0    0    0    0    0    0   2w2   0    0
    0    0    0    0    0    0    0  w1+2w2 0
    0    0    0    0    0    0    0    0 2(w1+w2)
    This can be split between mode 1 and 2:
           0  0  0  0  0  0  0  0  0        0  0  0  0  0  0  0  0  0
           0  w1 0  0  0  0  0  0  0        0  0  0  0  0  0  0  0  0
    m=0 -> 0  0 2w1 0  0  0  0  0  0 m=1 -> 0  0  0  0  0  0  0  0  0
           0  0  0  0  0  0  0  0  0        0  0  0  w2 0  0  0  0  0
           0  0  0  0  w1 0  0  0  0        0  0  0  0  w2 0  0  0  0
           0  0  0  0  0 2w1 0  0  0        0  0  0  0  0  w2 0  0  0
           0  0  0  0  0  0  0  0  0        0  0  0  0  0  0 2w2 0  0
           0  0  0  0  0  0  0  w1 0        0  0  0  0  0  0  0 2w2 0
           0  0  0  0  0  0  0  0 2w1       0  0  0  0  0  0  0  0 2w2
    One can see that the mode m=0 has a 3x3 base that is repeated three times, while the m=1 has a 9x9 which is
    repeated only once.
    Therefore, the algorithm will build the appropriate base and repeat it so that the correct dimension is reached.
    """
    def __init__(self, hamiltonian: LightMatterHamiltonian):
        """Initialize the CavityModesEnergyTerm"""
        super().__init__(hamiltonian=hamiltonian)
        # Override the type of the hamiltonian attribute to be more specific
        self.hamiltonian: LightMatterHamiltonian = hamiltonian

    def get_hamiltonian_term(self):
        """
        This method generates the uncoupled light Hamiltonian term. It returns the Hamiltonian term as a matrix
        """
        # Include the vacuum fock state
        number_of_fock_states = self.hamiltonian.maximum_mode_occupation + 1
        hamiltonian_term_diagonal: npt.NDArray[np.float64] = np.zeros(
                self.hamiltonian.get_hamiltonian_size(), dtype=np.float64)
        for mode in range(self.hamiltonian.number_of_effective_modes):
            number_of_frequency_repetition_in_base: int = number_of_fock_states**mode
            w_c: np.float64 = self.hamiltonian.modes_energies[mode]
            # 1. Build the base photon hamiltonian for this mode, which already includes th zero point energy
            base: npt.NDArray[np.float64] = \
                0.5 * w_c * np.ones(number_of_fock_states ** (mode+1), dtype=np.float64)
            # Then add the \\omega_{\\alpha} a_{\\alpha}^{\\dagger} a_{\\alpha} contribution for each Fock state
            for n in range(1, number_of_fock_states):
                base[n * number_of_frequency_repetition_in_base:(n + 1) * number_of_frequency_repetition_in_base] += \
                    n * w_c * np.ones(number_of_frequency_repetition_in_base, dtype=np.float64)
            # 2. Tensor product with and identity matrix representing the repetitions to
            #    reach the correct dimensions
            number_of_base_repetitions: int = number_of_fock_states**(self.hamiltonian.number_of_effective_modes - mode - 1)
            light_diagonal = np.kron(np.ones(number_of_base_repetitions, dtype=np.float64), base)
            # 3. Tensor product with and the matter hamiltonian
            hamiltonian_term_diagonal += np.kron(
                light_diagonal, np.ones(self.hamiltonian.matter_hamiltonian.get_hamiltonian_size() + 1, dtype=np.float64))
        # Return the result
        return hamiltonian_term_diagonal
