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

"""This module contains the BilinearCouplingTerm class"""

import numpy as np
import numpy.typing as npt

from openqed.hamiltonians.terms import HamiltonianTerm
from openqed.hamiltonians.light_matter import LightMatterHamiltonian

class BilinearCouplingTerm(HamiltonianTerm):
    """
    This class defines the bilinear coupling term (or paramagnetic term), which is the term that describes the
    interaction between matter and light in the Pauli-Fierz Hamiltonian. It is also call `p dot A` term.

    To build this Hamiltonian term, one should notice that we can exploit a repetition pattern
    In a two mode example with 3 Fock states we would have:
     0  pA1  0  pA2  0   0   0   0   0
    pA1  0  pA1  0  pA2  0   0   0   0
     0  pA1  0   0   0  pA2  0   0   0
    pA2  0   0   0  pA1  0  pA2  0   0
     0  pA2  0  pA1  0  pA1  0  pA2  0
     0   0  pA2  0  pA1  0   0   0  pA2
     0   0   0  pA2  0   0   0  pA1  0
     0   0   0   0  pA2  0  pA1  0  pA1
     0   0   0   0   0  pA2  0  pA1  0
     This can be split between mode 1 and 2:
            0  pA1  0   0   0   0   0   0   0        0   0   0  pA2  0   0   0   0   0
           pA1  0  pA1  0   0   0   0   0   0        0   0   0   0  pA2  0   0   0   0
     m=0 -> 0  pA1  0   0   0   0   0   0   0 m=1 -> 0   0   0   0   0  pA2  0   0   0
            0   0   0   0  pA1  0   0   0   0       pA2  0   0   0   0   0  pA2  0   0
            0   0   0  pA1  0  pA1  0   0   0        0  pA2  0   0   0   0   0  pA2  0
            0   0   0   0  pA1  0   0   0   0        0   0  pA2  0   0   0   0   0  pA2
            0   0   0   0   0   0   0  pA1  0        0   0   0  pA2  0   0   0   0   0
            0   0   0   0   0   0  pA1  0  pA1       0   0   0   0  pA2  0   0   0   0
            0   0   0   0   0   0   0  pA1  0        0   0   0   0   0  pA2  0   0   0
     Again, the mode 0 has a 3x3 base repeated three times, while the m=1 has a 9x9 base which is repeated only once.
    """
    def __init__(self, hamiltonian: LightMatterHamiltonian):
        """Initialize the BilinearCouplingTerm"""
        super().__init__(hamiltonian=hamiltonian)
        # Override the type of the hamiltonian attribute to be more specific
        self.hamiltonian: LightMatterHamiltonian = hamiltonian

    def get_hamiltonian_term(self) -> npt.NDArray[np.float64 | np.complex128]:
        """
        This method generates the bilinear coupling Hamiltonian term. It returns the Hamiltonian term as a matrix
        """
        # Include the vacuum fock state
        number_of_fock_states = self.hamiltonian.maximum_mode_occupation + 1
        # Build the P dot A matrix (same dimension as H QED)
        mat_dim: int = self.hamiltonian.get_hamiltonian_size()
        h_pa: npt.NDArray[np.complex128] = np.zeros((mat_dim, mat_dim), dtype=np.complex128)
        for mode in range(self.hamiltonian.number_of_effective_modes):
            # Build the interaction Hamiltonian for this mode
            h_int = self.hamiltonian.matter_hamiltonian.get_interaction_hamiltonian(
                    self.hamiltonian.modes_parallel_momenta[mode], self.hamiltonian.modes_polarization)
            matter_interaction = np.asarray(h_int.get("x")*self.hamiltonian.modes_polarization[0] + \
                                 h_int.get("y")*self.hamiltonian.modes_polarization[1], dtype=np.complex128)
            base_block_size: int = number_of_fock_states ** mode
            a_0: np.float64 = np.float64(self.hamiltonian.maximum_mode_occupation)
            # 1. First, define the base
            # Build the coefficient for the Fock state: it will look like [1. sqrt(2) sqrt(3) ...]
            fock_states_coefficients: list[float] = []
            for ph in range(0, self.hamiltonian.maximum_mode_occupation - 1):
                fock_states_coefficients.append(np.sqrt(ph + 1) * np.ones(base_block_size))
            base = -a_0 * np.ones(self.hamiltonian.maximum_mode_occupation * base_block_size, dtype=np.complex128) * np.concatenate(fock_states_coefficients)
            for i in range(2):
                # 2. Place the base on the correct diagonal
                h_temp: npt.NDArray[np.complex128] = np.diag(base, k=(-1) ** i * base_block_size)
                h_temp = np.asarray(
                        np.kron(np.identity(number_of_fock_states ** (self.hamiltonian.number_of_effective_modes - (mode + 1)), dtype=np.complex128), h_temp), dtype=np.complex128)
                h_temp = np.asarray(np.kron(h_temp, matter_interaction), dtype=np.complex128) \
                    if i == 0 else np.asarray(np.kron(h_temp, np.asarray(matter_interaction.conj().T, dtype=np.complex128)), dtype=np.complex128)
                h_pa += h_temp
        # Return the result
        return h_pa
