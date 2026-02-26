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

"""This module contains the DiamagneticTerm class"""

import numpy as np
import numpy.typing as npt

from openqed.hamiltonians.terms import HamiltonianTerm
from openqed.hamiltonians.light_matter import LightMatterHamiltonian

class DiamagneticTerm(HamiltonianTerm):
    """
    This class defines the diamagnetic term (or paramagnetic term), which is the term that describes the
    interaction between matter and light in the Pauli-Fierz Hamiltonian. It is also call :math:`A_0^2` term.

    To build this Hamiltonian term, one should notice that we can exploit a repetition pattern
    In a two mode example with 3 Fock states we would have:
     0   0  A_1  0   0   0  A_2  0   0
     0   0   0   0   0   0   0  A_2  0
    A_1  0   0   0   0   0   0   0  A_2
     0   0   0   0   0  A_1  0   0   0
     0   0   0   0   0   0   0   0   0
     0   0   0  A_1  0   0   0   0   0
    A_2  0   0   0   0   0   0   0  A_1
     0  A_2  0   0   0   0   0   0   0
     0   0  A_2  0   0   0  A_1  0   0
     This can be split between mode 1 and 2:
            0   0  A_1  0   0   0   0   0   0        0   0   0   0   0   0  A_2  0   0
            0   0   0   0   0   0   0   0   0        0   0   0   0   0   0   0  A_2  0
           A_1  0   0   0   0   0   0   0   0        0   0   0   0   0   0   0   0  A_2
     m=0 -> 0   0   0   0   0  A_1  0   0   0 m=1 -> 0   0   0   0   0   0   0   0   0
            0   0   0   0   0   0   0   0   0        0   0   0   0   0   0   0   0   0
            0   0   0  A_1  0   0   0   0   0        0   0   0   0   0   0   0   0   0
            0   0   0   0   0   0   0   0  A_1      A_2  0   0   0   0   0   0   0   0
            0   0   0   0   0   0   0   0   0        0  A_2  0   0   0   0   0   0   0
            0   0   0   0   0   0  A_1  0   0        0   0  A_2  0   0   0   0   0   0
     Again, the mode 0 has a 3x3 base repeated three times, while the m=1 has a 9x9 base which is repeated only once.
    """
    def __init__(self, hamiltonian: LightMatterHamiltonian):
        """Initialize the DiamagneticTerm"""
        super().__init__(hamiltonian=hamiltonian)
        # Override the type of the hamiltonian attribute to be more specific
        self.hamiltonian: LightMatterHamiltonian = hamiltonian

    def get_hamiltonian_term(self) -> npt.NDArray[np.float64 | np.complex128]:
        """
        This method generates the bilinear coupling Hamiltonian term. It returns the Hamiltonian term as a matrix
        """
        polarization_directions = polarization_vector_to_directions(self.hamiltonian.modes_polarization)
        # Include the vacuum fock state
        number_of_fock_states = self.hamiltonian.maximum_mode_occupation + 1
        # Build the P dot A matrix
        mat_dim: int = self.hamiltonian.get_hamiltonian_size()
        hamiltonian_term: npt.NDArray[np.complex128] = np.zeros((mat_dim, mat_dim), dtype=np.complex128)
        for mode in range(self.hamiltonian.number_of_effective_modes):
            # First, get the interaction coming from the matter hamiltonian (also accounting for light polarization).
            # This is a matrix of shape (matter_hamiltonian_dimension, matter_hamiltonian_dimension)
            h_int = self.hamiltonian.matter_hamiltonian.get_interaction_hamiltonian(
                    self.hamiltonian.modes_parallel_momenta[mode], polarization_directions)
            matter_interaction = None
            for key in h_int.keys():
                if matter_interaction is None:
                    matter_interaction = h_int[key] * self.hamiltonian.modes_polarization[direction_to_index(key)]
                else:
                    matter_interaction += h_int[key] * self.hamiltonian.modes_polarization[direction_to_index(key)]
            matter_interaction_hermitian_conjugate = matter_interaction.conj().T  # Get a view, no copy
            del h_int

            # Then, build the base photon hamiltonian for this mode. This is a vector of length
            # (number_of_fock_states ** (mode+1)) which accounts for the light matter coupling and the Fock states
            # coefficients. The base is built in such a way that it can be repeated to get the correct dimensions.
            preceding_modes_dimension: int = number_of_fock_states ** mode
            a_0 = self.hamiltonian.modes_couplings[mode]
            fock_states_coefficients = np.sqrt(np.arange(1, number_of_fock_states, dtype=np.float64))
            base = -a_0 * np.repeat(fock_states_coefficients, preceding_modes_dimension).astype(np.complex128)
            del fock_states_coefficients

            identity = np.identity(
                    number_of_fock_states ** (self.hamiltonian.number_of_effective_modes - (mode + 1)),
                    dtype=np.complex128)
            # The bilinear coupling term has two contributions: the one coming from the matter interaction (above
            # the main diagonal, representing the photonic annihilation operator) and the one coming from its hermitian
            # conjugate (below the main diagonal, representing the photonic creation operator).
            for i in range(2):
                # Build the Fock matrix, which represents the light matter coupling and the fock states coefficients
                fock_coupling_op = np.kron(identity, np.diag(base, k=(-1) ** i * preceding_modes_dimension))
                # Tensor product with the interaction coming from the matter hamiltonian
                hamiltonian_term += np.kron(fock_coupling_op,
                                            matter_interaction if i == 0 else matter_interaction_hermitian_conjugate)
                del fock_coupling_op
            # Clean up the variables to free memory
            del identity, matter_interaction, matter_interaction_hermitian_conjugate, base
            gc.collect()
        return hamiltonian_term
