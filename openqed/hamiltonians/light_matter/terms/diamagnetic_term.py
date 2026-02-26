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

import gc

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
        if self.hamiltonian.maximum_mode_occupation < 2:
            raise ValueError("The diamagnetic term requires at least 2 Fock states per mode to be represented.")

    def get_hamiltonian_term(self) -> npt.NDArray[np.float64]:
        """
        Generates the diamagnetic Hamiltonian term (diagonal A^2 + off-diagonal A^2).
        Returns a dense NumPy array.
        """
        number_of_fock_states: int = self.hamiltonian.maximum_mode_occupation + 1
        matter_dim: int = self.hamiltonian.matter_hamiltonian.get_hamiltonian_size() + 1
        mat_dim: int = self.hamiltonian.get_hamiltonian_size()

        hamiltonian_term: npt.NDArray[np.float64] = np.zeros((mat_dim, mat_dim), dtype=np.float64)
        full_diagonal: npt.NDArray[np.float64] = np.zeros(mat_dim, dtype=np.float64)
        for mode in range(self.hamiltonian.number_of_effective_modes):
            a_0: np.float64 = self.hamiltonian.modes_couplings[mode] ** 2
            preceding_modes_dimension: int = number_of_fock_states ** mode
            following_dim: int = number_of_fock_states ** (self.hamiltonian.number_of_effective_modes - mode - 1)

            # --- Diagonal part ---
            # Closed form: 0.5 * a_0 * (1 + 2n) per block of size preceding_dim
            fock_indices = np.repeat(np.arange(number_of_fock_states), preceding_modes_dimension)
            diag_base: npt.NDArray[np.float64] = 0.5 * a_0 * (1.0 + 2.0 * fock_indices)
            del fock_indices
            gc.collect()

            light_diagonal: npt.NDArray[np.float64] = np.tile(diag_base, following_dim)
            del diag_base
            gc.collect()

            full_diagonal += np.repeat(light_diagonal, matter_dim)
            del light_diagonal
            gc.collect()

            # --- Off-diagonal (A^2) part ---
            # Coefficients: 0.5 * a_0^2 * sqrt((n+1)(n+2)) for n = 0, ..., N_fock-3
            ph = np.arange(number_of_fock_states - 2, dtype=np.float64)
            fock_states_coefficients: npt.NDArray[np.float64] = (
                    0.5 * a_0 ** 2 * np.sqrt((ph + 1.0) * (ph + 2.0))
            )
            base: npt.NDArray[np.float64] = np.repeat(fock_states_coefficients, preceding_modes_dimension)
            del ph, fock_states_coefficients
            gc.collect()

            # Build the off-diagonal in the mode subspace
            h_a2: npt.NDArray[np.float64] = (
                    np.diag(base, k=2 * preceding_modes_dimension) + np.diag(base, k=-2 * preceding_modes_dimension)
            )
            del base
            gc.collect()

            # Tensor with following modes identity
            h_a2 = np.kron(np.eye(following_dim, dtype=np.float64), h_a2)
            gc.collect()

            # Tensor with matter identity
            h_a2 = np.kron(h_a2, np.eye(matter_dim, dtype=np.float64))
            gc.collect()

            hamiltonian_term += h_a2
            del h_a2
            gc.collect()

        # Add the diagonal once after the loop
        hamiltonian_term += np.diag(full_diagonal)
        del full_diagonal
        gc.collect()

        return hamiltonian_term
