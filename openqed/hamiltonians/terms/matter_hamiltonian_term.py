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

"""This module contains the MatterHamiltonianTerm class, which couples matter Hamiltonians to LightMatterHamiltonians"""

import numpy as np
import numpy.typing as npt
import scipy.linalg as spl

from .hamiltonian_term import HamiltonianTerm
from openqed.hamiltonians import MatterHamiltonian

class MatterHamiltonianTerm(HamiltonianTerm):
    """
    This class is a HamiltonianTerm that couples matter Hamiltonians to LightMatterHamiltonians. It is used to
    generate the term of the Hamiltonian that couples the matter Hamiltonian to the LightMatterHamiltonian.
    """
    def __init__(self, hamiltonian: MatterHamiltonian) -> None:
        """Initialize the MatterHamiltonianTerm

        Args:
            hamiltonian: The MatterHamiltonian object to which the term belongs
        """
        super().__init__(hamiltonian)

    @classmethod
    def from_matter_hamiltonian(cls, hamiltonian: MatterHamiltonian) -> "MatterHamiltonianTerm":
        """Create a MatterHamiltonianTerm from a MatterHamiltonian

        Args:
            hamiltonian: The MatterHamiltonian object to which the term belongs
        """
        return cls(hamiltonian)


    def get_hamiltonian_term(self) -> npt.NDArray[np.float64 | np.complex128]:
        """
        This method generates the Hamiltonian term. It returns the Hamiltonian term as a matrix. The final matrix
        includes the polaritonic ground state, which is added as the first row and column of the matrix.
        The original matter Hamiltonian is then placed in the bottom right corner of the matrix.

        Raises:
            ValueError: If the matter Hamiltonian is not Hermitian.

        Returns:
            The Hamiltonian term as a matrix, including the polaritonic ground state.
        """
        matter_hamiltonian = self.hamiltonian.get_hamiltonian()
        # Allocate the new matrix with the correct size (original size + 1 for the polaritonic ground state)
        expanded_hamiltonian = np.zeros(
            (matter_hamiltonian.shape[0] + 1, matter_hamiltonian.shape[1] + 1), dtype=matter_hamiltonian.dtype)
        # Copy data
        expanded_hamiltonian[1:, 1:] = matter_hamiltonian
        if not spl.ishermitian(expanded_hamiltonian, atol=1e-12):
            raise ValueError(f"The matter hamiltonian {self.hamiltonian.name} is not Hermitian")
        # Explicitly delete the reference to the old matrix to free memory
        del matter_hamiltonian
        return expanded_hamiltonian

