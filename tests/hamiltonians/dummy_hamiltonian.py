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

"""This module contains a dummy Hamiltonian class, to be used in the tests"""

import numpy as np
import numpy.typing as npt
import scipy.sparse.linalg as ssl

from openqed.hamiltonians.hamiltonian import Hamiltonian

class DummyHamiltonian(Hamiltonian):
    """Dummy Hamiltonian class"""
    def __init__(self, terms: list[str], *, grid) -> None:
        super().__init__("Mott-Wannier", terms, grid=grid)

    def get_hamiltonian(self, **kwargs):
        pass

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
