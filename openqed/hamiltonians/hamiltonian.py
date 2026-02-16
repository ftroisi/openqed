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

"""This module contains the base abstract Hamiltonian class"""

from abc import ABC, abstractmethod
import os
import numpy as np
import scipy.sparse.linalg as ssl
import numpy.typing as npt

class Hamiltonian(ABC):
    """This class is the base abstract Hamiltonian class. All modules in the `hamiltonians` folder
    and subfolders must inherit from this class.
    """
    def __init__(self, name: str, terms: list[str]) -> None:
        """Initializes the Hamiltonian

        Args:
            name: The name of the Hamiltonian.
            terms: a list of strings representing the terms of the Hamiltonian.
        """
        self.name: str = name.strip()
        if len(self.name) == 0:
            raise ValueError("The name of the Hamiltonian cannot be empty or contain only whitespace.")
        self.terms: list[str] = terms

    @abstractmethod
    def get_hamiltonian(self) -> npt.NDArray[np.float64] | npt.NDArray[np.complex128]:
        """
        This method generates the Hamiltonian from the terms. It returns the Hamiltonian as a matrix
        """
        pass

    @classmethod
    def diagonalize_hamiltonian(cls,
                                hamiltonian: npt.NDArray[np.float64] | npt.NDArray[np.complex128],
                                number_of_eigenvalues: int | None = None,
                                restart_path: str | None = None
        ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64] | npt.NDArray[np.complex128]]:
        """
        Diagonalize the Hamiltonian

        Args:
            hamiltonian: The Hamiltonian to diagonalize
            number_of_eigenvalues: The number of eigenvalues to compute. If None, all eigenvalues will be computed.
                Note that computing all eigenvalues can be very expensive for large Hamiltonians
            restart_path: The path to the file where the results of the diagonalization will be saved. If the file
                already exists, the results will be loaded from the file instead of diagonalizing the Hamiltonian again.
                It should be an absolute path in the desired location in the filesystem. Note that the file will be
                saved in the .npz format. The file will contain two arrays:
                - `eigenval`: The eigenvalues of the Hamiltonian
                - `eigenvec`: The eigenvectors of the Hamiltonian

        Returns:
            A tuple containing the eigenvalues and the eigenvectors of the Hamiltonian. The eigenvalues are returned as a 1D array, while the eigenvectors are returned as a 2D array, where each column is an eigenvector.
        """
        if restart_path is not None and not restart_path.endswith(".npz"):
            raise ValueError("The restart_path must end with .npz extension")

        # Check for restart data
        if restart_path is not None and os.path.exists(restart_path):
            data = np.load(restart_path)
            eigenval: npt.NDArray[np.float64] = data['eigenval']
            eigenvec: npt.NDArray[np.complex128] = data['eigenvec']
            return eigenval, eigenvec

        # Diagonalize the Hamiltonian
        if number_of_eigenvalues is None:
            eigenval, eigenvec = np.linalg.eigh(hamiltonian)
        else:
            eigenval, eigenvec = ssl.eigsh(
                hamiltonian,
                k=number_of_eigenvalues,
                which="SM",
                tol=1e-7, # type: ignore
                return_eigenvectors=True
            )

        # Save the results to file
        if restart_path is not None:
            # Only the root node should save the data. `self.get_size > 1` corresponds to the case
            # where we have parallelized over the cavity frequencies, thus every node in the group
            # is handling a different cavity frequency. Thus, all can output safely.
            # `self.get_size == 1 and self.am_i_world_root` correspond to the case where the code
            # is serial over the cavity frequencies, thus only the world root node should output
            # (all nodes will diagonalize the Hamiltonian for the all cavity frequencies)
            # TODO: Add MPI support!
            # if (self.get_size == 1 and self.am_i_world_root) or self.get_size > 1:
            np.savez(restart_path, eigenval=eigenval, eigenvec=eigenvec)
        return eigenval, eigenvec
