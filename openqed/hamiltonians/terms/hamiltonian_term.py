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

"""This module contains the base abstract HamiltonainTerm class"""

from abc import ABC, abstractmethod
from openqed.hamiltonians.hamiltonian import Hamiltonian

class HamiltonianTerm(ABC):
    """
    This class is the base abstract HamiltonainTerm class. All modules in the `hamiltonians/terms`
    folder and subfolders must inherit from this class.
    """
    def __init__(self, hamiltonian: Hamiltonian) -> None:
        """Initialize the HamiltonianTerm

        Args:
            hamiltonian: The Hamiltonian object to which the term belongs
        """
        self.hamiltonian: Hamiltonian = hamiltonian

    @abstractmethod
    def get_hamiltonian_term(self):
        """
        This method generates the Hamiltonian term. It returns the Hamiltonian term as a matrix
        """
        raise NotImplementedError()
