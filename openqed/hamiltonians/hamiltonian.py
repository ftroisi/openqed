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

"""This module contains the base abstract Hamiltonain class"""

from abc import ABC, abstractmethod
from openqed.grid.grid import Grid

class Hamiltonian(ABC):
    """This class is the base abstract Hamiltonain class. All modules in the `hamiltonians` folder
    and subfolders must inherit from this class.

    Attributes:
        `name`: The name of the Hamiltonian.
        `terms`: a list of sctrings representing the terms of the Hamiltonian.
    """
    def __init__(self,
                name: str,
                terms: list[str],
                grid: Grid) -> None:
        """Initialize the Hamiltonain

        Args:
            name: The name of the Hamiltonian.
            terms: a list of sctrings representing the terms of the Hamiltonian.
        """
        self.name: str = name
        self.terms: list[str] = terms
        self.grid: Grid = grid

    @abstractmethod
    def get_hamiltonain(self):
        """
        This method generates the Hamiltonian from the terms. It returns the Hamiltonian as a matrix
        """
        raise NotImplementedError()

    @abstractmethod
    def diagonalize_hamiltonian(self):
        """Diagonalize the Hamiltonian"""
        raise NotImplementedError()