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

"""This module contains the base abstract MatterHamiltonain class"""

from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt

from openqed.hamiltonians.hamiltonian import Hamiltonian

class MatterHamiltonian(Hamiltonian, ABC):
    """This class is the base abstract MatterHamiltonain class.
    All classes representing matter systems must inherit from this class.
    """

    @abstractmethod
    def get_matter_hamiltonian(self, **kwargs) -> npt.NDArray[np.float64 | np.complex128]:
        """
        This method generates the uncoupled matter hamiltonian for this class. It is the term that
        describes the matter system without the radiation field
        """
        raise NotImplementedError()

    @abstractmethod
    def get_interaction_hamiltonian(self, **kwargs) -> npt.NDArray[np.float64 | np.complex128]:
        """
        This method generates the interaction hamiltonian for this class. In a light-matter system,
        this term describes the interaction between the matter system and the radiation field
        (:math:`\\p \\cdot \\mathbf{A}`).
        """
        raise NotImplementedError()
