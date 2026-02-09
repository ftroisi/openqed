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

"""This module contains the base abstract MatterHamiltonian class"""

from abc import ABC, abstractmethod
from typing import Literal, Dict

import numpy as np
import numpy.typing as npt

from openqed.hamiltonians.hamiltonian import Hamiltonian

class MatterHamiltonian(Hamiltonian, ABC):
    """
    This class is the base abstract MatterHamiltonian class.
    All classes representing single particle matter systems must inherit from this class.
    """

    @abstractmethod
    def get_interaction_hamiltonian(self,
                                    light_momentum: npt.NDArray[np.float64],
                                    light_polarization: Literal['x', 'y', 'z', 'xy', 'xz', 'yz', '+', '-']
        ) -> Dict[Literal['x', 'y', 'z'], npt.NDArray[np.float64 | np.complex128]]:
        """
        This method generates the interaction hamiltonian for this class. In a light-matter system,
        this term describes the interaction between the matter system and the radiation field
        (:math:`\\p \\cdot \\mathbf{A}`).

        Args:
            light_momentum: The momentum of the light field. It is a 2D array of shape:
                (number of modes, direction) representing the momentum in the x, y and z directions for each mode.
            light_polarization: The polarization of the light field. It can be one of the following:
                - 'x': linear polarization along the x direction
                - 'y': linear polarization along the y direction
                - 'z': linear polarization along the z direction
                - 'xy': linear polarization in the xy plane
                - 'xz': linear polarization in the xz plane
                - 'yz': linear polarization in the yz plane
                - '+': circular polarization with positive helicity
                - '-': circular polarization with negative helicity

        Returns:
            A dictionary with keys 'x', 'y' and 'z' representing the interaction hamiltonian in the x, y and z
            polarization directions.
            The values are 3D arrays of shape: (number of modes, interaction matrix size, interaction matrix size).
        """
        pass
