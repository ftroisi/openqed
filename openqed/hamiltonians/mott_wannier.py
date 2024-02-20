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

"""This model contains the Mott-Wannier Hamiltonian class"""

from .hamiltonian import Hamiltonian

class MottWannier(Hamiltonian):
    """
    This class is the Mott-Wannier Hamiltonian class. It inherits from the base Hamiltonian class.

    Attributes:
        `name`: The name of the Hamiltonian.
        `terms`: a list of sctrings representing the terms of the Hamiltonian.
    """
    def __init__(self, terms: list[str]) -> None:
        """Initialize the Mott-Wannier Hamiltonain

        Args:
            name: The name of the Hamiltonian.
            terms: a list of sctrings representing the terms of the Hamiltonian.
        """
        # First, initialize the base Hamiltonian class
        super().__init__("Mott-Wannier", terms)
        
