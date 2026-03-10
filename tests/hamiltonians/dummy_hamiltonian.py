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

from openqed.hamiltonians.matter_hamiltonian import MatterHamiltonian

class DummyHamiltonian(MatterHamiltonian):
    """Dummy Hamiltonian class"""
    def __init__(self, terms: list[str], *, grid) -> None:
        super().__init__("Mott-Wannier", terms, mode="many_body", grid=grid)

    def get_hamiltonian_size(self) -> int:
        return np.size(self.grid.flat_grid(), 0)

    def _get_many_body_hamiltonian(self):
        n = self.get_hamiltonian_size()
        return np.zeros((n, n), dtype=np.float64)

    def get_interaction_hamiltonian(self, light_momentum, light_polarization):
        raise NotImplementedError()
