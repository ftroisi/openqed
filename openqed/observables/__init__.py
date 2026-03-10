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

"""
Observables module for OpenQED.

.. currentmodule:: openqed.observables

Classes and utilities to handle the Observables in OpenQED.
Observables are computed from the Polaritonic states and eigenvalues obtained from the Hamiltonian diagonalization.
They can be used to compute physical quantities such as the linear response and the spectral function.

.. autosummary::
    :toctree: generated/

    LinearResponse

"""

from .observable import Observable
from .linear_response import LinearResponse

__all__: list[str] = [
    "Observable",
    "LinearResponse"
]
