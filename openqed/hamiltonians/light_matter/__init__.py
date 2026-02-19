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
LightMatterHamiltonians module for OpenQED.
These classes represent the various Hamiltonians that can be used to describe light-matter systems in OpenQED.

.. currentmodule:: openqed.hamiltonians.light_matter

Classes and utilities to handle the LightMatterHamiltonians in OpenQED.

.. autosummary::
    :toctree: generated/

    LightMatterHamiltonian
"""

from .light_matter_hamiltonian import LightMatterHamiltonian, LightMatterHamiltonianTerms

__all__: list[str] = [
    "LightMatterHamiltonian",
    "LightMatterHamiltonianTerms"
]
