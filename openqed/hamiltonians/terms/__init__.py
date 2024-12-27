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

"""
Hamiltonian terms module for OpenQED.

.. currentmodule:: openqed.terms

Classes and utilities to handle the Terms that can appear in any Hamiltonian in OpenQED.

.. autosummary::
    :toctree: generated/

    Effective2DCoulombPotential
"""

from .hamiltonian_term import HamiltonianTerm
from .effective_2d_coulomb import Effective2DCoulombPotential
from .free_exciton import FreeExciton

__all__: list[str] = [
    "HamiltonianTerm",
    "Effective2DCoulombPotential",
    "FreeExciton"]
