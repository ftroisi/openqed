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
Hamiltonian types module for OpenQED.

.. currentmodule:: openqed.types

Classes and types to handle the Terms that can appear in any Hamiltonian in OpenQED.

.. autosummary::
    :toctree: generated/

    BilayerStructure
    BilayerExcitons
    excitons_key_to_structure_key
    structure_key_to_exciton_key
"""

from .bilayer_excitons import (BilayerStructure,
                               BilayerExcitons,
                               excitons_key_to_structure_key,
                               structure_key_to_exciton_key)

__all__: list[str] = ["BilayerStructure",
                      "BilayerExcitons",
                      "excitons_key_to_structure_key",
                      "structure_key_to_exciton_key"]
