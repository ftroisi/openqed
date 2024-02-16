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
Input/Output module for OpenQED.

.. currentmodule:: openqed.io

Classes and utilities to handle the input/output of OpenQED.

.. autosummary::
    :toctree: generated/

    InputFile
    Parser
    Dumper

"""

from .input import InputFile
from .parser import Parser, Dumper

__all__: list[str] = [
    "InputFile",
    "Parser",
    "Dumper"]