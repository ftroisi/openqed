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

"""This module contains the types used for describing excitons in bilayer structures"""
from typing import TypedDict
import numpy as np
import numpy.typing as npt

class BilayerStructure(TypedDict, total=False):
    """This class is used to define a certain property for a bilayer structure."""
    substrate1: np.float64
    layer1: np.float64 | npt.NDArray[np.float64]
    interlayer: np.float64
    layer2: np.float64 | npt.NDArray[np.float64]
    substrate2: np.float64

class BilayerExcitons(TypedDict, total=False):
    """
    This class defines the excitons of the system. It is used to define several parameters
    of the system
    """
    l1_l1: np.float64 | npt.NDArray[np.float64] | np.complex128
    l1_l2: np.float64 | npt.NDArray[np.float64] | np.complex128
    l2_l1: np.float64 | npt.NDArray[np.float64] | np.complex128
    l2_l2: np.float64 | npt.NDArray[np.float64] | np.complex128

def excitons_key_to_system_key(key: str) -> str:
    """This method converts the key of the excitons to the key of the system"""
    if key == "l1":
        return "layer1"
    if key == "l2":
        return "layer2"
    raise ValueError("The key is not valid")

def structure_key_to_exciton_key(key: str) -> str:
    """This method converts the key of the excitons to the key of the system"""
    if key == "layer1":
        return "l1"
    if key == "layer2":
        return "l2"
    raise ValueError("The key is not valid")
