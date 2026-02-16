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

"""This module contains the types used for describing optical cavities"""

from dataclasses import dataclass
from enum import Enum

import numpy as np
import numpy.typing as npt

class ModePolarization(Enum):
    LINEAR_X = "x"
    LINEAR_Y = "y"
    LINEAR_Z = "z"
    LINEAR_XY = "xy"
    LINEAR_XZ = "xz"
    LINEAR_YZ = "yz"
    CHIRAL_PLUS = "+"
    CHIRAL_MINUS = "-"

def mode_polarization_to_vector(
        polarization: ModePolarization | tuple[np.complex128, np.complex128]) -> npt.NDArray[np.complex128]:
    """This method converts the mode polarization to a vector"""
    if isinstance(polarization, ModePolarization):
        if polarization == ModePolarization.LINEAR_X:
            return np.array([1, 0, 0], dtype=np.complex128)
        if polarization == ModePolarization.LINEAR_Y:
            return np.array([0, 1, 0], dtype=np.complex128)
        if polarization == ModePolarization.LINEAR_Z:
            return np.array([0, 0, 0], dtype=np.complex128)
        if polarization == ModePolarization.LINEAR_XY:
            return np.array([1, 1, 0], dtype=np.complex128) / np.sqrt(2)
        if polarization == ModePolarization.LINEAR_XZ:
            return np.array([1, 0, 0], dtype=np.complex128) / np.sqrt(2)
        if polarization == ModePolarization.LINEAR_YZ:
            return np.array([0, 1, 0], dtype=np.complex128) / np.sqrt(2)
        if polarization == ModePolarization.CHIRAL_PLUS:
            return np.array([1, 1j, 0], dtype=np.complex128) / np.sqrt(2)
        if polarization == ModePolarization.CHIRAL_MINUS:
            return np.array([1, -1j, 0], dtype=np.complex128) / np.sqrt(2)
        raise ValueError("Unsupported polarization")
    else:
        # Normalize the vector
        polarization = np.array([polarization[0], polarization[1], 0], dtype=np.complex128)
        norm = np.linalg.norm(polarization)
        if np.isclose(norm, 0):
            raise ValueError("The polarization vector cannot be the zero vector")
        return polarization / norm

@dataclass
class CavityParameters:
    """This class defines the parameters of the cavity."""
    number_of_effective_modes: int
    """The number of effective modes in the cavity (as in Ref. 1). Must be at least 1."""
    maximum_mode_occupation: int
    """The maximum occupation of each mode (i.e. the number of Fock states to be used to represent this mode).
        Must be at least 0. E.g. if maximum_mode_occupation is 2, the Fock states |0>, |1> and |2> will be used."""
    modes_energies: npt.NDArray[np.float64]
    """A 1D array representing the energies of the cavity modes ($\omega_{q}$). Its length must match the number of
        effective modes in the cavity."""
    modes_couplings: npt.NDArray[np.float64]
    """A 1D array representing the coupling strength of each cavity mode to the matter system ($A_{0,q}$). Its length
        must match the number of effective modes in the cavity."""
    modes_parallel_momenta: npt.NDArray[np.float64]
    """A 2D array of shape (number of modes, direction) representing the momentum in the x, y and z directions for each
        mode ($\\mathbf{q}_{\\parallel}$). Its length must match the number of effective modes in the cavity."""
    modes_polarization: ModePolarization | tuple[np.complex128, np.complex128]
    """The polarization of the cavity modes. It can be one of the following:
        - 'x': linear polarization along the x direction
        - 'y': linear polarization along the y direction
        - 'z': linear polarization along the z direction
        - 'xy': linear polarization in the xy plane
        - 'xz': linear polarization in the xz plane
        - 'yz': linear polarization in the yz plane
        - '+': chiral polarization with positive helicity (i.e. right-handed circular polarization)
        - '-': chiral polarization with negative helicity (i.e. left-handed circular polarization)
        
        Alternatively, the polarization can be defined as a tuple of two complex numbers representing the x and y
        components of the polarization vector, respectively. The z component is assumed to be zero.
        The polarization vector will be automatically normalized to have unit length."""

