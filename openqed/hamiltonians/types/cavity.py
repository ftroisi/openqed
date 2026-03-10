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

class LightPolarization(Enum):
    LINEAR_X = "x"
    LINEAR_Y = "y"
    LINEAR_Z = "z"
    LINEAR_XY = "xy"
    LINEAR_XZ = "xz"
    LINEAR_YZ = "yz"
    CHIRAL_PLUS = "+"
    CHIRAL_MINUS = "-"

def direction_to_index(direction: str) -> int:
    """This method converts a direction string to an index"""
    if direction == "x":
        return 0
    if direction == "y":
        return 1
    if direction == "z":
        return 2
    raise ValueError("Unsupported direction")

def mode_polarization_to_vector(
        polarization: LightPolarization | tuple[np.complex128, np.complex128]) -> npt.NDArray[np.complex128]:
    """This method converts the mode polarization to a vector"""
    if isinstance(polarization, LightPolarization):
        if polarization == LightPolarization.LINEAR_X:
            return np.array([1, 0, 0], dtype=np.complex128)
        if polarization == LightPolarization.LINEAR_Y:
            return np.array([0, 1, 0], dtype=np.complex128)
        if polarization == LightPolarization.LINEAR_Z:
            return np.array([0, 0, 0], dtype=np.complex128)
        if polarization == LightPolarization.LINEAR_XY:
            return np.array([1, 1, 0], dtype=np.complex128) / np.sqrt(2)
        if polarization == LightPolarization.LINEAR_XZ:
            return np.array([1, 0, 0], dtype=np.complex128) / np.sqrt(2)
        if polarization == LightPolarization.LINEAR_YZ:
            return np.array([0, 1, 0], dtype=np.complex128) / np.sqrt(2)
        if polarization == LightPolarization.CHIRAL_PLUS:
            return np.array([1, 1j, 0], dtype=np.complex128) / np.sqrt(2)
        if polarization == LightPolarization.CHIRAL_MINUS:
            return np.array([1, -1j, 0], dtype=np.complex128) / np.sqrt(2)
        raise ValueError("Unsupported polarization")
    else:
        # Normalize the vector
        polarization = np.array([polarization[0], polarization[1], 0], dtype=np.complex128)
        norm = np.linalg.norm(polarization)
        if np.isclose(norm, 0):
            raise ValueError("The polarization vector cannot be the zero vector")
        return (polarization / norm).astype(np.complex128)

def polarization_vector_to_directions(polarization: npt.NDArray[np.complex128]) -> LightPolarization:
    """
    Converts a 3D polarization vector to a ModePolarization enum value.

    Args:
        polarization: A 3-element complex array representing the polarization vector.
                      Must be non-zero and have a valid combination of non-zero components.

    Returns:
        The corresponding ModePolarization enum value.

    Raises:
        ValueError: If the polarization vector is zero, has an invalid shape,
                    or does not map to a known ModePolarization.
    """
    if polarization.ndim != 1 or polarization.shape[0] != 3:
        raise ValueError(
            f"Polarization vector must be a 1D array of length 3, "
            f"got shape {polarization.shape}."
        )
    if not np.isclose(np.linalg.norm(polarization), 1.0):
        raise ValueError(
            f"Polarization vector must be normalized (unit norm), "
            f"got norm {np.linalg.norm(polarization):.6f}."
        )
    pol = ""
    if not np.isclose(polarization[0], 0):
        pol += "x"
    if not np.isclose(polarization[1], 0):
        pol += "y"
    if not np.isclose(polarization[2], 0):
        pol += "z"
    if not pol:
        raise ValueError(
            "Polarization vector is effectively zero after tolerance check. "
            "Cannot determine polarization direction."
        )
    try:
        return LightPolarization(pol)
    except ValueError:
        raise ValueError(
            f"Polarization vector {polarization} maps to direction string '{pol}', "
            f"which does not correspond to any ModePolarization member. "
            f"Valid members are: {[m.value for m in LightPolarization]}."
        )


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
    modes_polarization: LightPolarization | tuple[np.complex128, np.complex128]
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

