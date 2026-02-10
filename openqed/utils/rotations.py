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

"""This module contains useful functions for rotatating a vector or a matrix"""

import numpy as np
import numpy.typing as npt

def get_2d_rotation_matrix(theta: np.float64) -> npt.NDArray[np.float64]:
    """This method computes the 2D rotation matrix for a given angle theta.

    Args:
        theta: The angle in degrees.

    Returns:
        A 2x2 rotation matrix as a NumPy array.
    """
    theta_rad = np.radians(theta)
    return np.array([[np.cos(theta_rad), -np.sin(theta_rad)],
                     [np.sin(theta_rad),  np.cos(theta_rad)]], dtype=np.float64)

def rotate_vector(theta: np.float64, vector: npt.NDArray) -> npt.NDArray:
    """
    This method rotates a vector by multiplying by the rotation matrix

    Args:
        theta: the angle in degrees
        vector: the vector to rotate

    Returns:
        the rotated vector
    """
    if vector.ndim != 1 or vector.shape[0] not in (2, 3):
        raise ValueError("Only 2D and 3D vectors are supported")
    if vector.shape[0] == 2:
        rot_matrix = get_2d_rotation_matrix(theta)
    else:
        raise NotImplementedError("Only 2D vectors are supported")
    return rot_matrix.dot(vector.reshape(vector.shape[0], 1))
