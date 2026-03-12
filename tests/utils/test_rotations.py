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

"""Tests for the rotations utility module"""

import unittest
import numpy as np

from tests.openqed_test import BaseOpenqedTest
from openqed.utils.rotations import get_2d_rotation_matrix, rotate_vector


class TestRotationMatrix(BaseOpenqedTest):
    """Test class for get_2d_rotation_matrix"""

    def test_identity_rotation(self):
        """A 0-degree rotation must be the identity matrix"""
        rotation = get_2d_rotation_matrix(np.float64(0.0))
        self.assertEqual(rotation.shape, (2, 2))
        self.assertTrue(np.allclose(rotation, np.eye(2)))

    def test_90_degree_rotation(self):
        """A 90-degree rotation matrix must satisfy R @ [1, 0] = [0, 1]"""
        rotation = get_2d_rotation_matrix(np.float64(90.0))
        rotated = rotation @ np.array([1.0, 0.0])
        self.assertTrue(np.allclose(rotated, [0.0, 1.0], atol=1e-14))

    def test_180_degree_rotation(self):
        """A 180-degree rotation must negate the vector"""
        rotation = get_2d_rotation_matrix(np.float64(180.0))
        rotated = rotation @ np.array([1.0, 0.0])
        self.assertTrue(np.allclose(rotated, [-1.0, 0.0], atol=1e-14))

    def test_360_degree_rotation(self):
        """A 360-degree rotation must return to identity"""
        rotation = get_2d_rotation_matrix(np.float64(360.0))
        self.assertTrue(np.allclose(rotation, np.eye(2), atol=1e-14))

    def test_rotation_matrix_is_orthogonal(self):
        """R^T R must be the identity and det(R) must be 1 (proper rotation)"""
        for angle in [0.0, 30.0, 45.0, 60.0, 90.0, 123.456]:
            rotation = get_2d_rotation_matrix(np.float64(angle))
            self.assertTrue(np.allclose(rotation.T @ rotation, np.eye(2), atol=1e-14))
            self.assertTrue(np.isclose(np.linalg.det(rotation), 1.0, atol=1e-14))

    def test_rotation_preserves_norm(self):
        """Rotation must preserve the vector norm"""
        rotation = get_2d_rotation_matrix(np.float64(47.0))
        v = np.array([3.0, 4.0])
        rotated = rotation @ v
        self.assertTrue(np.isclose(np.linalg.norm(rotated), np.linalg.norm(v)))

    def test_inverse_rotation(self):
        """Rotating by theta then by -theta must restore the original vector"""
        theta = np.float64(37.5)
        rotation = get_2d_rotation_matrix(theta)
        inverse_rotation = get_2d_rotation_matrix(-theta)
        v = np.array([2.5, -1.3])
        self.assertTrue(np.allclose(inverse_rotation @ rotation @ v, v, atol=1e-14))


class TestRotateVector(BaseOpenqedTest):
    """Test class for rotate_vector"""

    def test_rotate_unit_x_by_90(self):
        """Rotating [1, 0] by 90 degrees must give [0, 1]"""
        v = np.array([1.0, 0.0])
        result = rotate_vector(np.float64(90.0), v)
        self.assertTrue(np.allclose(result, [0.0, 1.0], atol=1e-14))

    def test_rotate_arbitrary_vector(self):
        """Rotation by 60 degrees of [1, 0] must give [cos60, sin60]"""
        v = np.array([1.0, 0.0])
        result = rotate_vector(np.float64(60.0), v)
        expected = np.array([np.cos(np.radians(60)), np.sin(np.radians(60))])
        self.assertTrue(np.allclose(result, expected, atol=1e-14))

    def test_rotate_preserves_norm(self):
        """Rotation must preserve the norm of the vector"""
        v = np.array([3.0, 4.0])
        result = rotate_vector(np.float64(123.0), v)
        self.assertTrue(np.isclose(np.linalg.norm(result), np.linalg.norm(v)))

    def test_rotate_zero_angle(self):
        """Rotating by 0 degrees must return the original vector"""
        v = np.array([7.0, -2.0])
        result = rotate_vector(np.float64(0.0), v)
        self.assertTrue(np.allclose(result, v, atol=1e-14))

    def test_3d_vector_raises(self):
        """3D vectors are not yet supported and should raise"""
        with self.assertRaises(NotImplementedError):
            rotate_vector(np.float64(45.0), np.array([1.0, 0.0, 0.0]))

    def test_wrong_shape_raises(self):
        """Vectors with wrong dimensions must raise ValueError"""
        with self.assertRaises(ValueError):
            rotate_vector(np.float64(45.0), np.array([1.0]))
        with self.assertRaises(ValueError):
            rotate_vector(np.float64(45.0), np.array([[1.0, 0.0]]))


if __name__ == "__main__":
    unittest.main()
