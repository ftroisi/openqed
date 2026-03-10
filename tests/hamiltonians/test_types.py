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

"""Tests for hamiltonians/types module"""

import unittest
import numpy as np

from tests.openqed_test import BaseOpenqedTest
from openqed.hamiltonians.types.bilayer_excitons import (
    excitons_key_to_structure_key,
    structure_key_to_exciton_key,
)
from openqed.hamiltonians.types.cavity import (
    LightPolarization,
    mode_polarization_to_vector,
    polarization_vector_to_directions,
    direction_to_index,
)


class TestBilayerKeyConversions(BaseOpenqedTest):
    """Test key conversion utilities for bilayer excitons"""

    def test_structure_to_exciton_layer1(self):
        self.assertEqual(structure_key_to_exciton_key("layer1"), "l1")

    def test_structure_to_exciton_layer2(self):
        self.assertEqual(structure_key_to_exciton_key("layer2"), "l2")

    def test_structure_to_exciton_invalid(self):
        with self.assertRaises(ValueError):
            structure_key_to_exciton_key("layer3")

    def test_exciton_to_structure_l1(self):
        self.assertEqual(excitons_key_to_structure_key("l1"), "layer1")

    def test_exciton_to_structure_l2(self):
        self.assertEqual(excitons_key_to_structure_key("l2"), "layer2")

    def test_exciton_to_structure_invalid(self):
        with self.assertRaises(ValueError):
            excitons_key_to_structure_key("l3")

    def test_roundtrip(self):
        """Converting back and forth must preserve the keys"""
        for struct_key in ["layer1", "layer2"]:
            exc = structure_key_to_exciton_key(struct_key)
            self.assertEqual(excitons_key_to_structure_key(exc), struct_key)


class TestCavityTypes(BaseOpenqedTest):
    """Test cavity type utilities"""

    def test_direction_to_index(self):
        self.assertEqual(direction_to_index("x"), 0)
        self.assertEqual(direction_to_index("y"), 1)
        self.assertEqual(direction_to_index("z"), 2)

    def test_direction_to_index_invalid(self):
        with self.assertRaises(ValueError):
            direction_to_index("w")

    def test_linear_x_polarization_vector(self):
        v = mode_polarization_to_vector(LightPolarization.LINEAR_X)
        self.assertTrue(np.allclose(v, [1, 0, 0]))

    def test_linear_y_polarization_vector(self):
        v = mode_polarization_to_vector(LightPolarization.LINEAR_Y)
        self.assertTrue(np.allclose(v, [0, 1, 0]))

    def test_linear_xy_polarization_vector(self):
        v = mode_polarization_to_vector(LightPolarization.LINEAR_XY)
        expected = np.array([1, 1, 0], dtype=np.complex128) / np.sqrt(2)
        self.assertTrue(np.allclose(v, expected))

    def test_chiral_plus_polarization_vector(self):
        v = mode_polarization_to_vector(LightPolarization.CHIRAL_PLUS)
        expected = np.array([1, 1j, 0], dtype=np.complex128) / np.sqrt(2)
        self.assertTrue(np.allclose(v, expected))

    def test_chiral_minus_polarization_vector(self):
        v = mode_polarization_to_vector(LightPolarization.CHIRAL_MINUS)
        expected = np.array([1, -1j, 0], dtype=np.complex128) / np.sqrt(2)
        self.assertTrue(np.allclose(v, expected))

    def test_polarization_vector_is_normalized(self):
        """Pure in-plane polarization vectors should have unit norm.

        LINEAR_Z, LINEAR_XZ, LINEAR_YZ have z-components that are discarded
        in the 2D cavity model, so their in-plane projections are not unit-normed.
        """
        in_plane = {LightPolarization.LINEAR_X, LightPolarization.LINEAR_Y,
                    LightPolarization.LINEAR_XY,
                    LightPolarization.CHIRAL_PLUS, LightPolarization.CHIRAL_MINUS}
        for pol in in_plane:
            v = mode_polarization_to_vector(pol)
            self.assertTrue(np.isclose(np.linalg.norm(v), 1.0),
                            f"{pol.name} vector norm = {np.linalg.norm(v)}")

    def test_tuple_polarization_normalized(self):
        """Custom tuple polarization should be normalized"""
        v = mode_polarization_to_vector((np.complex128(3.0), np.complex128(4.0)))
        self.assertTrue(np.isclose(np.linalg.norm(v), 1.0))

    def test_zero_tuple_polarization_raises(self):
        with self.assertRaises(ValueError):
            mode_polarization_to_vector((np.complex128(0.0), np.complex128(0.0)))

    def test_polarization_vector_to_directions_roundtrip(self):
        """For simple linear polarizations, roundtrip should work"""
        for pol in [LightPolarization.LINEAR_X, LightPolarization.LINEAR_Y,
                    LightPolarization.LINEAR_XY]:
            v = mode_polarization_to_vector(pol)
            recovered = polarization_vector_to_directions(v)
            self.assertEqual(recovered, pol)

    def test_polarization_vector_to_directions_unnormalized_raises(self):
        v = np.array([2.0, 0.0, 0.0], dtype=np.complex128)
        with self.assertRaises(ValueError):
            polarization_vector_to_directions(v)

    def test_polarization_vector_to_directions_wrong_shape_raises(self):
        v = np.array([1.0, 0.0], dtype=np.complex128)
        with self.assertRaises(ValueError):
            polarization_vector_to_directions(v)


if __name__ == "__main__":
    unittest.main()
