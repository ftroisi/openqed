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

"""Tests for UnitCell and HexagonalUnitCell"""

import unittest
import numpy as np
from ase.units import Bohr

from tests.openqed_test import BaseOpenqedTest
from openqed.grid.grid import GridType
from openqed.grid.k_space_grid import KSpaceGrid
from openqed.unit_cells.hexagonal_unit_cell import HexagonalUnitCell

class TestHexagonalUnitCell(BaseOpenqedTest):
    """Test class for HexagonalUnitCell"""

    def setUp(self) -> None:
        super().setUp()
        # MoS2-like lattice parameter: 3.16 Angstrom -> atomic units
        self.a = np.float64(3.16) / Bohr
        self.cell = HexagonalUnitCell(self.a)

    # --- Construction ---

    def test_dimension(self):
        """The hexagonal unit cell must be 2D"""
        self.assertEqual(self.cell.dimension, 2)

    def test_lattice_parameter_stored(self):
        """The lattice parameter must be stored correctly"""
        self.assertTrue(np.isclose(self.cell.lattice_parameter, self.a))
        self.assertTrue(np.isclose(self.cell.a, self.a))

    def test_invalid_lattice_parameter_zero(self):
        """A zero lattice parameter must be rejected"""
        with self.assertRaises(ValueError):
            HexagonalUnitCell(np.float64(0.0))

    def test_invalid_lattice_parameter_negative(self):
        """A negative lattice parameter must be rejected"""
        with self.assertRaises(ValueError):
            HexagonalUnitCell(np.float64(-1.0))

    # --- Real-space lattice vectors ---

    def test_real_space_lattice_vectors_shape(self):
        """Real-space lattice vectors must be a 2x2 matrix"""
        vecs = self.cell.get_lattice_vectors(GridType.REAL_SPACE)
        self.assertEqual(vecs.shape, (2, 2))

    def test_real_space_vector_norms(self):
        """Both real-space lattice vectors must have length a"""
        vecs = self.cell.get_lattice_vectors(GridType.REAL_SPACE)
        for i in range(2):
            norm = np.linalg.norm(vecs[i])
            self.assertTrue(np.isclose(norm, self.a),
                            f"Vector {i} norm {norm} != a={self.a}")

    def test_real_space_angle_120_degrees(self):
        """The angle between hexagonal lattice vectors must be 60 degrees
        (the two vectors span a 60-degree angle in the standard convention)"""
        vecs = self.cell.get_lattice_vectors(GridType.REAL_SPACE)
        cos_angle = np.dot(vecs[0], vecs[1]) / (np.linalg.norm(vecs[0]) * np.linalg.norm(vecs[1]))
        angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
        # Standard hexagonal lattice has 60-degree angle between a1 and a2
        self.assertTrue(np.isclose(angle_deg, 60.0, atol=1e-10),
                        f"Angle {angle_deg} degrees, expected 60")

    def test_real_space_first_vector_along_x(self):
        """The first real-space vector must lie along the x-axis: [a, 0]"""
        vecs = self.cell.get_lattice_vectors(GridType.REAL_SPACE)
        self.assertTrue(np.isclose(vecs[0, 0], self.a))
        self.assertTrue(np.isclose(vecs[0, 1], 0.0))

    # --- Reciprocal-space lattice vectors ---

    def test_k_space_lattice_vectors_shape(self):
        """K-space lattice vectors must be a 2x2 matrix"""
        vecs = self.cell.get_lattice_vectors(GridType.K_SPACE)
        self.assertEqual(vecs.shape, (2, 2))

    def test_orthogonality_relation(self):
        """Real and reciprocal lattice vectors must satisfy a_i . b_j = 2*pi*delta_ij"""
        real_vecs = self.cell.get_lattice_vectors(GridType.REAL_SPACE)
        k_vecs = self.cell.get_lattice_vectors(GridType.K_SPACE)
        product = real_vecs @ k_vecs.T
        expected = 2 * np.pi * np.eye(2)
        self.assertTrue(np.allclose(product, expected, atol=1e-10),
                        f"Orthogonality relation violated: {product}")

    def test_k_space_vector_norms(self):
        """Both reciprocal lattice vectors must have the same length (hexagonal symmetry)"""
        k_vecs = self.cell.get_lattice_vectors(GridType.K_SPACE)
        norm0 = np.linalg.norm(k_vecs[0])
        norm1 = np.linalg.norm(k_vecs[1])
        self.assertTrue(np.isclose(norm0, norm1, atol=1e-10))
        # The reciprocal lattice parameter for a hexagonal cell: b = 4*pi / (a*sqrt(3))
        expected_b = 4 * np.pi / (self.a * np.sqrt(3))
        self.assertTrue(np.isclose(norm0, expected_b, atol=1e-10),
                        f"|b| = {norm0}, expected {expected_b}")

    def test_k_space_angle_120_degrees(self):
        """The angle between reciprocal lattice vectors should be 120 degrees for hexagonal"""
        k_vecs = self.cell.get_lattice_vectors(GridType.K_SPACE)
        cos_angle = np.dot(k_vecs[0], k_vecs[1]) / (np.linalg.norm(k_vecs[0]) * np.linalg.norm(k_vecs[1]))
        angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
        self.assertTrue(np.isclose(angle_deg, 120.0, atol=1e-10),
                        f"Reciprocal angle {angle_deg} degrees, expected 120")

    # --- Unit cell area ---

    def test_real_space_unit_cell_area(self):
        """The area of the hexagonal real-space unit cell must be a^2 * sqrt(3)/2"""
        vecs = self.cell.get_lattice_vectors(GridType.REAL_SPACE)
        v0 = np.append(vecs[0], 0.0)
        v1 = np.append(vecs[1], 0.0)
        area = abs(np.cross(v0, v1)[2])
        expected_area = self.a**2 * np.sqrt(3) / 2
        self.assertTrue(np.isclose(area, expected_area, atol=1e-10),
                        f"Area {area}, expected {expected_area}")

    def test_real_times_reciprocal_area(self):
        """Product of real and reciprocal cell areas should be (2*pi)^2"""
        real_vecs = self.cell.get_lattice_vectors(GridType.REAL_SPACE)
        k_vecs = self.cell.get_lattice_vectors(GridType.K_SPACE)
        r0 = np.append(real_vecs[0], 0.0)
        r1 = np.append(real_vecs[1], 0.0)
        k0 = np.append(k_vecs[0], 0.0)
        k1 = np.append(k_vecs[1], 0.0)
        real_area = abs(np.cross(r0, r1)[2])
        k_area = abs(np.cross(k0, k1)[2])
        self.assertTrue(np.isclose(real_area * k_area, (2 * np.pi)**2, atol=1e-8))

    # --- fit_lattice_vectors_to_grid ---

    def test_fit_lattice_vectors_to_grid(self):
        """Fitted vectors must have magnitude equal to the inscribed circle radius"""
        grid = KSpaceGrid(
            boundaries={
                'x': (np.float64(-0.05), np.float64(0.05)),
                'y': (np.float64(-0.05), np.float64(0.05))
            },
            spacing={'x': np.float64(0.005), 'y': np.float64(0.005)}
        )
        fitted = self.cell.fit_lattice_vectors_to_grid(grid)
        radius = grid.get_max_inscribed_circle_radius()
        # The norm of the first fitted vector should equal the radius
        self.assertTrue(np.isclose(np.linalg.norm(fitted[0]), radius, atol=1e-10))
        # Direction must be preserved: fitted vectors should be parallel to the originals
        k_vecs = self.cell.get_lattice_vectors(GridType.K_SPACE)
        for i in range(2):
            cos_angle = np.dot(fitted[i], k_vecs[i]) / (
                np.linalg.norm(fitted[i]) * np.linalg.norm(k_vecs[i]))
            self.assertTrue(np.isclose(abs(cos_angle), 1.0, atol=1e-10))


if __name__ == "__main__":
    unittest.main()
