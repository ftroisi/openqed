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

"""Tests for the Hamiltonian base class"""

import os
import tempfile
import unittest
import numpy as np

from tests.openqed_test import BaseOpenqedTest
from openqed.hamiltonians.hamiltonian import Hamiltonian


class TestDiagonalizeHamiltonian(BaseOpenqedTest):
    """Test the classmethod diagonalize_hamiltonian on known matrices"""

    def test_identity_matrix(self):
        """Diagonalizing the identity must return all-ones eigenvalues"""
        H = np.eye(3, dtype=np.float64)
        eigenval, eigenvec = Hamiltonian.diagonalize_hamiltonian(H)
        self.assertTrue(np.allclose(eigenval, np.ones(3)))
        # Eigenvectors should be orthonormal
        self.assertTrue(np.allclose(eigenvec @ eigenvec.T, np.eye(3), atol=1e-12))

    def test_diagonal_matrix(self):
        """Eigenvalues of a diagonal matrix must be the diagonal entries, sorted"""
        diag = np.array([3.0, 1.0, 2.0])
        H = np.diag(diag)
        eigenval, _ = Hamiltonian.diagonalize_hamiltonian(H)
        self.assertTrue(np.allclose(eigenval, np.sort(diag)))

    def test_hermitian_2x2(self):
        """Standard 2x2 Hermitian: eigenvalues must be real"""
        H = np.array([[1.0, 0.5], [0.5, 2.0]])
        eigenval, eigenvec = Hamiltonian.diagonalize_hamiltonian(H)
        self.assertTrue(np.all(np.isreal(eigenval)))
        # Verify H @ v = lambda * v
        for i in range(len(eigenval)):
            lhs = H @ eigenvec[:, i]
            rhs = eigenval[i] * eigenvec[:, i]
            self.assertTrue(np.allclose(lhs, rhs, atol=1e-12))

    def test_partial_diagonalization(self):
        """With number_of_eigenvalues, only that many values should be returned"""
        H = np.diag(np.arange(1.0, 11.0))
        eigenval, eigenvec = Hamiltonian.diagonalize_hamiltonian(H, number_of_eigenvalues=3)
        self.assertEqual(len(eigenval), 3)
        self.assertEqual(eigenvec.shape[1], 3)
        # Should be the 3 smallest eigenvalues (which="SM")
        self.assertTrue(np.allclose(np.sort(eigenval), [1.0, 2.0, 3.0], atol=1e-5))

    def test_restart_save_and_load(self):
        """diagonalize_hamiltonian should save to and load from restart file"""
        H = np.array([[4.0, 1.0], [1.0, 3.0]])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "restart.npz")
            # First call: computes and saves
            eigenval1, eigenvec1 = Hamiltonian.diagonalize_hamiltonian(H, restart_path=path)
            self.assertTrue(os.path.exists(path))
            # Second call: loads from file
            eigenval2, eigenvec2 = Hamiltonian.diagonalize_hamiltonian(H, restart_path=path)
            self.assertTrue(np.allclose(eigenval1, eigenval2))
            self.assertTrue(np.allclose(eigenvec1, eigenvec2))

    def test_restart_path_must_be_npz(self):
        """restart_path must end with .npz"""
        H = np.eye(2)
        with self.assertRaises(ValueError):
            Hamiltonian.diagonalize_hamiltonian(H, restart_path="/tmp/bad.txt")

    def test_eigenvalues_are_real(self):
        """Eigenvalues of a Hermitian matrix must be real"""
        # Complex Hermitian matrix
        H = np.array([[2.0+0j, 1.0-1j], [1.0+1j, 3.0+0j]])
        eigenval, _ = Hamiltonian.diagonalize_hamiltonian(H)
        self.assertTrue(np.all(np.isreal(eigenval)))
        self.assertGreater(eigenval[0], 0.0)


if __name__ == "__main__":
    unittest.main()
