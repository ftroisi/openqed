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

"""Tests for the MatterHamiltonianTerm class"""

import unittest
import numpy as np

from tests.openqed_test import BaseOpenqedTest
from tests.hamiltonians.dummy_hamiltonian import DummyHamiltonian

from openqed.grid.k_space_grid import KSpaceGrid
from openqed.hamiltonians.terms.matter_hamiltonian_term import MatterHamiltonianTerm


class TestMatterHamiltonianTerm(BaseOpenqedTest):
    """Test class for MatterHamiltonianTerm"""

    def setUp(self) -> None:
        super().setUp()
        spacing = {'x': np.float64(0.01), 'y': np.float64(0.01)}
        boundaries = {
            'x': (np.float64(-0.02), np.float64(0.02)),
            'y': (np.float64(-0.02), np.float64(0.02))
        }
        k_space_grid = KSpaceGrid(spacing=spacing, boundaries=boundaries)
        k_space_grid.flat_grid(cache_result=True)
        self.hamiltonian = DummyHamiltonian(terms=[], grid=k_space_grid)

    def test_get_hamiltonian_term_shape(self):
        """The expanded matrix must be (N+1) x (N+1) where N is the matter Hamiltonian size"""
        term = MatterHamiltonianTerm(self.hamiltonian)
        result = term.get_hamiltonian_term()
        n = self.hamiltonian.get_hamiltonian_size()
        self.assertEqual(result.shape, (n + 1, n + 1))

    def test_ground_state_row_col_zero(self):
        """The first row and column (polaritonic ground state) should be zero for a zero Hamiltonian"""
        term = MatterHamiltonianTerm(self.hamiltonian)
        result = term.get_hamiltonian_term()
        self.assertTrue(np.allclose(result[0, :], 0.0))
        self.assertTrue(np.allclose(result[:, 0], 0.0))

    def test_hermiticity(self):
        """The expanded matrix must remain Hermitian"""
        term = MatterHamiltonianTerm(self.hamiltonian)
        result = term.get_hamiltonian_term()
        self.assertTrue(np.allclose(result, result.conj().T, atol=1e-12))

    def test_from_matter_hamiltonian_factory(self):
        """The classmethod factory should produce the same result"""
        term = MatterHamiltonianTerm.from_matter_hamiltonian(self.hamiltonian)
        result = term.get_hamiltonian_term()
        n = self.hamiltonian.get_hamiltonian_size()
        self.assertEqual(result.shape, (n + 1, n + 1))


if __name__ == "__main__":
    unittest.main()
