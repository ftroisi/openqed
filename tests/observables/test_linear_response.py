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

"""This module contains tests for the LinearResponse class."""

import os
import tempfile
import unittest

import numpy as np

from openqed.observables import LinearResponse
from tests.openqed_test import BaseOpenqedTest


class TestLinearResponse(BaseOpenqedTest):
    """Test class for the LinearResponse observable."""

    def setUp(self) -> None:
        super().setUp()
        self.w_probe = np.array([0.0, 2.0], dtype=np.float64)
        self.eta = 0.1
        self.eigenvalues = np.array([0.0, 2.0], dtype=np.float64)
        self.eigenvectors = np.eye(2, dtype=np.complex128)

        self.probe_operator = np.zeros((3, 2, 2), dtype=np.complex128)
        self.probe_operator[0] = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)

    def test_linear_response_from_direct_states(self):
        """Linear response should match the analytical two-level model."""
        linear_response = LinearResponse(
            w_probe=self.w_probe,
            eta=self.eta,
            polarization="x",
            eigenvalues=self.eigenvalues,
            eigenvectors=self.eigenvectors
        )
        response = linear_response.compute(probe_operator=self.probe_operator)

        expected = np.array([
            self.eta / (4.0 + self.eta ** 2),
            1.0 / self.eta
        ], dtype=np.float64)
        self.assertTrue(np.allclose(response, expected, atol=1e-12))

    def test_linear_response_from_states_file(self):
        """States should be loadable from a restart file in __init__."""
        with tempfile.TemporaryDirectory() as tmpdir:
            states_path = os.path.join(tmpdir, "states.npz")
            np.savez(states_path, eigenval=self.eigenvalues, eigenvec=self.eigenvectors)

            linear_response = LinearResponse(
                w_probe=self.w_probe,
                eta=self.eta,
                polarization="x",
                states_path=states_path
            )
            response = linear_response.compute(probe_operator=self.probe_operator)

            expected_peak = 1.0 / self.eta
            self.assertTrue(np.isclose(response[1], expected_peak, atol=1e-12))

    def test_save_results(self):
        """Computed response should be saved to npz with metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            linear_response = LinearResponse(
                w_probe=self.w_probe,
                eta=self.eta,
                polarization="x",
                eigenvalues=self.eigenvalues,
                eigenvectors=self.eigenvectors,
                results_path=tmpdir
            )
            response = linear_response.compute(probe_operator=self.probe_operator)
            output_path = linear_response.save_results(filename="linear_response.npz", response=response)

            self.assertTrue(os.path.isfile(output_path))
            loaded = np.load(output_path)
            self.assertTrue(np.allclose(loaded["w_probe"], self.w_probe))
            self.assertTrue(np.allclose(loaded["response"], response))
            self.assertTrue(np.isclose(loaded["eta"], self.eta))
            self.assertEqual(str(loaded["polarization"]), "x")


if __name__ == "__main__":
    unittest.main()
