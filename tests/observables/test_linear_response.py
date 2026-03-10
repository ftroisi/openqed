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
from typing import Dict, Literal

import numpy as np
import numpy.typing as npt

from openqed.observables import LinearResponse
from openqed.observables.observable import ObservableFileKey, PolaritonicStatesRestartKeys
from openqed.observables.linear_response import LinearResponseFileKey
from openqed.hamiltonians.types.cavity import LightPolarization
from tests.openqed_test import BaseOpenqedTest


class TestLinearResponse(BaseOpenqedTest):
    """Test class for the LinearResponse observable."""

    def setUp(self) -> None:
        super().setUp()
        self.w_probe = np.array([0.0, 2.0], dtype=np.float64)
        self.eta = 0.1
        self.eigenvalues = np.array([0.0, 2.0], dtype=np.float64)
        self.eigenvectors = np.eye(2, dtype=np.complex128)

        self.probe_operator: Dict[Literal["x", "y", "z"], npt.NDArray[np.complex128]] = {
            "x": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
            "y": np.zeros((2, 2), dtype=np.complex128),
            "z": np.zeros((2, 2), dtype=np.complex128)
        }

    def test_linear_response_from_direct_states(self):
        """Linear response should match the analytical two-level model."""
        linear_response = LinearResponse(
            omega_probe=self.w_probe,
            eta=self.eta,
            probe_polarization=LightPolarization.LINEAR_X,
            probe_operator=self.probe_operator,
            eigenvalues=self.eigenvalues,
            eigenvectors=self.eigenvectors
        )
        response = linear_response.compute()

        # Kubo response: both absorption and emission
        # For omega=0: chi = eta/(4+eta^2) - eta/(4+eta^2) = 0
        # For omega=2: chi = 1/eta - eta/(16+eta^2)
        expected = np.array([
            0.0,
            1.0 / self.eta - self.eta / (16.0 + self.eta ** 2)
        ], dtype=np.float64)
        self.assertTrue(np.allclose(response, expected, atol=1e-12))

    def test_linear_response_from_states_file(self):
        """States should be loadable from a restart file in __init__."""
        with tempfile.TemporaryDirectory() as tmpdir:
            states_path = os.path.join(tmpdir, "states.npz")
            np.savez(
                states_path,
                **{
                    PolaritonicStatesRestartKeys.EIGENVALUES.value: self.eigenvalues,
                    PolaritonicStatesRestartKeys.EIGENVECTORS.value: self.eigenvectors
                }
            )

            linear_response = LinearResponse(
                omega_probe=self.w_probe,
                eta=self.eta,
                probe_polarization=LightPolarization.LINEAR_X,
                probe_operator=self.probe_operator,
                states_path=states_path
            )
            response = linear_response.compute()

            # Kubo response: both absorption and emission
            expected_peak = 1.0 / self.eta - self.eta / (16.0 + self.eta ** 2)
            self.assertTrue(np.isclose(response[1], expected_peak, atol=1e-12))

    def test_save_results(self):
        """Computed response should be saved to npz with metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            linear_response = LinearResponse(
                omega_probe=self.w_probe,
                eta=self.eta,
                probe_polarization=LightPolarization.LINEAR_X,
                probe_operator=self.probe_operator,
                eigenvalues=self.eigenvalues,
                eigenvectors=self.eigenvectors,
                results_path=tmpdir
            )
            response = linear_response.compute()
            output_path = linear_response.save(filename="linear_response.npz", data=response)

            self.assertTrue(os.path.isfile(output_path))
            loaded = np.load(output_path)
            self.assertTrue(np.allclose(loaded[LinearResponseFileKey.OMEGA_PROBE.value], self.w_probe))
            self.assertTrue(np.allclose(loaded[ObservableFileKey.OBSERVABLE.value], response))
            self.assertTrue(np.isclose(loaded[LinearResponseFileKey.ETA.value], self.eta))
            self.assertEqual(str(loaded[LinearResponseFileKey.PROBE_POLARIZATION.value]), "x")

    def test_spectral_peaks(self):
        """Response should show clear absorption peaks at the positive transition frequencies.

        A three-level system is used so that two distinct resonances can be observed
        simultaneously.  The probe operator couples only the ground state to the two
        excited states, with different matrix elements so the relative peak heights
        are also testable.

        The full Kubo formula is used (counter-rotating / emission terms are kept),
        which slightly reduces the peak heights and can produce a small negative
        background between peaks, but the dominant features are the two positive
        absorption peaks.

        System:
            E_0 = 0,  E_1 = 1.5,  E_2 = 3.0  (Hartree)
            |<0|P_x|1>|^2 = 1.0  →  peak height at omega=1.5 ≈ 1/eta = 20
            |<0|P_x|2>|^2 = 0.25 →  peak height at omega=3.0 ≈ 0.25/eta = 5
        """
        eta = 0.05
        eigenvalues = np.array([0.0, 1.5, 3.0], dtype=np.float64)
        eigenvectors = np.eye(3, dtype=np.complex128)

        probe_x = np.zeros((3, 3), dtype=np.complex128)
        probe_x[0, 1] = probe_x[1, 0] = 1.0   # |<0|P|1>|^2 = 1.0
        probe_x[0, 2] = probe_x[2, 0] = 0.5   # |<0|P|2>|^2 = 0.25
        probe_operator: Dict[Literal["x", "y", "z"], npt.NDArray[np.complex128]] = {
            "x": probe_x,
            "y": np.zeros((3, 3), dtype=np.complex128),
            "z": np.zeros((3, 3), dtype=np.complex128),
        }

        omega_probe = np.linspace(0.3, 3.8, 400)
        linear_response = LinearResponse(
            omega_probe=omega_probe,
            eta=eta,
            probe_polarization=LightPolarization.LINEAR_X,
            probe_operator=probe_operator,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
        )
        response = linear_response.compute()

        # Locate the two absorption peaks.
        mask1 = (omega_probe > 1.0) & (omega_probe < 2.0)
        mask2 = (omega_probe > 2.5) & (omega_probe < 3.5)
        peak1_omega = omega_probe[mask1][np.argmax(response[mask1])]
        peak2_omega = omega_probe[mask2][np.argmax(response[mask2])]
        peak1_val = response[mask1].max()
        peak2_val = response[mask2].max()

        # Peak positions should coincide with the transition energies E_1-E_0 and E_2-E_0.
        self.assertAlmostEqual(peak1_omega, 1.5, delta=0.05,
                               msg="First absorption peak should be at E_1 - E_0 = 1.5")
        self.assertAlmostEqual(peak2_omega, 3.0, delta=0.05,
                               msg="Second absorption peak should be at E_2 - E_0 = 3.0")

        # Peak heights: near resonance chi ≈ |<0|P|n>|^2/eta (Kubo formula slightly corrects
        # this via the counter-rotating term, so an absolute tolerance of 1.0 is sufficient).
        self.assertAlmostEqual(peak1_val, 1.0 / eta, delta=1.0,
                               msg="Absorption peak-1 height should be ≈ |M_01|^2/eta")
        self.assertAlmostEqual(peak2_val, 0.25 / eta, delta=1.0,
                               msg="Absorption peak-2 height should be ≈ |M_02|^2/eta")

        # Both peaks must be positive (absorption dominates at positive probe frequencies).
        self.assertGreater(peak1_val, 0.0, msg="Absorption peak-1 must be positive")
        self.assertGreater(peak2_val, 0.0, msg="Absorption peak-2 must be positive")

        # The peak with the larger matrix element must be taller.
        self.assertGreater(peak1_val, peak2_val,
                           msg="Peak at 1.5 (larger matrix element) should dominate")

    def test_response_is_real(self):
        """The linear response (imaginary part of susceptibility) must be real-valued"""
        linear_response = LinearResponse(
            omega_probe=self.w_probe,
            eta=self.eta,
            probe_polarization=LightPolarization.LINEAR_X,
            probe_operator=self.probe_operator,
            eigenvalues=self.eigenvalues,
            eigenvectors=self.eigenvectors
        )
        response = linear_response.compute()
        self.assertTrue(np.all(np.isreal(response)))

    def test_negative_eta_raises(self):
        """Negative broadening eta must raise ValueError"""
        with self.assertRaises(ValueError):
            LinearResponse(
                omega_probe=self.w_probe,
                eta=-0.1,
                probe_polarization=LightPolarization.LINEAR_X,
                probe_operator=self.probe_operator,
                eigenvalues=self.eigenvalues,
                eigenvectors=self.eigenvectors
            )

    def test_save_bad_extension_raises(self):
        """Saving to a non-.npz file must raise ValueError"""
        linear_response = LinearResponse(
            omega_probe=self.w_probe,
            eta=self.eta,
            probe_polarization=LightPolarization.LINEAR_X,
            probe_operator=self.probe_operator,
            eigenvalues=self.eigenvalues,
            eigenvectors=self.eigenvectors
        )
        response = linear_response.compute()
        with self.assertRaises(ValueError):
            linear_response.save(filename="output.txt", data=response)

    def test_load_states_missing_file_raises(self):
        """Loading states from a non-existent file must raise FileNotFoundError"""
        from openqed.observables.observable import Observable
        with self.assertRaises(FileNotFoundError):
            Observable._load_states("/tmp/nonexistent_states_file.npz")

    def test_load_states_bad_extension_raises(self):
        """Loading states from a non-.npz file must raise ValueError"""
        with tempfile.NamedTemporaryFile(suffix=".txt") as f:
            from openqed.observables.observable import Observable
            with self.assertRaises(ValueError):
                Observable._load_states(f.name)


if __name__ == "__main__":
    unittest.main()
