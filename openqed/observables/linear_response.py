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

"""This module contains the LinearResponse observable class."""

from __future__ import annotations

from typing import Literal
import os

import numpy as np
import numpy.typing as npt

from .observable import Observable

ProbePolarization = Literal["x", "y", "z", "xy", "xz", "yz", "+", "-"]
ProbeOperatorType = (npt.NDArray[np.float64] |
                     npt.NDArray[np.complex128] |
                     dict[Literal["x", "y", "z"], npt.NDArray[np.float64] | npt.NDArray[np.complex128]])


def _direction_to_index(direction: str) -> int:
    if direction == "x":
        return 0
    if direction == "y":
        return 1
    if direction == "z":
        return 2
    raise ValueError(f"Unsupported direction '{direction}'.")


class LinearResponse(Observable):
    """
    This class computes the linear response function from precomputed polaritonic states.

    The class does not perform Hamiltonian diagonalization. Polaritonic states can be provided
    directly via ``eigenvalues`` and ``eigenvectors`` or loaded from a ``.npz`` restart file that
    contains ``eigenval`` and ``eigenvec`` arrays.
    """

    def __init__(
            self,
            *,
            w_probe: npt.NDArray[np.float64],
            eta: float,
            polarization: ProbePolarization = "x",
            eigenvalues: npt.NDArray[np.float64] | None = None,
            eigenvectors: npt.NDArray[np.float64] | npt.NDArray[np.complex128] | None = None,
            states_path: str | None = None,
            results_path: str | None = None
    ) -> None:
        super().__init__(name="LinearResponse", results_path=results_path)

        self.w_probe = np.asarray(w_probe, dtype=np.float64)
        if self.w_probe.ndim != 1 or self.w_probe.shape[0] == 0:
            raise ValueError("`w_probe` must be a non-empty 1D array.")

        if eta <= 0:
            raise ValueError("`eta` must be strictly positive.")
        self.eta: np.float64 = np.float64(eta)

        self.polarization: ProbePolarization = polarization

        if states_path is not None and (eigenvalues is not None or eigenvectors is not None):
            raise ValueError(
                "Specify either `states_path` or (`eigenvalues`, `eigenvectors`), but not both."
            )

        if states_path is not None:
            self.eigenvalues, self.eigenvectors = self._load_states(states_path)
        else:
            if eigenvalues is None or eigenvectors is None:
                raise ValueError(
                    "Both `eigenvalues` and `eigenvectors` must be provided when `states_path` is None."
                )
            self.eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
            self.eigenvectors = np.asarray(eigenvectors, dtype=np.complex128)

        self._validate_states()
        self._sort_states_by_energy()

        self._last_response: npt.NDArray[np.float64] | None = None

    def compute(
            self,
            probe_operator: ProbeOperatorType | None = None,
            number_of_states: int | None = None
        ) -> npt.NDArray[np.float64]:
        """
        Compute the linear response function for the selected probe polarization.

        For now, only transitions from the ground state to the excited states (0 -> m) are
        included.

        Args:
            probe_operator: Probe operator in the polaritonic basis. Supported formats are:
                - 2D array ``(N, N)``: already polarization-projected operator.
                - 3D array ``(3, N, N)``: operators along ``x, y, z`` directions.
                - dict with keys ``'x'``, ``'y'``, ``'z'`` and values ``(N, N)`` arrays.
            number_of_states: Number of polaritonic states to include from the lowest-energy ones.
                If ``None``, all available states are used.

        Returns:
            The linear response function evaluated on ``self.w_probe``.
        """
        if probe_operator is None:
            raise ValueError("`probe_operator` must be provided.")

        n_states = self.eigenvalues.shape[0] if number_of_states is None else number_of_states
        if n_states < 2 or n_states > self.eigenvalues.shape[0]:
            raise ValueError(
                f"`number_of_states` must be in [2, {self.eigenvalues.shape[0]}], got {n_states}."
            )

        energies: npt.NDArray[np.float64] = self.eigenvalues[:n_states]
        states: npt.NDArray[np.complex128] = self.eigenvectors[:, :n_states]

        projected_probe_operator = self._project_probe_operator(probe_operator, expected_size=states.shape[0])
        transition_matrix = np.conj(states.T) @ projected_probe_operator @ states

        transition_elements: npt.NDArray[np.float64] = np.abs(transition_matrix[0, 1:]) ** 2
        energy_differences: npt.NDArray[np.float64] = energies[1:] - energies[0]

        non_zero_energy_diff = np.abs(energy_differences) > 1e-14
        if np.any(~non_zero_energy_diff):
            energy_differences = energy_differences[non_zero_energy_diff]
            transition_elements = transition_elements[non_zero_energy_diff]

        if transition_elements.shape[0] == 0:
            response = np.zeros_like(self.w_probe, dtype=np.float64)
            self._last_response = response
            return response

        denominator = self.w_probe[:, np.newaxis] - energy_differences[np.newaxis, :] + 1j * self.eta
        response = -np.imag(np.sum(transition_elements[np.newaxis, :] / denominator, axis=1))
        self._last_response = response
        return response

    def save_results(
            self,
            *,
            filename: str,
            response: npt.NDArray[np.float64] | None = None
    ) -> str:
        """
        Save linear response results to ``.npz`` format.

        Args:
            filename: Output file name (or absolute path). Must end with ``.npz``.
            response: Response values to save. If ``None``, the most recent value computed
                by :meth:`compute` is used.

        Returns:
            The absolute path of the written file.
        """
        if response is None:
            if self._last_response is None:
                raise ValueError("No response available. Run `compute` first or pass `response` explicitly.")
            response_data = self._last_response
        else:
            response_data = np.asarray(response, dtype=np.float64)

        if response_data.shape != self.w_probe.shape:
            raise ValueError(
                f"`response` must have shape {self.w_probe.shape}, got {response_data.shape}."
            )

        output_path = self.resolve_output_path(filename)
        if not output_path.endswith(".npz"):
            raise ValueError("Output filename must end with '.npz'.")

        np.savez(
            output_path,
            w_probe=self.w_probe,
            response=response_data,
            eta=self.eta,
            polarization=self.polarization
        )
        return output_path

    @staticmethod
    def _load_states(states_path: str
        ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.complex128]]:
        if not os.path.exists(states_path):
            raise FileNotFoundError(f"States file not found: {states_path}")
        if not states_path.endswith(".npz"):
            raise ValueError("`states_path` must point to a '.npz' file.")

        data = np.load(states_path)
        if "eigenval" not in data or "eigenvec" not in data:
            raise ValueError("States file must contain `eigenval` and `eigenvec` arrays.")

        eigenvalues = np.asarray(data["eigenval"], dtype=np.float64)
        eigenvectors = np.asarray(data["eigenvec"], dtype=np.complex128)
        return eigenvalues, eigenvectors

    def _validate_states(self) -> None:
        if self.eigenvalues.ndim != 1 or self.eigenvalues.shape[0] == 0:
            raise ValueError("`eigenvalues` must be a non-empty 1D array.")
        if self.eigenvectors.ndim != 2:
            raise ValueError("`eigenvectors` must be a 2D array.")

        n_states = self.eigenvalues.shape[0]
        if self.eigenvectors.shape[1] != n_states:
            raise ValueError(
                "`eigenvectors` columns must match the number of `eigenvalues`. "
                f"Got eigenvectors shape {self.eigenvectors.shape} and {n_states} eigenvalues."
            )

    def _sort_states_by_energy(self) -> None:
        sorted_indices = np.argsort(self.eigenvalues)
        self.eigenvalues = self.eigenvalues[sorted_indices]
        self.eigenvectors = self.eigenvectors[:, sorted_indices]

    def _project_probe_operator(
            self,
            probe_operator: ProbeOperatorType,
            *,
            expected_size: int
    ) -> npt.NDArray[np.complex128]:
        if isinstance(probe_operator, dict):
            return self._project_probe_operator_from_dict(probe_operator, expected_size=expected_size)

        probe_operator_array = np.asarray(probe_operator, dtype=np.complex128)
        if probe_operator_array.ndim == 2:
            if (probe_operator_array.shape[0] != expected_size or
                    probe_operator_array.shape[1] != expected_size):
                raise ValueError(
                    f"Probe operator shape must be ({expected_size}, {expected_size}), "
                    f"got {probe_operator_array.shape}."
                )
            return probe_operator_array

        if probe_operator_array.ndim == 3:
            if probe_operator_array.shape[0] != 3:
                raise ValueError(
                    "A 3D probe operator must have shape (3, N, N), with cartesian components "
                    "ordered as (x, y, z)."
                )
            if (probe_operator_array.shape[1] != expected_size or
                    probe_operator_array.shape[2] != expected_size):
                raise ValueError(
                    "Probe operator dimensions are inconsistent with the eigenvectors size. "
                    f"Expected (3, {expected_size}, {expected_size}), got {probe_operator_array.shape}."
                )
            return self._project_probe_operator_from_components(probe_operator_array)

        raise ValueError("Unsupported `probe_operator` format.")

    def _project_probe_operator_from_dict(
            self,
            probe_operator: dict[Literal["x", "y", "z"], npt.NDArray[np.float64] | npt.NDArray[np.complex128]],
            *,
            expected_size: int
    ) -> npt.NDArray[np.complex128]:
        components: npt.NDArray[np.complex128] = np.zeros((3, expected_size, expected_size), dtype=np.complex128)
        for idx, key in enumerate(["x", "y", "z"]):
            if key not in probe_operator:
                continue
            component = np.asarray(probe_operator[key], dtype=np.complex128)
            if component.shape != (expected_size, expected_size):
                raise ValueError(
                    f"Probe operator component '{key}' has wrong shape {component.shape}; "
                    f"expected ({expected_size}, {expected_size})."
                )
            components[idx] = component

        return self._project_probe_operator_from_components(components)

    def _project_probe_operator_from_components(
            self,
            probe_operator: npt.NDArray[np.complex128]
    ) -> npt.NDArray[np.complex128]:
        if self.polarization in ["x", "y", "z"]:
            return probe_operator[_direction_to_index(self.polarization), :, :]
        if self.polarization in ["xy", "xz", "yz"]:
            first_idx = _direction_to_index(self.polarization[0])
            second_idx = _direction_to_index(self.polarization[1])
            return (probe_operator[first_idx, :, :] + probe_operator[second_idx, :, :]) / np.sqrt(2)
        if self.polarization == "+":
            return (probe_operator[0, :, :] + 1j * probe_operator[1, :, :]) / np.sqrt(2)
        if self.polarization == "-":
            return (probe_operator[0, :, :] - 1j * probe_operator[1, :, :]) / np.sqrt(2)
        raise ValueError(f"Unsupported polarization '{self.polarization}'.")
