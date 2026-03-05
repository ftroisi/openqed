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

"""Linear response observable for precomputed polaritonic states."""

from __future__ import annotations

from enum import Enum
from typing import Literal

import numpy as np
import numpy.typing as npt

from .observable import Observable
from openqed.hamiltonians.types.cavity import LightPolarization


class LinearResponseFileKey(str, Enum):
    """Keys used in ``LinearResponse`` output ``.npz`` files."""
    OMEGA_PROBE = "omega_probe"
    ETA = "eta"
    PROBE_POLARIZATION = "probe_polarization"


class LinearResponse(Observable):
    r"""
    Compute the optical linear response from precomputed polaritonic eigenstates.

    This class does **not** perform diagonalization; it only post-processes already available
    eigenvalues/eigenvectors. The implemented response keeps only ground-to-excited transitions
    (``0 -> I``):

    .. math::
        \chi(\omega) = -Im\left[\sum_{I>0}
        \frac{|\langle \Psi_0|\hat{P}_{\alpha}|\Psi_I\rangle|^2}
        {\omega - (E_I - E_0) + i\eta}\right]

    where :math:`\hat{P}_{\alpha}` is the probe operator projected onto the requested probe
    polarization :math:`\alpha`.
    """
    def __init__(
            self,
            *,
            omega_probe: npt.NDArray[np.float64],
            eta: float,
            probe_polarization: LightPolarization,
            probe_operator: dict[Literal["x", "y", "z"], npt.NDArray[np.float64 | np.complex128]],
            number_of_states: int | None = None,
            eigenvalues: npt.NDArray[np.float64] | None = None,
            eigenvectors: npt.NDArray[np.float64 | np.complex128] | None = None,
            states_path: str | None = None,
            results_path: str | None = None
    ) -> None:
        """
        Initialize the linear response observable.

        Args:
            omega_probe: Probe frequency grid where the response is evaluated.
            eta: Positive imaginary broadening.
            probe_polarization: Probe polarization selector used to project the operator components.
            probe_operator: Probe operator components as a dictionary with keys ``'x'``, ``'y'``, ``'z'`` (subset
                allowed depending on ``probe_polarization``). Values must be square matrices in the polaritonic basis.
            number_of_states: Number of lowest-energy states to include. If ``None``, all states are used.
            eigenvalues: Eigenvalues of the polaritonic Hamiltonian.
            eigenvectors: Eigenvectors of the polaritonic Hamiltonian (columns are eigenvectors).
            states_path: Optional restart path to an ``.npz`` file containing the polaritonic states and energies.
                Mutually exclusive with ``eigenvalues/eigenvectors``.
            results_path: Optional output directory for saved results.

        Raises:
            ValueError: If inputs are inconsistent or invalid.
        """
        super().__init__(name="LinearResponse", results_path=results_path)

        self.omega_probe = np.asarray(omega_probe, dtype=np.float64)
        if self.omega_probe.ndim != 1 or self.omega_probe.shape[0] == 0:
            raise ValueError("`omega_probe` must be a non-empty 1D array.")

        if eta <= 0:
            raise ValueError("`eta` must be strictly positive.")
        self.eta: np.float64 = np.float64(eta)

        self.polarization: LightPolarization = probe_polarization
        self.probe_operator: dict[Literal["x", "y", "z"], npt.NDArray[np.float64 | np.complex128]] = probe_operator
        self.number_of_states: int | None = number_of_states

        if states_path is not None and (eigenvalues is not None or eigenvectors is not None):
            raise ValueError("Specify either `states_path` or (`eigenvalues`, `eigenvectors`), but not both.")

        if states_path is not None:
            self.eigenvalues, self.eigenvectors = self._load_states(states_path)
        else:
            if eigenvalues is None or eigenvectors is None:
                raise ValueError("Both `eigenvalues` and `eigenvectors` must be provided when `states_path` is None.")
            self.eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
            self.eigenvectors = np.asarray(eigenvectors, dtype=np.complex128)

        self._validate_states()
        self._sort_states_by_energy()

    def compute(self) -> npt.NDArray[np.float64]:
        """
        Compute the linear response on ``self.omega_probe``.

        Returns:
            Real-valued response array with the same shape as ``self.omega_probe``.

        Raises:
            ValueError: If the selected number of states is invalid or if the
                probe operator is inconsistent with the polaritonic basis size.
        """
        n_states = self._resolve_number_of_states()

        energies: npt.NDArray[np.float64] = self.eigenvalues[:n_states]
        states: npt.NDArray[np.complex128] = self.eigenvectors[:, :n_states]

        projected_probe_operator = self._project_probe_operator(expected_size=states.shape[0])
        transition_matrix = np.conj(states.T) @ projected_probe_operator @ states

        transition_elements: npt.NDArray[np.float64] = np.abs(transition_matrix[0, 1:]) ** 2
        energy_differences: npt.NDArray[np.float64] = energies[1:] - energies[0]

        non_zero_energy_diff = np.abs(energy_differences) > 1e-14
        if np.any(~non_zero_energy_diff):
            energy_differences = energy_differences[non_zero_energy_diff]
            transition_elements = transition_elements[non_zero_energy_diff]

        if transition_elements.shape[0] == 0:
            return np.zeros_like(self.omega_probe, dtype=np.float64)

        denominator = self.omega_probe[:, np.newaxis] - energy_differences[np.newaxis, :] + 1j * self.eta
        response = -np.imag(np.sum(transition_elements[np.newaxis, :] / denominator, axis=1))
        return response

    def _resolve_number_of_states(self) -> int:
        """
        Resolve and validate the number of states used in the response sum.

        Returns:
            Number of states to include.

        Raises:
            ValueError: If the requested number is outside valid bounds.
        """
        n_states = self.eigenvalues.shape[0] if self.number_of_states is None else self.number_of_states
        if n_states < 2 or n_states > self.eigenvalues.shape[0]:
            raise ValueError(f"`number_of_states` must be in [2, {self.eigenvalues.shape[0]}], got {n_states}.")
        return n_states

    def _validate_states(self) -> None:
        """Validate dimensions and consistency of eigenvalues/eigenvectors."""
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
        """Sort states in ascending eigenenergy and reorder eigenvectors accordingly."""
        sorted_indices = np.argsort(self.eigenvalues)
        self.eigenvalues = self.eigenvalues[sorted_indices]
        self.eigenvectors = self.eigenvectors[:, sorted_indices]

    def _project_probe_operator(self, expected_size: int) -> npt.NDArray[np.complex128]:
        """
        Build the polarization-projected probe operator.

        Args:
            expected_size: Expected matrix dimension in the polaritonic basis.

        Raises:
            ValueError: If the polarization is unsupported or if required probe components are missing/inconsistent.

        Returns:
            Probe operator projected onto the selected polarization.
        """
        pol = self.polarization.value
        if pol in ["x", "y", "z"]:
            return self._get_probe_component(pol, expected_size)
        if pol in ["xy", "xz", "yz"]:
            first = self._get_probe_component(pol[0], expected_size)
            second = self._get_probe_component(pol[1], expected_size)
            return (first + second) / np.sqrt(2)
        if pol == "+":
            return (
                self._get_probe_component("x", expected_size)
                + 1j * self._get_probe_component("y", expected_size)
            ) / np.sqrt(2)
        if pol == "-":
            return (
                self._get_probe_component("x", expected_size)
                - 1j * self._get_probe_component("y", expected_size)
            ) / np.sqrt(2)
        raise ValueError(f"Unsupported polarization '{self.polarization.value}'.")

    def _get_probe_component(self, direction: Literal["x", "y", "z"], expected_size: int) -> npt.NDArray[np.complex128]:
        """
        Retrieve and validate a probe-operator Cartesian component.

        Args:
            direction: Cartesian component key (``'x'``, ``'y'``, or ``'z'``).
            expected_size: Expected matrix dimension in the polaritonic basis.

        Returns:
            Complex probe-operator matrix for the requested direction.

        Raises:
            ValueError: If the component is missing or has inconsistent shape.
        """
        if direction not in self.probe_operator:
            raise ValueError(
                f"Missing probe-operator component '{direction}. Provide required keys in `probe_operator`."
            )
        component = np.asarray(self.probe_operator[direction], dtype=np.complex128)
        if component.shape != (expected_size, expected_size):
            raise ValueError(
                f"Probe-operator component '{direction}' has shape {component.shape}, "
                f"expected ({expected_size}, {expected_size})."
            )
        return component
