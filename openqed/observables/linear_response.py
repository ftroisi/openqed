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
            include_non_ground_state_transitions: bool = False,
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
            include_non_ground_state_transitions: Whether to include non-ground state transitions.
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

        self.include_non_ground_state_transitions = include_non_ground_state_transitions
        if eta <= 0:
            raise ValueError("`eta` must be strictly positive.")
        self.eta: np.float64 = np.float64(eta)

        self.polarization: LightPolarization = probe_polarization
        self.probe_operator: dict[Literal["x", "y", "z"], npt.NDArray[np.float64 | np.complex128]] = probe_operator
        self.number_of_states: int | None = number_of_states
        if self.number_of_states is not None:
            raise NotImplementedError(
                    "`number_of_states` selection is not yet implemented. Please set it to None to use all states.")

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

        projected_probe_operator = self._project_probe_operator()
        transition_matrix = np.conj(states.T) @ projected_probe_operator @ states

        # For the full Kubo sum (include_non_ground_state_transitions=True) all off-diagonal
        # pairs are kept, but the counter-rotating upper-triangle elements receive a (-1) sign
        # from the occupation difference (f_m - f_n) = -1 for ground-to-excited pairs.
        if not self.include_non_ground_state_transitions:
            transition_matrix[1:, 1:] = 0.  # zero excited-to-excited transitions
        transition_elements: npt.NDArray[np.float64] = np.abs(transition_matrix) ** 2
        del transition_matrix, projected_probe_operator, states
        # Account for occupation differences: upper-triangle elements T[m,n] with m<n use the
        # emission denominator omega+(E_n-E_m), so their occupation weight (f_m-f_n)<0
        # contributes a minus sign.  Lower-triangle elements T[n,m] with n>m are the absorption
        # terms and carry a positive occupation weight.
        rows, cols = transition_elements.shape
        transition_elements[np.arange(rows)[:, None] < np.arange(cols)] *= -1

        # Compute the energy differences between all eigenvalues
        energy_differences: npt.NDArray[np.float64] = np.add.outer(energies, -energies)
        energy_differences = energy_differences[:n_states, :n_states]
        energy_differences[np.where(np.abs(energy_differences) < 1e-5)] = self.omega_probe[1] - self.omega_probe[0]
        del energies

        non_zero_transition_elements = np.abs(transition_elements) > 1e-14
        if np.any(~non_zero_transition_elements):
            energy_differences = energy_differences[non_zero_transition_elements]
            transition_elements = transition_elements[non_zero_transition_elements]

        response_function: npt.NDArray[np.float64] = np.zeros_like(self.omega_probe, dtype=np.float64)
        if transition_elements.shape[0] > 0:
            for omega_idx, _ in enumerate(self.omega_probe):
                # First, compute the denominator.
                denominator = self.omega_probe[omega_idx] - energy_differences + 1j * self.eta
                # Then, compute the response function by doing transition_elem / denominator.
                response_function[omega_idx] = -np.imag(np.sum(transition_elements / denominator))
        return response_function

    def save(self, filename: str, data: npt.NDArray[np.float64], **kwargs) -> str:
        """
        Save the linear response together with its metadata (omega_probe, eta,
        probe_polarization) to an ``.npz`` file.

        Args:
            filename: Output file name (or absolute path). Must end with ``.npz``.
            data: Response array as returned by :meth:`compute`.
            kwargs: Additional arrays/scalars forwarded to the base :meth:`save`.

        Returns:
            Absolute path of the written file.
        """
        return super().save(
            filename,
            data,
            **{
                LinearResponseFileKey.OMEGA_PROBE.value: self.omega_probe,
                LinearResponseFileKey.ETA.value: np.array(self.eta),
                LinearResponseFileKey.PROBE_POLARIZATION.value: np.array(
                    str(self.polarization.value)
                ),
            },
            **kwargs,
        )

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

    def _project_probe_operator(self) -> npt.NDArray[np.complex128]:
        """
        Build the polarization-projected probe operator.

        Raises:
            ValueError: If the polarization is unsupported or if required probe components are missing/inconsistent.

        Returns:
            Probe operator projected onto the selected polarization.
        """
        pol = self.polarization.value
        if pol in ["x", "y", "z"]:
            matter_response: npt.NDArray[np.complex128] = self._get_probe_component(pol)
        elif pol in ["xy", "xz", "yz"]:
            first = self._get_probe_component(pol[0])
            second = self._get_probe_component(pol[1])
            matter_response: npt.NDArray[np.complex128] = (first + second) / np.sqrt(2)
            del first, second
        elif pol == "+" or pol == "-":
            first = self._get_probe_component("x")
            second = self._get_probe_component("y")
            matter_response: npt.NDArray[np.complex128] =\
                (first + 1j*second) / np.sqrt(2) if pol == "+" else (first - 1j*second) / np.sqrt(2)
            del first, second
        else:
            raise ValueError(f"Unsupported polarization '{self.polarization.value}'.")
        # Get the correct matrix size for the linear response operator. Its representation is:
        # mat_res X {1_m0 X ... X 1_mn} where mat_res is the matter response and 1_mi is the
        # identity operator for the i-th photon mode
        # First, check that the matter response size is consistent with the polaritonic basis size.
        if self.eigenvalues.shape[0] % matter_response.shape[0] != 0:
            raise ValueError(f"Probe operator size {matter_response.shape} is inconsistent" +
                             f"with the polaritonic basis size {self.eigenvalues.shape[0]}.")
        # Then, build the full probe operator by taking the Kronecker product with the identity over the photon modes.
        matter_repetitions = self.eigenvalues.shape[0] // matter_response.shape[0]
        return np.kron(matter_response, np.eye(matter_repetitions, dtype=np.complex128))

    def _get_probe_component(self, direction: Literal["x", "y", "z"]) -> npt.NDArray[np.complex128]:
        """
        Retrieve and validate a probe-operator Cartesian component.

        Args:
            direction: Cartesian component key (``'x'``, ``'y'``, or ``'z'``).

        Returns:
            Complex probe-operator matrix for the requested direction.

        Raises:
            ValueError: If the component is missing or has inconsistent shape.
        """
        if direction not in self.probe_operator:
            raise ValueError(
                f"Missing probe-operator component '{direction}. Provide required keys in `probe_operator`."
            )
        return np.asarray(self.probe_operator[direction], dtype=np.complex128)
