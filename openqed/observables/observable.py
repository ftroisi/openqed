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

"""Base abstractions and shared utilities for observable calculations."""

from abc import ABC, abstractmethod
from enum import Enum
import os
import numpy as np
import numpy.typing as npt

from openqed.io import PolaritonicStatesRestartKeys

class ObservableFileKey(str, Enum):
    """Keys used in generic observable output ``.npz`` files."""
    OBSERVABLE = "observable"

class Observable(ABC):
    """
    Base abstract class for all observables in ``openqed.observables``.

    Subclasses are responsible for implementing :meth:`compute` and may use :meth:`save`, :meth:`resolve_output_path`,
    and :meth:`_load_states` to share common file I/O logic.
    """

    def __init__(self, *, name: str, results_path: str | None = None) -> None:
        """Initialize an observable instance.

        Args:
            name: Name of the observable.
            results_path: Directory where results are written. If ``None``, relative output paths are resolved against
                the current working directory.

        Raises:
            ValueError: If ``name`` is empty or only whitespace.
        """
        self.name: str = name.strip()
        if len(self.name) == 0:
            raise ValueError("The name of the Observable cannot be empty or contain only whitespace.")
        self.results_path: str | None = results_path
        # Check if the results path exists, if not create it
        if self.results_path is not None and not os.path.exists(self.results_path):
            os.makedirs(self.results_path)

    def resolve_output_path(self, filename: str) -> str:
        """
        Resolve the absolute output path for a result file.

        If ``filename`` is already an absolute path, it is returned unchanged. Otherwise, it is
        interpreted as relative to ``results_path`` when available, or to the current working
        directory when ``results_path`` is ``None``.

        Args:
            filename: Output file name (absolute or relative).

        Returns:
            Absolute output file path.

        Raises:
            ValueError: If ``filename`` is empty or only whitespace.
        """
        if len(filename.strip()) == 0:
            raise ValueError("The output filename cannot be empty or contain only whitespace.")

        if os.path.isabs(filename):
            output_path = filename
        elif self.results_path is not None:
            output_path = os.path.join(self.results_path, filename)
        else:
            output_path = os.path.join(os.getcwd(), filename)

        parent_dir = os.path.dirname(output_path)
        if len(parent_dir) > 0 and not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        return output_path

    def save(self, filename: str, data: npt.NDArray[np.float64 | np.complex128], **kwargs) -> str:
        """
        Save observable data to ``.npz`` format.

        Args:
            filename: Output file name (or absolute path). Must end with ``.npz``.
            data: Observable values to save (as returned by :meth:`compute`).
            kwargs: Additional metadata arrays/scalars to store in the same file.

        Returns:
            The absolute path of the written file.

        Raises:
            ValueError: If ``filename`` does not end with ``.npz``.
        """
        output_path = self.resolve_output_path(filename)
        if not output_path.endswith(".npz"):
            raise ValueError("Output filename must end with '.npz'.")

        np.savez(output_path, **{ObservableFileKey.OBSERVABLE.value: data}, **kwargs)
        return output_path

    @staticmethod
    def _load_states(states_path: str) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64 | np.complex128]]:
        """
        Load precomputed polaritonic states from a restart ``.npz`` file.

        Args:
            states_path: Path to a ``.npz`` file containing eigenvalues and eigenvectors.

        Returns:
            A tuple ``(eigenvalues, eigenvectors)``.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If file extension is not ``.npz`` or required keys are missing.
        """
        if not os.path.exists(states_path):
            raise FileNotFoundError(f"States file not found: {states_path}")
        if not states_path.endswith(".npz"):
            raise ValueError("`states_path` must point to a '.npz' file.")

        data = np.load(states_path)
        if (PolaritonicStatesRestartKeys.EIGENVALUES.value not in data
                or PolaritonicStatesRestartKeys.EIGENVECTORS.value not in data):
            raise ValueError(
                f"States file must contain arrays with keys `{PolaritonicStatesRestartKeys.EIGENVALUES.value}` " +
                f"and `{PolaritonicStatesRestartKeys.EIGENVECTORS.value}`."
            )

        eigenvalues = np.asarray(data[PolaritonicStatesRestartKeys.EIGENVALUES.value], dtype=np.float64)
        eigenvectors = np.asarray(data[PolaritonicStatesRestartKeys.EIGENVECTORS.value], dtype=np.complex128)
        return eigenvalues, eigenvectors

    @abstractmethod
    def compute(self) -> npt.NDArray[np.float64 | np.complex128]:
        """Compute the observable quantity and return it as a NumPy array."""
        pass
