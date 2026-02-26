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

"""This module contains the base abstract MatterHamiltonian class"""

from abc import ABC, abstractmethod
from typing import Literal, Dict
from enum import Enum

import numpy as np
import numpy.typing as npt

from openqed.grid.grid import Grid
from openqed.hamiltonians.hamiltonian import Hamiltonian
from openqed.hamiltonians.types import ModePolarization

class MatterHamiltonianFormalism(Enum):
    MANY_BODY = "many_body"
    TIGHT_BINDING = "tight_binding"

class MatterHamiltonian(Hamiltonian, ABC):
    """
    This class is the base abstract MatterHamiltonian class.
    All classes representing single particle matter systems must inherit from this class.
    """
    def __init__(self,
                 name: str,
                 terms: list[str],
                 mode: str | MatterHamiltonianFormalism,
                 grid: Grid
        ) -> None:
        """Initializes the MatterHamiltonian

        Args:
            name: The name of the hamiltonian.
            terms: a list of strings representing the terms of the Hamiltonian.
            mode: The construction mode of the Hamiltonian. Must be one of ``"many_body"`` or ``"tight_binding"``. In
                ``"many_body"`` mode :meth:`get_hamiltonian` returns the full ``(N, N)`` matrix over all k-points. In
                ``"tight_binding"`` mode it returns a stacked array of shape ``(n_k, n_bands, n_bands)``,
                 one small matrix per k-point.
            grid: The grid object to use. The Hamiltonian will be represented on this grid.
        """
        super().__init__(name, terms)
        self.grid: Grid = grid
        self.mode = mode

    @property
    def mode(self) -> MatterHamiltonianFormalism:
        """The construction mode of the Hamiltonian (``"many-body"`` or ``"tight-binding"``)."""
        return self._mode

    @mode.setter
    def mode(self, value: str | MatterHamiltonianFormalism) -> None:
        if not isinstance(value, (str, MatterHamiltonianFormalism)):
            raise TypeError(f"mode must be a str or MatterHamiltonianFormalism, got {type(value).__name__!r}")
        self._mode = MatterHamiltonianFormalism(value)


    def get_hamiltonian(self) -> npt.NDArray[np.float64 | np.complex128]:
        """Build and return the Hamiltonian matrix.

        Dispatches to :meth:`_get_many_body_h` or :meth:`_get_tight_binding_h`
        depending on the current value of :attr:`mode`.

        Returns:
            * ``"many-body"`` mode — 2-D array of shape ``(N, N)``.
            * ``"tight-binding"`` mode — 3-D array of shape ``(n_k, n_bands, n_bands)``.
        """
        if self.mode == MatterHamiltonianFormalism.MANY_BODY:
            return self._get_many_body_hamiltonian()
        return self._get_tight_binding_hamiltonian()

    def _get_many_body_hamiltonian(self) -> npt.NDArray[np.float64 | np.complex128]:
        """Build the Hamiltonian as a single ``(N, N)`` matrix over all k-points.

        Subclasses that support ``"many-body"`` mode must override this method.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement 'many-body' mode."
        )

    def _get_tight_binding_hamiltonian(self) -> npt.NDArray[np.float64 | np.complex128]:
        """Build the Hamiltonian as a stacked ``(n_k, n_bands, n_bands)`` array.

        Subclasses that support ``"tight-binding"`` mode must override this
        method.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement 'tight-binding' mode."
        )

    @abstractmethod
    def get_interaction_hamiltonian(self,
                                    light_momentum: npt.NDArray[np.float64],
                                    light_polarization: ModePolarization
        ) -> Dict[Literal['x', 'y', 'z'], npt.NDArray[np.float64 | np.complex128]]:
        """
        This method generates the interaction hamiltonian for this class. In a light-matter system,
        this term describes the interaction between the matter system and the radiation field
        (:math:`\\mathbf{p} \\cdot \\mathbf{A}`).

        Args:
            light_momentum: The momentum of the light field. It is a 1D array of shape:
                where each entry represents the momentum in the x, y or z directions for each mode.
            light_polarization: The polarization of the light field. It can be one of the following:
                - 'x': linear polarization along the x direction
                - 'y': linear polarization along the y direction
                - 'z': linear polarization along the z direction
                - 'xy': linear polarization in the xy plane
                - 'xz': linear polarization in the xz plane
                - 'yz': linear polarization in the yz plane
                - '+': circular polarization with positive helicity
                - '-': circular polarization with negative helicity

        Returns:
            A dictionary with keys 'x', 'y' and 'z' representing the interaction hamiltonian in the x, y and z
            polarization directions for a certain mode.
            The values are 2D arrays of shape: (interaction matrix size, interaction matrix size).
        """
        pass
