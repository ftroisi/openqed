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

"""This module contains the Mott-Wannier Hamiltonian class"""

from abc import ABC
from enum import Enum

import numpy as np

from openqed.hamiltonians import Hamiltonian, MatterHamiltonian
from openqed.hamiltonians.types import CavityParameters, mode_polarization_to_vector

class LightMatterHamiltonianTerms(Enum):
    UNCOUPLED_LIGHT = "uncoupled_light"
    UNCOUPLED_MATTER = "uncoupled_matter"
    BILINEAR = "bilinear"
    DIAMAGNETIC = "diamagnetic"

class LightMatterHamiltonian(Hamiltonian, ABC):
    """
    This class defines the generalized QED light-matter problem. It shall be used to couple a
    matter Hamiltonian with the radiation field.
    """
    def __init__(self,
                *,
                name: str,
                terms: list[str],
                cavity_parameters: CavityParameters,
                matter_hamiltonian: MatterHamiltonian
        ) -> None:
        """Initializes the LightMatterHamiltonian

        Args:
            name: The name of the Hamiltonian.
            terms: a list of strings representing the terms of the Hamiltonian.
            cavity_parameters: The parameters of the cavity.
            matter_hamiltonian: The matter Hamiltonian to couple with the radiation field.

        References:
            - [1] Svendsen, M.K. et al., Effective equilibrium theory of quantum light-matter interaction in cavities
                for extended systems and the long wavelength approximation, Commun Phys 8, 425 (2025).
        """
        # First, initialize the base Hamiltonian class
        super().__init__(name, terms)
        # Verify and save the parameters
        if cavity_parameters.number_of_effective_modes is None or cavity_parameters.number_of_effective_modes < 1:
            raise ValueError("Number of effective modes in the cavity must be at least 1")
        self.number_of_effective_modes = cavity_parameters.number_of_effective_modes
        if cavity_parameters.maximum_mode_occupation is None or cavity_parameters.maximum_mode_occupation < 0:
            raise ValueError("Maximum mode occupation must be at least 0")
        self.maximum_mode_occupation = cavity_parameters.maximum_mode_occupation
        if cavity_parameters.modes_energies is None or np.any(cavity_parameters.modes_energies <= 0):
            raise ValueError("Energies of the modes must be positive")
        if (cavity_parameters.modes_energies.ndim != 1 and
                cavity_parameters.modes_energies.shape[0] != self.number_of_effective_modes):
            raise ValueError("The number of the modes energies array must match the number of effective modes."
                + f" Expected shape: ({self.number_of_effective_modes},), got {cavity_parameters.modes_energies.shape}")
        self.modes_energies = cavity_parameters.modes_energies
        if cavity_parameters.modes_couplings is None or np.any(cavity_parameters.modes_couplings < 0):
            raise ValueError("Coupling strength of the modes must be non-negative")
        if (cavity_parameters.modes_couplings.ndim != 1 and
                cavity_parameters.modes_couplings.shape[0] != self.number_of_effective_modes):
            raise ValueError("The number of the modes couplings array must match the number of effective modes."
                + f" Expected shape: ({self.number_of_effective_modes},), got {cavity_parameters.modes_couplings.shape}")
        self.modes_couplings = cavity_parameters.modes_couplings
        if (cavity_parameters.modes_parallel_momenta is None or cavity_parameters.modes_parallel_momenta.ndim != 2 or
            cavity_parameters.modes_parallel_momenta.shape[0] != self.number_of_effective_modes or
            cavity_parameters.modes_parallel_momenta.shape[1] != 3):
            raise ValueError(
                "The modes parallel momenta array must be a 2D array of shape (number of effective modes, 3). Expected "
                + f"shape: ({self.number_of_effective_modes}, 3), got {cavity_parameters.modes_parallel_momenta.shape}")
        self.modes_parallel_momenta = cavity_parameters.modes_parallel_momenta
        if cavity_parameters.modes_polarization is None:
            raise ValueError("The polarization of the modes must be specified")
        self.modes_polarization = mode_polarization_to_vector(cavity_parameters.modes_polarization)
        # Check the matter Hamiltonian
        if matter_hamiltonian is None:
            raise ValueError("The matter Hamiltonian must be specified")
        self.matter_hamiltonian = matter_hamiltonian
        # Ensure that the momentum of the photonic modes is compatible with the grid of the matter Hamiltonian
        for mode_momentum in self.modes_parallel_momenta:
            if self.matter_hamiltonian.grid.get_point_index(mode_momentum) == -1:
                raise ValueError(f"The momentum of the photonic mode {mode_momentum}" +
                                 " is not compatible with the grid of the matter Hamiltonian")


