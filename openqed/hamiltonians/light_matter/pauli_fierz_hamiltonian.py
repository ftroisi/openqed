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

"""This module contains the Pauli-Fierz Hamiltonian class"""

from collections import OrderedDict

import numpy as np

from . import LightMatterHamiltonian
from openqed.hamiltonians import MatterHamiltonian
from openqed.hamiltonians.types import CavityParameters
from openqed.hamiltonians.light_matter import LightMatterHamiltonianTerms
from openqed.hamiltonians.terms import HamiltonianTerm

class PauliFierzHamiltonian(LightMatterHamiltonian):
    """
    This class represents the Pauli-Fierz Hamiltonian, which is the non-relativistic limit of the full QED theory.
    It describes the interaction between matter and the quantized electromagnetic field in a cavity for low-energy.

    References:
        - [1] Svendsen, M.K. et al., "Effective equilibrium theory of quantum light-matter interaction in cavities for
            extended systems and the long wavelength approximation", Commun Phys 8, 425 (2025)
        - [2] Ruggenthaler, M. et al. “From a quantum-electrodynamical light–matter description to novel
            spectroscopies”, Nature Reviews Chemistry 2.3 (2018)
    """
    def __init__(self,
                *,
                terms: list[str],
                cavity_parameters: CavityParameters,
                matter_hamiltonian: MatterHamiltonian
        ) -> None:
        # First, initialize the base LightMatterHamiltonian class
        super().__init__("PauliFierz", terms, cavity_parameters=cavity_parameters, matter_hamiltonian=matter_hamiltonian)
        # Initialize the Hamiltonian terms
        self.terms_to_instances: OrderedDict[LightMatterHamiltonianTerms, HamiltonianTerm] = OrderedDict()
        for term in [LightMatterHamiltonianTerms(t) for t in terms]:
            if term == LightMatterHamiltonianTerms.UNCOUPLED_MATTER:
                self.terms_to_instances[term] = matter_hamiltonian.to_hamiltonian_term()
