# Copyright 2024 Francesco Troisi
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

"""This model contains the Mott-Wannier Hamiltonian class"""

import numpy as np
from openqed.io.input_file import InputFile
from openqed.grid.k_space_grid import KSpaceGrid
from .hamiltonian import Hamiltonian

class MottWannier(Hamiltonian):
    """
    This class is the Mott-Wannier Hamiltonian class. It inherits from the base Hamiltonian class.

    Attributes:
        `name`: The name of the Hamiltonian.
        `terms`: a list of sctrings representing the terms of the Hamiltonian.
    """
    def __init__(self, input_file: InputFile, terms: list[str]) -> None:
        """Initialize the Mott-Wannier Hamiltonain

        Args:
            name: The name of the Hamiltonian.
            terms: a list of sctrings representing the terms of the Hamiltonian.
        """
        # First, initialize the base Hamiltonian class
        super().__init__("Mott-Wannier", terms, KSpaceGrid(input_file))

    # def get_hamiltonian(self):
    #     """This method builds the Mott-Wannier Hamiltonian fora couple of layers.
    #     See equation (13) of the supplementary information of
    #     https://pubs.acs.org/doi/10.1021/acs.nanolett.0c03019?fig=fig4&ref=pdf

    #     Args:
    #         layer (str): the layer for which the Hamiltonian is computed. Its value can be
    #             any key of the Excitons class

    #     Returns:
    #         h_ll: a matrix, corresponding to the Hamiltonians that describe the exciton
    #             wavefunction for the chosen layers
    #     """
    #     # First compute mesh size
    #     mesh_size = np.size(self.k_grid, 1)
    #     try:
    #         eff_mass = self.eff_mass[layer]
    #     except ValueError as exc:
    #         raise ValueError(f"The effective mass for layer {layer} is not valid") from exc
    #     # Precompute the main diagonal (without the 1/mr factor)
    #     h_mat = np.zeros((mesh_size, 1), dtype=float)
    #     h_mat = 0.5 * np.linalg.norm(self.flat_k_grid, axis=1)**2
    #     # Define the coulomb potential
    #     w_k = self.get_w_q_analytical(np.sqrt(self.k_grid), layer=layer)
    #     # Build matrix
    #     h_ll = np.zeros((mesh_size, mesh_size), dtype=float)
    #     np.fill_diagonal(h_ll, h_mat / eff_mass)
    #     h_ll -= w_k
    #     return h_ll
