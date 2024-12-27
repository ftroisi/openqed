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

"""This module contains the FreeExciton class"""

import numpy as np
import numpy.typing as npt
from openqed.hamiltonians.terms.hamiltonian_term import HamiltonianTerm
from openqed.hamiltonians.hamiltonian import Hamiltonian
from openqed.hamiltonians.types import BilayerStructure, BilayerExcitons
from openqed.hamiltonians.types.bilayer_excitons import structure_key_to_exciton_key

class FreeExciton(HamiltonianTerm):
    """
    This class represents the Free Exciton Hamiltonian term. It is a subclass of the
    HamiltonianTerm class.
    """
    def __init__(self,
                 hamiltonian: Hamiltonian,
                 *,
                 electron_effective_mass: BilayerStructure,
                 hole_effective_mass: BilayerStructure) -> None:
        """Initialize the FreeExciton class

        Args:
            hamiltonian: The Hamiltonian object to which the term belongs
        """
        super().__init__(hamiltonian)
        self.exciton_effective_mass: BilayerExcitons = {}
        # Then, define the masses
        if electron_effective_mass.keys() != hole_effective_mass.keys():
            raise ValueError(
                "The effective masses for electrons and holes must be defined for the same layers")
        # https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.53.1391
        for elec_key in electron_effective_mass.keys():
            l1 = structure_key_to_exciton_key(elec_key)
            for hole_key in hole_effective_mass.keys():
                l2 = structure_key_to_exciton_key(hole_key)
                key = f"{l1}_{l2}"
                self.exciton_effective_mass[key] = \
                    (1 / electron_effective_mass[l1] + 1 / hole_effective_mass[l2])**(-1)

    def get_hamiltonian_term(self, **kwargs) -> npt.NDArray[np.float64]:
        """
        This method generates the Free Exciton Hamiltonian term. It returns the Hamiltonian term as
        a 1D array, as it is a diagonal matrix.

        Args:
            exciton: the exciton for which the analytical screened potential is computed. It is
                a string with the format `li_lj`, where `i` and `j` are numbers and the whole string
                is a key of `:class:BilayerExcitons`. For instance, specifying `l1_l1` will compute
                the analytical screened potential for the intraleyer excitons of the first layer

        Returns:
            The Free Exciton Hamiltonian term as a matrix
        """
        exciton: str = kwargs.get('exciton', None)
        if exciton is None:
            raise ValueError("The exciton argument is missing")
        # Get the effective mass
        try:
            eff_mass: np.float64 = self.exciton_effective_mass[exciton]
        except ValueError as exc:
            raise ValueError(f"The effective mass for exciton {exciton} is not valid") from exc
        # The free exciton Hamiltonian term is a diagonal matrix with the squared norm of the
        # momentum divided by the effective mass. Thus, we only return it as a 1D array
        h_mat: npt.NDArray[np.float64] = \
            np.linalg.norm(self.hamiltonian.grid.flat_grid(), axis=1)**2 / (2 * eff_mass)
        return h_mat
