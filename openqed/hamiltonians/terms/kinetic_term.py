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

"""This module contains the KineticTerm class"""

import numpy as np
import numpy.typing as npt

from openqed.hamiltonians.terms.hamiltonian_term import HamiltonianTerm
from openqed.hamiltonians.hamiltonian import Hamiltonian
from openqed.grid.real_space_grid import RealSpaceGrid

class KineticTerm(HamiltonianTerm):
    """
    This class represents the KineticTerm. The kinetic term is the term in the Hamiltonian that
    describes the kinetic energy of the particle. Its formula depends on the grid used.
    - For a k-space grid, the kinetic term is: :math:`\\frac{\\hbar^2 * k^2}{2 * m}`
    - For a real-space grid, the kinetic term is: :math:`-\\frac{\\hbar^2}{2 * m} \\nabla^2`
    """
    def __init__(self,
                 hamiltonian: Hamiltonian,
                 *,
                 mass: np.float64) -> None:
        """Initialize the KineticTerm class

        Args:
            hamiltonian: The Hamiltonian object to which the term belongs
            mass: The mass of the particle
        """
        super().__init__(hamiltonian)
        if isinstance(hamiltonian.grid, RealSpaceGrid):
            raise NotImplementedError("RealSpaceGrid is not supported yet")
        # Save the mass of the particle
        self.mass = mass

    @property
    def mass(self) -> np.float64:
        """The mass of the particle."""
        return self._mass

    @mass.setter
    def mass(self, mass: np.float64) -> None:
        if np.isclose(mass, 0):
            raise ValueError("The mass of the particle cannot be zero")
        self._mass: np.float64 = mass

    def get_hamiltonian_term(self) -> npt.NDArray[np.float64]:
        """
        This method generates the Kinetic term. The formula depends on the grid used.
        For a k-space grid, the kinetic term is:
        :math:`\\frac{\\hbar^2 * k^2}{2 * m}`

        Returns:
            The Kinetic term as a 1D array
        """
        # The kinetic term is a diagonal matrix with the squared norm of the
        # momentum divided by the effective mass. Thus, we only return it as a 1D array
        return np.linalg.norm(self.hamiltonian.grid.flat_grid(), axis=1)**2 / (2 * self.mass)
