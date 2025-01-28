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

"""This module contains the tests for the MottWannier class"""

import unittest
import numpy as np
from ase.units import Bohr

from tests.openqed_test import BaseOpenqedTest

from openqed.grid.k_space_grid import KSpaceGrid
from openqed.hamiltonians.mott_wannier_hamiltonian import MottWannierHamiltonian
from openqed.hamiltonians.types import BilayerStructure

class TestMottWannierHamiltonian(BaseOpenqedTest):
    """Test class for the Effective2DCoulombPotential class

    The following tests are referenced against the Supplementary Note 1 of the Supporting Info
    of the paper: https://doi.org/10.1038/s42005-019-0122-z
    """
    def setUp(self):
        super().setUp()
        spacing = {'x': np.float64(0.05) / Bohr, 'y': np.float64(0.05) / Bohr}
        boundaries = {
            'x': (np.float64(-1.5) / Bohr, np.float64(1.5) / Bohr),
            'y': (np.float64(-1.5) / Bohr, np.float64(1.5) / Bohr)
        }
        k_space_grid = KSpaceGrid(spacing=spacing, boundaries=boundaries)
        k_space_grid.flat_grid(cache_result=True)

        # Create an instance of the MottWannier class
        terms = ['free_exciton', 'effective_2d_coulomb']
        electron_effective_mass: BilayerStructure = { 'layer1': np.float64(0.38) }
        hole_effective_mass: BilayerStructure = { 'layer1': np.float64(0.38) }
        thicknesses: BilayerStructure = {
            'layer1': np.float64(1.),
            'interlayer': np.float64(0.),
            'layer2': np.float64(0.)
        }
        dielectric_constants: BilayerStructure = {
            'substrate1': np.float64(1.),
            'layer1': np.float64(4.505),
            'interlayer': np.float64(1.),
            'layer2': np.float64(1.),
            'substrate2': np.float64(1.)
        }
        self.mott_wannier = MottWannierHamiltonian(
            terms,
            grid=k_space_grid,
            electron_effective_mass=electron_effective_mass,
            hole_effective_mass=hole_effective_mass,
            thicknesses=thicknesses,
            dielectric_constants=dielectric_constants
        )

    def test_hydrogen_2d(self):
        """Test the hydrogen 2D case"""
        hamiltonian = self.mott_wannier.get_hamiltonian(exciton="l1_l1")
        eigenval, _ = self.mott_wannier.diagonalize_hamiltonian(hamiltonian)
        print(eigenval[0:5])

if __name__ == "__main__":
    unittest.main()
