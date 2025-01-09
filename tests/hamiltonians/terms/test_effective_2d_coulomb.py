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

"""This module contains the tests for the Effective2DCoulombPotential class"""

import unittest
import numpy as np

from tests.openqed_test import BaseOpenqedTest
from tests.hamiltonians.dummy_hamiltonian import DummyHamiltonian

from openqed.grid.k_space_grid import KSpaceGrid
from openqed.hamiltonians.terms.effective_2d_coulomb import Effective2DCoulombPotential
from openqed.hamiltonians.types import BilayerStructure

class TestEffective2DCoulombPotential(BaseOpenqedTest):
    """Test class for the Effective2DCoulombPotential class

    The following tests are referenced against the Supplementary Note 1 of the Supporting Info
    of the paper: https://doi.org/10.1038/s42005-019-0122-z
    """
    def setUp(self):
        super().setUp()
        # First, init the Grid
        spacing = {'x': np.float64(0.025), 'y': np.float64(0.025)}
        boundaries = {
            'x': (np.float64(-0.75), np.float64(0.75)),
            'y': (np.float64(-0.75), np.float64(0.75))
        }
        k_space_grid = KSpaceGrid(spacing=spacing, boundaries=boundaries)
        k_space_grid.flat_grid(cache_result=True)
        # Init a dummy Hamiltonian
        self.hamiltonian = DummyHamiltonian(terms=['free_exciton'], grid=k_space_grid)
        # Set the effective masses
        self.thicknesses: BilayerStructure = {
            'layer1': np.float64(0.645),
            'interlayer': np.float64(0.0),
            'layer2': np.float64(0.645)
        }
        self.hole_effective_mass: BilayerStructure = {
            'substrate1': np.float64(1.),
            'layer1': np.float64(15.1),
            'interlayer': np.float64(1.),
            'layer2': np.float64(16.5),
            'substrate2': np.float64(3.9)
        }
        # Create the FreeExciton class
        self.coulomb_potential = Effective2DCoulombPotential(
            self.hamiltonian,
            thicknesses=self.thicknesses,
            dielectric_constants=self.hole_effective_mass
        )

    # 1. Compute the intralayer Coulomb potential

    def test_intralayer_coulomb_potential(self):
        """Test the intra-layer Coulomb potential"""
        coulomb_kernel = self.coulomb_potential.generate_coulomb_kernel()
        potential = self.coulomb_potential.get_intralayer_dielectric_function(
                ref_layer="layer2",
                coupled_layer='layer1',
                k_grid=coulomb_kernel)
        # Find a suitable point for comparison
        points_to_compare = np.where(np.isclose(coulomb_kernel, 2.))
        ref = 27.231862419
        for i in range(len(points_to_compare[0])):
            self.assertTrue(
                np.isclose(potential[points_to_compare[0][i], points_to_compare[1][i]], ref))
        # Compare another point
        points_to_compare = \
            np.where(np.isclose(coulomb_kernel, 4., atol=0.005))
        ref = 31.399844165640637
        for i in range(len(points_to_compare[0])):
            self.assertTrue(
                np.isclose(potential[points_to_compare[0][i], points_to_compare[1][i]], ref))

    # 2. Compute the interlayer Coulomb potential

    def test_interlayer_coulomb_potential(self):
        """Test the intra-layer Coulomb potential"""
        coulomb_kernel = self.coulomb_potential.generate_coulomb_kernel()
        potential = self.coulomb_potential.get_interlayer_dielectric_function(
                ref_layer="layer1",
                coupled_layer='layer2',
                k_grid=coulomb_kernel)
        # Compare one point
        points_to_compare = np.where(np.isclose(coulomb_kernel, 0.5))
        ref = 15.806956786596157
        for i in range(len(points_to_compare[0])):
            self.assertTrue(
                np.isclose(potential[points_to_compare[0][i], points_to_compare[1][i]], ref))
        # Compare another point
        points_to_compare = np.where(np.isclose(coulomb_kernel, 1.0))
        ref = 29.988017788744465
        for i in range(len(points_to_compare[0])):
            self.assertTrue(
                np.isclose(potential[points_to_compare[0][i], points_to_compare[1][i]], ref))

if __name__ == '__main__':
    unittest.main()
