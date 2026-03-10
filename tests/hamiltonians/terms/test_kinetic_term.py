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

"""This module contains the tests for the KineticTerm class"""

import unittest
import numpy as np
from ase.units import Bohr

from tests.openqed_test import BaseOpenqedTest
from tests.hamiltonians.dummy_hamiltonian import DummyHamiltonian

from openqed.grid.k_space_grid import KSpaceGrid
from openqed.hamiltonians.terms.kinetic_term import KineticTerm
from openqed.hamiltonians.types import BilayerStructure
from openqed.unit_cells.hexagonal_unit_cell import HexagonalUnitCell

class TestKineticTerm(BaseOpenqedTest):
    """Test class for the KineticTerm class"""
    def setUp(self) -> None:
        super().setUp()
        # First, init the Grid
        spacing = {'x': np.float64(0.001), 'y': np.float64(0.001)}
        boundaries = {
            'x': (np.float64(-0.02), np.float64(0.02)),
            'y': (np.float64(-0.02), np.float64(0.02))
        }
        lattice_params: BilayerStructure = {
            'layer1': np.float64(3.32) / Bohr,
            'layer2': np.float64(3.319) / Bohr
        }
        k_space_grid = KSpaceGrid(spacing=spacing, boundaries=boundaries)
        unit_cell = HexagonalUnitCell(np.float64(lattice_params["layer1"]))
        k_space_grid.project_to_unit_cell(unit_cell, cache_result=True)
        # Init a dummy Hamiltonian
        self.hamiltonian = DummyHamiltonian(terms=['free_exciton'], grid=k_space_grid)
        # Set the effective masses
        self.electron_effective_mass: BilayerStructure = { 'layer1': np.float64(0.5) }
        self.hole_effective_mass: BilayerStructure = { 'layer1': np.float64(0.6) }
        self.eff_mass = \
            (1/self.electron_effective_mass['layer1'] + 1/self.hole_effective_mass['layer1'])**(-1)
        # Create the FreeExciton class
        self.free_exciton_term = KineticTerm(self.hamiltonian, mass=np.float64(self.eff_mass))

    # 1. Test the Hamiltonian term is correctly generated

    def test_get_hamiltonian_term(self):
        """Test the get_hamiltonian_term method"""
        hamiltonian_term = self.free_exciton_term.get_hamiltonian_term()
        for q_idx, q in enumerate(self.free_exciton_term.hamiltonian.grid.flat_grid()):
            self.assertTrue(np.isclose(hamiltonian_term[q_idx], (q[0]**2 + q[1]**2) / (2*self.eff_mass)))
        # Check that the minimum value is 0
        self.assertTrue(np.isclose(np.min(hamiltonian_term), 0.0))

    # 2. Check that the Hamiltonian term is diagonal

    def test_diagonalize_hamiltonian(self):
        """Test the get_hamiltonian_term method"""
        hamiltonian_term = self.free_exciton_term.get_hamiltonian_term()
        eigenval, _ = \
            self.free_exciton_term.hamiltonian.diagonalize_hamiltonian(np.diag(hamiltonian_term))
        # Check that the minimum value is 0
        self.assertTrue(np.isclose(np.min(hamiltonian_term), eigenval[0]))
        # Check that the maximum value is the maximum of the grid
        self.assertTrue(np.isclose(np.max(hamiltonian_term), eigenval[-1]))

    def test_kinetic_energy_all_non_negative(self):
        """Kinetic energy T = k^2 / (2m) must be >= 0 for all k-points"""
        hamiltonian_term = self.free_exciton_term.get_hamiltonian_term()
        self.assertTrue(np.all(hamiltonian_term >= -1e-15))

    def test_kinetic_energy_is_real(self):
        """Kinetic energy must be purely real"""
        hamiltonian_term = self.free_exciton_term.get_hamiltonian_term()
        self.assertTrue(np.all(np.isreal(hamiltonian_term)))

    def test_kinetic_energy_zero_at_gamma(self):
        """The kinetic energy at the Gamma point (k=0) must be exactly zero"""
        hamiltonian_term = self.free_exciton_term.get_hamiltonian_term()
        grid = self.free_exciton_term.hamiltonian.grid.flat_grid()
        # Find the k-point closest to Gamma (origin)
        k_norms = np.linalg.norm(grid, axis=1)
        gamma_idx = np.argmin(k_norms)
        self.assertTrue(np.isclose(hamiltonian_term[gamma_idx], 0.0, atol=1e-12))

    def test_kinetic_energy_scales_with_mass(self):
        """Halving the effective mass should double the kinetic energy"""
        T_original = self.free_exciton_term.get_hamiltonian_term()
        light_term = KineticTerm(self.hamiltonian, mass=np.float64(self.eff_mass / 2))
        T_light = light_term.get_hamiltonian_term()
        # T = k^2/(2m), so halving m doubles T
        self.assertTrue(np.allclose(T_light, 2.0 * T_original, atol=1e-12))

if __name__ == "__main__":
    unittest.main()
