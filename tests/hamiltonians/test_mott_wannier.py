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
        terms = ['kinetic_term', 'effective_2d_coulomb']
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
            "many_body",
            terms,
            "l1_l1",
            grid=k_space_grid,
            electron_effective_mass=electron_effective_mass,
            hole_effective_mass=hole_effective_mass,
            thicknesses=thicknesses,
            dielectric_constants=dielectric_constants
        )

    def test_hydrogen_2d(self):
        """Test the hydrogen 2D case"""
        hamiltonian = self.mott_wannier.get_hamiltonian()
        eigenval, eigenvec = self.mott_wannier.diagonalize_hamiltonian(hamiltonian)
        # Eigenvalues must be real (Hermitian Hamiltonian)
        self.assertTrue(np.all(np.isreal(eigenval)))
        # The ground state energy should be negative (bound state)
        self.assertLess(eigenval[0], 0.0)

    def test_hamiltonian_is_hermitian(self):
        """The Mott-Wannier Hamiltonian must be Hermitian"""
        hamiltonian = self.mott_wannier.get_hamiltonian()
        self.assertTrue(np.allclose(hamiltonian, hamiltonian.conj().T, atol=1e-12))

    def test_hamiltonian_size(self):
        """Hamiltonian size must match the grid size"""
        size = self.mott_wannier.get_hamiltonian_size()
        hamiltonian = self.mott_wannier.get_hamiltonian()
        self.assertEqual(hamiltonian.shape, (size, size))

    def test_eigenvalue_ordering(self):
        """Eigenvalues from eigh must be sorted in ascending order"""
        hamiltonian = self.mott_wannier.get_hamiltonian()
        eigenval, _ = self.mott_wannier.diagonalize_hamiltonian(hamiltonian)
        self.assertTrue(np.all(np.diff(eigenval) >= -1e-12))

    def test_ground_state_eigenvalue_pinned(self):
        """Pin the ground state energy to detect breaking changes.

        This value was computed on 2026-03-10 and should remain stable unless the
        physics implementation intentionally changes.
        """
        hamiltonian = self.mott_wannier.get_hamiltonian()
        eigenval, _ = self.mott_wannier.diagonalize_hamiltonian(hamiltonian)
        # Pin to a reference value with a tolerance broad enough for numerical effects
        # but tight enough to catch actual physics regressions
        self.assertTrue(np.isclose(eigenval[0], -0.16784, atol=1e-3),
                        f"Ground state energy {eigenval[0]} drifted from expected -0.16784 Ha")

    def test_eigenvectors_orthonormal(self):
        """Eigenvectors must be orthonormal"""
        hamiltonian = self.mott_wannier.get_hamiltonian()
        _, eigenvec = self.mott_wannier.diagonalize_hamiltonian(hamiltonian)
        overlap = eigenvec.conj().T @ eigenvec
        self.assertTrue(np.allclose(overlap, np.eye(overlap.shape[0]), atol=1e-10))

    def test_effective_mass_computation(self):
        """The reduced mass for equal-mass case must be m/2"""
        masses = self.mott_wannier.effective_masses
        # For m_e = m_h = 0.38, reduced mass = (1/0.38 + 1/0.38)^(-1) = 0.19
        self.assertTrue(np.isclose(masses['l1_l1'], 0.19))

    def test_invalid_exciton_type(self):
        """Setting an unsupported exciton type must raise ValueError"""
        with self.assertRaises(ValueError):
            self.mott_wannier.exciton = "invalid_type"

    def test_mismatched_effective_mass_keys(self):
        """Electron and hole effective masses with different keys must raise"""
        with self.assertRaises(ValueError):
            MottWannierHamiltonian._get_effective_mass(
                electron_effective_mass={'layer1': np.float64(0.5)},
                hole_effective_mass={'layer2': np.float64(0.5)}
            )

if __name__ == "__main__":
    unittest.main()
