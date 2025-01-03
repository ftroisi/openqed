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

""" Test class for the k space grid """

import unittest
import numpy as np
from ase.units import Bohr

from openqed.grid.grid import GridType
from openqed.grid.k_space_grid import KSpaceGrid

class TestKSpaceGrid(unittest.TestCase):
    """ Test class for the k space grid """
    def setUp(self) -> None:
        super().setUp()
        # Define 1D grid
        self.grid_1d = KSpaceGrid(
            boundaries={'x': (np.float64(0.0), np.float64(0.02))},
            spacing={'x': np.float64(0.001)}
        )
        # Define 2D grid
        spacing = {'x': np.float64(0.001), 'y': np.float64(0.001)}
        boundaries = {
            'x': (np.float64(-0.02), np.float64(0.02)),
            'y': (np.float64(-0.02), np.float64(0.02))
        }
        self.grid_2d = KSpaceGrid(boundaries=boundaries, spacing=spacing)

    # 1. Test the grid attribute (mesh and type)

    def test_1d_mesh(self):
        """ Test the mesh attribute for the 1D grid """
        self.assertEqual(len(self.grid_1d.mesh), 1)
        self.assertEqual(self.grid_1d.mesh[0].shape[0], 20)

    def test_2d_mesh(self):
        """ Test the mesh attribute for the 2D grid """
        self.assertEqual(len(self.grid_2d.mesh), 2)
        self.assertEqual(self.grid_2d.mesh[0].shape, (40, 40))
        self.assertEqual(self.grid_2d.mesh[1].shape, (40, 40))

    def test_grid_type(self):
        """ Test the grid_type attribute """
        self.assertEqual(self.grid_1d.grid_type, GridType.K_SPACE)

    # 2. Try to initialize the class with wrong inputs

    def test_wrong_inputs(self):
        """ Test the initialization of the class with wrong inputs """
        with self.assertRaises(KeyError):
            KSpaceGrid(
                boundaries={'x': (np.float64(0.0), np.float64(0.02))},
                spacing={'y': np.float64(0.001)}
            )
        with self.assertRaises(ValueError):
            KSpaceGrid(
                boundaries={'x': (np.float64(0.0), np.float64(0.02))},
                spacing={'x': np.float64(0.001), 'y': np.float64(0.001)}
            )

    # 3. Test the flat_grid method

    def test_flat_1d_grid(self):
        """ Test the flat_grid method for the 1D grid """
        # Check that if the flat grid is not cached, the private attribute is None
        self.grid_1d.flat_grid()
        #pylint: disable=protected-access
        self.assertEqual(self.grid_1d._flat_grid, None)
        #pylint: enable=protected-access
        flat_grid = self.grid_1d.flat_grid(cache_result=True)
        #pylint: disable=protected-access
        self.assertIsInstance(self.grid_1d._flat_grid, np.ndarray)
        #pylint: enable=protected-access

        # Check the shape of the flat grid
        self.assertEqual(flat_grid.shape, (20, 1))
        self.assertTrue(np.isclose(np.max(flat_grid, axis=0), 0.019)) # Boundary - spacing

    def test_flat_2d_grid(self):
        """ Test the flat_grid method for the 2D grid """
        flat_grid = self.grid_2d.flat_grid(cache_result=True)
        # Check the shape of the flat grid
        self.assertEqual(flat_grid.shape, (1600, 2))
        # Check the maximum values of the flat grid
        self.assertTrue(
            np.isclose(np.max(np.linalg.norm(flat_grid, axis=1)), np.sqrt(2*(-0.02)**2)))
        self.assertTrue(np.isclose(np.min(np.linalg.norm(flat_grid, axis=1)), 0.0))
        # Find one point in the grid
        point = np.array([0.01, 0.015])
        index = np.where(np.linalg.norm(flat_grid - point, axis=1) < 0.001 / 2)[0]
        self.assertEqual(len(index), 1)

    # 4. Test get the index of a point in the grid

    def test_get_index_1d(self):
        """ Test the get_index method for the 1D grid """
        point = np.array([0.01])
        index = self.grid_1d.get_point_index(point)
        self.assertEqual(index, 10)
        # Test the case where the point has the wrong number of dimensions
        with self.assertRaises(ValueError):
            self.grid_1d.get_point_index(np.array([0.01, 0.02]))

    def test_get_index_2d(self):
        """ Test the get_index method for the 2D grid """
        # Point 1: origin
        point = np.array([0.0, 0.0])
        index = self.grid_2d.get_point_index(point)
        self.assertEqual(index, 820)
        # Point 2, last of the grid, defined as tuple
        point = (0.019, 0.019)
        index = self.grid_2d.get_point_index(point)
        self.assertEqual(index, 1599)
        # Test the case where the point has the wrong number of dimensions
        with self.assertRaises(ValueError):
            self.grid_2d.get_point_index(np.array([0.01]))

    # 5. Test the project_to_unit_cell method

    def test_project_to_unit_cell(self):
        """ Test the project_to_unit_cell method """
        self.grid_2d.flat_grid(cache_result=True)
        # Find one point in the grid
        point = np.array([0.01, 0.015])
        index_before_proj = self.grid_2d.get_point_index(point)
        # Check that the projection method does not raise any error
        self.grid_2d.project_to_unit_cell(real_space_lattice_parameter=np.float64(3.32) / Bohr)
        # Check that the point was projected to the unit cell only in the new grid
        index_after_proj = self.grid_2d.get_point_index(point)
        self.assertEqual(index_after_proj, index_before_proj)
        # Now cache the projected grid
        self.grid_2d.project_to_unit_cell(
            real_space_lattice_parameter=np.float64(3.32) / Bohr, cache_result=True)
        index_after_proj = self.grid_2d.get_point_index(point)
        self.assertNotEqual(index_after_proj, index_before_proj)
        # Finally, check that the new value of the reference point is correct
        projected_point = np.array([0.01001481, 0.01156411])
        index_after_proj = self.grid_2d.get_point_index(projected_point)
        self.assertEqual(index_after_proj, index_before_proj)


if __name__ == "__main__":
    unittest.main()
