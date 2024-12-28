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

"""This module contains the class to handle the real space grid"""

import gc
import numpy as np
import numpy.typing as npt
from openqed.io.input_file import InputFile
from .grid import Grid, GridType

def docstring_inherit(parent):
    def inherit(obj):
        spaces = "    "
        if "Attributes:" not in str(obj.__doc__):
            obj.__doc__ += "\n" + spaces + "Attributes:\n"
        obj.__doc__ = str(obj.__doc__).rstrip() + "\n"
        for attribute in parent.__doc__.split("Attributes:\n")[-1].lstrip().split("\n"):
            obj.__doc__ += spaces * 2 + str(attribute).lstrip().rstrip() + "\n"
        return obj

    return inherit

@docstring_inherit(Grid)
class KSpaceGrid(Grid):
    """
    This class is the k space (or reciprocal space) grid class. It inherits from the base Grid class

    Attributes:
        mesh: The mesh of the grid, as returned from `numpy.meshgrid`. It is a list of 1D, 2D or 3D
            arrays, depending on the dimensionality of the grid.
        _flat_grid: The flat grid. It is a 2D array where each row represents a point in the grid,
            and each column represents a dimension of the grid. DO NOT ACCESS THIS ATTRIBUTE
            DIRECTLY, but call the `flat_grid` method instead.
    """
    def __init__(self,
                 *,
                input_file: InputFile | None = None,
                boundaries: dict[str, tuple[np.float64, np.float64]] | None = None,
                spacing: dict[str, np.float64] | None = None):
        # First, initialize the base Grid class
        super().__init__(input_file, boundaries, spacing)
        self.grid_type = GridType.K_SPACE
        # Define the range of the grid in each dimension
        space: list[npt.NDArray] = []
        for dim in self.spacing.keys():
            start, end = self.boundaries[dim]
            space.append(np.arange(start, end, self.spacing[dim]))
        # Then build a mesh, flatten it (meaning that the mesh becomes a 1D
        # array) and finally stack the x-y-z-arrays
        if self.dimensions == 1:
            self.mesh: tuple[npt.NDArray[np.float64], ...] = np.meshgrid(space[0], indexing='ij')
        elif self.dimensions == 2:
            self.mesh = np.meshgrid(space[0], space[1], indexing='ij')
        elif self.dimensions == 3:
            self.mesh = np.meshgrid(space[0], space[1], space[2], indexing='ij')
        self._flat_grid: npt.NDArray[np.float64] | None = None

    @property
    def grid_size(self) -> int:
        """Get the size of the grid"""
        return np.shape(self.mesh[0])[0]

    def deallocate_flat_grid(self) -> None:
        """Deallocate the flat grid"""
        self._flat_grid = None
        gc.collect()

    def flat_grid(self, cache_result: bool = False) -> npt.NDArray[np.float64]:
        """Returns the flat grid

        The flat grid is a 2D array where each row represents a point in the grid, and each column
        represents a dimension of the grid. Thus:
        - calling `flat_grid[0, 1]` returns the coordinate for the second dimension (`y`) of the
            first point of the grid.
        - calling `flat_grid[1]` returns an array with all the coordinates of the second point

        Args:
            cache_result: If True, the flate grid will be stored in memory. The next time that this
                method is called, the grid will be returned without recomputing it.
        """
        # If the grid is already computed, return it
        if self._flat_grid is not None:
            return self._flat_grid
        # Otherwise, compute the grid
        flat_gr: npt.NDArray[np.float64] = np.array([])
        if self.dimensions == 1:
            flat_gr = self.mesh[0].flatten().reshape((self.grid_size, 1))
        if self.dimensions == 2:
            flat_gr = np.vstack((self.mesh[0].flatten(), self.mesh[1].flatten())).T
        if self.dimensions == 3:
            flat_gr = np.vstack(
                (self.mesh[0].flatten(), self.mesh[1].flatten(), self.mesh[2].flatten())).T
        # If the cache is enabled, store the grid to save computation time
        if cache_result:
            self._flat_grid = flat_gr
        return flat_gr

    def project_to_unit_cell(self,
                            real_space_lattice_parameter: np.float64 | None = None,
                            k_space_lattice_parameter: np.float64 | None = None,
                            cache_result: bool = False) -> npt.NDArray[np.float64]:
        """This method projects the grid to the unit cell

        Args:
            real_space_lattice_parameter: the lattice constant in the real space (in atomic units).
            k_space_lattice_parameter: lattice constant in the reciprocal lattice (in atomic units).
            cache_result: If True, the flate grid will be stored in memory. This will overwrite the
                previous grid if it was already computed

        Returns:
            The flat grid projected to the unit cell
        """
        if self.dimensions != 2:
            raise NotImplementedError("This method is only available for 2D grids")
        flat_gr: npt.NDArray[np.float64] = np.dot(
            self.flat_grid(),
            get_k_space_hexagonal_cell(
                real_space_lattice_parameter=real_space_lattice_parameter,
                k_space_lattice_parameter=k_space_lattice_parameter
            ))
        if cache_result:
            self._flat_grid = flat_gr
        return flat_gr

def get_k_space_hexagonal_cell(
        real_space_lattice_parameter: np.float64 | None = None,
        k_space_lattice_parameter: np.float64 | None = None) -> npt.NDArray[np.float64]:
    """This method computes the hexagonal cell in k space of a 2D material.

    Args:
        real_space_lattice_parameter: the lattice constant in the real space (in atomic units).
        k_space_lattice_parameter: lattice constant in the reciprocal lattice (in atomic units).

    Returns:
        The hexagonal cell in k space
    """
    if real_space_lattice_parameter is not None:
        real_lattice = [
            [real_space_lattice_parameter, 0],
            [real_space_lattice_parameter / 2, real_space_lattice_parameter * 3**0.5 / 2]
        ]
        return np.array(2. * np.pi * np.linalg.inv(real_lattice).T, dtype=np.float64)
    if k_space_lattice_parameter is not None:
        rec_lattice = [
            [k_space_lattice_parameter, 0],
            [k_space_lattice_parameter / 2, k_space_lattice_parameter * 3**0.5 / 2]
        ]
        return np.array(rec_lattice, dtype=np.float64)
    raise ValueError("You must provide either the real lattice or the lattice constant")
