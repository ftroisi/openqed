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

"""This module contains the base abstract Grid class"""

from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
import numpy.typing as npt
from openqed.io.input_file import InputFile

class GridType(Enum):
    """This class is an enumeration of the possible grid types"""
    REAL_SPACE = "real_space"
    K_SPACE = "k_space"

class Grid(ABC):
    """This class is the base abstract Grid class. All modules in the `grid` folder
    and subfolders must inherit from this class.

    Attributes:
        `name`: The name of the Grid.
        `spacing`: The spacing of the grid.
    """
    def __init__(self,
                input_file: InputFile | None = None,
                boundaries: dict[str, tuple[np.float64, np.float64]] | None = None,
                spacing: dict[str, np.float64] | None = None) -> None:
        """Initialize the Grid

        Args:
            boundaries: The boundaries of the grid. It is a dictionary where each key represents a
                        dimension (e.g. `x`, `y`...) and the values are tuples with the minimum and
                        maximum values of the grid.
            spacing: The spacing of the grid for each dimension. It is a dictionary where each key

        Raises:
            ValueError: If the keys of the `boundaries` and `spacing` dictionaries are not the same.
        """
        if input_file is None:
            if boundaries is None or spacing is None:
                raise ValueError(
                    "If `input_file` is not provided, both `boundaries` and `spacing` must be.")
            self.spacing: dict[str, np.float64] = spacing
            self.boundaries: dict[str, tuple[np.float64, np.float64]] = boundaries
        else:
            self.spacing: dict[str, np.float64] = input_file.input["grid_spacing"]
            self.boundaries: dict[str, tuple[np.float64, np.float64]] = \
                input_file.input["grid_boundaries"]
        # Ensure that the keys of the `boundaries` and `spacing` dictionaries are the same
        if len(self.spacing.keys()) != len(self.boundaries.keys()):
            raise ValueError(
                "The keys of the `boundaries` and `spacing` dictionaries must be the same." +
                f"Spacing: {self.spacing.keys()}; Boundaries {self.boundaries.keys()}")

    @property
    def dimensions(self) -> int:
        """Get the number of dimensions of the grid"""
        return len(self.spacing.keys())

    @abstractmethod
    def flat_grid(self) -> npt.NDArray[np.float64]:
        """Get the flat grid"""
        raise NotImplementedError()

    def get_point_index(self,
                        point: tuple[np.float64 | float, ...] | npt.NDArray[np.float64]) -> int:
        """Get the index of a point in the flat grid"""
        flat_grid = self.flat_grid()
        # First, check that the dimensions of the point are correct
        point = np.array(point, dtype=np.float64)
        if point.shape[0] != self.dimensions:
            raise ValueError("The point must have the same number of dimensions as the grid")
        # Then, return the index of the point in the flat grid
        index: npt.NDArray[np.int64] = np.array([])
        spacing_keys = list(self.spacing.keys())
        # Define a bitmap to find the point in the grid
        grid_bitmap = np.abs(flat_grid[:, 0] - point[0]) < self.spacing[spacing_keys[0]] / 2
        for i in range(1, self.dimensions):
            grid_bitmap &= (np.abs(flat_grid[:, i] - point[i]) < self.spacing[spacing_keys[i]] / 2)
        index = np.where(grid_bitmap)[0]
        # If more than one point is found, raise an error
        if index.shape[0] > 1:
            raise ValueError(f"More than one point found in the grid: {index}")
        return int(index[0])
