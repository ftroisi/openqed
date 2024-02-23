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
                input_file: InputFile,
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
        self.grid_type: GridType = GridType(input_file.input["grid_type"])
        self.spacing: dict[str, np.float64] = input_file.input["grid_spacing"]
        self.boundaries: dict[str, tuple[np.float64, np.float64]] = \
            input_file.input["grid_boundaries"]

    @property
    def dimensions(self) -> int:
        """Get the number of dimensions of the grid"""
        return len(self.spacing.keys())
