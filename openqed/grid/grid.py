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

"""This module contains the base abstract Grid class"""

from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
import numpy.typing as npt
from scipy.spatial import ConvexHull
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
        # Ensure that the keys of the `boundaries` and `spacing` dictionaries are the same
        if len(self.spacing.keys()) != len(self.boundaries.keys()):
            raise ValueError(
                "The keys of the `boundaries` and `spacing` dictionaries must be the same." +
                f"Spacing: {self.spacing.keys()}; Boundaries {self.boundaries.keys()}")
        self.grid_type = None

    @property
    def dimensions(self) -> int:
        """Get the number of dimensions of the grid"""
        return len(self.spacing.keys())

    @abstractmethod
    def flat_grid(self) -> npt.NDArray[np.float64]:
        """Get the flat grid"""
        pass

    @abstractmethod
    def get_spacing(self) -> np.float64:
        """Get the spacing of the grid"""
        pass

    def visualize_flattened_grid(self, axis: int, markersize=10, color='g'):
        """
        This method prints a flattened grid to a figure

        Args:
            grid: the flattened grid to plot. Must be generated with .flatten() method, which can be
            used after creating a meshgrid

            axis: The grid is a 2D array, on on dimension we have the coordinates of the points and on
            the other we have all the points. This parameter represents the latter

            markersize: size of markers
            color: color of markers
        """
        import matplotlib.pyplot as plt
        # Define the number of points in the grid and the dimension
        size = self.flat_grid().shape[axis]
        dim = self.flat_grid().shape[0] if axis == 1 else self.flat_grid().shape[1]
        if dim == 1:
            for i in range(size):
                plt.plot(self.flat_grid()[i, 0], 0, f'{color}-o', markersize=markersize)
        elif dim == 2:
            for i in range(size):
                plt.plot(self.flat_grid()[i, 0], self.flat_grid()[i, 1], f'{color}-o', markersize=markersize)
        elif dim == 3:
            ax = plt.axis(projection='3d')
            for i in range(size):
                ax.plot3D(self.flat_grid()[i, 0], self.flat_grid()[i, 1], self.flat_grid()[i, 2], f'{color}-o', markersize=markersize)  # type: ignore

        plt.axis("equal")
        plt.show()

    def get_max_inscribed_circle_radius(self) -> np.float64:
        """Find the radius of the largest circle fully contained by the grid.

        For a 2D grid, this computes the minimum distance from the grid centroid
        to any edge of the convex hull (not to vertices).

        Returns:
            The inscribed circle radius in the same units as the grid.

        Raises:
            ValueError: If the grid is not 2-dimensional.
        """
        if self.dimensions != 2:
            raise ValueError(
                    f"Max inscribed circle is only supported for 2D grids. "
                    f"Got {self.dimensions}D grid."
            )

        grid_points = self.flat_grid()

        # Compute centroid
        centroid = np.mean(grid_points, axis=0)

        # Get convex hull
        hull = ConvexHull(grid_points)
        boundary_points = grid_points[hull.vertices]

        # Find minimum distance from centroid to any hull edge
        min_distance = np.inf
        n_vertices = len(hull.vertices)

        for i in range(n_vertices):
            # Get edge endpoints
            p1 = boundary_points[i]
            p2 = boundary_points[(i + 1) % n_vertices]

            # Compute perpendicular distance from centroid to edge
            edge_vector = p2 - p1
            edge_length = np.linalg.norm(edge_vector)

            if edge_length > 1e-10:  # Avoid division by zero
                # Vector from p1 to centroid
                to_centroid = centroid - p1

                # Distance formula: |edge_vector × to_centroid| / |edge_vector|
                # In 2D: cross product gives scalar
                cross = edge_vector[0] * to_centroid[1] - edge_vector[1] * to_centroid[0]
                distance = abs(cross) / edge_length

                min_distance = min(min_distance, distance)

        return np.float64(min_distance)

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
        # If more than one point is found, raise an error
        if index.shape[0] > 1:
            raise ValueError(f"More than one point found in the grid: {index}")
        if index.shape[0] == 0:
            return -1
        return int(index[0])
