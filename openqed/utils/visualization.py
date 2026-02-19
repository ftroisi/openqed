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

"""This module contains utilities to visualize matrices and vectors"""

from typing import Literal
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

def visualize_matrix(
        matrix: npt.NDArray,
        type_to_plot: Literal['abs', 'real', 'imag'] = 'abs',
        title: str = "",
        log_plot: bool = False
) -> tuple[plt.Figure, plt.Axes]:
    """
    Visualizes a matrix using matplotlib.

    Args:
        matrix: The matrix to visualize.
        type_to_plot: The type of the matrix to plot. Can be 'abs', 'real', or 'imag'.
        title: The title of the plot.
        log_plot: Whether to plot the logarithm of the matrix values.
    """
    # Create a new figure and axes
    fig, ax = plt.subplots()

    # Select data based on type
    if type_to_plot == 'abs':
        data = np.abs(matrix)
    elif type_to_plot == 'real':
        data = np.real(matrix)
    elif type_to_plot == 'imag':
        data = np.imag(matrix)
    else:
        raise ValueError(f'Unknown type to plot: {type_to_plot}. Must be one of "abs", "real", or "imag".')

    # Apply log scaling if requested
    if log_plot:
        # Avoid log(0) issues by adding a tiny epsilon or using specialized norms
        data = np.log(data + 1e-15)

    # Plot
    cax = ax.matshow(data, interpolation='none')

    # Add colorbar
    cbar = fig.colorbar(cax, ax=ax)
    cbar.set_label(f'M.{type_to_plot}', rotation=270, labelpad=15)

    ax.set_title(title)
    plt.show()

    del data
    return fig, ax
