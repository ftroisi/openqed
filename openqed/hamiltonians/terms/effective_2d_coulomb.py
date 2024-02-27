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

"""This module contains the analytical Effective 2D Coulomb Hamiltonian-Term class"""

from typing import TypedDict
import numpy as np
import numpy.typing as npt
from openqed.hamiltonians.hamiltonian import Hamiltonian

class BilayerStructure(TypedDict, total=False):
    """This class is used to define a certain property for a bilayer structure."""
    substrate1: np.float64
    layer1: np.float64 | npt.NDArray[np.float64]
    interlayer: np.float64
    layer2: np.float64 | npt.NDArray[np.float64]
    substrate2: np.float64

class Effective2DCoulombPotential():
    """
    This class defines the analytical effective 2D Coulomb potential Hamiltonian term.\\
    It implements the formulas of the Supplementary Note 1 of the Supporting Info of the paper:
    https://doi.org/10.1038/s42005-019-0122-z

    Attributes:
        `hamiltonian`: The Hamiltonian object of which this term is a part of.

    Note:
        The formulas are taylored for a bilayer heterostructure, so you should use this class only
        for this kind of system.
    """
    def __init__(self,
                hamiltonian: Hamiltonian,
                *,
                thicknesses: BilayerStructure,
                dielectric_constants: BilayerStructure):
        self.hamiltonian: Hamiltonian = hamiltonian
        # Now set the properties of the bilayer structure
        self.dielectric_constants: BilayerStructure = dielectric_constants
        self.thicknesses: BilayerStructure = thicknesses

    def _get_dielectric_constants(self, ref_layer: str, coupled_layer: str):
        """
        This method is a getter for the dielectric constants of the bilayer structure.

        Args:
            ref_layer: the layer for which the auxiliary variables are computed. Its value can be
                either `layer1` or `layer2`
            coupled_layer: the layer coupled to the reference layer. Its value can be either
                `layer1` or `layer2`

        Returns:
            A tuple containing the following auxiliary variables:
            - eps_1: dielectric constant of the surrounding medium of reference layer (`substrate1`)
            - eps_2: dielectric constant of the surrounding medium of coupled layer (`substrate2`)
            - eps_r: dielectric constant of the interlayer medium (`interlayer`)
            - eps_l1: dielectric constant of the reference layer (`layer1`)
            - eps_l2: dielectric constant of the coupled layer (`layer2`)
        """
        if not ref_layer in ['layer1', 'layer2'] or not coupled_layer in ['layer1', 'layer2']:
            raise ValueError(f"Argument {ref_layer} or {coupled_layer} is not valid")
        # Surrounding dielectric constants
        eps_1 = np.float64(1.)
        try:
            eps_r: np.float64 = self.dielectric_constants[
                'substrate1' if ref_layer == 'layer1' else 'substrate2']
        except ValueError as exc:
            raise ValueError(f"Invalid dielectric constant for {ref_layer}'s substrate") from exc
        eps_2 = np.float64(1.)
        try:
            eps_2: np.float64 = self.dielectric_constants[
                'substrate1' if coupled_layer == 'layer1' else 'substrate2']
        except ValueError as exc:
            raise ValueError(f"Invalid dielectric constant for {coupled_layer} substrate") from exc
        eps_r = np.float64(1.)
        try:
            eps_r: np.float64 = self.dielectric_constants['interlayer']
        except ValueError as exc:
            raise ValueError("The dielectric constant of the interlayer is not valid") from exc
        # Layer dielectric constants
        try:
            eps_l1: np.float64 = self.dielectric_constants[ref_layer]
            eps_l2: np.float64 = self.dielectric_constants[coupled_layer]
        except ValueError as exc:
            raise ValueError("The layers dielectric constant for are not valid") from exc
        return eps_1, eps_2, eps_r, eps_l1, eps_l2

    def _get_thicknesses(self,
                        ref_layer: str,
                        coupled_layer: str) -> tuple[np.float64, np.float64, np.float64]:
        """
        This method is a getter for the thicknesses of the bilayer structure.

        Args:
            ref_layer: the layer for which the auxiliary variables are computed. Its value can be
                either `layer1` or `layer2`
            coupled_layer: the layer coupled to the reference layer. Its value can be either
                `layer1` or `layer2`
            k_gr: the grid in k space

        Returns:
            A tuple containing the following auxiliary variables:
            - th_kr: quantity tanh(k * R), k = k grid, R = interlayer distance
            - d_1: thickness of the reference layer
            - d_2: thickness of the coupled layer
        """
        try:
            d_l1: np.float64 = self.thicknesses[ref_layer]
            d_l2: np.float64 = self.thicknesses[coupled_layer]
            d_inter: np.float64 = self.thicknesses['interlayer']
            return d_l1, d_l2, d_inter
        except ValueError as exc:
            raise ValueError("The thicknesses are not valid") from exc

    def _get_f_k(self,
                ref_layer: str,
                coupled_layer: str,
                k_grid: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """This method computes f(k), a the term for the dielectric function. It is equation 4 of
        the paper in the class docstring.

        Args:
            ref_layer: the layer for which the auxiliary variables are computed. Its value can be
                either `layer1` or `layer2`
            coupled_layer: the layer coupled to the reference layer. Its value can be either
                `layer1` or `layer2`
            k_grid: the grid in k space

        Return:
            the value of f(k) on all the k grid
        """
        # First, define some useful quantitites
        eps_1, eps_2, eps_r, eps_l1, eps_l2 = \
            self._get_dielectric_constants(ref_layer=ref_layer,coupled_layer=coupled_layer)
        eps_12: np.float64 = eps_1 * eps_2
        eps_l12: np.float64 = eps_l1 * eps_l2
        d_l1, d_l2, d_inter = \
            self._get_thicknesses(ref_layer=ref_layer, coupled_layer=coupled_layer)
        th_kr: npt.NDArray[np.float64] = np.tanh(k_grid * d_inter)
        # Then compute the various terms of f_k
        f_k: npt.NDArray[np.float64] = (eps_1 + eps_2) + (eps_12 / eps_r + eps_r) * th_kr
        f_k += ((eps_l1 + eps_12 / eps_l1) + (eps_l1 * eps_2 / eps_r +
               eps_1 * eps_r / eps_l1) * th_kr) * np.tanh(k_grid * d_l1)
        f_k += ((eps_l2 + eps_12 / eps_l2) + (eps_l2 * eps_1 / eps_r +
               eps_2 * eps_r / eps_l2) * th_kr) * np.tanh(k_grid * d_l2)
        f_k += (((eps_l2 * eps_1 / eps_l1 + eps_l1 * eps_2 / eps_l2) +
               (eps_l12 / eps_r + eps_12 * eps_r / eps_l12) * th_kr) *
               np.tanh(k_grid * d_l1) * np.tanh(k_grid * d_l2))
        return f_k

    def get_intralayer_dielectric_function(self,
                                            ref_layer: str,
                                            coupled_layer: str,
                                            k_grid: npt.NDArray[np.float64]):
        """This method computes `epsilon_ll`, the dielectric function for intralyer excitons.
        It is equation 3 of the paper in the class docstring.

        Args:
            ref_layer: the layer for which the auxiliary variables are computed. Its value can be
                either `layer1` or `layer2`
            coupled_layer: the layer coupled to the reference layer. Its value can be either
                `layer1` or `layer2`
            k_grid: the grid in k space

        Return:
            epsilon_ll(k): the value of the dielectric function on all the k grid
        """
        # First, define some useful quantitites
        eps_1, eps_2, eps_r, eps_l1, eps_l2 = \
            self._get_dielectric_constants(ref_layer=ref_layer,coupled_layer=coupled_layer)
        eps_l12: np.float64 = eps_l1 * eps_l2
        d_l1, d_l2, d_inter = \
            self._get_thicknesses(ref_layer=ref_layer, coupled_layer=coupled_layer)
        th_kr: npt.NDArray[np.float64] = np.tanh(k_grid * d_inter)
        # Then compute f(k)
        f_k = self._get_f_k(ref_layer=ref_layer, coupled_layer=coupled_layer, k_grid=k_grid)
        # Compute the various terms of the dielectric function
        denom = (np.cosh(k_grid * d_l1 / 2)**2 * (1 + (eps_1/eps_l1) * np.tanh(k_grid * d_l1 / 2)))
        f_k_denom: npt.NDArray[np.float64] = 1 + (eps_2 / eps_r) * th_kr
        f_k_denom += (eps_2/eps_l2 + eps_l2/eps_r * th_kr) * np.tanh(k_grid * d_l2)
        f_k_denom += (eps_2/eps_l1 + eps_r/eps_l1 * th_kr) * np.tanh(k_grid * d_l1 / 2)
        f_k_denom += ((eps_l2/eps_l1 + eps_2*eps_r / eps_l12 * th_kr) *
            np.tanh(k_grid * d_l2) * np.tanh(k_grid * d_l1 / 2))
        # Combine the terms to get the dielectric function
        return (np.cosh(k_grid * d_l1) / denom) * (f_k / f_k_denom)

    def get_interlayer_dielectric_function(self,
                                        ref_layer: str,
                                        coupled_layer: str,
                                        k_grid: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """This method computes `epsilon_ll'`, the dielectric function for interlyer excitons.
        It is equation 3 of the paper in the class docstring.

        Args:
            ref_layer: the layer for which the auxiliary variables are computed. Its value can be
                either `layer1` or `layer2`
            coupled_layer: the layer coupled to the reference layer. Its value can be either
                `layer1` or `layer2`
            k_grid: the grid in k space

        Return:
            epsilon_ll'(k): the value of the dielectric function on all the k grid
        """
        # First, define some useful quantitites
        eps_1, eps_2, _, eps_l1, eps_l2 = \
            self._get_dielectric_constants(ref_layer=ref_layer,coupled_layer=coupled_layer)
        d_l1, d_l2, d_inter = \
            self._get_thicknesses(ref_layer=ref_layer, coupled_layer=coupled_layer)
        cosh_kr = np.cosh(k_grid * d_inter)
        # Then compute f(k)
        f_k = self._get_f_k(ref_layer=ref_layer, coupled_layer=coupled_layer, k_grid=k_grid)
        # Now compute the dielectic function
        denom: npt.NDArray[np.float64] = np.cosh(k_grid * d_l1 / 2)
        denom *= np.cosh(k_grid * d_l2 / 2)
        denom *= (1 + eps_1/eps_l1 * np.tanh(k_grid * d_l1 / 2))
        denom *= (1 + eps_2/eps_l2 * np.tanh(k_grid * d_l2 / 2))
        # Note that eps will be an array with the dimensions of the k grid
        return cosh_kr * f_k * ((np.cosh(k_grid * d_l1) * np.cosh(k_grid * d_l2)) / denom)

    def get_w_q_analytical(self, k_gr: npt.NDArray[np.float64], layer: str):
        """
        This function computes the analytical effective 2D Coulomb potential.
        It is equation 2 of the paper in the class docstring.

        Args:
            k_vector (real, array): grid in k space
            layer_one, layer_two (int, scalar): index of the layer

        Returns:
            W_q (array, real): w_q is an array of the dimension of the k grid
                containing the analytical screened potential
        """
        # Generate coulomb kernel
        k_grid = self.generate_coulomb_kernel()
        # First, define the two layers. Since the layer argument contains one of the keys of
        # the Excitons class, we need to convert it to the key of the SystemStructure class
        ref_layer = excitons_key_to_system_key(layer.split("_")[0])
        coupled_layer = excitons_key_to_system_key(layer.split("_")[1])
        # Compute the in-plane momentum (k parallel)
        eps = 1.
        # Compute the dielectric function.
        if ref_layer == coupled_layer:
            # Intra-layer casex
            eps = self.get_intralayer_dielectric_function(
                ref_layer=ref_layer, coupled_layer='layer1' if ref_layer == 'layer2' else 'layer2',
                k_grid=k_grid)
        else:
            # Inter-layer case
            eps = self.get_interlayer_dielectric_function(ref_layer=ref_layer,
                                                        coupled_layer=coupled_layer, k_grid=k_grid)
        # Element-wise multiplication. The 4pi factor come from the conversion
        # of the vacuum permittivity eps_0 in atomic units
        q_c = np.float64(np.linalg.norm(self.flat_k_grid[1] - self.flat_k_grid[0]))
        np.fill_diagonal(k_grid, q_c / 4.)
        w_q = (q_c / (2*np.pi))**2 * np.divide(4 * np.pi, (np.multiply(k_grid, eps)))
        return w_q

    def generate_coulomb_kernel(self) -> npt.NDArray[np.float64]:
        """
        This method generates the Coulomb kernel 1/|r - r'|.
        TODO: This will probably go to some abstract class, since it is not specific to this class
        """
        # Define the grid with kx and ky components. This grid has the correct
        # number of columns and twice the number of rows (for kx and ky)
        flat_grid = self.hamiltonian.grid.flat_grid()
        # Generate x and y matrix
        kx_grid = np.add.outer(flat_grid[:,0], -flat_grid[:,0])
        ky_grid = np.add.outer(flat_grid[:,1], -flat_grid[:,1])
        # Finally set up the actual grid on which to compute the matrices
        return np.array(np.power(kx_grid, 2) + np.power(ky_grid, 2), dtype=np.float64)