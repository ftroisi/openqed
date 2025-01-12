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

"""This module contains the Mott-Wannier Hamiltonian class"""

from typing import Literal
import numpy as np
import numpy.typing as npt

from openqed.hamiltonians.hamiltonian import Hamiltonian

class LightMatterHamiltonian(Hamiltonian):
    """
    This class defines the generalized QED light-matter problem. It shall be used to couple a
    matter Hamiltonian with the radiation field.
    """
    def __init__(self,
                *,
                mode: Literal['QED', 'QED-Floquet'] = 'QED',
                hamiltonian_terms: list[Literal['light', 'matter', 'bilinear', 'diamagnetic']] =
                    ['light', 'matter', 'bilinear', 'diamagnetic'],
                n_modes: int = 1,
                mode_symm: Literal['odd', 'even'] | None = None,
                w_c: npt.NDArray[np.float64] | None = None,
                a_0: npt.NDArray[np.float64] | None = None,
                polarization: Literal['x', 'y', 'z', 'xy', 'xz', 'yz', '+', '-'] = 'x',
                n_photons: int = 0,
                h_mat: npt.NDArray[np.complex128] | None = None,
                p_mat: npt.NDArray[np.complex128] | None = None,
                n_k: int = 1,
                a2_mat: npt.NDArray[np.float64] | None = None,
                kappa=1.,
                w_fl=0.,
                a_fl=None,
                occ_mat=None,
                occ_ph=None):
        # Calculation mode
        self.mode: Literal['QED', 'QED-Floquet'] = mode
        # Defining the terms to include in the hamiltonian
        self.hamiltonian_terms = hamiltonian_terms

