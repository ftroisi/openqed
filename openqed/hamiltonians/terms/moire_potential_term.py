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

"""This module contains the Moiré potential class"""

from typing import Literal, cast
import numpy as np
import numpy.typing as npt

from openqed.grid.real_space_grid import RealSpaceGrid

from openqed.hamiltonians.hamiltonian import Hamiltonian
from openqed.hamiltonians.terms.hamiltonian_term import HamiltonianTerm
from openqed.hamiltonians.types.bilayer_excitons import BilayerStructure

from openqed.utils.rotations import rotate_vector

class MoirePotentialTerm(HamiltonianTerm):
    """
    This class represents the Moiré potential term.
    The Moiré potential term describes the potential energy of the particle due to the Moiré pattern.

    References:
        [1] Tunable Phases of Moiré Excitons in van der Waals Heterostructures:
            https://pubs.acs.org/doi/10.1021/acs.nanolett.0c03019
        [2] Moiré straintronics: a universal platform for reconfigurable quantum materials:
            https://www.nature.com/articles/s41699-023-00382-4
    """
    def __init__(self,
                hamiltonian: Hamiltonian,
                 *,
                layers_lattice_vectors: BilayerStructure,
                twist_angle: np.float64 | None,
                moire_lattice_vectors: npt.NDArray[np.float64] | None,
                moire_prefactor: np.complex128,
                moire_symmetry: Literal["C2", "C4", "C6"] = "C6") -> None:
        """Initialize the MoirePotential class

        Args:
            hamiltonian: The Hamiltonian object to which the term belongs
            layers_lattice_vectors: The lattice vectors of the layers. ``layer1`` is the one at the
                bottom, which is untwisted, and ``layer2`` is the one at the top. All other keys will
                be ignored. The values represent the lattice vectors of the layers.
            twist_angle: The twist angle between the two layers of the bilayer structure. Only the top
                layer will be twisted, while the bottom will remain unchanged. The angle must be provided
                in degrees. If this is None, the moire_lattice_vectors must be provided.
            moire_lattice_vectors: The lattice vectors of the Moiré pattern. If this is None, the
                twist_angle must be provided.
            moire_prefactor: The prefactor of the Moiré potential. This is a complex number. It is
                defined as in Eq. 7 of the Supporting Info of Ref. [1].
            moire_symmetry: The symmetry of the Moiré pattern. It can be either "C2", "C4", or "C6".
                The default is "C6". This parameter controls the number of terms in the sum in Eq. 6 of
                the Supporting Info of Ref. [1]. If the symmetry is "C2", only the term with angle 0 is
                considered (linear Moire pattern). If the symmetry is "C4", the terms with angles 0 and
                120 are considered. If the symmetry is "C6", the terms with angles 0, 120, and 240 are
                considered (complete Moire pattern).
        """
        super().__init__(hamiltonian)
        if isinstance(hamiltonian.grid, RealSpaceGrid):
            raise NotImplementedError("RealSpaceGrid is not supported yet")
        # Check that the lattice vectors are provided
        if not all(key in layers_lattice_vectors for key in ["layer1", "layer2"]):
            raise ValueError("The lattice vectors of both layers must be provided")
        # Check that either twist_angle or moire_lattice_vectors is not None
        if (all(arg is None for arg in [twist_angle, moire_lattice_vectors]) or
            all(arg is not None for arg in [twist_angle, moire_lattice_vectors])):
            raise ValueError("Either twist_angle or moire_periodicity must be provided")
        # Save periodicity parameters
        self.moire_prefactor = moire_prefactor
        self.moire_symmetry = moire_symmetry

        # Compute the Moire lattice vectors if the twist angle is provided
        bottom_layer_lattice_vectors = \
            cast(npt.NDArray[np.float64], layers_lattice_vectors.get("layer1"))
        top_layer_lattice_vectors = \
            cast(npt.NDArray[np.float64], layers_lattice_vectors.get("layer2"))
        if twist_angle is not None:
            flat_k_grid = hamiltonian.grid.flat_grid()
            # Compute the discretized twist angle and use it to rotate the top layer lattice vectors
            discretized_twist_angle = self.get_discretized_twist_angle(
                flat_k_grid, top_layer_lattice_vectors, twist_angle)
            rotated_top_layer_lattice_vectors =  np.array(
                [rotate_vector(discretized_twist_angle, v) for v in top_layer_lattice_vectors],
                dtype=np.float64)
            # Compute the moire lattice vectors
            self.moire_lattice_vectors: npt.NDArray =\
                bottom_layer_lattice_vectors - rotated_top_layer_lattice_vectors
            # Save the twist angle
            self.twist_angle = discretized_twist_angle
            # TODO: WARN THE USER!
        # If the user provided the Moire, then we save it directly, and compute the twist angle from them
        else:
            self.moire_lattice_vectors = cast(npt.NDArray[np.float64], moire_lattice_vectors)
            # The twist angle is the angle between the rotated top layer lattice vector and the bottom
            # layer lattice vector. The former is expressed as the unrotated top layer lattice vector
            # times the rotation matrix
            self.twist_angle: np.float64 = np.degrees(
                -np.pi / 4 + np.arccos(
                    (self.moire_lattice_vectors[0, 0] + bottom_layer_lattice_vectors[0, 0]) /
                    (np.sqrt(2) * top_layer_lattice_vectors[0, 0])
                ))
            # Rotate the top layer lattice vectors
            rotated_top_layer_lattice_vectors =  np.array(
                [rotate_vector(self.twist_angle, v) for v in top_layer_lattice_vectors],
                dtype=np.float64)

        # Compute the sum of the lattice vectors
        self.sum_lattice_vectors = bottom_layer_lattice_vectors + rotated_top_layer_lattice_vectors
        # Finally, compute the moire lattice vectors in real space
        self.moire_lattice_vectors_real_space =\
            self._get_real_space_moire_vectors(self.moire_lattice_vectors)

    def _get_real_space_moire_vectors(
            self, k_space_vectors: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        This method computes the real space Moiré vectors using equation 5 of the Supporting Info of Ref.
        [2].
        """
        rotated_k_space_vectors = np.array(
            [rotate_vector(np.float64(90), v) for v in k_space_vectors], dtype=np.float64)
        denominator: np.float64 = np.linalg.norm(
            np.dot(k_space_vectors[0, :].T, rotated_k_space_vectors[1, :]), axis=0)
        return cast( npt.NDArray[np.float64], (2 * np.pi / denominator) * rotated_k_space_vectors)

    def get_discretized_twist_angle(self,
                                    flat_k_grid: npt.NDArray[np.float64],
                                    lattice_vectors: npt.NDArray[np.float64],
                                    twist_angle: np.float64) -> np.float64:
        """
        This method computes the discretized the twist angle after the rotation of the lattice vectors.
        Since the grid is discrete, after the rotation of the vector there is no guarantee that the
        rotated vector will be exactly on the grid, thus the real twsit angle will be slightly different
        from the one provided by the user.

        Args:
            flat_k_grid: The grid in k-space
            lattice_vectors: The lattice vectors to rotate
            twist_angle: The twist angle

        Returns:
            The discretized twist angle in degrees
        """
        # First, rotate the lattice vectors by the twist angle
        first_rotated_lattice_vectors: npt.NDArray[np.float64] = np.array(
                rotate_vector(twist_angle, lattice_vectors[0]), dtype=np.float64)
        # Find the index of the k-point that is closest to the rotated top lattice vector
        index_closest_k_point = self.hamiltonian.grid.get_point_index(first_rotated_lattice_vectors)
        # Compute the discretized twist angle
        denominator = np.linalg.norm(flat_k_grid[index_closest_k_point]) *\
                                     np.linalg.norm(first_rotated_lattice_vectors)
        discretized_twist_angle = np.float64(np.degrees(np.arccos(
            np.dot(flat_k_grid[index_closest_k_point], first_rotated_lattice_vectors) / denominator)
        ))
        return discretized_twist_angle

    def get_hamiltonian_term(self) -> npt.NDArray[np.complex128]:
        """
        This method computes the Moiré potential, implementing equation 6 of the Supporting Info of Ref.
        [1].

        Returns:
            A 1D array, with the size of the grid, representing the Moiré potential
        """
        flat_k_grid = self.hamiltonian.grid.flat_grid()
        spacing: np.float64 = np.float64(np.linalg.norm(flat_k_grid[1] - flat_k_grid[0])) / 2
        v_q = np.zeros((np.size(flat_k_grid, 0)), dtype=np.complex128)
        # Generate an array of ones with lenght = number_of_kpt.
        # This array will then be multiplied by either 1 (if the kpt on the grid corresponds to
        # C_3^n(G_l - G_l')) or 0 (corresponding to the Kroenecker delta)
        deltas = np.ones((np.size(flat_k_grid, 0)), dtype=np.int64)
        # Define the rotation angles
        rotation_angles = [0]
        if self.moire_symmetry == "C4":
            rotation_angles.append(120)
        elif self.moire_symmetry == "C6":
            rotation_angles.extend([120, 240])
        for angle in rotation_angles:
            # First compute the correct rotation for g0_1 +/- g0_2
            g_sum = np.array(
                [rotate_vector(np.float64(angle), v) for v in self.sum_lattice_vectors],
                dtype=np.float64)
            g_diff = np.array(
                [rotate_vector(np.float64(angle), v) for v in self.moire_lattice_vectors],
                dtype=np.float64)
            # Compute C_3^n(G_l + G_l')) * D_l
            displacement: np.float64 = np.sum(
                [np.dot(g_sum[i, :], self.moire_lattice_vectors_real_space[i, :])
                    for i in range(g_sum.shape[0])])
            # Then add the contibution to v_q
            v_q += np.exp(1j * displacement / 2) * deltas * (
                (np.abs(flat_k_grid[:, 0] - g_diff[0]) < spacing) *
                (np.abs(flat_k_grid[:, 1] - g_diff[1]) < spacing))
        # Finally, return the Moiré potential
        return cast(npt.NDArray[np.complex128], self.moire_prefactor * v_q)
