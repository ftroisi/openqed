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

"""This module contains the Moiré potential class"""
from dataclasses import dataclass
from typing import Literal, cast, Dict
import numpy as np
import numpy.typing as npt

from openqed.grid.real_space_grid import RealSpaceGrid

from openqed.hamiltonians.hamiltonian import Hamiltonian
from openqed.hamiltonians.terms.hamiltonian_term import HamiltonianTerm
from openqed.hamiltonians.types.bilayer_excitons import BilayerStructure, BilayerExcitons, excitons_key_to_structure_key
from openqed.unit_cells import UnitCell

from openqed.utils.rotations import rotate_vector

@dataclass
class MoirePotentialParameters:
    """
    layers_lattice_vectors: The lattice vectors of the layers. ``layer1`` is the one at the
        bottom, which is untwisted, and ``layer2`` is the one at the top. All other keys will
        be ignored. The values represent the lattice vectors of the layers.
    twist_angle: The twist angle between the two layers of the bilayer structure. Only the top
        layer will be twisted, while the bottom will remain unchanged. The angle must be provided
        in degrees. If this is None, the moire_lattice_vectors must be provided.
    moire_lattice_vectors: The lattice vectors of the Moiré pattern. If this is None, the twist_angle must be provided.
    moire_prefactor_alpha_valence_band: The prefactor alpha of the valence band of the Moiré potential as defined in
        Eq. 7 of the Supporting Info of Ref. [1]. This is a complex number, defined for each layer of the structure.
    moire_prefactor_alpha_conduction_band: The prefactor alpha of the conduction band of the Moiré potential as defined
        in Eq. 7 of the Supporting Info of Ref. [1]. This is a complex number, defined for each layer of the structure.
    moire_prefactor_beta_valence_band: The prefactor beta of the valence band of the Moiré potential as defined in
        Eq. 7 of the Supporting Info of Ref. [1]. This is a complex number, defined for each layer of the structure.
    moire_prefactor_beta_conduction_band: The prefactor beta of the conduction band of the Moiré potential as defined in
        Eq. 7 of the Supporting Info of Ref. [1]. This is a complex number, defined for each layer of the structure.
    moire_symmetry: The symmetry of the Moiré pattern. It can be either "C2", "C4", or "C6". The default is "C6".
        This parameter controls the number of terms in the sum in Eq. 6 of the Supporting Info of Ref. [1].
        If the symmetry is "C2", only the term with angle 0 is considered (linear Moiré pattern).
        If the symmetry is "C4", the terms with angles 0 and 120 are considered. If the symmetry is "C6",
        the terms with angles 0, 120, and 240 are considered (complete Moiré pattern).
    """
    layers_lattice_vectors: Dict[Literal["layer1", "layer2"], UnitCell]
    twist_angle: np.float64 | None
    moire_lattice_vectors: npt.NDArray[np.float64] | None
    moire_prefactor_alpha_valence_band: BilayerStructure
    moire_prefactor_alpha_conduction_band: BilayerStructure
    moire_prefactor_beta_valence_band: BilayerStructure
    moire_prefactor_beta_conduction_band: BilayerStructure
    moire_symmetry: Literal["C2", "C4", "C6"] = "C6"

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
                wannier_eigenvectors: BilayerExcitons,
                wannier_eigenvalues: BilayerExcitons,
                moire_parameters: MoirePotentialParameters
        ) -> None:
        """Initialize the MoirePotential class

        Args:
            hamiltonian: The Hamiltonian object to which the term belongs
            wannier_eigenvectors: The eigenvectors of the Wannier Hamiltonian for the excitons of the system.
                It must have the same keys of ´wannier_eigenvalues´. Note that the number of columns passed (i.e. the
                number of states) will be used to determine what to include in the Hamiltonian.
                For example, if only the 1s state is included, only the 1s state will be included in the Hamiltonian.
            wannier_eigenvalues: The eigenvalues of the Wannier Hamiltonian for the excitons of the system.
                It must have the same keys of ´wannier_eigenvectors´. Note that the number of eigenvalues passed (i.e.
                the number of states) will be used to determine what to include in the Hamiltonian.
                For example, if only the 1s state is included, only the 1s state will be included in the Hamiltonian.
            moire_parameters: The parameters of the Moiré potential.
        """
        super().__init__(hamiltonian)
        if isinstance(hamiltonian.grid, RealSpaceGrid):
            raise NotImplementedError("RealSpaceGrid is not supported yet")
        # Check that the lattice vectors are provided
        if not all(key in moire_parameters.layers_lattice_vectors for key in ["layer1", "layer2"]):
            raise ValueError("The lattice vectors of both layers must be provided")
        # Check that either twist_angle or moire_lattice_vectors is not None
        if (all(arg is None for arg in [moire_parameters.twist_angle, moire_parameters.moire_lattice_vectors]) or
            all(arg is not None for arg in [moire_parameters.twist_angle, moire_parameters.moire_lattice_vectors])):
            raise ValueError("Either twist_angle or moire_periodicity must be provided")
        # Save periodicity parameters
        self.moire_prefactor_alpha_conduction_band = moire_parameters.moire_prefactor_alpha_conduction_band
        self.moire_prefactor_alpha_valence_band = moire_parameters.moire_prefactor_alpha_valence_band
        self.moire_prefactor_beta_conduction_band = moire_parameters.moire_prefactor_beta_conduction_band
        self.moire_prefactor_beta_valence_band = moire_parameters.moire_prefactor_beta_valence_band
        self.moire_symmetry = moire_parameters.moire_symmetry
        # Save the Wannier eigenvectors and eigenvalues
        self.wannier_eigenvectors = wannier_eigenvectors
        self.wannier_eigenvalues = wannier_eigenvalues
        self.exciton = next(k for k, v in wannier_eigenvectors.items() if v is not None)

        # Compute the Moiré lattice vectors if the twist angle is provided
        bottom_layer_lattice_vectors = \
            moire_parameters.layers_lattice_vectors.get("layer1").fit_lattice_vectors_to_grid(self.hamiltonian.grid)
        top_layer_lattice_vectors = \
            moire_parameters.layers_lattice_vectors.get("layer2").fit_lattice_vectors_to_grid(self.hamiltonian.grid)
        if moire_parameters.twist_angle is not None:
            # Compute the discretized twist angle and use it to rotate the top layer lattice vectors
            discretized_twist_angle = self.get_discretized_twist_angle(
                    top_layer_lattice_vectors, moire_parameters.twist_angle)
            print(discretized_twist_angle)
            rotated_top_layer_lattice_vectors = np.array(
                [rotate_vector(discretized_twist_angle, v) for v in top_layer_lattice_vectors], dtype=np.float64)
            # Compute the moiré lattice vectors
            self.moire_lattice_vectors: npt.NDArray[np.float64] =\
                bottom_layer_lattice_vectors - rotated_top_layer_lattice_vectors
            print(bottom_layer_lattice_vectors)
            print(rotated_top_layer_lattice_vectors)
            print(self.moire_lattice_vectors)
            # Save the twist angle
            self.twist_angle = discretized_twist_angle
            # TODO: WARN THE USER!
        # If the user provided the vector of the Moiré BZ, save it directly and compute the twist angle from them
        else:
            self.moire_lattice_vectors = cast(npt.NDArray[np.float64], moire_parameters.moire_lattice_vectors)
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
                [rotate_vector(self.twist_angle, v) for v in top_layer_lattice_vectors], dtype=np.float64)

        # Compute the sum of the lattice vectors
        self.sum_lattice_vectors = bottom_layer_lattice_vectors + rotated_top_layer_lattice_vectors
        # Finally, compute the Moiré lattice vectors in real space
        self.moire_lattice_vectors_real_space = self._get_real_space_moire_vectors(self.moire_lattice_vectors)

    def _get_real_space_moire_vectors(
            self, k_space_vectors: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute real space Moiré lattice vectors from reciprocal space vectors.

        This method transforms reciprocal space (k-space) Moiré lattice vectors to
        real space using the duality relation between direct and reciprocal lattices.
        The transformation follows Eq. 5 of the Supporting Info of Ref. [2].

        For 2D lattices, the real space vectors are obtained by:
        1. Rotating k-space vectors by 90 degrees (perpendicular in reciprocal space)
        2. Scaling by 2π divided by the unit cell area in k-space

        Args:
            k_space_vectors: Array of shape (2, 2) containing two reciprocal lattice
                vectors as rows, where k_space_vectors[i] = [kx, ky] for vector i.

        Returns:
            Real space Moiré lattice vectors as array of shape (2, 2), where each
            row is a real space vector [x, y].

        Note:
            The transformation uses the reciprocal lattice relation:
            a_i · b_j = 2π delta_ij, where a_i are real space vectors and b_j are
            reciprocal space vectors.
        """
        # Rotate k-space vectors by 90° (counterclockwise)
        rotated_k_space_vectors = np.array(
                [rotate_vector(np.float64(90), v) for v in k_space_vectors],
                dtype=np.float64
        )
        # Compute the area of the reciprocal space unit cell
        # Area = |k1 × k2| = |k1_x * k2_y - k1_y * k2_x|
        cross_product = (k_space_vectors[0, 0] * k_space_vectors[1, 1] -
                         k_space_vectors[0, 1] * k_space_vectors[1, 0])
        k_space_area = np.abs(cross_product)
        # Real space vectors = (2π / k_space_area) * rotated_k_vectors
        return (2 * np.pi / k_space_area) * rotated_k_space_vectors

    def get_discretized_twist_angle(self,
            lattice_vectors: npt.NDArray[np.float64], twist_angle: np.float64) -> np.float64:
        """Compute the actual twist angle after discretization to the grid.

        Since the grid is discrete, the rotated lattice vector must snap to the
        nearest grid point. This method computes the actual twist angle that results
        from this discretization, which may differ slightly from the requested angle.

        Args:
            lattice_vectors: The lattice vectors before rotation, shape (2, 2)
            twist_angle: The requested twist angle in degrees

        Returns:
            The actual (discretized) twist angle in degrees between the original
            and discretized rotated lattice vector.
        """
        # Original (unrotated) lattice vector
        original_vector = lattice_vectors[0]
        # Rotate the lattice vector by the requested twist angle
        rotated_vector_ideal = np.array(
                rotate_vector(twist_angle, original_vector), dtype=np.float64
        )
        # Find the closest grid point to the ideal rotated vector
        index_closest_k_point = self.hamiltonian.grid.get_point_index(rotated_vector_ideal)
        discretized_vector = self.hamiltonian.grid.flat_grid()[index_closest_k_point]
        # Compute the angle between the original and discretized vectors
        dot_product = np.dot(original_vector, discretized_vector)
        norm_product = np.linalg.norm(original_vector) * np.linalg.norm(discretized_vector)
        # Clamp to [-1, 1] to avoid numerical issues with arccos
        cos_angle = np.clip(dot_product / norm_product, -1.0, 1.0)
        # The discretized twist angle
        return np.float64(np.degrees(np.arccos(cos_angle)))

    def get_moire_potential_coefficients(self,
            band: Literal['cb', 'vb'], layer: Literal['layer1', 'layer2']) -> npt.NDArray[np.complex128]:
        """
        This method computes the Moiré potential, implementing equation 6 of the Supporting Info of Ref.
        [1].

        Returns:
            A 1D array, with the size of the grid, representing the Moiré potential
        """
        flat_k_grid = self.hamiltonian.grid.flat_grid()
        spacing: np.float64 = self.hamiltonian.grid.get_spacing()
        # Define the rotation angles
        rotation_angles = [0, 180]
        if self.moire_symmetry == "C4":
            rotation_angles.extend([90, 270])
        elif self.moire_symmetry == "C6":
            rotation_angles.extend([60, 120, 240, 300])

        # Since the lattice vectors of the layers were shrunk, the Moire lattice vectors might be too small for the grid
        # Find the optimal multiplier for the Moiré vectors. We need to use the same multiplier for all rotation angles
        moire_magnitude = np.linalg.norm(self.moire_lattice_vectors[0])
        min_multiplier = max(1, int(np.ceil(spacing / moire_magnitude)))
        # Find the smallest multiplier that works for ALL rotation angles
        multiplier = None
        for n in range(min_multiplier, 100):
            all_angles_valid = True
            point_indexes = []
            for angle in rotation_angles:
                g_diff = np.array(
                        [rotate_vector(np.float64(angle), v) for v in self.moire_lattice_vectors],
                        dtype=np.float64
                )
                test_vector = n * g_diff[0, :]

                try:
                    # Check if this multiple can be found on the grid
                    point_idx = self.hamiltonian.grid.get_point_index(test_vector)
                    grid_point = flat_k_grid[point_idx]
                    distance = np.linalg.norm(grid_point - test_vector)

                    # Ensure it's reasonably close (within half grid spacing)
                    if distance > spacing or point_idx in point_indexes:
                        all_angles_valid = False
                        break
                    point_indexes.append(point_idx)
                except (IndexError, ValueError):
                    all_angles_valid = False
                    break

            if all_angles_valid and len(point_indexes) == len(rotation_angles):
                multiplier = n
                break

        if multiplier is None:
            raise ValueError(
                    f"Could not find suitable multiplier for Moiré vectors with magnitude "
                    f"{moire_magnitude:.2e}. Grid spacing: {spacing:.2e}. Consider refining the grid."
            )

        v_q = np.zeros((np.size(flat_k_grid, 0)), dtype=np.complex128)
        for angle in rotation_angles:
            # Compute rotated vectors
            g_sum = np.array(
                    [rotate_vector(np.float64(angle), v) for v in self.sum_lattice_vectors], dtype=np.float64)
            g_diff = np.array(
                    [rotate_vector(np.float64(angle), v) for v in self.moire_lattice_vectors], dtype=np.float64)
            # Scale by multiplier
            g_diff_scaled = multiplier * g_diff
            # Compute displacement term: C_3^n(G_l + G_l') · D_l
            displacement: np.float64 = np.sum(
                    [np.dot(g_sum[i, :], self.moire_lattice_vectors_real_space[i, :])
                     for i in range(g_sum.shape[0])]
            )
            # Find the grid point for the scaled Moiré vector
            point = self.hamiltonian.grid.get_point_index(g_diff_scaled[0, :])
            v_q[point] += np.exp(1j * displacement / 2)

        # Finally, return the Moiré potential
        sigma = 1.
        if band == "cb":
            try:
                alpha = self.moire_prefactor_alpha_conduction_band[layer]
                beta = self.moire_prefactor_beta_conduction_band[layer]
            except ValueError as exc:
                raise ValueError(f"Invalid alpha or beta for {layer}") from exc
        else:
            try:
                alpha = self.moire_prefactor_alpha_valence_band[layer]
                beta = self.moire_prefactor_beta_valence_band[layer]
            except ValueError as exc:
                raise ValueError(f"Invalid alpha or beta for {layer}") from exc
        prefactor: np.complex128 = alpha + np.exp(2j * np.pi * sigma / 3) * beta
        return prefactor * v_q

    def get_form_factor(
            self, psi, q_indexes: list[int] | npt.NDArray[np.int64]) -> tuple[list[np.complex128], list[np.complex128]]:
        """
        This method computes for the form factor appearing in equation 16.
        The idea is to shift the vector psi (which is known on the k-k') base
        by the vector q (a vector on the kpt grid).

        Args:
            psi: the wavefunction of the exciton
            q_indexes: the indexes of the q vectors for which the form factor is computed

        Returns:
            A tuple where each element is an array containing the form factor for each q point:
            [F(q) = sum_k conj(psi(k))*psi(k+q), F^*(q) = sum_k psi(k)*conj(psi(k+q))]
        """
        flat_k_grid = self.hamiltonian.grid.flat_grid()
        kgr_size: int = np.size(flat_k_grid, 0)
        form_factor: list[np.complex128] = []
        form_factor_conj: list[np.complex128] = []
        for qq in q_indexes:
            # Define k + q
            k_plus_q = flat_k_grid[:, :] + flat_k_grid[qq, :]
            total: np.complex128 = np.complex128(0)
            total_conj: np.complex128 = np.complex128(0)
            for i in range(kgr_size):
                if np.abs(psi[i]) < 1e-9:
                    continue
                # Find the indexes on x and y for which the k+q grid matches the original k grid
                idx_x = np.where(np.abs(flat_k_grid[:, 0] - k_plus_q[i, 0]) < 1e-9)[0]
                idx_y = np.where(np.abs(flat_k_grid[:, 1] - k_plus_q[i, 1]) < 1e-9)[0]
                # Intersect the two to find a unique index that matches both x and y
                idx = np.intersect1d(idx_x, idx_y, assume_unique=True)
                if np.size(idx) == 0:
                    continue
                # Compute the form factor
                total += (np.conj(psi[i]) * psi[idx])[0]
                total_conj += (psi[i] * np.conj(psi[idx]))[0]
            form_factor.append(total)
            form_factor_conj.append(total_conj)
        return form_factor, form_factor_conj

    def get_hamiltonian_term(self) -> npt.NDArray[np.complex128]:
        """
        This method builds the Moirè Hamiltonian, that is equation 15 of the Supplementary Information of Ref. [1].

        Returns:
            The Moiré Hamiltonian for the request exciton. It will be a matrix of size
            (1 + num_exc X Q, 1 + num_exc X Q), where num_exc is the number of excitons (1s, 2s ...) that we want to
            include in the Hamiltonian, and Q is the number of k-points in the grid.
        """
        # First build an empty matrix with dimensions (1 + num_exc X q, 1 + num_exc X Q)
        flat_k_grid = self.hamiltonian.grid.flat_grid()
        kgr_size: int = np.size(flat_k_grid, 0)
        number_of_excitonic_states = self.wannier_eigenvectors.get(self.exciton, np.array([])).shape[1]
        size: int = kgr_size * number_of_excitonic_states
        moire_hamiltonian: npt.NDArray[np.complex128] = np.zeros((1 + size, 1 + size), dtype=np.complex128)

        # Define the layers involved. The reference layer is the one in which the hole is located,
        # while the coupled layer is the one in which the electron is located.
        # For intralayer excitons, the two layers are the same, while for interlayer excitons they are different.
        reference_layer: str = excitons_key_to_structure_key(self.exciton.split("_")[0])
        coupled_layer: str = excitons_key_to_structure_key(self.exciton.split("_")[1])

        # Get the Moiré potential
        moire_potential_valence_band: dict[str, npt.NDArray[np.complex128]] = {
            reference_layer: self.get_moire_potential_coefficients(band="vb", layer=reference_layer),
            coupled_layer: self.get_moire_potential_coefficients(band="vb", layer=coupled_layer)
        }
        moire_potential_conduction_band: dict[str, npt.NDArray[np.complex128]] = {
            reference_layer: self.get_moire_potential_coefficients(band="cb", layer=reference_layer),
            coupled_layer: self.get_moire_potential_coefficients(band="cb", layer=coupled_layer)
        }
        # Determine the indexes of the q vectors for which the Moiré potential is not zero. We expect 6 values
        moire_idx_valence_band = np.where(np.abs(moire_potential_valence_band[reference_layer]) > 0)[0]
        moire_idx_conduction_band = np.where(np.abs(moire_potential_conduction_band[coupled_layer]) > 0)[0]
        print(self.hamiltonian.grid.flat_grid()[moire_idx_valence_band])
        assert len(moire_idx_valence_band) == 6 and len(moire_idx_conduction_band) == 6

        # 3 of these value yield diagonals in the lower triangular part of the matrix, the other 3 in the upper
        # triangular. Therefore, we can keep only three values (as the Hamiltonian is hermitian). Depending on the
        # symmetry of the lattice, these points will either be on a hexagon lying horizontally or vertically.
        n_points: int = 3 if self.moire_symmetry == 'C6' else 1
        # 1. Hexagon lying horizontally, take points with positive x
        moire_idx_reduced: dict[str, npt.NDArray[np.complex128]] = {
            reference_layer: np.array(moire_idx_valence_band[flat_k_grid[moire_idx_valence_band, 0] > 0], dtype=int),
            coupled_layer: np.array(moire_idx_conduction_band[flat_k_grid[moire_idx_valence_band, 0] > 0], dtype=int)
        }
        if len(moire_idx_reduced[reference_layer]) != n_points or len(moire_idx_reduced[coupled_layer]) != n_points:
            # 2. Hexagon lying vertically, take points with positive y
            moire_idx_reduced: dict[str, npt.NDArray[np.complex128]] = {
                reference_layer: np.array(moire_idx_valence_band[flat_k_grid[moire_idx_valence_band, 1] > 0],dtype=int),
                coupled_layer: np.array(moire_idx_conduction_band[flat_k_grid[moire_idx_valence_band, 1] > 0],dtype=int)
            }
        assert len(moire_idx_reduced[reference_layer]) == n_points or len(moire_idx_reduced[coupled_layer]) == n_points

        for i in range(number_of_excitonic_states):
            q_moire = moire_idx_reduced[reference_layer]
            # Get the form factor
            form_factor, form_factor_conj = self.get_form_factor(
                    self.wannier_eigenvectors[self.exciton][:, i], q_indexes=q_moire)
            # Compute the Moiré Hamiltonian coefficients
            moire_coefficients = (np.multiply(moire_potential_conduction_band[reference_layer][q_moire], form_factor[i])
                - np.multiply(moire_potential_valence_band[coupled_layer][q_moire], np.conj(form_factor_conj[i])))
            for j, q_i in enumerate(q_moire):
                # The shape of the Moiré potential is: 6 diagonals (parallel to the main diagonal) that are non-0.
                # All the rest is 0.
                q = flat_k_grid[q_i, :]
                for k in range(kgr_size):
                    # Define the Q + q grid and get the index of the corresponding element.
                    # Note that this will yield either one or zero elements (if the potential is zero in that point)
                    Q_plus_q = flat_k_grid[k, :] + q
                    idx = self.hamiltonian.grid.get_point_index(Q_plus_q)
                    if idx == -1:
                        continue
                    # Fill diag in the lower and upper triangular part
                    i_h: int = i * kgr_size + k
                    j_h: int = i * kgr_size + idx
                    if i_h > j_h:  # Element is in lower triangular part
                        moire_hamiltonian[i_h, j_h] = np.conj(moire_coefficients[j])
                        moire_hamiltonian[j_h, i_h] = moire_coefficients[j]
                    else:  # Element is in upper triangular part
                        moire_hamiltonian[i_h, j_h] = moire_coefficients[j]
                        moire_hamiltonian[j_h, i_h] = np.conj(moire_coefficients[j])
        return moire_hamiltonian
