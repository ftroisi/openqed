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

"""
This script shows how to couple Moiré excitons to light using OpenQED.

References:
    [1] Cavity-QED-controlled two-dimensional Moiré excitons without twisting
            https://www.nature.com/articles/s41467-025-67570-2
"""

from ase.units import Bohr
import numpy as np

from openqed.grid import KSpaceGrid
from openqed.unit_cells import HexagonalUnitCell

from openqed.hamiltonians import MottWannierHamiltonian, MoireExcitonHamiltonian, Hamiltonian
from openqed.hamiltonians.light_matter.pauli_fierz_hamiltonian import PauliFierzHamiltonian
from openqed.hamiltonians.light_matter.light_matter_hamiltonian import LightMatterHamiltonianTerms
from openqed.hamiltonians.moire_exciton_hamiltonian import MoireExcitonHamiltonianTerms, MoireExcitonsMomentumMatrixElements
from openqed.hamiltonians.mott_wannier_hamiltonian import MottWannierHamiltonianTerms
from openqed.hamiltonians.terms.moire_potential_term import MoirePotentialParameters

from openqed.hamiltonians.types import BilayerExcitons, BilayerStructure, CavityParameters, ModePolarization

HA_TO_EV = 27.2114079527
spacing = np.float64(0.0015)
grid_boundaries = np.float64(0.03)
grid = KSpaceGrid(
        spacing={'x': spacing, 'y': spacing},
        boundaries={'x': (-grid_boundaries, grid_boundaries), 'y': (-grid_boundaries, grid_boundaries)})
unit_cell_mo_se2 = HexagonalUnitCell(lattice_parameter=np.float64(3.32 / Bohr))
unit_cell_w_se2 = HexagonalUnitCell(lattice_parameter=np.float64(3.319 / Bohr))
grid.project_to_unit_cell(unit_cell_mo_se2, cache_result=True)

electron_effective_mass = BilayerStructure(
        layer1=np.float64(0.5), # Effective mass of the electrons in the first layer
        layer2=np.float64(0.29)  # Effective mass of the electrons in the second layer
)
hole_effective_mass = BilayerStructure(
        layer1=np.float64(0.6), # Effective mass of the holes in the first layer
        layer2=np.float64(0.36)  # Effective mass of the holes in the second layer
)
# Dielectric constant value for substrate: https://pubs.acs.org/doi/epdf/10.1021/acs.nanolett.5b01251
dielectric_constants = BilayerStructure(
        substrate1=np.float64(4.), # Dielectric constant of the substrate
        layer1=np.float64(16.5), # Dielectric constant of the first layer
        interlayer=np.float64(1.), # Dielectric constant of the interlayer region
        layer2=np.float64(15.1), # Dielectric constant of the second layer
        substrate2=np.float64(4.) # Dielectric constant of the substrate
)
thicknesses = BilayerStructure(
        substrate1=np.float64(1.), # Thickness of the substrate
        layer1=np.float64(6.2926 / Bohr), # Thickness of the first layer
        interlayer=np.float64(6.22705 / Bohr), # Thickness of the interlayer region
        layer2=np.float64(6.1615 / Bohr), # Thickness of the second layer
        substrate2=np.float64(1.) # Thickness of the substrate
)

mott_wannier = MottWannierHamiltonian(
        terms=[MottWannierHamiltonianTerms.KINETIC.value, MottWannierHamiltonianTerms.EFFECTIVE_2D_COULOMB.value],
        exciton="l1_l1", # Intralayer exciton of the first layer
        grid=grid,
        electron_effective_mass=electron_effective_mass,
        hole_effective_mass=hole_effective_mass,
        thicknesses=thicknesses,
        dielectric_constants=dielectric_constants
)
# Get and diagonalize the Hamiltonian
mott_wannier_hamiltonian = mott_wannier.get_hamiltonian()
wannier_binding_energies, wannier_states = Hamiltonian.diagonalize_hamiltonian(mott_wannier_hamiltonian)
print(wannier_binding_energies[0:5] * HA_TO_EV)
wannier_states = BilayerExcitons(
    l1_l1 = wannier_states[:, 0].reshape((np.size(grid.flat_grid(), 0), 1)) # 1s Intralayer exciton of the first layer
)
wannier_binding_energies = BilayerExcitons(
    l1_l1 = wannier_binding_energies[0], # 1s Intralayer exciton of the first layer
)

# Energy gap of the VB/CB of the layers: https://arxiv.org/pdf/1802.05310
band_gaps = BilayerExcitons(
    l1_l1 = np.float64(1.837 / HA_TO_EV),
    l1_l2 = np.float64(1.44 / HA_TO_EV),
    l2_l2 = np.float64(1.867 / HA_TO_EV)
)

# Moire potential parameters
moire_params = MoirePotentialParameters(
    layers_lattice_vectors = dict(
        layer1=unit_cell_mo_se2,
        layer2=unit_cell_w_se2
    ),
    twist_angle = np.float64(2.0),
    moire_lattice_vectors = None,
    moire_prefactor_alpha_valence_band=BilayerStructure(
        layer1=np.float64(-20 / (HA_TO_EV * 1000)),
        layer2=np.float64(-16.1 / (HA_TO_EV * 1000))
    ),
    moire_prefactor_alpha_conduction_band=BilayerStructure(
        layer1=np.float64(-14.7 / (HA_TO_EV * 1000)),
        layer2=np.float64(-11.1 / (HA_TO_EV * 1000))
    ),
    moire_prefactor_beta_valence_band=BilayerStructure(
        layer1=np.float64(-7.3 / (HA_TO_EV * 1000)),
        layer2=np.float64(-8 / (HA_TO_EV * 1000))
    ),
    moire_prefactor_beta_conduction_band=BilayerStructure(
        layer1=np.float64(-5.2 / (HA_TO_EV * 1000)),
        layer2=np.float64(-5.4 / (HA_TO_EV * 1000))
    ),
    moire_symmetry="C6"
)

# Matrix elements
wse2_matrix_element_x = np.complex128(-1.53180762E-01 + 1j*2.42061800E-01)
wse2_matrix_element_y = np.complex128(-2.42061809E-01 - 1j*1.53180757E-01)
mose2_matrix_element_x = np.complex128(8.34722539E-03 + 1j*2.26891415E-01)
mose2_matrix_element_y = np.complex128(-2.26891410E-01 + 1j*8.34721640E-03)
matrix_elements = MoireExcitonsMomentumMatrixElements(
    x = BilayerExcitons(
        l1_l1 = mose2_matrix_element_x,
        l1_l2 = 0.5*(mose2_matrix_element_x + wse2_matrix_element_x),
        l2_l2 = wse2_matrix_element_x
    ),
    y = BilayerExcitons(
        l1_l1 = mose2_matrix_element_y,
        l1_l2 = 0.5*(mose2_matrix_element_y + wse2_matrix_element_y),
        l2_l2 = wse2_matrix_element_y
    ),
    z = None
)

moire_excitons = MoireExcitonHamiltonian(
    terms=[
        MoireExcitonHamiltonianTerms.FREE_EXCITON.value,
        MoireExcitonHamiltonianTerms.MOIRE_POTENTIAL.value
    ],
    grid=grid,
    wannier_states=wannier_states,
    wannier_binding_energies=wannier_binding_energies,
    band_gap=band_gaps,
    moire_matrix_elements=matrix_elements,
    moire_parameters=moire_params,
    electron_effective_mass=electron_effective_mass,
    hole_effective_mass=hole_effective_mass,
)
print(f"h_mat size: {moire_excitons.get_hamiltonian_size()}")

# Now create light matter coupling
cavity_parameters = CavityParameters(
    number_of_effective_modes=2,
    maximum_mode_occupation=2,
    modes_energies=np.array([0.01, 0.02], dtype=np.float64),
    modes_couplings=np.array([0.01, 0.01], dtype=np.float64),
    modes_parallel_momenta=np.array([[0., 0.],[0., 0.]], dtype=np.float64),
    modes_polarization=ModePolarization.LINEAR_X
)
lm = PauliFierzHamiltonian(
        terms=[
            LightMatterHamiltonianTerms.UNCOUPLED_MATTER.value,
            LightMatterHamiltonianTerms.UNCOUPLED_LIGHT.value
        ],
        cavity_parameters=cavity_parameters,
        matter_hamiltonian=moire_excitons
)
lm.get_hamiltonian()

