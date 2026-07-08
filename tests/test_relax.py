# This source code is part of the Hydride package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import itertools
from os.path import join
import biotite.structure as struc
import biotite.structure.info as info
import biotite.structure.io.pdbx as pdbx
import numpy as np
import pytest
import hydride
from hydride.relax import _find_rotatable_bonds
from tests.util import data_dir, place_over_periodic_boundary


@pytest.fixture
def ethane():
    # Construct ethane in staggered conformation
    ethane = struc.AtomArray(8)
    ethane.element = np.array(["C", "C", "H", "H", "H", "H", "H", "H"])
    ethane.coord = np.array(
        [
            [-0.756, 0.000, 0.000],
            [0.756, 0.000, 0.000],
            [-1.140, 0.659, 0.7845],
            [-1.140, 0.350, -0.9626],
            [-1.140, -1.009, 0.1781],
            [1.140, -0.350, 0.9626],
            [1.140, 1.009, -0.1781],
            [1.140, -0.659, -0.7845],
        ]
    )
    ethane.bonds = struc.BondList(
        8,
        np.array(
            [
                [0, 1, 1],
                [0, 2, 1],
                [0, 3, 1],
                [0, 4, 1],
                [1, 5, 1],
                [1, 6, 1],
                [1, 7, 1],
            ]
        ),
    )
    ethane.set_annotation("charge", np.zeros(ethane.array_length(), dtype=int))

    # Check if created ethane is in optimal staggered conformation
    # -> Dihedral angle of 60 degrees
    dihed = struc.dihedral(ethane[2], ethane[0], ethane[1], ethane[5])
    assert np.rad2deg(dihed) % 120 == pytest.approx(60, abs=1)

    return ethane


@pytest.mark.parametrize(
    "seed, periodic_dim", itertools.product(range(10), [None, 0, 1, 2])
)
def test_staggered(ethane, seed, periodic_dim):
    """
    :func:`relax_hydrogen()` should be able to restore a staggered
    conformation of ethane from any other conformation.
    """
    BOX_SIZE = 100

    # Move the ethane molecule away
    # from the optimal staggered conformation
    np.random.seed(seed)
    angle = np.random.rand() * 2 * np.pi
    ethane.coord[5:] = struc.rotate_about_axis(
        ethane.coord[5:],
        angle=angle,
        axis=ethane.coord[1] - ethane.coord[0],
        support=ethane.coord[0],
    )

    # Check if new conformation ethane is not staggered anymore
    dihed = struc.dihedral(ethane[2], ethane[0], ethane[1], ethane[5])
    assert np.rad2deg(dihed) % 120 != pytest.approx(60, abs=1)

    if periodic_dim is None:
        box = None
    else:
        box = np.identity(3) * BOX_SIZE
        # Move molecule to the border of the box
        # to enforce interatomic interactions
        # using minimum image convention
        ethane = place_over_periodic_boundary(ethane, periodic_dim, BOX_SIZE)

    # Try to restore staggered conformation via relax_hydrogen()
    ethane.coord = hydride.relax_hydrogen(
        ethane,
        # The angle increment must be smaller
        # than the expected accuracy (abs=1)
        angle_increment=np.deg2rad(0.5),
        box=box,
    )

    if periodic_dim is not None:
        # Remove PBC again
        ethane.coord = struc.remove_pbc_from_coord(ethane.coord, box)

    # Check if staggered conformation is restored
    dihed = struc.dihedral(ethane[2], ethane[0], ethane[1], ethane[5])
    assert np.rad2deg(dihed) % 120 == pytest.approx(60, abs=1)


@pytest.mark.parametrize("periodic_dim", [None, 0, 1, 2])
def test_hydrogen_bonds(periodic_dim):
    """
    Check whether the relaxation algorithm is able to restore most of
    the original hydrogen bonds.
    The number of bonds found without relaxation is handled as baseline.
    The residues at the biotin binding pocket of streptavidin (including
    biotin itself) are used as test case.
    """
    # The percentage of recovered hydrogen bonds
    PERCENTAGE = 1.0
    # The relevant residues of the streptavidin binding pocket
    RES_IDS = (27, 43, 45, 47, 90, 300)
    # The size of the box if PBCs are enabled
    BOX_SIZE = 1000

    pdbx_file = pdbx.BinaryCIFFile.read(join(data_dir(), "2rtg.bcif"))
    atoms = pdbx.get_structure(
        pdbx_file, model=1, include_bonds=True, extra_fields=["charge"]
    )
    atoms = atoms[atoms.chain_id == "B"]
    mask = np.isin(atoms.res_id, RES_IDS)
    ref_num = len(struc.hbond(atoms, mask, mask))

    atoms = atoms[atoms.element != "H"]
    atoms, _ = hydride.add_hydrogen(atoms)
    mask = np.isin(atoms.res_id, RES_IDS)
    base_num = len(struc.hbond(atoms, mask, mask))

    if periodic_dim is None:
        box = None
    else:
        box = np.identity(3) * BOX_SIZE
        # Move molecule to the border of the box
        # to enforce interatomic interactions
        # using minimum image convention
        atoms = place_over_periodic_boundary(atoms, periodic_dim, BOX_SIZE)

    atoms.coord = hydride.relax_hydrogen(atoms, box=box)

    if periodic_dim is not None:
        # Remove PBC again
        atoms.coord = struc.remove_pbc_from_coord(atoms.coord, box)

    test_num = len(struc.hbond(atoms, mask, mask))

    if base_num == ref_num:
        ValueError(
            "Invalid test case, "
            "no further hydrogen bonds can be found via relaxation"
        )
    assert (test_num - base_num) / (ref_num - base_num) >= PERCENTAGE


@pytest.mark.parametrize(
    "res_name, ref_bonds",
    [
        # Fructopyranose
        (
            "FRU",
            [
                ("O1", "C1", True, ("HO1",)),
                ("O2", "C2", True, ("HO2",)),
                ("O3", "C3", True, ("HO3",)),
                ("O4", "C4", True, ("HO4",)),
                ("O6", "C6", True, ("HO6",)),
            ],
        ),
        # Arginine with positive side chain
        (
            "ARG",
            [
                ("N", "CA", True, ("H", "H2")),
                ("OXT", "C", True, ("HXT",)),
            ],
        ),
        # Isoleucine
        (
            "ILE",
            [
                ("N", "CA", True, ("H", "H2")),
                ("OXT", "C", True, ("HXT",)),
                ("CG2", "CB", True, ("HG21", "HG22", "HG23")),
                ("CD1", "CG1", True, ("HD11", "HD12", "HD13")),
            ],
        ),
        # 1-phenylguanidine
        (
            "PL0",
            [
                ("N3", "C7", False, ("HN3",)),
            ],
        ),
        # Water
        ("HOH", []),
    ],
)
def test_bond_identification(res_name, ref_bonds):
    """
    Test whether rotatable bonds for the relaxation are correctly
    identified based on known molecules.
    """
    molecule = info.residue(res_name)
    rotatable_bonds = _find_rotatable_bonds(molecule)

    ref_bonds = set(ref_bonds)

    assert len(rotatable_bonds) == len(ref_bonds)
    for center_atom_i, bonded_atom_i, is_free, h_indices in rotatable_bonds:
        bond_tuple = (
            molecule.atom_name[center_atom_i],
            molecule.atom_name[bonded_atom_i],
            is_free,
            tuple(np.sort(molecule.atom_name[h_indices])),
        )
        assert bond_tuple in ref_bonds


def test_return_trajectory(atoms):
    """
    Test whether the `return_trajectory` parameter works properly.
    It is expected that :func:`relax_hydrogen()` returns multiple
    models.
    """

    traj_coord = hydride.relax_hydrogen(atoms, return_trajectory=True)

    assert traj_coord.ndim == 3
    # Last model in trajectory should be the same result
    # as running 'relax_hydrogen()' without 'return_trajectory=True'
    assert np.array_equal(traj_coord[-1], hydride.relax_hydrogen(atoms))


def test_return_energies(atoms):
    """
    Test whether the `return_energies` parameter works properly.
    It is expected that :func:`relax_hydrogen()` returns an array of
    energies.
    """

    _, energies = hydride.relax_hydrogen(atoms, return_energies=True)
    assert isinstance(energies, np.ndarray)
    # Energies should monotonically decrease
    assert (np.diff(energies) <= 0).all()

    traj_coord, energies = hydride.relax_hydrogen(
        atoms, return_energies=True, return_trajectory=True
    )
    assert len(traj_coord) == len(energies)

    assert traj_coord.ndim == 3
    # Last model in trajectory should be the same result
    # as running 'relax_hydrogen()' without 'return_trajectory=True'
    assert np.array_equal(traj_coord[-1], hydride.relax_hydrogen(atoms))


@pytest.mark.parametrize("repulsive", [False, True])
def test_partial_charges(ethane, repulsive):
    """
    Test whether the `partial_charges` parameter is properly used, by
    giving one hydrogen atom on each carbon atom of ethane an
    unphysical charge, either attractive or repulsive to each other.
    This should give result conformations that strongly deviate from
    the staggered conformation, since the electrostatic term should
    minimize or maximize the distance between these hydrogen atoms,
    respectively.
    """
    if repulsive:
        charges = np.array([0, 0, 1, 0, 0, 1, 0, 0])
    else:
        charges = np.array([0, 0, -1, 0, 0, 1, 0, 0])

    ethane.coord = hydride.relax_hydrogen(
        ethane,
        # The angle increment must be smaller
        # than the expected accuracy (abs=1)
        angle_increment=np.deg2rad(0.5),
        partial_charges=charges,
    )

    # Check if staggered conformation is restored
    dihed = struc.dihedral(ethane[2], ethane[0], ethane[1], ethane[5])
    if repulsive:
        exp_angle = 180
    else:
        exp_angle = 0
    deg = np.rad2deg(dihed)
    # shortest path to circle (eg 359.99 and 179.99 should work)
    diff = (deg - exp_angle + 180) % 360 - 180
    assert diff == pytest.approx(0, abs=1)


def test_limited_iterations(atoms):
    """
    Test whether the `iterations` parameter works properly.
    It is expected that the number of returned models,
    if `return_trajectory is set to true, is equal to the given number
    of maximum iterations.
    That is only true, if the number of iterations is low enough,
    so that the relaxation does not terminate before.
    """
    ITERATIONS = 4

    traj_coord = hydride.relax_hydrogen(atoms, ITERATIONS, return_trajectory=True)

    assert traj_coord.shape[0] == ITERATIONS


@pytest.mark.parametrize(
    "iterations, return_trajectory, return_energies",
    itertools.product([None, 100], [False, True], [False, True]),
)
def test_shortcut_return(iterations, return_trajectory, return_energies):
    """
    Test whether the shortcut return, that happens if no rotatable bonds
    are found, has the same return types as the regular return.
    Therefore the output types of two molecules, one with and one
    without rotatable bonds, are compared.
    """
    # Rotatable
    ref_atoms = info.residue("GLY")
    # Non-rotatable
    test_atoms = info.residue("HOH")

    ref_output = hydride.relax_hydrogen(
        ref_atoms,
        iterations,
        return_trajectory=return_trajectory,
        return_energies=return_energies,
    )
    test_output = hydride.relax_hydrogen(
        test_atoms,
        iterations,
        return_trajectory=return_trajectory,
        return_energies=return_energies,
    )

    if isinstance(ref_output, tuple):
        assert isinstance(test_output, tuple)
        assert len(test_output) == len(ref_output)
        for i in range(len(ref_output)):
            assert isinstance(test_output[i], type(ref_output[i]))
    else:
        assert isinstance(test_output, type(ref_output))


def test_atom_mask(atoms):
    """
    Test atom mask usage by relaxing only part of the model and
    expect that no unmasked hydrogen positions changed.
    """
    MASKED_RES_IDS = np.arange(1, 11)

    ref_coord = atoms.coord.copy()

    mask = np.isin(atoms.res_id, MASKED_RES_IDS)
    test_coord = hydride.relax_hydrogen(atoms, mask=mask)

    assert (test_coord[~mask] == ref_coord[~mask]).all()
    assert not (test_coord[mask] == ref_coord[mask]).all()

def test_relax_hydrogen_differentiability(ethane):
    """
    Test that the unrolled hydrogen relaxation loop can be differentiated
    with respect to the initial heavy atom coordinates, producing non-vanishing
    gradients due to the dynamic center and axis kinematics mapping.
    """
    import jax
    import jax.numpy as jnp

    # Unpack the relaxation parameters from the ethane fixture
    params = hydride.get_relaxation_params(ethane)
    assert params is not None

    # Perturb a heavy atom (Carbon at index 0) to push the system out of
    # equilibrium and guarantee non-zero geometric gradients
    perturbed_coord = ethane.coord.copy()
    perturbed_coord[0] += np.array([0.1, -0.1, 0.05], dtype=np.float32)
    init_coord = jnp.array(perturbed_coord)

    # Define a scalar loss function with respect to the starting coordinates
    def scalar_loss(coords):
        final_coord, _, energies = hydride.relax_hydrogen_jit(
            coords,
            *params,
            iterations=5,  # Low iteration count suffices to verify gradient flow
        )
        # Use the final optimized energy as the scalar target
        return energies[-1]

    # Differentiate the loop with respect to the initial coordinates
    grads = jax.grad(scalar_loss)(init_coord)

    # Assertions to ensure gradients flow correctly and cleanly
    assert jnp.isfinite(grads).all(), "Gradients contain NaN or Inf values."
    assert jnp.any(grads != 0.0), "Gradients completely vanished to zero."

    # Verify explicitly that heavy atoms (indices 0 and 1) receive gradients
    # This confirms that moving a heavy atom correctly alters the hydrogen potential energy
    heavy_atom_grads = grads[:2]
    assert jnp.any(
        heavy_atom_grads != 0.0
    ), "Heavy atom gradients vanished; kinematic chain rule broken."

def test_hydrogen_kinematics_corotation(atoms):
    """
    Test that the implicit hydrogen placement (NeRF kinematics) perfectly
    co-rotates and co-translates hydrogens when the parent heavy atoms undergo
    a rigid body SE(3) transformation[cite: 7].
    """
    import jax.numpy as jnp

    # Extract the relaxation parameters containing the NeRF topology[cite: 7]
    params = hydride.get_relaxation_params(atoms)
    assert params is not None

    # Unpack the specific NeRF kinematic arrays appended to the base parameters
    p1_idx = params[14]
    p2_idx = params[15]
    p3_idx = params[16]
    ref_v = params[17]
    use_ref_v = params[18]
    lengths = params[19]
    angles = params[20]
    torsions = params[21]
    h_mask = params[22]

    X_initial = jnp.array(atoms.coord)

    # 1. Place hydrogens in the original baseline frame
    X_H_baseline = hydride.relax.place_hydrogens_jit(
        X_initial, p1_idx, p2_idx, p3_idx, ref_v, use_ref_v,
        lengths, angles, torsions, h_mask
    )

    # 2. Generate a random rigid body SE(3) transformation
    np.random.seed(42)
    translation = np.random.uniform(-10.0, 10.0, size=3).astype(np.float32)
    angle_x, angle_y, angle_z = np.random.uniform(-np.pi, np.pi, size=3)

    # 3. Transform heavy atoms
    transformed_atoms = atoms.copy()
    transformed_atoms.coord = struc.rotate(transformed_atoms.coord, (angle_x, angle_y, angle_z))
    transformed_atoms.coord += translation
    X_heavy_transformed = jnp.array(transformed_atoms.coord)

    # 4. Place hydrogens using the rotated and translated heavy atoms
    X_H_transformed = hydride.relax.place_hydrogens_jit(
        X_heavy_transformed, p1_idx, p2_idx, p3_idx, ref_v, use_ref_v,
        lengths, angles, torsions, h_mask
    )

    # 5. Explicitly apply the identical transformation to the baseline coordinates
    dummy_array = atoms.copy()
    dummy_array.coord = np.array(X_H_baseline)
    dummy_array.coord = struc.rotate(dummy_array.coord, (angle_x, angle_y, angle_z))
    dummy_array.coord += translation
    X_H_expected = dummy_array.coord

    # 6. Assert that implicitly placed hydrogens perfectly track the heavy atom frames
    # We exclusively check hydrogens governed by internal p3 frames (use_ref_v == False)
    # as global fallback reference vectors for 2-atom systems do not co-rotate by definition.
    mask_to_check = np.array(h_mask) & ~np.array(use_ref_v)

    placed_hydrogens = np.array(X_H_transformed)[mask_to_check]
    expected_hydrogens = X_H_expected[mask_to_check]

    np.testing.assert_allclose(placed_hydrogens, expected_hydrogens, atol=1e-4)

def test_nerf_topology_adversarial():
    """
    Adversarial test: Construct a molecule where the closest spatial atom
    is NOT the covalent parent. The implicit hydrogen must resolve its
    local frame using the covalent bond, not distance.
    """
    # Create a system: C1 - C2, with H attached to C1.
    # Place a non-bonded C3 very close to H to tempt a distance-based frame selector.
    atoms = struc.AtomArray(4)
    atoms.element = np.array(["C", "C", "C", "H"])
    # C1 at origin, C2 at 1.5A, C3 at 1.0A (but not bonded to H), H bonded to C1 at 1.1A
    atoms.coord = np.array([
        [0.0, 0.0, 0.0],    # C1
        [1.5, 0.0, 0.0],    # C2
        [0.0, 1.0, 0.0],    # C3 (Close to H, but not bonded)
        [0.0, 0.0, 1.1]     # H (Bonded to C1)
    ])
    atoms.bonds = struc.BondList(4, np.array([[0, 1, 1], [0, 3, 1]]))

    # Trigger the NeRF parameter calculation
    atom_to_bond_idx = np.array([-1, -1, -1, -1], dtype=np.int32)
    center_indices = np.zeros(0, dtype=np.int32)
    axis_indices = np.zeros(0, dtype=np.int32)

    # This call uses the internal graph-traversal logic
    p1, p2, p3, ref_v, use_ref_v, lengths, angles, torsions, h_mask = hydride.relax._get_nerf_params(
        atoms, atom_to_bond_idx, center_indices, axis_indices
    )

    # ADVERSARIAL CHECK:
    # If the logic incorrectly picked C3 (closest distance) as p2,
    # it would fail this check. It must pick C2 (covalent neighbor).
    hydrogen_idx = 3
    c1_idx = 0
    c2_idx = 1

    assert p1[hydrogen_idx] == c1_idx, "NeRF frame failed to identify covalent parent P1"
    assert p2[hydrogen_idx] == c2_idx, "NeRF frame incorrectly prioritized distance over covalent connectivity"

def test_nerf_topology_independence_from_spatial_proximity():
    """
    Adversarial test: A topologically rigid NeRF frame must be immune to the 
    spatial translation of nearby, non-bonded atoms. 
    Distance-based heuristics will fail this by latching onto the nearby atom.
    """
    import numpy as np
    import jax.numpy as jnp
    import biotite.structure as struc
    import hydride
    
    # Construct: H(4) - C1(0) - C2(1) - C3(2) 
    # O4(3) is an unbonded oxygen placed very close to C1 to act as a spatial trap.
    atoms = struc.AtomArray(5)
    atoms.element = np.array(["C", "C", "C", "O", "H"])
    atoms.coord = np.array([
        [0.0, 0.0, 0.0],    # 0: C1
        [1.5, 0.0, 0.0],    # 1: C2
        [2.0, 1.5, 0.0],    # 2: C3
        [0.0, 1.2, 0.0],    # 3: O4 (Trap: closer to C1 than C2/C3)
        [0.0, 0.0, 1.0]     # 4: H
    ])
    
    # Define the covalent graph, explicitly omitting O4
    atoms.bonds = struc.BondList(5, np.array([
        [0, 1, 1], # C1 - C2
        [1, 2, 1], # C2 - C3
        [0, 4, 1]  # C1 - H
    ]))
    
    # Extract parameters using the current implementation
    params = hydride.get_relaxation_params(atoms)
    
    p1_idx, p2_idx, p3_idx = params[14], params[15], params[16]
    ref_v, use_ref_v = params[17], params[18]
    lengths, angles, torsions, h_mask = params[19], params[20], params[21], params[22]
    
    # Baseline placement
    X_heavy_base = jnp.array(atoms.coord)
    H_base = hydride.relax.place_hydrogens_jit(
        X_heavy_base, p1_idx, p2_idx, p3_idx, ref_v, use_ref_v, lengths, angles, torsions, h_mask
    )[4] # Get the H atom coords
    
    # Move the trap atom (O4) far away
    atoms_moved = atoms.copy()
    atoms_moved.coord[3] += np.array([10.0, 10.0, 10.0])
    
    # Re-place the hydrogen using the NEW heavy atom coordinates but the SAME NeRF parameters
    X_heavy_moved = jnp.array(atoms_moved.coord)
    H_moved = hydride.relax.place_hydrogens_jit(
        X_heavy_moved, p1_idx, p2_idx, p3_idx, ref_v, use_ref_v, lengths, angles, torsions, h_mask
    )[4]
    
    # Explicitly verify the topological graph traversal worked
    assert p2_idx[4] == 1, "P2 should be the covalently bonded C2, not the closer O4."
    assert p3_idx[4] == 2, "P3 should be the next bonded C3, not the closer O4."

    # If the NeRF frame was topologically sound, it ignored O4 entirely.
    # Thus, the H position should be completely unchanged relative to the C1-C2-C3 frame.
    np.testing.assert_allclose(
        np.array(H_moved), 
        np.array(H_base), 
        atol=1e-5, 
        err_msg="Hydrogen frame was corrupted by non-bonded spatial neighbors!"
    )

def test_nerf_topology_local_bisector_priority():
    """
    Adversarial test: Ensure that the NeRF frame prioritizes local branches
    (other heavy atoms bonded to P1) over looking backward down the chain
    (heavy atoms bonded to P2). This ensures methylene hydrogens correctly
    bisect the angle between their two immediate heavy neighbors.
    """
    import numpy as np
    import biotite.structure as struc
    import hydride

    # Construct a chain: C0 - C1 - C2 - C3
    # Attach H4 to C2.
    # C2 is P1. C1 is P2 (first neighbor of C2).
    # P3 should be C3 (local neighbor of C2), not C0 (backward neighbor of C1).
    atoms = struc.AtomArray(5)
    atoms.element = np.array(["C", "C", "C", "C", "H"])

    # Coordinates must not be perfectly collinear to pass the cross product check.
    atoms.coord = np.array([
        [0.0, 0.0, 0.0],    # 0: C0
        [1.5, 0.0, 0.0],    # 1: C1
        [2.0, 1.0, 0.0],    # 2: C2
        [3.5, 1.0, 0.0],    # 3: C3
        [2.0, 1.0, 1.0]     # 4: H4 (Bonded to C2)
    ], dtype=np.float32)

    atoms.bonds = struc.BondList(5, np.array([
        [0, 1, 1], # C0 - C1
        [1, 2, 1], # C1 - C2
        [2, 3, 1], # C2 - C3
        [2, 4, 1]  # C2 - H4
    ]))

    # Setup dummy rotatable parameters so H4 is treated as locked/fixed for the frame check
    atom_to_bond_idx = np.array([-1, -1, -1, -1, -1], dtype=np.int32)
    center_indices = np.zeros(0, dtype=np.int32)
    axis_indices = np.zeros(0, dtype=np.int32)

    p1, p2, p3, ref_v, use_ref_v, lengths, angles, torsions, h_mask = hydride.relax._get_nerf_params(
        atoms, atom_to_bond_idx, center_indices, axis_indices
    )

    h_idx = 4
    c0_idx = 0
    c1_idx = 1
    c2_idx = 2
    c3_idx = 3

    assert p1[h_idx] == c2_idx, "P1 should be C2 (the covalently bonded parent)."
    assert p2[h_idx] == c1_idx, "P2 should be C1 (the first heavy neighbor of C2)."

    # ADVERSARIAL CHECK:
    # The old code (backwards priority) would assign P3 to C0 (neighbor of P2).
    # The new code (local branch priority) must assign P3 to C3 (neighbor of P1).
    assert p3[h_idx] == c3_idx, (
        f"P3 was incorrectly assigned to {p3[h_idx]}. It must be {c3_idx} (local bisector neighbor), "
        f"not {c0_idx} (backward chain neighbor)."
    )
