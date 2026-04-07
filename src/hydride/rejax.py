# This source code is part of the Hydride package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "hydride"
__author__ = "Patrick Kunzmann, Jacob Marcel Anter, Differentiable JAX Port"
__all__ = ["relax_hydrogen"]

import warnings
import numpy as np
import biotite.structure as struc
import jax
import jax.numpy as jnp

# Values are taken from Rappé et al. (UFF)
NB_VALUES = {
    "H" : (2.886, 0.044),
    "HE": (2.362, 0.056),
    "LI": (2.451, 0.025),
    "BE": (2.745, 0.085),
    "B" : (4.083, 0.180),
    "C" : (3.851, 0.105),
    "N" : (3.660, 0.069),
    "O" : (3.500, 0.060),
    "F" : (3.364, 0.050),
    "NE": (3.243, 0.042),
    "NA": (2.983, 0.030),
    "MG": (3.021, 0.111),
    "AL": (4.499, 0.505),
    "SI": (4.295, 0.402),
    "P" : (4.147, 0.305),
    "S" : (4.035, 0.274),
    "CL": (3.947, 0.227),
    "AR": (3.868, 0.185),
    "K" : (3.812, 0.035),
    "CA": (3.399, 0.238),
    "SC": (3.295, 0.019),
    "TI": (3.175, 0.017),
    "V" : (3.144, 0.016),
    "CR": (3.023, 0.015),
    "MN": (2.961, 0.013),
    "FE": (2.912, 0.013),
    "CO": (2.872, 0.014),
    "NI": (2.834, 0.015),
    "CU": (3.495, 0.005),
    "ZN": (2.763, 0.124),
    "GA": (4.383, 0.415),
    "GE": (4.280, 0.379),
    "AS": (4.230, 0.309),
    "SE": (4.205, 0.291),
    "BR": (4.189, 0.251),
    "KR": (4.141, 0.220),
    "RB": (4.114, 0.040),
    "SR": (3.641, 0.235),
    "Y" : (3.345, 0.072),
    "ZR": (3.124, 0.069),
    "NB": (3.165, 0.059),
    "MO": (3.052, 0.056),
    "TC": (2.998, 0.048),
    "RU": (2.963, 0.056),
    "RH": (2.929, 0.053),
    "PD": (2.899, 0.048),
    "AG": (3.148, 0.036),
    "CD": (2.848, 0.228),
    "IN": (4.463, 0.599),
    "SN": (4.392, 0.567),
    "SB": (4.420, 0.449),
    "TE": (4.470, 0.398),
    "I" : (4.500, 0.339),
    "XE": (4.404, 0.332),
    "CS": (4.517, 0.045),
    "BA": (3.703, 0.364),
    "LA": (3.522, 0.017),
    "CE": (3.556, 0.013),
    "PR": (3.606, 0.010),
    "ND": (3.575, 0.010),
    "PM": (3.547, 0.009),
    "SM": (3.520, 0.008),
    "EU": (3.493, 0.008),
    "GD": (3.368, 0.009),
    "TB": (3.451, 0.007),
    "DY": (3.428, 0.007),
    "HO": (3.409, 0.007),
    "ER": (3.391, 0.007),
    "TM": (3.374, 0.006),
    "YB": (3.355, 0.228),
    "LU": (3.640, 0.041),
    "HF": (3.141, 0.072),
    "TA": (3.170, 0.081),
    "W" : (3.069, 0.067),
    "RE": (2.954, 0.066),
    "OS": (3.120, 0.037),
    "IR": (2.840, 0.073),
    "PT": (2.754, 0.080),
    "AU": (3.293, 0.039),
    "HG": (2.705, 0.385),
    "TL": (4.347, 0.680),
    "PB": (4.297, 0.663),
    "BI": (4.370, 0.518),
    "PO": (4.709, 0.325),
    "AT": (4.750, 0.284),
    "RN": (4.765, 0.248),
    "FR": (4.900, 0.050),
    "RA": (3.677, 0.404),
    "AC": (3.478, 0.033),
    "TH": (3.396, 0.026),
    "PA": (3.424, 0.022),
    "U" : (3.395, 0.022),
    "NP": (3.424, 0.019),
    "PU": (3.424, 0.016),
    "AM": (3.381, 0.014),
    "CM": (3.326, 0.013),
    "BK": (3.339, 0.013),
    "CF": (3.313, 0.013),
    "ES": (3.299, 0.012),
    "FM": (3.286, 0.012),
    "MD": (3.274, 0.011),
    "NO": (3.248, 0.011),
    "LW": (3.236, 0.011),
}

HBOND_ELEMENTS = ("N", "O", "F", "S", "CL")
HBOND_FACTOR = 0.79


def _setup_energy_parameters(atoms, rotatable_bonds, partial_charges=None, force_cutoff=10.0, box=None):
    """
    Standard NumPy preprocessing to build pairwise interacting arrays.
    This creates fixed-size arrays that are passed into the compiled JAX functions.
    """
    if partial_charges is None:
        partial_charges = struc.partial_charges(atoms)
    partial_charges[np.isnan(partial_charges)] = 0.0
    
    # Map atoms to their rotational group
    atom_to_group = np.full(atoms.array_length(), -1, dtype=np.int32)
    for bond_idx, (_, _, _, h_indices) in enumerate(rotatable_bonds):
        atom_to_group[h_indices] = bond_idx
        
    if box is None:
        cell_list = struc.CellList(atoms, cell_size=force_cutoff)
    else:
        cell_list = struc.CellList(atoms, cell_size=force_cutoff, periodic=True, box=box)
        
    relevant_indices = np.where(atom_to_group != -1)[0].astype(np.int32)
    adj_indices = cell_list.get_atoms(atoms.coord[relevant_indices], radius=force_cutoff)
    bond_indices = atoms.bonds.get_all_bonds()[0]
    elements = np.char.upper(atoms.element)
    
    pairs = []
    r_6 = []
    r_12 = []
    eps = []
    elec_param = []
    
    hbond_mask = np.isin(elements, HBOND_ELEMENTS)
    
    for i_idx, atom_i in enumerate(relevant_indices):
        group_i = atom_to_group[atom_i]
        bonded_atom_i = bond_indices[atom_i, 0]
        
        for atom_j in adj_indices[i_idx]:
            if atom_j == -1 or atom_j <= atom_i:
                continue # Skip padding and deduplicate H-H interactions
            if group_i == atom_to_group[atom_j]:
                continue # Skip intra-group
            if bonded_atom_i == atom_j:
                continue # Skip direct bond
                
            element_j = elements[atom_j]
            if element_j not in NB_VALUES:
                warnings.warn(f"Unknown element '{element_j}'")
                continue
                
            pairs.append((atom_i, atom_j))
            
            # Electrostatics
            elec = 332.0673 * (partial_charges[atom_i] * partial_charges[atom_j])
            elec_param.append(elec)
            
            # Lennard-Jones
            r_i, scale_i = NB_VALUES[elements[atom_i]]
            r_j, scale_j = NB_VALUES[element_j]
            
            hb_factor = HBOND_FACTOR if (bonded_atom_i != -1 and hbond_mask[bonded_atom_i] and hbond_mask[atom_j]) else 1.0
            
            r6 = (hb_factor * 0.5 * (r_i + r_j))**6
            r_6.append(r6)
            r_12.append(r6**2)
            eps.append(np.sqrt(scale_i * scale_j))
            
    return (
        jnp.array(pairs, dtype=jnp.int32), 
        jnp.array(elec_param, dtype=jnp.float32),
        jnp.array(eps, dtype=jnp.float32), 
        jnp.array(r_6, dtype=jnp.float32), 
        jnp.array(r_12, dtype=jnp.float32),
        jnp.array(atom_to_group, dtype=jnp.int32)
    )

@jax.jit
def apply_rotations(init_coord, thetas, rot_centers, rot_axes, atom_to_bond_idx):
    """
    Differentiable Forward Kinematics: Rotates all hydrogen atoms based on theta array.
    This avoids all scatter/indexing operations by doing a global vector-mapped transform.
    """
    # For heavy atoms (bond_idx = -1), map to dummy index 0 but force theta to 0.0
    safe_bond_idx = jnp.where(atom_to_bond_idx == -1, 0, atom_to_bond_idx)
    
    centers = rot_centers[safe_bond_idx] 
    axes = rot_axes[safe_bond_idx]       
    t = thetas[safe_bond_idx]            
    
    # If it's a heavy atom, force theta=0 so it does not move
    t = jnp.where(atom_to_bond_idx == -1, 0.0, t)
    
    vecs = init_coord - centers
    
    cos_t = jnp.cos(t)[:, None]
    sin_t = jnp.sin(t)[:, None]
    
    cross = jnp.cross(axes, vecs)
    dot = jnp.sum(axes * vecs, axis=-1, keepdims=True)
    
    rotated_vecs = vecs * cos_t + cross * sin_t + axes * dot * (1 - cos_t)
    return centers + rotated_vecs

@jax.jit
def compute_energy(coord, pairs, elec_param, eps, r_6, r_12, box=None):
    """
    Pure JAX Energy function evaluating Electrostatics and Non-bonded potential.
    """
    coord_i = coord[pairs[:, 0]]
    coord_j = coord[pairs[:, 1]]
    
    delta = coord_i - coord_j
    
    if box is not None:
        # Minimum Image Convention for Generalized Triclinic Box
        box_inv = jnp.linalg.inv(box)
        frac_delta = delta @ box_inv
        frac_delta = frac_delta - jnp.round(frac_delta)
        delta = frac_delta @ box
        
    dist_sq = jnp.sum(delta**2, axis=-1)
    dist = jnp.sqrt(dist_sq)
    
    e_el = elec_param / dist
    
    dist_6 = dist_sq ** 3
    dist_12 = dist_6 ** 2
    e_nb = eps * (r_12 / dist_12 - 2 * r_6 / dist_6)
    
    return jnp.sum(e_el + e_nb)


def relax_hydrogen(atoms, iterations=200, mask=None, angle_increment=None, 
                   return_trajectory=False, return_energies=False, partial_charges=None, box=None):
    """
    Differentiable optimizer for hydrogen relaxation.
    """
    # Standard Biotite pre-processing
    atoms = atoms.copy()
    init_coord = jnp.array(atoms.coord)
    
    if box is True: box = atoms.box
    j_box = jnp.array(box) if box is not None else None
    
    rotatable_bonds = _find_rotatable_bonds(atoms, mask)
    if len(rotatable_bonds) == 0:
        return (init_coord, np.zeros(0)) if return_energies else init_coord

    # Extract rotation parameters
    B = len(rotatable_bonds)
    rot_centers = np.zeros((B, 3))
    rot_axes = np.zeros((B, 3))
    is_free_mask = np.ones(B, dtype=bool)
    
    for i, (c_idx, b_idx, is_free, h_indices) in enumerate(rotatable_bonds):
        rot_centers[i] = atoms.coord[c_idx]
        axis = atoms.coord[c_idx] - atoms.coord[b_idx]
        if box is not None:
            axis = struc.displacement(atoms.coord[b_idx], atoms.coord[c_idx], box)
        rot_axes[i] = axis / np.linalg.norm(axis)
        is_free_mask[i] = is_free
        
    rot_centers = jnp.array(rot_centers)
    rot_axes = jnp.array(rot_axes)
    is_free_mask = jnp.array(is_free_mask)

    pairs, elec_param, eps, r_6, r_12, atom_to_bond_idx = _setup_energy_parameters(
        atoms, rotatable_bonds, partial_charges, box=box
    )

    @jax.jit
    def loss_fn(thetas):
        coord = apply_rotations(init_coord, thetas, rot_centers, rot_axes, atom_to_bond_idx)
        return compute_energy(coord, pairs, elec_param, eps, r_6, r_12, j_box)

    grad_fn = jax.jit(jax.value_and_grad(loss_fn))

    thetas = jnp.zeros(B)
    m = jnp.zeros(B)
    v = jnp.zeros(B)
    lr = 0.1
    
    trajectory = []
    energies = []
    
    for step in range(iterations):
        loss, grads = grad_fn(thetas)
        
        # Enforce restricted bonds (is_free == False) by masking their gradients to 0.0
        grads = jnp.where(is_free_mask, grads, 0.0)
        
        # Adam Update rules
        m = 0.9 * m + 0.1 * grads
        v = 0.999 * v + 0.001 * (grads ** 2)
        m_hat = m / (1 - 0.9 ** (step + 1))
        v_hat = v / (1 - 0.999 ** (step + 1))
        thetas = thetas - lr * m_hat / (jnp.sqrt(v_hat) + 1e-8)
        
        if return_trajectory:
            trajectory.append(jnp.array(apply_rotations(init_coord, thetas, rot_centers, rot_axes, atom_to_bond_idx)))
        energies.append(float(loss))

    final_coord = apply_rotations(init_coord, thetas, rot_centers, rot_axes, atom_to_bond_idx)

    if return_trajectory:
        return_coord = jnp.stack(trajectory)
    else:
        return_coord = jnp.array(final_coord)
        
    if return_energies:
        return return_coord, jnp.array(energies)
    return return_coord

@jax.jit
def _relaxation_kernel(init_coord, iterations, rot_centers, rot_axes, is_free_mask, 
                       pairs, elec_param, eps, r_6, r_12, atom_to_bond_idx, j_box):
    """Pure JAX kernel for optimization."""
    B = rot_centers.shape[0]
    thetas = jnp.zeros(B)
    m = jnp.zeros(B)
    v = jnp.zeros(B)
    lr = 0.1
    
    def scan_body(carry, _):
        thetas, m, v, step = carry
        
        # Differentiable loss and gradient
        def loss_fn(t):
            coord = apply_rotations(init_coord, t, rot_centers, rot_axes, atom_to_bond_idx)
            return compute_energy(coord, pairs, elec_param, eps, r_6, r_12, j_box)
        
        loss, grads = jax.value_and_grad(loss_fn)(thetas)
        grads = jnp.where(is_free_mask, grads, 0.0)
        
        # Adam Update
        m = 0.9 * m + 0.1 * grads
        v = 0.999 * v + 0.001 * (grads ** 2)
        m_hat = m / (1 - 0.9 ** (step + 1))
        v_hat = v / (1 - 0.999 ** (step + 1))
        new_thetas = thetas - lr * m_hat / (jnp.sqrt(v_hat) + 1e-8)
        
        return (new_thetas, m, v, step + 1), loss

    # Use jax.lax.scan for high-performance loops inside JIT
    (final_thetas, _, _, _), energies = jax.lax.scan(
        scan_body, (thetas, m, v, 0), jnp.arange(iterations)
    )
    
    final_coord = apply_rotations(init_coord, final_thetas, rot_centers, rot_axes, atom_to_bond_idx)
    return final_coord, energies

import warnings
import numpy as np
import biotite.structure as struc

# Extract Biotite constants
ANY = struc.BondType.ANY
SINGLE = struc.BondType.SINGLE
DOUBLE = struc.BondType.DOUBLE
AROMATIC_DOUBLE = struc.BondType.AROMATIC_DOUBLE

def _find_rotatable_bonds(atoms, mask=None):
    """
    Identify rotatable bonds between two heavy atoms, where one atom
    has only one heavy bond partner and one or multiple hydrogen partners.
    """
    if mask is None:
        atom_mask = np.ones(atoms.array_length(), dtype=bool)
    else:
        if len(mask) != atoms.array_length():
            raise IndexError(
                f"Mask has length {len(mask)}, "
                f"but there are {atoms.array_length()} atoms"
            )
        atom_mask = np.asarray(mask, dtype=bool)

    if atoms.bonds is None:
        raise struc.BadStructureError(
            "The input structure must have an associated BondList"
        )
        
    all_bond_indices, all_bond_types = atoms.bonds.get_all_bonds()

    is_hydrogen = (atoms.element == "H")
    is_nitrogen = (atoms.element == "N")

    rotatable_bonds = []

    for i in range(all_bond_indices.shape[0]):
        if is_hydrogen[i] or not atom_mask[i]:
            continue

        hydrogen_indices = []
        bonded_heavy_index = -1
        bonded_heavy_btype = -1
        is_rotatable = True

        # Check for number of bonded heavy atoms and store bonded hydrogens
        for j in range(all_bond_indices.shape[1]):
            bonded_i = all_bond_indices[i, j]
            if bonded_i == -1:
                # -1 is a padding value that appears after all actual values
                break
                
            if is_hydrogen[bonded_i]:
                hydrogen_indices.append(bonded_i)
            elif bonded_heavy_index == -1:
                bonded_heavy_index = bonded_i
                bonded_heavy_btype = all_bond_types[i, j]
            else:
                # There is already a bonded heavy atom
                # -> at least two bonded heavy atoms -> no rotational freedom
                is_rotatable = False
                break

        # There must be at least one bonded hydrogen atom
        if len(hydrogen_indices) == 0:
            is_rotatable = False

        is_free = False
        
        # Determine specific rotation constraints based on bond types
        if is_rotatable:
            if bonded_heavy_btype == SINGLE:
                is_free = True
                # Check for partial double bonds (Nitrogen is the relevant case)
                if is_nitrogen[i]:
                    for j in range(all_bond_indices.shape[1]):
                        rem_index = all_bond_indices[bonded_heavy_index, j]
                        if rem_index == -1:
                            break
                        rem_btype = all_bond_types[bonded_heavy_index, j]
                        if rem_btype == AROMATIC_DOUBLE or rem_btype == DOUBLE:
                            is_free = False
                            break
                            
            elif bonded_heavy_btype == DOUBLE:
                is_free = False
            elif bonded_heavy_btype == ANY:
                warnings.warn(
                    "The given structure contains 'BondType.ANY' bonds, "
                    "which cannot be rotated about"
                )
                is_rotatable = False
                is_free = False
            else:
                # Triple bond etc. -> no rotational freedom
                is_rotatable = False
                is_free = False

        # 180 degrees rotation makes only sense if there is only one
        # hydrogen atom, as two would simply replace each other
        if is_rotatable and not is_free and len(hydrogen_indices) > 1:
            is_rotatable = False

        if is_rotatable:
            rotatable_bonds.append((
                i, 
                bonded_heavy_index, 
                is_free,
                np.array(hydrogen_indices, dtype=np.int32)
            ))

    return rotatable_bonds
