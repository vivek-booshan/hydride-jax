# This source code is part of the Hydride package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "hydride"
__author__ = "Patrick Kunzmann, Jacob Marcel Anter, Differentiable JAX Port"
__all__ = ["relax_hydrogen", "relax_hydrogen_jit", "get_relaxation_params"]

import warnings
import numpy as np
import biotite.structure as struc
import jax
import jax.numpy as jnp
import functools

ANY = struc.BondType.ANY
SINGLE = struc.BondType.SINGLE
DOUBLE = struc.BondType.DOUBLE
AROMATIC_DOUBLE = struc.BondType.AROMATIC_DOUBLE

NB_VALUES = {
    "H" : (2.886, 0.044), "HE": (2.362, 0.056), "LI": (2.451, 0.025), "BE": (2.745, 0.085),
    "B" : (4.083, 0.180), "C" : (3.851, 0.105), "N" : (3.660, 0.069), "O" : (3.500, 0.060),
    "F" : (3.364, 0.050), "NE": (3.243, 0.042), "NA": (2.983, 0.030), "MG": (3.021, 0.111),
    "AL": (4.499, 0.505), "SI": (4.295, 0.402), "P" : (4.147, 0.305), "S" : (4.035, 0.274),
    "CL": (3.947, 0.227), "AR": (3.868, 0.185), "K" : (3.812, 0.035), "CA": (3.399, 0.238),
    "SC": (3.295, 0.019), "TI": (3.175, 0.017), "V" : (3.144, 0.016), "CR": (3.023, 0.015),
    "MN": (2.961, 0.013), "FE": (2.912, 0.013), "CO": (2.872, 0.014), "NI": (2.834, 0.015),
    "CU": (3.495, 0.005), "ZN": (2.763, 0.124), "GA": (4.383, 0.415), "GE": (4.280, 0.379),
    "AS": (4.230, 0.309), "SE": (4.205, 0.291), "BR": (4.189, 0.251), "KR": (4.141, 0.220),
    "RB": (4.114, 0.040), "SR": (3.641, 0.235), "Y" : (3.345, 0.072), "ZR": (3.124, 0.069),
    "NB": (3.165, 0.059), "MO": (3.052, 0.056), "TC": (2.998, 0.048), "RU": (2.963, 0.056),
    "RH": (2.929, 0.053), "PD": (2.899, 0.048), "AG": (3.148, 0.036), "CD": (2.848, 0.228),
    "IN": (4.463, 0.599), "SN": (4.392, 0.567), "SB": (4.420, 0.449), "TE": (4.470, 0.398),
    "I" : (4.500, 0.339), "XE": (4.404, 0.332), "CS": (4.517, 0.045), "BA": (3.703, 0.364),
    "LA": (3.522, 0.017), "CE": (3.556, 0.013), "PR": (3.606, 0.010), "ND": (3.575, 0.010),
    "PM": (3.547, 0.009), "SM": (3.520, 0.008), "EU": (3.493, 0.008), "GD": (3.368, 0.009),
    "TB": (3.451, 0.007), "DY": (3.428, 0.007), "HO": (3.409, 0.007), "ER": (3.391, 0.007),
    "TM": (3.374, 0.006), "YB": (3.355, 0.228), "LU": (3.640, 0.041), "HF": (3.141, 0.072),
    "TA": (3.170, 0.081), "W" : (3.069, 0.067), "RE": (2.954, 0.066), "OS": (3.120, 0.037),
    "IR": (2.840, 0.073), "PT": (2.754, 0.080), "AU": (3.293, 0.039), "HG": (2.705, 0.385),
    "TL": (4.347, 0.680), "PB": (4.297, 0.663), "BI": (4.370, 0.518), "PO": (4.709, 0.325),
    "AT": (4.750, 0.284), "RN": (4.765, 0.248), "FR": (4.900, 0.050), "RA": (3.677, 0.404),
    "AC": (3.478, 0.033), "TH": (3.396, 0.026), "PA": (3.424, 0.022), "U" : (3.395, 0.022),
    "NP": (3.424, 0.019), "PU": (3.424, 0.016), "AM": (3.381, 0.014), "CM": (3.326, 0.013),
    "BK": (3.339, 0.013), "CF": (3.313, 0.013), "ES": (3.299, 0.012), "FM": (3.286, 0.012),
    "MD": (3.274, 0.011), "NO": (3.248, 0.011), "LW": (3.236, 0.011),
}

HBOND_ELEMENTS = ("N", "O", "F", "S", "CL")
HBOND_FACTOR = 0.79

def _original_get_relaxation_params(atoms, mask=None, partial_charges=None, force_cutoff=10.0, box=None):
    rotatable_bonds = _find_rotatable_bonds(atoms, mask)
    if len(rotatable_bonds) == 0:
        return None

    if box is True or (box is None):
        box = atoms.box
    elif box is False:
        box = None

    if box is not None:
        box = np.asarray(box, dtype=np.float32)
        box_inv = np.linalg.inv(box).astype(np.float32)
    else:
        box_inv = None

    B = len(rotatable_bonds)
    center_indices = np.zeros(B, dtype=np.int32)
    axis_indices = np.zeros(B, dtype=np.int32)
    is_free_mask = np.ones(B, dtype=bool)

    for i, (c_idx, b_idx, is_free, h_indices) in enumerate(rotatable_bonds):
        center_indices[i] = c_idx
        axis_indices[i] = b_idx
        is_free_mask[i] = is_free

    if partial_charges is None:
        partial_charges = struc.partial_charges(atoms)
    partial_charges[np.isnan(partial_charges)] = 0.0

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

    pairs, r_6, r_12, eps, elec_param = [], [], [], [], []
    hbond_mask = np.isin(elements, HBOND_ELEMENTS)

    for i_idx, atom_i in enumerate(relevant_indices):
        group_i = atom_to_group[atom_i]
        bonded_atom_i = bond_indices[atom_i, 0]

        for atom_j in adj_indices[i_idx]:
            if atom_j == -1 or atom_j <= atom_i: continue
            if group_i == atom_to_group[atom_j]: continue
            if bonded_atom_i == atom_j: continue

            element_j = elements[atom_j]
            if element_j not in NB_VALUES:
                continue

            pairs.append((atom_i, atom_j))
            elec = 332.0673 * (partial_charges[atom_i] * partial_charges[atom_j])
            elec_param.append(elec)

            r_i, scale_i = NB_VALUES[elements[atom_i]]
            r_j, scale_j = NB_VALUES[element_j]
            hb_factor = HBOND_FACTOR if (bonded_atom_i != -1 and hbond_mask[bonded_atom_i] and hbond_mask[atom_j]) else 1.0
            r6 = (hb_factor * 0.5 * (r_i + r_j))**6
            r_6.append(r6)
            r_12.append(r6**2)
            eps.append(np.sqrt(scale_i * scale_j))

    pairs_np = np.array(pairs, dtype=np.int32)
    num_pairs = len(pairs_np)

    if num_pairs > 0:
        sort_idx = np.argsort(pairs_np[:, 0])
        pairs_np = pairs_np[sort_idx]
        elec_param = np.array(elec_param, dtype=np.float32)[sort_idx]
        eps = np.array(eps, dtype=np.float32)[sort_idx]
        r_6 = np.array(r_6, dtype=np.float32)[sort_idx]
        r_12 = np.array(r_12, dtype=np.float32)[sort_idx]

        # Generate atomic-free reduction map structures to prevent memory locks
        flat_indices = np.concatenate([pairs_np[:, 0], pairs_np[:, 1]])
        flat_signs = np.concatenate([np.ones(num_pairs, dtype=np.float32), np.full(num_pairs, -1.0, dtype=np.float32)])
        flat_pair_map = np.concatenate([np.arange(num_pairs, dtype=np.int32), np.arange(num_pairs, dtype=np.int32)])

        sort_reduction = np.argsort(flat_indices)
        reduction_indices = flat_indices[sort_reduction]
        reduction_signs = flat_signs[sort_reduction]
        reduction_pair_map = flat_pair_map[sort_reduction]
    else:
        reduction_indices = np.zeros(0, dtype=np.int32)
        reduction_signs = np.zeros(0, dtype=np.float32)
        reduction_pair_map = np.zeros(0, dtype=np.int32)

    return (
        jnp.array(center_indices), jnp.array(axis_indices), jnp.array(is_free_mask),
        jnp.array(pairs_np, dtype=jnp.int32), jnp.array(elec_param, dtype=jnp.float32),
        jnp.array(eps, dtype=jnp.float32), jnp.array(r_6, dtype=jnp.float32),
        jnp.array(r_12, dtype=jnp.float32), jnp.array(atom_to_group, dtype=jnp.int32),
        jnp.array(box) if box is not None else None,
        jnp.array(box_inv) if box_inv is not None else None,
        jnp.array(reduction_indices, dtype=jnp.int32),
        jnp.array(reduction_signs, dtype=jnp.float32),
        jnp.array(reduction_pair_map, dtype=jnp.int32)
    )

@jax.custom_vjp
def compute_energy(coord, pairs, elec_param, eps, r_6, r_12, box=None, box_inv=None, 
                   reduction_indices=None, reduction_signs=None, reduction_pair_map=None):
    return _compute_energy_fwd(coord, pairs, elec_param, eps, r_6, r_12, box, box_inv, 
                               reduction_indices, reduction_signs, reduction_pair_map)[0]

def _compute_energy_fwd(coord, pairs, elec_param, eps, r_6, r_12, box=None, box_inv=None,
                       reduction_indices=None, reduction_signs=None, reduction_pair_map=None):
    delta = coord[pairs[:, 0]] - coord[pairs[:, 1]]
    if box is not None and box_inv is not None:
        frac_x = delta[:, 0] * box_inv[0, 0] + delta[:, 1] * box_inv[1, 0] + delta[:, 2] * box_inv[2, 0]
        frac_y = delta[:, 0] * box_inv[0, 1] + delta[:, 1] * box_inv[1, 1] + delta[:, 2] * box_inv[2, 1]
        frac_z = delta[:, 0] * box_inv[0, 2] + delta[:, 1] * box_inv[1, 2] + delta[:, 2] * box_inv[2, 2]
        
        frac_x = frac_x - jnp.round(frac_x)
        frac_y = frac_y - jnp.round(frac_y)
        frac_z = frac_z - jnp.round(frac_z)
        
        delta_x = frac_x * box[0, 0] + frac_y * box[1, 0] + frac_z * box[2, 0]
        delta_y = frac_x * box[0, 1] + frac_y * box[1, 1] + frac_z * box[2, 1]
        delta_z = frac_x * box[0, 2] + frac_y * box[1, 2] + frac_z * box[2, 2]
        dist_sq = delta_x * delta_x + delta_y * delta_y + delta_z * delta_z
        wrapped_delta = jnp.stack([delta_x, delta_y, delta_z], axis=-1)
    else:
        dist_sq = jnp.sum(delta * delta, axis=-1)
        wrapped_delta = delta
        
    dist_6 = dist_sq * dist_sq * dist_sq
    dist_12 = dist_6 * dist_6
    e_nb = eps * (r_12 / dist_12 - 2.0 * r_6 / dist_6)
    energy = jnp.sum(elec_param / jnp.sqrt(dist_sq + 1e-8) + e_nb)
    
    return energy, (coord, pairs, elec_param, eps, r_6, r_12, box, box_inv, wrapped_delta, dist_sq, dist_6, 
                    reduction_indices, reduction_signs, reduction_pair_map)

def _compute_energy_bwd(res, g):
    (coord, pairs, elec_param, eps, r_6, r_12, box, box_inv, wrapped_delta, dist_sq, dist_6, 
     reduction_indices, reduction_signs, reduction_pair_map) = res
    
    inv_dist_sq = 1.0 / (dist_sq + 1e-8)
    inv_dist_1 = jnp.sqrt(inv_dist_sq)
    
    de_elec = -0.5 * elec_param * (inv_dist_sq * inv_dist_1)
    de_vdw = eps * 6.0 * inv_dist_sq * (-r_12 / (dist_6 * dist_6) + r_6 / dist_6)
    dE_ddist_sq = (de_elec + de_vdw) * g
    
    pair_forces = 2.0 * dE_ddist_sq[..., None] * wrapped_delta
    
    # Run coalesced GPU Segment Reductions to map pairwise interactions instantly
    mapped_forces = pair_forces[reduction_pair_map] * reduction_signs[..., None]
    grad_coord = jax.ops.segment_sum(mapped_forces, reduction_indices, num_segments=coord.shape[0])
    
    return grad_coord, None, None, None, None, None, None, None, None, None, None

compute_energy.defvjp(_compute_energy_fwd, _compute_energy_bwd)

def _get_nerf_params(atoms, atom_to_bond_idx, center_indices, axis_indices, box=None):
    """Calculates chemically rigid NeRF parameters using exact covalent graph traversal."""
    num_atoms = atoms.array_length()
    
    p1_idx = np.zeros(num_atoms, dtype=int)
    p2_idx = np.zeros(num_atoms, dtype=int)
    p3_idx = np.zeros(num_atoms, dtype=int)
    ref_v = np.zeros((num_atoms, 3), dtype=np.float32)
    use_ref_v = np.zeros(num_atoms, dtype=bool)
    
    bonds, _ = atoms.bonds.get_all_bonds()
    heavy_mask = (atoms.element != "H") & (atoms.element != "D")
    h_mask = ~heavy_mask
    
    h_indices = np.where(h_mask)[0]
    heavy_indices = np.where(heavy_mask)[0]
    coord = atoms.coord
    
    def min_image_np(vecs):
        if box is None: return vecs
        box_inv = np.linalg.inv(box)
        frac = vecs @ box_inv
        frac -= np.round(frac)
        return frac @ box
    
    for h in h_indices:
        b = atom_to_bond_idx[h]
        if b != -1:
            # Enforce strict alignment with Hydride's energy axis for rotatable bonds
            p1 = center_indices[b]
            p2 = axis_indices[b]
        else:
            # For fixed hydrogens, traverse the covalent graph to find P1 and P2
            h_neighbors = [n for n in bonds[h] if n != -1 and heavy_mask[n]]
            p1 = h_neighbors[0] if h_neighbors else heavy_indices[0]
            
            p1_neighbors = [n for n in bonds[p1] if n != -1 and heavy_mask[n] and n != h]
            p2 = p1_neighbors[0] if p1_neighbors else p1
            
        # Topologically discover P3 to complete the local rotameric frame
        p3_candidates = []
        
        # 1. PRIORITY: Other heavy atoms bonded directly to P1 (e.g., C_epsilon for a CH2 on C_delta)
        # This guarantees methylene and methine protons are perfectly symmetric to local heavy neighbors.
        p3_candidates.extend([n for n in bonds[p1] if n != -1 and heavy_mask[n] and n != p2 and n != h])
        
        # 2. FALLBACK: Heavy atoms bonded to P2 (e.g., C_beta for a terminal CH3 on C_gamma)
        if p2 != p1:
            p3_candidates.extend([n for n in bonds[p2] if n != -1 and heavy_mask[n] and n != p1])
        
        found_p3 = False
        v1_vec = min_image_np(coord[p2] - coord[p1])        

        # Ensure the chosen P3 is not collinear with P1-P2
        for candidate in p3_candidates:
            v2_vec = min_image_np(coord[candidate] - coord[p1])
            v2_vec = v2_vec / (np.linalg.norm(v2_vec) + 1e-8)
            if np.linalg.norm(np.cross(v1_vec, v2_vec)) > 1e-2:
                p3 = candidate
                found_p3 = True
                break
                
        if found_p3:
            p1_idx[h] = p1
            p2_idx[h] = p2
            p3_idx[h] = p3
            use_ref_v[h] = False
        else:
            # Fallback for perfectly collinear or isolated molecules (e.g., water, ethane)
            v_r = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            if np.abs(np.dot(v1_vec, v_r)) > 0.9:
                v_r = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            v_r = v_r - np.dot(v1_vec, v_r) * v1_vec
            v_r = v_r / (np.linalg.norm(v_r) + 1e-8)
            
            p1_idx[h] = p1
            p2_idx[h] = p2
            p3_idx[h] = p1
            ref_v[h] = v_r
            use_ref_v[h] = True
            
    v1 = min_image_np(coord[p2_idx] - coord[p1_idx])
    v1 = v1 / (np.linalg.norm(v1, axis=-1, keepdims=True) + 1e-8)
    
    v2 = np.where(use_ref_v[:, None], ref_v, min_image_np(coord[p3_idx] - coord[p1_idx]))
    
    n = np.cross(v1, v2)
    n = n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-8)
    
    v = np.cross(n, v1)
    
    delta = min_image_np(coord - coord[p1_idx])
    x = np.sum(delta * v1, axis=-1)
    y = np.sum(delta * v, axis=-1)
    z = np.sum(delta * n, axis=-1)
    
    lengths = np.linalg.norm(delta, axis=-1)
    angles = np.arctan2(np.hypot(y, z), x)
    torsions = np.arctan2(z, y)
    
    return p1_idx, p2_idx, p3_idx, ref_v, use_ref_v, lengths, angles, torsions, h_mask

def get_relaxation_params(atoms, mask=None, partial_charges=None, box=None):
    base_params = _original_get_relaxation_params(atoms, mask, partial_charges, box=box)
    if base_params is None:
        return None
        
    atom_to_bond_idx = base_params[8]
    center_indices = base_params[0]
    axis_indices = base_params[1]
    
    p1, p2, p3, ref_v, use_ref_v, lengths, angles, torsions, h_mask = _get_nerf_params(
        atoms, atom_to_bond_idx, center_indices, axis_indices, box=box
    )
    return tuple(base_params) + (p1, p2, p3, ref_v, use_ref_v, lengths, angles, torsions, h_mask)


@jax.jit
def place_hydrogens_jit(X_heavy, p1_idx, p2_idx, p3_idx, ref_v, use_ref_v, lengths, angles, torsions, h_mask, box=None, box_inv=None):
    p1 = X_heavy[p1_idx]
    p2 = X_heavy[p2_idx]
    p3 = X_heavy[p3_idx]
    
    v1 = p2 - p1
    if box is not None and box_inv is not None:
        frac = jnp.einsum('...i,ij->...j', v1, box_inv)
        frac = frac - jnp.round(frac)
        v1 = jnp.einsum('...i,ij->...j', frac, box)
        
    v1 = v1 / jnp.sqrt(jnp.sum(v1**2, axis=-1, keepdims=True) + 1e-12)
    
    v2 = p3 - p1
    if box is not None and box_inv is not None:
        frac2 = jnp.einsum('...i,ij->...j', v2, box_inv)
        frac2 = frac2 - jnp.round(frac2)
        v2 = jnp.einsum('...i,ij->...j', frac2, box)
        
    v2 = jnp.where(use_ref_v[:, None], ref_v, v2)
    
    n = jnp.cross(v1, v2)
    n = n / jnp.sqrt(jnp.sum(n**2, axis=-1, keepdims=True) + 1e-12)
    
    v = jnp.cross(n, v1)
    
    x = lengths * jnp.cos(angles)
    y = lengths * jnp.sin(angles) * jnp.cos(torsions)
    z = lengths * jnp.sin(angles) * jnp.sin(torsions)
    
    X_H = p1 + x[:, None]*v1 + y[:, None]*v + z[:, None]*n
    
    if box is not None and box_inv is not None:
        diff = X_H - X_heavy
        frac = jnp.einsum('...i,ij->...j', diff, box_inv)
        frac = frac - jnp.round(frac)
        X_H = X_heavy + jnp.einsum('...i,ij->...j', frac, box)
    
    # We always rebuild all hydrogens from NeRF to guarantee they co-move safely with heavy atoms
    return jnp.where(h_mask[:, None], X_H, X_heavy)


@functools.partial(jax.jit, static_argnames=("iterations", "return_trajectory"))
def relax_hydrogen_jit(
    X_initial, center_indices, axis_indices, is_free_mask, pairs, elec_param, eps, r_6, r_12,
    atom_to_bond_idx, box, box_inv, reduction_indices, reduction_signs, reduction_pair_map,
    p1_idx, p2_idx, p3_idx, ref_v, use_ref_v, lengths, angles, initial_torsions, h_mask,
    iterations=200, return_trajectory=False, start_angle=0.0
):
    num_atoms = X_initial.shape[0]
    B = center_indices.shape[0]
    
    init_delta_torsions = jnp.full(B, start_angle + 1e-3)
    init_m = jnp.zeros(B)
    init_v = jnp.zeros(B)
    base_lr = 0.1
    
    def compute_X(d_torsions):
        d_t = jnp.where(is_free_mask, d_torsions, 0.0)
        d_t_padded = jnp.append(d_t, 0.0)
        atom_shifts = d_t_padded[atom_to_bond_idx]
        current_torsions = initial_torsions + atom_shifts
        return place_hydrogens_jit(
            X_initial, p1_idx, p2_idx, p3_idx, ref_v, use_ref_v, lengths, angles, current_torsions, h_mask, box, box_inv
        )

    def scan_body(carry, i):
        d_torsions, m_t, v_t = carry
        
        def energy_fn(d_t):
            X_current = compute_X(d_t)
            from hydride.relax import compute_energy
            return compute_energy(X_current, pairs, elec_param, eps, r_6, r_12,
                                  box, box_inv, reduction_indices, reduction_signs, reduction_pair_map)
        
        ener, grads = jax.value_and_grad(energy_fn)(d_torsions)
        grads = jnp.where(is_free_mask, grads, 0.0)
        
        progress = i / iterations
        lr_t = base_lr * 0.5 * (1.0 + jnp.cos(jnp.pi * progress))
        
        m_next = 0.9 * m_t + 0.1 * grads
        v_next = 0.999 * v_t + 0.001 * (grads ** 2)
        m_hat = m_next / (1.0 - 0.9 ** (i + 1))
        v_hat = v_next / (1.0 - 0.999 ** (i + 1))
        
        d_torsions_new = d_torsions - lr_t * m_hat / (jnp.sqrt(v_hat) + 1e-8)
        
        X_current = jax.lax.cond(
            return_trajectory,
            lambda: compute_X(d_torsions_new),
            lambda: jnp.zeros_like(X_initial)
        )
        return (d_torsions_new, m_next, v_next), (X_current, ener)
        
    if return_trajectory:
        (final_delta_torsions, _, _), (traj, ener) = jax.lax.scan(scan_body, (init_delta_torsions, init_m, init_v), jnp.arange(iterations))
        ener = jax.lax.cummin(ener)
        final_X = compute_X(final_delta_torsions)
        return final_X, traj, ener
    else:
        def step_body_simple(i, carry):
            d_torsions, m_t, v_t = carry
            def energy_fn_simple(dt):
                X_current = compute_X(dt)
                from hydride.relax import compute_energy
                return compute_energy(X_current, pairs, elec_param, eps, r_6, r_12,
                                      box, box_inv, reduction_indices, reduction_signs, reduction_pair_map)
            
            grads = jax.grad(energy_fn_simple)(d_torsions)
            grads = jnp.where(is_free_mask, grads, 0.0)
            
            progress = i / iterations
            lr_t = base_lr * 0.5 * (1.0 + jnp.cos(jnp.pi * progress))
            
            m_next = 0.9 * m_t + 0.1 * grads
            v_next = 0.999 * v_t + 0.001 * (grads ** 2)
            m_hat = m_next / (1.0 - 0.9 ** (i + 1))
            v_hat = v_next / (1.0 - 0.999 ** (i + 1))
            
            d_torsions_new = d_torsions - lr_t * m_hat / (jnp.sqrt(v_hat) + 1e-8)
            return (d_torsions_new, m_next, v_next)
            
        final_delta_torsions, _, _ = jax.lax.fori_loop(0, iterations, step_body_simple, (init_delta_torsions, init_m, init_v))
        
        X_final = compute_X(final_delta_torsions)
        from hydride.relax import compute_energy
        final_energy = compute_energy(X_final, pairs, elec_param, eps, r_6, r_12,
                                      box, box_inv, reduction_indices, reduction_signs, reduction_pair_map)
        
        return X_final, jnp.zeros((0, num_atoms, 3)), jnp.array([final_energy])


def relax_hydrogen(atoms, iterations=200, mask=None, angle_increment=None, 
                   return_trajectory=False, return_energies=False, partial_charges=None, box=None):
    if iterations is None:
        warnings.warn("passing None to iterations. Defaulting to 200")
        iterations = 200

    atoms = atoms.copy()
    init_coord_np = atoms.coord
    params = get_relaxation_params(atoms, mask, partial_charges, box=box)
    
    if params is None:
        if return_trajectory and return_energies: return np.array([init_coord_np]), np.zeros(0)
        if return_energies: return init_coord_np, np.zeros(0)
        if return_trajectory: return np.array([init_coord_np])
        return init_coord_np

    best_coord, best_traj, best_energies = None, None, None
    min_final_energy = float('inf')

    for ang in [0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0]:
        f_coord, traj, ener = relax_hydrogen_jit(
            jnp.array(init_coord_np), 
            *params,
            iterations=iterations, return_trajectory=return_trajectory,
            start_angle=ang
        )
        f_energy = ener[-1] if len(ener) > 0 else 0.0
        
        if f_energy < min_final_energy:
            min_final_energy = f_energy
            best_coord = f_coord
            best_traj = traj
            best_energies = ener

    # params[8] is atom_to_bond_idx. params[-1] is h_mask.
    mobile_h_mask_np = np.array(params[-1] & (params[8] != -1))
    if mask is not None:
        mobile_h_mask_np &= mask

    if return_trajectory:
        out_coord = np.array(best_traj, copy=True).astype(np.float32)
        # Restore multi-frame 3D coordinates for locked atoms
        out_coord[:, ~mobile_h_mask_np, :] = init_coord_np[~mobile_h_mask_np, :]
    else:
        out_coord = np.array(best_coord, copy=True).astype(np.float32)
        out_coord[~mobile_h_mask_np, :] = init_coord_np[~mobile_h_mask_np, :]
        
    if return_energies:
        return out_coord, np.array(best_energies, copy=True)
    return out_coord

def _find_rotatable_bonds(atoms, mask=None):
    if mask is None:
        atom_mask = np.ones(atoms.array_length(), dtype=bool)
    else:
        if len(mask) != atoms.array_length():
            raise IndexError(f"Mask has length {len(mask)}, but there are {atoms.array_length()} atoms")
        atom_mask = np.asarray(mask, dtype=bool)

    if atoms.bonds is None:
        raise struc.BadStructureError("The input structure must have an associated BondList")
        
    all_bond_indices, all_bond_types = atoms.bonds.get_all_bonds()
    is_hydrogen, is_nitrogen = (atoms.element == "H"), (atoms.element == "N")
    rotatable_bonds = []

    for i in range(all_bond_indices.shape[0]):
        if is_hydrogen[i] or not atom_mask[i]: continue

        hydrogen_indices = []
        bonded_heavy_index, bonded_heavy_btype = -1, -1
        is_rotatable = True

        for j in range(all_bond_indices.shape[1]):
            bonded_i = all_bond_indices[i, j]
            if bonded_i == -1: break
            if is_hydrogen[bonded_i]:
                hydrogen_indices.append(bonded_i)
            elif bonded_heavy_index == -1:
                bonded_heavy_index, bonded_heavy_btype = bonded_i, all_bond_types[i, j]
            else:
                is_rotatable = False
                break

        if len(hydrogen_indices) == 0: is_rotatable = False
        is_free = False
        
        if is_rotatable:
            if bonded_heavy_btype == SINGLE:
                is_free = True
                if is_nitrogen[i]:
                    for j in range(all_bond_indices.shape[1]):
                        rem_index = all_bond_indices[bonded_heavy_index, j]
                        if rem_index == -1: break
                        rem_btype = all_bond_types[bonded_heavy_index, j]
                        if rem_btype in (AROMATIC_DOUBLE, DOUBLE):
                            is_free = False
                            break
            elif bonded_heavy_btype == DOUBLE: is_free = False
            elif bonded_heavy_btype == ANY:
                warnings.warn("Structure contains 'BondType.ANY' bonds.")
                is_rotatable, is_free = False, False
            else:
                is_rotatable, is_free = False, False

        if is_rotatable and not is_free and len(hydrogen_indices) > 1:
            is_rotatable = False

        if is_rotatable:
            rotatable_bonds.append((i, bonded_heavy_index, is_free, np.array(hydrogen_indices, dtype=np.int32)))

    return rotatable_bonds
