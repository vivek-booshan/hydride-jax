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

# Extract Biotite constants
ANY = struc.BondType.ANY
SINGLE = struc.BondType.SINGLE
DOUBLE = struc.BondType.DOUBLE
AROMATIC_DOUBLE = struc.BondType.AROMATIC_DOUBLE

# Values are taken from Rappé et al. (UFF)
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

def get_relaxation_params(atoms, mask=None, partial_charges=None, force_cutoff=10.0, box=None):
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
    if pairs_np.size > 0:
        sort_idx = np.argsort(pairs_np[:, 0])
        pairs_np = pairs_np[sort_idx]
        elec_param = np.array(elec_param, dtype=np.float32)[sort_idx]
        eps = np.array(eps, dtype=np.float32)[sort_idx]
        r_6 = np.array(r_6, dtype=np.float32)[sort_idx]
        r_12 = np.array(r_12, dtype=np.float32)[sort_idx]

    return (
        jnp.array(center_indices), jnp.array(axis_indices), jnp.array(is_free_mask),
        jnp.array(pairs_np, dtype=jnp.int32), jnp.array(elec_param, dtype=jnp.float32),
        jnp.array(eps, dtype=jnp.float32), jnp.array(r_6, dtype=jnp.float32), 
        jnp.array(r_12, dtype=jnp.float32), jnp.array(atom_to_group, dtype=jnp.int32),
        jnp.array(box) if box is not None else None,
        jnp.array(box_inv) if box_inv is not None else None
    )

@jax.jit
def apply_rotations(init_coord, thetas, rot_centers, rot_axes, atom_to_bond_idx, box=None, box_inv=None):
    safe_bond_idx = jnp.where(atom_to_bond_idx == -1, 0, atom_to_bond_idx)
    centers = rot_centers[safe_bond_idx] 
    axes = rot_axes[safe_bond_idx]       
    t = thetas[safe_bond_idx]            
    
    t = jnp.where(atom_to_bond_idx == -1, 0.0, t)
    vecs = init_coord - centers
    
    if box is not None and box_inv is not None:
        frac_x = vecs[:, 0] * box_inv[0, 0] + vecs[:, 1] * box_inv[1, 0] + vecs[:, 2] * box_inv[2, 0]
        frac_y = vecs[:, 0] * box_inv[0, 1] + vecs[:, 1] * box_inv[1, 1] + vecs[:, 2] * box_inv[2, 1]
        frac_z = vecs[:, 0] * box_inv[0, 2] + vecs[:, 1] * box_inv[1, 2] + vecs[:, 2] * box_inv[2, 2]
        
        frac_x = frac_x - jnp.round(frac_x)
        frac_y = frac_y - jnp.round(frac_y)
        frac_z = frac_z - jnp.round(frac_z)
        
        vecs_x = frac_x * box[0, 0] + frac_y * box[1, 0] + frac_z * box[2, 0]
        vecs_y = frac_x * box[0, 1] + frac_y * box[1, 1] + frac_z * box[2, 1]
        vecs_z = frac_x * box[0, 2] + frac_y * box[1, 2] + frac_z * box[2, 2]
        vecs = jnp.stack([vecs_x, vecs_y, vecs_z], axis=-1)
        
    cos_t = jnp.cos(t)[:, None]
    sin_t = jnp.sin(t)[:, None]
    
    cross = jnp.cross(axes, vecs)
    dot = jnp.sum(axes * vecs, axis=-1, keepdims=True)
    rotated_vecs = vecs * cos_t + cross * sin_t + axes * dot * (1 - cos_t)
    
    new_coord = init_coord + (rotated_vecs - vecs)
    return jnp.where(atom_to_bond_idx[:, None] == -1, init_coord, new_coord)


# Decorate with custom_vjp up-front to prevent compilation NameError issues
@jax.custom_vjp
def compute_energy(coord, pairs, elec_param, eps, r_6, r_12, box=None, box_inv=None):
    return _compute_energy_fwd(coord, pairs, elec_param, eps, r_6, r_12, box, box_inv)[0]

def _compute_energy_fwd(coord, pairs, elec_param, eps, r_6, r_12, box=None, box_inv=None):
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
    
    return energy, (coord, pairs, elec_param, eps, r_6, r_12, box, box_inv, wrapped_delta, dist_sq, dist_6)

def _compute_energy_bwd(res, g):
    coord, pairs, elec_param, eps, r_6, r_12, box, box_inv, wrapped_delta, dist_sq, dist_6 = res
    
    inv_dist_sq = 1.0 / (dist_sq + 1e-8)
    inv_dist_1 = jnp.sqrt(inv_dist_sq)
    
    de_elec = -0.5 * elec_param * (inv_dist_sq * inv_dist_1)
    de_vdw = eps * 6.0 * inv_dist_sq * (-r_12 / (dist_6 * dist_6) + r_6 / dist_6)
    dE_ddist_sq = (de_elec + de_vdw) * g
    
    pair_forces = 2.0 * dE_ddist_sq[..., None] * wrapped_delta
    
    grad_coord = jnp.zeros_like(coord)
    grad_coord = grad_coord.at[pairs[:, 0]].add(pair_forces)
    grad_coord = grad_coord.at[pairs[:, 1]].add(-pair_forces)
    
    return grad_coord, None, None, None, None, None, None, None

compute_energy.defvjp(_compute_energy_fwd, _compute_energy_bwd)


@jax.jit(static_argnames=("iterations", "return_trajectory"))
def relax_hydrogen_jit(init_coord, center_indices, axis_indices, is_free_mask, pairs, 
                       elec_param, eps, r_6, r_12, atom_to_bond_idx, box=None, box_inv=None,
                       iterations: int = 200, return_trajectory: bool = False,
                       start_angle: float = 0.0):
    B = center_indices.shape[0]
    
    raw_axes = init_coord[center_indices] - init_coord[axis_indices]
    if box is not None and box_inv is not None:
        frac_x = raw_axes[:, 0] * box_inv[0, 0] + raw_axes[:, 1] * box_inv[1, 0] + raw_axes[:, 2] * box_inv[2, 0]
        frac_y = raw_axes[:, 0] * box_inv[0, 1] + raw_axes[:, 1] * box_inv[1, 1] + raw_axes[:, 2] * box_inv[2, 1]
        frac_z = raw_axes[:, 0] * box_inv[0, 2] + raw_axes[:, 1] * box_inv[1, 2] + raw_axes[:, 2] * box_inv[2, 2]
        
        frac_x = frac_x - jnp.round(frac_x)
        frac_y = frac_y - jnp.round(frac_y)
        frac_z = frac_z - jnp.round(frac_z)
        
        axes_x = frac_x * box[0, 0] + frac_y * box[1, 0] + frac_z * box[2, 0]
        axes_y = frac_x * box[0, 1] + frac_y * box[1, 1] + frac_z * box[2, 1]
        axes_z = frac_x * box[0, 2] + frac_y * box[1, 2] + frac_z * box[2, 2]
        raw_axes = jnp.stack([axes_x, axes_y, axes_z], axis=-1)

    rot_axes = raw_axes / (jnp.linalg.norm(raw_axes, axis=-1, keepdims=True) + 1e-8)
    rot_centers = init_coord[center_indices]
    
    thetas = jnp.full(B, start_angle + 1e-3)
    m = jnp.zeros(B)
    v = jnp.zeros(B)
    lr = 0.1
    
    @jax.checkpoint
    def scan_body(carry, _):
        t, m_t, v_t, step = carry
        
        def loss_fn(ang):
            c = apply_rotations(init_coord, ang, rot_centers, rot_axes, atom_to_bond_idx, box, box_inv)
            return compute_energy(c, pairs, elec_param, eps, r_6, r_12, box, box_inv)

        loss, grads = jax.value_and_grad(loss_fn)(t)
        grads = jnp.where(is_free_mask, grads, 0.0)
        
        progress = step / iterations
        lr_t = lr * 0.5 * (1.0 + jnp.cos(jnp.pi * progress))
        
        m_next = 0.9 * m_t + 0.1 * grads
        v_next = 0.999 * v_t + 0.001 * (grads ** 2)
        m_hat = m_next / (1 - 0.9 ** (step + 1))
        v_hat = v_next / (1 - 0.999 ** (step + 1))
        t_next = t - lr_t * m_hat / (jnp.sqrt(v_hat) + 1e-8)
        
        coord_t = jax.lax.cond(
            return_trajectory,
            lambda: apply_rotations(init_coord, t_next, rot_centers, rot_axes, atom_to_bond_idx, box, box_inv).astype(init_coord.dtype),
            lambda: jnp.zeros_like(init_coord) 
        )
        return (t_next, m_next, v_next, step + 1), (coord_t, loss)

    (final_thetas, _, _, _), (trajectory, energies) = jax.lax.scan(
        scan_body, (thetas, m, v, 0), jnp.arange(iterations)
    )

    energies = jax.lax.cummin(energies)
    final_coord = apply_rotations(init_coord, final_thetas, rot_centers, rot_axes, atom_to_bond_idx, box, box_inv)
    return final_coord, trajectory, energies

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
            params[0], params[1], params[2], params[3], params[4], 
            params[5], params[6], params[7], params[8], params[9], params[10],
            iterations=iterations, return_trajectory=return_trajectory,
            start_angle=ang
        )
        f_energy = ener[-1] if len(ener) > 0 else 0.0
        
        if f_energy < min_final_energy:
            min_final_energy = f_energy
            best_coord = f_coord
            best_traj = traj
            best_energies = ener

    if return_trajectory:
        out_coord = np.array(best_traj, copy=True).astype(np.float32)
    else:
        out_coord = np.array(best_coord, copy=True).astype(np.float32)
        
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
