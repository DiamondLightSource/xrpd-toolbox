from pathlib import Path

import numpy as np
from CifFile import ReadCif

from xrpd_toolbox.utils.utils import parse_numbers_with_error

# TODO: do the comment below
# Use U_iso for computation / modeling
# Accept B_iso when reading external data

# uiso- = direct measure of atomic displacement (mean-square displacement).
# biso - debye waller factor


# @njit()
def u_iso_to_b_iso(u_iso: np.ndarray) -> np.ndarray:
    return 8 * np.pi**2 * u_iso


# @njit()
def b_iso_to_u_iso(b_iso: np.ndarray) -> np.ndarray:
    return b_iso / (8 * np.pi**2)


###cif reader
def open_cif(cif_filepath: str | Path, block_number: int = 0) -> dict:
    """opens the cif and returns a dict-like representation of the cif"""
    cif = ReadCif(cif_filepath)
    block_name = list(cif.keys())[block_number]
    block = cif[block_name]

    return block


def get_symmetry_operation_from_cif(cif_filepath: str | Path, block_number: int = 0):
    block = open_cif(cif_filepath=cif_filepath, block_number=block_number)
    synmmetry_operations = np.array(block["_space_group_symop_operation_xyz"])

    return synmmetry_operations


def read_cif(cif_filepath: str | Path, block_number: int = 0) -> tuple:
    """reads data from a cif and returns
    list of Atom classes and a unit cell class"""

    # TODO: make this use the errors

    block = open_cif(cif_filepath=cif_filepath, block_number=block_number)

    x = np.array(block["_atom_site_fract_x"], dtype=str)
    x, x_e = parse_numbers_with_error(x)
    y = np.array(block["_atom_site_fract_y"], dtype=str)
    y, y_e = parse_numbers_with_error(y)

    z = np.array(block["_atom_site_fract_z"], dtype=str)
    z, z_e = parse_numbers_with_error(z)

    atom_labels = np.array(block["_atom_site_label"])

    if "_atom_site_type_symbol" in block:
        elements = np.array(block["_atom_site_type_symbol"])
    else:
        elements = atom_labels.copy()

    elements = np.array([el.capitalize() for el in elements])

    occupancies = (
        np.array(block["_atom_site_occupancy"], dtype=str)
        if "_atom_site_occupancy" in block
        else np.ones(len(x))
    )

    occupancies, occupancies_e = parse_numbers_with_error(occupancies)

    # --- B_iso or U_iso handling ---
    if "_atom_site_B_iso_or_equiv" in block:
        b_iso = np.array(block["_atom_site_B_iso_or_equiv"], dtype=str)

    elif "_atom_site_U_iso_or_equiv" in block:
        u_iso = np.array(block["_atom_site_U_iso_or_equiv"], dtype=str)
        u_iso, u_iso_e = parse_numbers_with_error(u_iso)
        b_iso = u_iso_to_b_iso(u_iso)

    else:
        b_iso = np.tile(0.5, len(x))  # fallback
        b_iso, b_iso_e = parse_numbers_with_error(b_iso)

    if "_symmetry_space_group_name_H-M" in block:
        spacegroup_symbol = block["_symmetry_space_group_name_H-M"]
    elif "_space_group_name_H-M_alt" in block:
        spacegroup_symbol = block["_space_group_name_H-M_alt"]
    else:
        print("_symmetry_space_group_name_H-M not in cif")
        spacegroup_symbol = "P1"

    if "_space_group_symop_operation_xyz" in block:
        symmetry_operations = np.array(block["_space_group_symop_operation_xyz"])
    elif "_symmetry_equiv_pos_as_xyz" in block:
        symmetry_operations = np.array(block["_symmetry_equiv_pos_as_xyz"])
    else:
        print("No symmetry operation in cif")
        symmetry_operations = np.array([])

    if "_chemical_name_mineral" in block:
        name = block["_chemical_name_mineral"]
    else:
        name = None

    atoms = {
        "labels": atom_labels,
        "elements": elements,
        "xyz": np.column_stack((x, y, z)),
        "b_iso": b_iso,
        "occupancies": occupancies,
    }

    a, a_e = parse_numbers_with_error(block["_cell_length_a"])
    b, b_e = parse_numbers_with_error(block["_cell_length_b"])
    c, c_e = parse_numbers_with_error(block["_cell_length_c"])
    alpha, alpha_e = parse_numbers_with_error(block["_cell_angle_alpha"])
    beta, beta_e = parse_numbers_with_error(block["_cell_angle_beta"])
    gamma, gamma_e = parse_numbers_with_error(block["_cell_angle_gamma"])

    lattice = {
        "a": float(a),
        "b": float(b),
        "c": float(c),
        "alpha": float(alpha),
        "beta": float(beta),
        "gamma": float(gamma),
    }

    return spacegroup_symbol, lattice, atoms, symmetry_operations, name
