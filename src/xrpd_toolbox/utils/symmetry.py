import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from yaml import safe_load

from xrpd_toolbox.utils.constants import get_spacegroup_symbol
from xrpd_toolbox.utils.settings import XRPDBaseModel

_FRAC_RE = re.compile(r"([+-]?\d+)/(\d+)")


def frac_to_decimal_string(sym_strs: np.ndarray, dp: int = 3):
    """
    Convert fractions in symmetry strings to decimals while preserving the + or - signs.
    Example: "x+1/4" -> "x+0.25"

    Parameters
    ----------
    sym_strs : array-like of str

    Returns
    -------
    np.ndarray of str
    """
    sym_strs = np.asarray(sym_strs, dtype=str)

    # Replacement function
    def _replace_frac(match):
        num, den = match.groups()
        val = float(num) / float(den)
        # Keep the explicit + if positive
        return f"{val:+.{dp}g}" if val >= 0 else f"{val:.{dp}g}"

    # Apply to all strings (still vectorized over array, no per-character loops)
    new_strs = np.array([_FRAC_RE.sub(_replace_frac, s) for s in sym_strs])
    return new_strs


def format_space_group_name(sg: str) -> str:
    """Format space group name to a consistent format for lookup."""

    sg = sg[0].upper() + sg[1:].lower()  # upper case first letter
    sg = re.sub(r"\s+", "", sg)  # remove all whitespace
    sg = re.sub(r"^([A-Za-z]+)3([A-Za-z]+)$", r"\1-3\2", sg)
    sg = re.sub(r"([234])_1", r"\g<1>1", sg)  # 2_1 → 21 etc.

    # remove trivial 1's (P121 → P21, P112 → P12 etc.)
    # sg = re.sub(r"1(?=[xyzabcmbcn/:-]|$)", "", sg)
    # sg = re.sub(r"^([A-Za-z])1+", r"\1", sg)  # drop leading 1's
    sg = re.sub(r"1$", "", sg)  # drop trailing 1

    return sg


def lattice_to_matrix(
    a: float,
    b: float,
    c: float,
    alpha: float,
    beta: float,
    gamma: float,
) -> np.ndarray:
    """Convert lattice parameters to Cartesian lattice matrix."""

    alpha_r, beta_r, gamma_r = np.radians([alpha, beta, gamma])

    cos_a, cos_b, cos_g = np.cos([alpha_r, beta_r, gamma_r])
    sin_g = np.sin(gamma_r)

    a_vec = np.array([a, 0.0, 0.0])
    b_vec = np.array([b * cos_g, b * sin_g, 0.0])

    cx = c * cos_b
    cy = c * (cos_a - cos_b * cos_g) / sin_g
    cz = np.sqrt(max(c**2 - cx**2 - cy**2, 0.0))

    c_vec = np.array([cx, cy, cz])

    return np.vstack([a_vec, b_vec, c_vec])


def _convert_symmetry_operations_yaml():
    constants_folder = os.path.join(os.path.dirname(__file__), "constants")

    symmetry_operation_filepath = os.path.join(
        constants_folder, "symmetry_operations.yaml"
    )

    with open(symmetry_operation_filepath) as file:
        try:
            symmetry_operations = safe_load(file)
        except Exception:
            raise

        new_spacegroups = {}

        for sg in symmetry_operations:
            sg["symops"] = frac_to_decimal_string(sg["symops"])
            sg["ncsym"] = frac_to_decimal_string(sg["ncsym"])

            sg_name = get_spacegroup_symbol(sg["number"])

            sg["spacegroup"] = sg_name

            if sg_name in new_spacegroups:
                continue

            new_spacegroups[sg_name] = sg

    itc = IntTabCryst(spacegroups=new_spacegroups)

    itc.save_to_yaml(
        "/workspaces/XRPD-Toolbox/src/xrpd_toolbox/utils/constants/international_tables_of_crystallography.yaml"
    )


class SpaceGroup(XRPDBaseModel):
    spacegroup: str
    crystal_class: str
    hall: str
    hermann_mauguin: str
    symops: list[str]
    ncsym: list[str]
    number: int
    schoenflies: str
    universal_h_m: str

    #  symop - full set

    # Use this for:

    # HKL reduction
    # Laue symmetry
    # structure factor
    # systematic absences
    # diffraction pattern

    # ncsym  (non-centrosymmetric set)
    # inversion symmetry is removed
    # Used for:

    # anomalous scattering (f′, f″)
    # Friedel pairs (hkl vs -hkl not equivalent)
    # non-centrosymmetric analysis
    # separating centrosymmetric vs non-centrosymmetric contributions

    def _normalise_symmetry_operations(self, symops):
        """Convert symmetry operation strings to a consistent format for parsing."""
        ops = np.asarray(symops, dtype=str)

        # Normalize signs and remove leading '+'
        ops = np.char.replace(ops, "-", "+-")
        ops = np.char.replace(ops, "++", "+")
        ops = np.char.lstrip(ops, "+")

        return ops

    def get_rotations_and_translations(self):
        ops = self._normalise_symmetry_operations(self.symops)

        # Split into coordinates (N,3)
        coords = np.char.split(ops, ",")

        # Split each coordinate into terms (e.g. "x+1/4" -> ["x", "+1/4"])
        terms = np.array([[c.split("+") for c in row] for row in coords], dtype=object)

        # ---------------------------------
        # FLATTEN TERMS
        # ---------------------------------
        flat_terms = np.concatenate([t for row in terms for t in row])

        # ---------------------------------
        # IDENTIFY VARIABLES
        # ---------------------------------
        is_x = np.char.find(flat_terms, "x") >= 0
        is_y = np.char.find(flat_terms, "y") >= 0
        is_z = np.char.find(flat_terms, "z") >= 0

        coeff = np.where(np.char.startswith(flat_terms, "-"), -1.0, 1.0)

        rx = np.where(is_x, coeff, 0.0)
        ry = np.where(is_y, coeff, 0.0)
        rz = np.where(is_z, coeff, 0.0)

        # ---------------------------------
        # CONSTANT TERMS
        # ---------------------------------
        is_const = ~(is_x | is_y | is_z)
        const_terms = np.where(is_const, flat_terms, "0")
        const_terms = np.where(const_terms == "", "0", const_terms)

        frac = np.char.find(const_terms, "/") >= 0
        num = np.where(frac, np.char.partition(const_terms, "/")[:, 0], const_terms)
        den = np.where(frac, np.char.partition(const_terms, "/")[:, 2], "1")

        const_vals = num.astype(float) / den.astype(float)

        # ---------------------------------
        # REASSEMBLE
        # ---------------------------------
        split_sizes = np.array([len(t) for row in terms for t in row])
        idx = np.repeat(np.arange(len(split_sizes)), split_sizes)

        rx_sum = np.bincount(idx, weights=rx)
        ry_sum = np.bincount(idx, weights=ry)
        rz_sum = np.bincount(idx, weights=rz)
        t_sum = np.bincount(idx, weights=const_vals)

        n_ops = len(ops)

        rotations = (
            np.stack([rx_sum, ry_sum, rz_sum], axis=1).reshape(n_ops, 3, 3).astype(int)
        )
        translations = t_sum.reshape(n_ops, 3) % 1.0

        return rotations, translations

    def apply_symmetry(self, xyz: np.ndarray) -> np.ndarray:
        """take the symmetry operation of the space group and
        applies them to the provides xyz fractional coordinations
        must have xyz for every position shape = (n, 3)"""

        assert xyz.shape[-1] == 3

        rotations, translations = self.get_rotations_and_translations()

        pos = (
            np.einsum("sij,aj->sai", rotations, xyz, optimize=True)
            + translations[:, None, :]
        ) % 1.0

        return pos


class IntTabCryst(XRPDBaseModel):
    """The international crystallographuc tables in digital format
    The space groups are loaded from a yaml file

    Once intialised itc = IntTabCryst.load()
    space groups can be accessed such as pnma = itc["Pnma"]
    If space group requires a setting and none is provided
    this will assume it's in the first setting"""

    spacegroups: dict[str, SpaceGroup]

    def __getitem__(self, spacegroup_name: str | int) -> SpaceGroup:
        if isinstance(spacegroup_name, int):
            sg_name = get_spacegroup_symbol(spacegroup_name)
        else:
            sg_name = spacegroup_name.replace(" ", "")

        sg = self.spacegroups.get(sg_name)

        if sg is None:
            first_setting_sg_name = self._assume_first_setting(sg_name)
            # print("Assuming first setting for space group:", first_setting_sg_name)
            sg = self.spacegroups.get(first_setting_sg_name)

        if sg is None:
            formatted_sg_name = format_space_group_name(sg_name)
            # print(f"Trying formatted space group name: {formatted_sg_name}")
            sg = self.spacegroups.get(formatted_sg_name)

        if sg is None:
            formatted_sg_name = format_space_group_name(sg_name)
            first_setting_formatted_sg_name = self._assume_first_setting(
                formatted_sg_name
            )
            # print(
            #     f"Trying formatted space group name in first setting: {first_setting_formatted_sg_name}"  # noqa
            # )
            sg = self.spacegroups.get(first_setting_formatted_sg_name)

        if sg is None:
            print("Available space groups:", list(self.spacegroups.keys()))
            raise KeyError(f"{spacegroup_name} not in ITC")

        return sg

    def __len__(self):
        return len(self.spacegroups)

    def keys(self):
        return self.spacegroups.keys()

    def values(self):
        return self.spacegroups.values()

    def items(self):
        return self.spacegroups.items()

    def _assume_first_setting(self, spacegroup_name):
        return str(spacegroup_name) + ":1"

    @classmethod
    def load(cls, filepath: str | Path | None = None) -> "IntTabCryst":
        constants_folder = os.path.join(os.path.dirname(__file__), "constants")

        default_filepath = os.path.join(
            constants_folder, "international_tables_of_crystallography.yaml"
        )

        return cls.load_from_yaml(filepath or default_filepath)


@lru_cache
def get_symmetry_tables():
    itc = IntTabCryst.load()
    return itc


if __name__ == "__main__":
    _convert_symmetry_operations_yaml()

    itc = get_symmetry_tables()

    print(itc.keys())

    for i in range(1, 231):
        print(i, itc[i].spacegroup, itc[i].crystal_class)

    # spacegroup = itc["P212121"]
    # spacegroup = itc["P21"]

    # print(spacegroup)

    # positions = np.array([[0, 0, 0]])

    # spacegroup.apply_symmetry(positions)

    # print(itc["P42/mnm"])
