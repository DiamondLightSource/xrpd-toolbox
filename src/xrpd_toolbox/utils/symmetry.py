import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from yaml import safe_load

from xrpd_toolbox.utils.settings import SettingsBase

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


def symmetry_ops_to_translations_and_rotations(ops):
    ops = np.asarray(ops, dtype=str)

    # Split into (N,3)
    coords = np.char.split(ops, ",")
    coords = np.array(coords.tolist(), dtype="U50")

    # ---------------------------------
    # NORMALISE: replace "-" with "+-" so we can split on "+"
    # ---------------------------------
    coords = np.char.replace(coords, "-", "+-")

    # remove leading "+"
    coords = np.char.lstrip(coords, "+")

    # split into terms
    terms = np.char.split(coords, "+")
    terms = np.array(terms.tolist(), dtype=object)  # (N,3,list)

    # ---------------------------------
    # Flatten terms for vectorisation
    # ---------------------------------
    flat_terms = np.concatenate(terms.reshape(-1))

    # Identify axis contributions
    is_x = np.char.find(flat_terms, "x") >= 0
    is_y = np.char.find(flat_terms, "y") >= 0
    is_z = np.char.find(flat_terms, "z") >= 0

    # Coefficients
    coeff = np.ones_like(flat_terms, dtype=float)

    coeff[np.char.startswith(flat_terms, "-")] = -1

    # Build rotation contributions
    rx = np.where(is_x, coeff, 0)
    ry = np.where(is_y, coeff, 0)
    rz = np.where(is_z, coeff, 0)

    # Translation terms (no x,y,z)
    is_const = ~(is_x | is_y | is_z)
    const_terms = np.where(is_const, flat_terms, "0")

    # Convert constants safely
    const_terms = np.where(const_terms == "", "0", const_terms)

    # Handle fractions
    frac_mask = np.char.find(const_terms, "/") >= 0

    num = np.where(frac_mask, np.char.partition(const_terms, "/")[:, 0], const_terms)
    den = np.where(frac_mask, np.char.partition(const_terms, "/")[:, 2], "1")

    const_vals = num.astype(float) / den.astype(float)

    # ---------------------------------
    # Reassemble per symmetry op
    # ---------------------------------
    n_ops = len(ops)

    # Each coord has variable number of terms → need grouping
    # lengths = np.array([len(t) for t in flat_terms.reshape(-1, 1)]).flatten()

    # Instead: rebuild via cumulative sums
    split_sizes = np.array([len(t) for t in terms.reshape(-1)])

    idx = np.repeat(np.arange(len(split_sizes)), split_sizes)

    # Sum contributions
    rx_sum = np.bincount(idx, weights=rx)
    ry_sum = np.bincount(idx, weights=ry)
    rz_sum = np.bincount(idx, weights=rz)
    t_sum = np.bincount(idx, weights=const_vals)

    # reshape back to (N,3)
    rotations = np.stack([rx_sum, ry_sum, rz_sum], axis=1).reshape(n_ops, 3, 3)
    translations = t_sum.reshape(n_ops, 3) % 1.0

    rotations = rotations.astype(int)

    return rotations, translations


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

        new_symmetry_operations = {}

        for sg in symmetry_operations:
            sg_name = sg["universal_h_m"].replace(" ", "")

            sg["symops"] = frac_to_decimal_string(sg["symops"])
            sg["ncsym"] = frac_to_decimal_string(sg["ncsym"])

            new_symmetry_operations[sg_name] = sg

    itc = IntTabCryst(spacegroups=new_symmetry_operations)

    itc.save_to_yaml(
        "/workspaces/XRPD-Toolbox/src/xrpd_toolbox/utils/constants/international_tables_of_crystallography.yaml"
    )

    sg = itc["P 1 21"]

    print(sg)


class SpaceGroup(SettingsBase):
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

    def get_rotations_and_translations(self):
        return symmetry_ops_to_translations_and_rotations(self.symops)

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


class IntTabCryst(SettingsBase):
    """The international crystallographuc tables in digital format
    The space groups are loaded from a yaml file

    Once intialised itc = IntTabCryst.load()
    space groups can be accessed such as pnma = itc["Pnma"]
    If space group requires a setting and none is provided
    this will assume it's in the first setting"""

    spacegroups: dict[str, SpaceGroup]

    def __getitem__(self, spacegroup_name: str) -> SpaceGroup:
        sg_name = spacegroup_name.replace(" ", "")

        try:
            sg = self.spacegroups[sg_name]
        except KeyError:
            try:
                sg = self.spacegroups[self._assume_first_setting(sg_name)]
            except Exception:
                sg = None

        if sg is None:
            raise KeyError(f"{spacegroup_name} not in ITC")

        return sg

    def __len__(self):
        return len(self.spacegroups)

    def _assume_first_setting(self, spacegroup_name):
        return spacegroup_name + ":1"

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

    pnma = itc["Fm-3m"]

    positions = np.array([[0, 0, 0]])

    pnma.apply_symmetry(positions)
