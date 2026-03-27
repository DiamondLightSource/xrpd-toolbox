import os
import re
from collections.abc import Collection, Sequence
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
from CifFile import ReadCif
from numba import njit
from pydantic import ConfigDict, Field, model_validator

from xrpd_toolbox.utils.constants import (
    ELEMENT_ATOMIC_NUMBER,
    get_spacegroup_number,
)
from xrpd_toolbox.utils.peaks import Peak
from xrpd_toolbox.utils.settings import SettingsBase
from xrpd_toolbox.utils.symmetry import get_symmetry_tables
from xrpd_toolbox.utils.unit_conversion import (
    beam_energy_to_wavelength,
)


@lru_cache
def load_xray_form_factors():
    folder = os.path.dirname(__file__)

    form_factor_filepath = os.path.join(folder, "constants", "atom_form_factors.csv")

    elements = np.genfromtxt(
        form_factor_filepath,
        skip_header=1,
        delimiter=None,
        dtype=None,
        usecols=[0],
    )

    xrff = np.genfromtxt(
        form_factor_filepath,
        skip_header=1,
        delimiter=None,
        dtype=None,
        usecols=range(1, 10),
    )

    elements = np.char.replace(elements, "val", "")

    return elements, xrff


def calculate_debye_waller_factor(b_iso: np.ndarray, q: np.ndarray):
    """q is q space in radians"""
    s = q / (4 * np.pi)
    dw = np.exp(-b_iso[:, None] * s[None, :] ** 2)
    return dw


def calculate_structure_factor(
    hkl: np.ndarray,
    positions: np.ndarray,
    occupancy: np.ndarray,
    b_iso: np.ndarray,
    elements: np.ndarray,
    q: np.ndarray,
) -> np.ndarray:
    """Compute complex structure factor F(hkl)."""

    phase = 2j * np.pi * (hkl @ positions.T)

    ff = calculate_form_factor(elements, q)

    dw = calculate_debye_waller_factor(b_iso, q)

    return np.sum(
        occupancy[None, :] * ff.T * dw.T * np.exp(phase),
        axis=1,
    )


def get_x_ray_form_factor_parameters(elements: Collection[str]) -> np.ndarray:
    """Place holder currently returns ones - should return array of xray form factors
    https://lampz.tugraz.at/~hadley/ss1/crystaldiffraction/atomicformfactors/formfactors.php
    """

    element_names, xrff = load_xray_form_factors()

    element_names = np.asarray(element_names)
    elements = np.asarray(elements)

    # Build lookup (fast and correct)
    lookup = {el: i for i, el in enumerate(element_names)}

    try:
        indices = np.array([lookup[e] for e in elements])
    except KeyError as exc:
        raise ValueError(f"{exc} not in form factor table") from exc

    return xrff[indices].astype(float)


def calculate_form_factor(elements: Collection[str], q: np.ndarray) -> np.ndarray:
    params = get_x_ray_form_factor_parameters(elements)
    # https://lampz.tugraz.at/~hadley/ss1/crystaldiffraction/atomicformfactors/formfactors.php

    a = params[:, [0, 2, 4, 6]]
    b = params[:, [1, 3, 5, 7]]
    c = params[:, 8]

    s = q / (4 * np.pi)

    s2 = s**2

    a = a[:, :, None]
    b = b[:, :, None]
    s2 = s2[None, None, :]

    ff = np.sum(a * np.exp(-b * s2), axis=1) + c[:, None]

    assert ff.shape[0] == len(elements)

    return ff


def hkl_laue_reduction(
    hkl: np.ndarray, rotations: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Laue reduction with positive-preference canonicalisation.
    """

    equiv = np.einsum("rij,nj->rni", rotations, hkl, optimize=True)
    equiv = np.rint(equiv).astype(int)  # (R,N,3)

    # → (N,R,3)
    equiv = np.transpose(equiv, (1, 0, 2))

    # enforce "first non-zero positive" rule
    signs = np.sign(equiv)

    # find first non-zero index along last axis
    first_nonzero_idx = np.argmax(equiv != 0, axis=2)

    # gather sign at that index
    first_sign = np.take_along_axis(
        signs,
        first_nonzero_idx[:, :, None],
        axis=2,
    ).squeeze(-1)

    # flip where negative
    flip_mask = first_sign < 0
    equiv = np.where(flip_mask[:, :, None], -equiv, equiv)

    # sort to get positive hkls
    order = np.lexsort((equiv[:, :, 2], equiv[:, :, 1], equiv[:, :, 0]))
    canonical = equiv[np.arange(len(equiv)), order[:, -1]]  # -1 not 0
    # --------------------------------------------------
    # Step 3: unique + multiplicity
    # --------------------------------------------------
    unique, counts = np.unique(canonical, axis=0, return_counts=True)

    return unique, counts


def q_vectors_from_hkl(
    hkl: np.ndarray,
    reciprocal_lattice_matrix: np.ndarray,
) -> np.ndarray:
    """Also know as G in single crystal"""

    return hkl @ reciprocal_lattice_matrix.T


def q_magnitude(q_vectors: np.ndarray) -> np.ndarray:
    """also known as G* in single crystal"""

    return np.linalg.norm(q_vectors, axis=1)


def lorentz_polarisation(
    theta: np.ndarray, polarisation: float = 0.0, azimuth: float | None = None
) -> np.ndarray:
    """
    Lorentz–polarisation factor for X-ray diffraction.

    Parameters
    ----------
    theta : np.ndarray
        Bragg angle in radians.
    polarisation : float, optional
        Degree of polarisation (0 = unpolarised, 1 = fully polarised).
    azimuth : float, optional
        Azimuthal angle (radians) between scattering and polarisation planes.
        If None, assumes powder-averaged synchrotron geometry.

    Returns
    -------
    np.ndarray
        Lorentz–polarisation factor.
    """

    two_theta = 2 * theta

    cos2theta = np.cos(two_theta) ** 2
    sin2theta = np.sin(two_theta)

    if azimuth is None:
        return (1.0 + cos2theta * (1.0 - 2.0 * polarisation)) / (sin2theta)
    else:
        return (
            (1.0 - polarisation) * (1.0 + cos2theta)
            + polarisation * (1.0 - cos2theta * np.cos(azimuth) ** 2)
        ) / (sin2theta * cos2theta)


def unit_cell_volume(
    a: float | np.ndarray,
    b: float | np.ndarray,
    c: float | np.ndarray,
    alpha: float | np.ndarray,
    beta: float | np.ndarray,
    gamma: float | np.ndarray,
    degrees: bool = True,
) -> np.ndarray:
    """
    Compute unit cell volume for arbitrary lattice.

    Parameters
    ----------
    a, b, c : float or np.ndarray
        Lattice parameters
    alpha, beta, gamma : float or np.ndarray
        Angles (degrees by default)
    degrees : bool
        If True, convert angles from degrees to radians

    Returns
    -------
    np.ndarray
        Unit cell volume
    """
    a = np.asarray(a)
    b = np.asarray(b)
    c = np.asarray(c)

    if degrees:
        alpha = np.radians(alpha)
        beta = np.radians(beta)
        gamma = np.radians(gamma)

    cos_a = np.cos(alpha)
    cos_b = np.cos(beta)
    cos_g = np.cos(gamma)

    volume = (
        a
        * b
        * c
        * np.sqrt(1 - cos_a**2 - cos_b**2 - cos_g**2 + 2 * cos_a * cos_b * cos_g)
    )

    return volume


####### calculate peak positions
def allowed_reflections_simple(hkl: np.ndarray, centering: str):
    """for a given array of hkl value it returns a mask of
    hkls which are forbidden by centering"""

    h, k, l = hkl[:, 0], hkl[:, 1], hkl[:, 2]  # noqa - crystallogrpahic nomenclature

    if centering == "P":
        # Primitive: no centering → all allowed
        return np.ones(len(hkl), dtype=bool)

    elif centering == "I":
        # Body-centred: (0,0,0) + (1/2,1/2,1/2)
        # h+k+l = 2n
        return (h + k + l) % 2 == 0

    elif centering == "F":
        # Face-centred: (0,0,0), (1/2,1/2,0), (1/2,0,1/2), (0,1/2,1/2)
        # all even or all odd
        even = (h % 2 == 0) & (k % 2 == 0) & (l % 2 == 0)
        odd = (h % 2 != 0) & (k % 2 != 0) & (l % 2 != 0)
        return even | odd

    elif centering == "C":
        # C-centred: centering on ab faces → (1/2,1/2,0)
        # h + k = 2n
        return (h + k) % 2 == 0

    elif centering == "A":
        # A-centred: centering on bc faces → (0,1/2,1/2)
        # k + l = 2n
        return (k + l) % 2 == 0

    elif centering == "B":
        # B-centred: centering on ac faces → (1/2,0,1/2)
        # h + l = 2n
        return (h + l) % 2 == 0

    elif centering == "R":
        # Rhombohedral (hex setting)
        # -h + k + l = 3n
        # Rhombohedral - my fave - magnets baby
        return (-h + k + l) % 3 == 0

    else:
        raise ValueError(f"Unknown centering: {centering}")


def generate_h_k_l(hkl_max: Sequence[int]) -> np.ndarray:
    """generate hkl and filter forbidden beam centre hkl = 0 0 0"""

    hmax = hkl_max[0]
    kmax = hkl_max[1]
    lmax = hkl_max[2]

    h_range = np.arange(-hmax, hmax + 1)
    k_range = np.arange(-kmax, kmax + 1)
    l_range = np.arange(-lmax, lmax + 1)

    hkl = np.array(np.meshgrid(h_range, k_range, l_range)).reshape(3, -1).T
    return hkl[np.any(hkl != 0, axis=1)]


def generate_hkl(reciprocal_lattice_matrix: np.ndarray, q_max: float) -> np.ndarray:
    norms = np.linalg.norm(reciprocal_lattice_matrix, axis=0)

    hmax = int(np.ceil(q_max / norms[0]))
    kmax = int(np.ceil(q_max / norms[1]))
    lmax = int(np.ceil(q_max / norms[2]))

    hkl = generate_h_k_l([hmax, kmax, lmax])

    q_vec = hkl @ reciprocal_lattice_matrix.T
    q = np.linalg.norm(q_vec, axis=1)

    return hkl[q <= q_max]


def reciprocal_lattice_matrix(
    a: float, b: float, c: float, alpha: float, beta: float, gamma: float
) -> np.ndarray:
    a_vec = np.array([a, 0.0, 0.0])
    b_vec = np.array([b * np.cos(gamma), b * np.sin(gamma), 0.0])

    cx = c * np.cos(beta)
    cy = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
    cz = np.sqrt(max(c**2 - cx**2 - cy**2, 0.0))
    c_vec = np.array([cx, cy, cz])

    volume = np.dot(a_vec, np.cross(b_vec, c_vec))

    a_star = 2 * np.pi * np.cross(b_vec, c_vec) / volume
    b_star = 2 * np.pi * np.cross(c_vec, a_vec) / volume
    c_star = 2 * np.pi * np.cross(a_vec, b_vec) / volume

    return np.vstack([a_star, b_star, c_star]).T


def allowed_reflections(
    hkl: np.ndarray,
    rotations: np.ndarray,
    translations: np.ndarray,
    tol: float = 1e-8,
) -> np.ndarray:
    """
    Correct systematic absence condition.
    """

    # Apply rotations
    hkl_rot = np.einsum("rij,nj->rni", rotations, hkl, optimize=True)

    # Check invariance modulo integer lattice
    delta = hkl_rot - hkl[None]
    invariant = np.all(np.isclose(delta, np.rint(delta), atol=tol), axis=2)

    # Phase from translations
    phase = 2 * np.pi * (hkl @ translations.T)

    # Sum ONLY invariant ops
    real = (np.cos(phase).T * invariant).sum(axis=0)
    imag = (np.sin(phase).T * invariant).sum(axis=0)

    extinct = np.isclose(real, 0.0, atol=tol) & np.isclose(imag, 0.0, atol=tol)

    return ~extinct


def d_spacing(hkl: np.ndarray, g_star: np.ndarray) -> np.ndarray:
    """calculate d spacing from reciprocal metric tensor and hkls"""

    g_hkl = np.einsum("ni,ij,nj->n", hkl, g_star, hkl)
    return 1.0 / np.sqrt(g_hkl)


class Atom(SettingsBase):
    """This describes an atom"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    label: str  # elemnt label ie Si1 Si2
    element: str  # element name eg Si
    xyz: np.ndarray | list  # fractional coorindates of xyz
    b_iso: float
    occupancy: float = 1.0

    @property
    def x(self):
        return self.xyz[0]

    @property
    def y(self):
        return self.xyz[1]

    @property
    def z(self):
        return self.xyz[2]


def apply_symmetry_operations_to_atoms(
    asymmetric_atoms: list[Atom], rotations: np.ndarray, translations: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Expand asymmetric unit → full unit cell (vectorised)."""

    base_pos = np.array([a.xyz for a in asymmetric_atoms])
    base_occ = np.array([a.occupancy for a in asymmetric_atoms])
    base_b = np.array([a.b_iso for a in asymmetric_atoms])
    base_el = np.array([a.element for a in asymmetric_atoms])

    positions = (
        np.einsum("sij,aj->sai", rotations, base_pos, optimize=True)
        + translations[:, None, :]
    ) % 1.0

    positions = positions.reshape(-1, 3)
    occupancies = np.tile(base_occ, len(rotations))
    b_iso = np.tile(base_b, len(rotations))
    elements = np.tile(base_el, len(rotations))

    positions = positions % 1.0
    key = np.round(positions * 1e6).astype(int)
    _, idx = np.unique(key, axis=0, return_index=True)

    positions = positions[idx]
    occupancies = occupancies[idx]
    b_iso = b_iso[idx]
    elements = elements[idx]

    return positions, occupancies, b_iso, elements


# pydantic models to store the data


class Lattice(SettingsBase):
    """This decribes the assymetric unit cell lattice -
    a, b, c refer to the length of the unit cell in A
    alpha, beta, gamma are the angles of the unit cell in degreee"""

    a: float
    b: float
    c: float
    alpha: float  # in degrees
    beta: float  # in degrees
    gamma: float  # in degrees

    @property
    def alpha_radians(self):
        return np.deg2rad(self.alpha)

    @property
    def beta_radians(self):
        return np.deg2rad(self.beta)

    @property
    def gamma_radians(self):
        return np.deg2rad(self.gamma)


class Radiation(SettingsBase):
    """This defines the type of radiation that is diffracting"""

    radiation: Literal["synchotron-x-ray", "tof-neutron", "cw-neutron", "lab-xray"]


class Structure(SettingsBase):
    """This should contain everything needed to calculate the structure factor,
    peak intensities and peak position of the diffraction peeaks"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    lattice: Lattice
    atoms: list[Atom]
    spacegroup_symbol: str = "P1"
    symmetry_operations: np.ndarray = Field(default=np.array([]))

    @classmethod
    def load_from_cif(cls, cif_filepath: str | Path) -> "Structure":
        return read_cif(cif_filepath)

    @cached_property
    def spacegroup(self):
        sg = self.spacegroup_symbol
        sg = sg[0].upper() + sg[1:].lower()  # upper case first letter
        # sg = re.sub(r"\(.*?\)", "", sg)  # remove "(...)"
        sg = re.sub(r"\s+", "", sg)  # remove all whitespace
        sg = re.sub(r"^([A-Za-z]+)3([A-Za-z]+)$", r"\1-3\2", sg)
        # sg = re.sub(r"([2346])1", r"\1_1", sg)  # 21 → 2_1 etc.
        # sg = re.sub(r"^([A-Za-z])1+", r"\1", sg)  # drop leading 1's
        # sg = re.sub(r"1$", "", sg)  # drop trailing 1
        return sg

    @cached_property
    def spacegroup_number(self) -> int:
        return get_spacegroup_number(self.spacegroup)

    @cached_property
    def element_array(self):
        element_array = np.array([a.element for a in self.atoms])  # (N_atoms, 3)
        return element_array

    @property
    def positions(self):
        """Turns the list of Atoms into a vectorised numpy array
        for fast computation later"""

        positions = np.stack([a.xyz for a in self.atoms])  # (N_atoms, 3)
        return positions

    @property
    def occupancies(self):
        occ_array = np.array([a.occupancy for a in self.atoms])  # (N_atoms,)
        return occ_array

    @property
    def b_iso_array(self):
        b_iso_array = np.array([a.b_iso for a in self.atoms])  # (N_atoms,)
        return b_iso_array

    @cached_property
    def atomic_numbers(self):
        atomic_numbers = np.array(
            [ELEMENT_ATOMIC_NUMBER[a.element] for a in self.atoms]
        )  # (N_atoms,)
        return atomic_numbers

    @cached_property
    def centering(self):
        return self.spacegroup.split()[0]

    def reciprocal_lattice_matrix(self):
        return reciprocal_lattice_matrix(
            a=self.lattice.a,
            b=self.lattice.b,
            c=self.lattice.c,
            alpha=self.lattice.alpha_radians,
            beta=self.lattice.beta_radians,
            gamma=self.lattice.gamma_radians,
        )

    @property
    def volume(self):
        return unit_cell_volume(
            a=self.lattice.a,
            b=self.lattice.b,
            c=self.lattice.c,
            alpha=self.lattice.alpha,
            beta=self.lattice.beta,
            gamma=self.lattice.gamma,
            degrees=True,
        )

    def get_symmetry_operations(self):
        itc = get_symmetry_tables()
        sg = itc[self.spacegroup]
        return sg.get_rotations_and_translations()

    def calculate_peaks(
        self,
        wavelength: float | int = 1.5406,
        mode: Literal["powder", "single_crystal"] = "powder",
        radiation: Literal["x-ray", "neutron"] = "x-ray",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """calculates peaks"""

        reciprocal_lattice_matrix = self.reciprocal_lattice_matrix()
        rotations, translations = self.get_symmetry_operations()

        q_max = 4 * np.pi / wavelength

        # generate hkls
        hkl = generate_hkl(reciprocal_lattice_matrix, q_max=q_max)
        # remove ssystematic absences
        mask = allowed_reflections(hkl, rotations, translations)
        hkl = hkl[mask]

        if mode == "powder":
            hkl, multiplicity = hkl_laue_reduction(hkl, rotations)
            # some pxrd refs are equal
        else:
            hkl = hkl
            multiplicity = np.ones(len(hkl))

        q_vec = q_vectors_from_hkl(hkl, reciprocal_lattice_matrix)
        q = q_magnitude(q_vec)

        sort_index = np.argsort(q)
        q = q[sort_index]
        q_vec = q_vec[sort_index]
        hkl = hkl[sort_index]
        multiplicity = multiplicity[sort_index]

        positions, occupancies, b_iso, elements = apply_symmetry_operations_to_atoms(
            self.atoms, rotations, translations
        )

        structure_factor = calculate_structure_factor(
            hkl=hkl,
            positions=positions,
            occupancy=occupancies,
            b_iso=b_iso,
            elements=elements,
            q=q,
        )

        intensity = np.abs(structure_factor) ** 2  # |F|**2
        # intensity = intensity / (self.volume**2)

        theta = np.arcsin(q * wavelength / (4 * np.pi))
        two_theta_degrees = np.degrees(2 * theta)

        if mode == "powder":
            lp_factor = lorentz_polarisation(theta, 0.9)
            intensity = intensity * lp_factor

            return hkl, two_theta_degrees, intensity

        else:
            return hkl, two_theta_degrees, intensity


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


def read_cif(cif_filepath: str | Path, block_number: int = 0) -> Structure:
    """reads data from a cif and returns
    list of Atom classes and a unit cell class"""

    block = open_cif(cif_filepath=cif_filepath, block_number=block_number)

    x = np.array(block["_atom_site_fract_x"], dtype=float)
    y = np.array(block["_atom_site_fract_y"], dtype=float)
    z = np.array(block["_atom_site_fract_z"], dtype=float)

    atom_labels = np.array(block["_atom_site_label"])

    if "_atom_site_type_symbol" in block:
        elements = np.array(block["_atom_site_type_symbol"])
    else:
        elements = atom_labels.copy()

    occ = (
        np.array(block["_atom_site_occupancy"], dtype=float)
        if "_atom_site_occupancy" in block
        else np.ones(len(x))
    )

    # --- B_iso or U_iso handling ---
    if "_atom_site_B_iso_or_equiv" in block:
        b_iso = np.array(block["_atom_site_B_iso_or_equiv"], dtype=float)

    elif "_atom_site_U_iso_or_equiv" in block:
        u_iso = np.array(block["_atom_site_U_iso_or_equiv"], dtype=float)
        b_iso = 8 * np.pi**2 * u_iso
    else:
        b_iso = np.zeros(len(x))  # fallback

    try:
        spacegroup_symbol = block["_symmetry_space_group_name_H-M"]
    except Exception:
        print("_symmetry_space_group_name_H-M not in cif")
        spacegroup_symbol = "P1"

    try:
        synmmetry_operations = np.array(block["_space_group_symop_operation_xyz"])
    except Exception:
        print("No symmetry operation in cif")
        synmmetry_operations = np.array([])

    atoms = [
        Atom(
            label=atom_labels[i],
            element=elements[i].capitalize(),
            xyz=np.array([x[i], y[i], z[i]], dtype=float),
            b_iso=float(b_iso[i]),
            occupancy=float(occ[i]),
        )
        for i in range(len(x))
    ]

    lattice = Lattice(
        a=float(block["_cell_length_a"]),
        b=float(block["_cell_length_b"]),
        c=float(block["_cell_length_c"]),
        alpha=float(block["_cell_angle_alpha"]),
        beta=float(block["_cell_angle_beta"]),
        gamma=float(block["_cell_angle_gamma"]),
    )

    structure = Structure(
        lattice=lattice,
        atoms=atoms,
        spacegroup_symbol=spacegroup_symbol,
        symmetry_operations=synmmetry_operations,
    )

    return structure


@njit(parallel=True)
def calculate_profile(
    x: np.ndarray,
    peaks: Collection[Peak],
    background: int | float | np.ndarray,
    phase_scale: int | float = 1,
    wdt: int | float = 5,
):
    """wdt (range) of calculated profile of a single Bragg reflection in units of FWHM
    (typically 4 for Gaussian and 20-30 for Lorentzian, 4-5 for TOF).

    peaks: list of class: Peak which contain (cen, amp, fwhm)

    background: scalar or array, if array must be same shape as x
    """

    if isinstance(background, np.ndarray):
        assert len(x) == len(background)

    intensity = np.zeros_like(x)

    for peak in peaks:
        assert peak.background == 0

        start_idx = np.searchsorted(x, peak.centre - (wdt * peak.fwhm))
        end_idx = np.searchsorted(x, peak.centre + (wdt * peak.fwhm), side="right")

        xi = x[start_idx:end_idx]
        peak_intensity = peak.calculate(xi)
        intensity[start_idx:end_idx] += peak_intensity

    intensity = (intensity * phase_scale) + background

    return intensity


##### more complex pydantic models to do whol profiles


class Profile(SettingsBase):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    x: np.ndarray
    peaks: Collection[Peak]
    background: np.ndarray
    phase_scale: int | float
    wdt: int | float = 5

    @model_validator(mode="after")
    def validate_parameters(self):
        assert len(self.x) == len(self.background)
        return self

    def calculate_profile(self):
        return calculate_profile(self.x, self.peaks, self.background, self.wdt)


class ProfileRefinement(SettingsBase):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    x_data: np.ndarray
    y_data: np.ndarray
    cif_filepath: str | Path | None
    wavelength: int | float = 1.5406  # CuKa1
    unit: Literal["tth"] = "tth"
    radiation: str = "x-ray"
    refinement: str = "Pawley"
    structure: Structure | list[Structure] | None = None

    def load_cif(self):
        assert self.cif_filepath is not None

        self.structure = read_cif(self.cif_filepath)

    def calculate_peaks(self):
        if self.structure is not None:
            if isinstance(self.structure, Structure):
                return self.structure.calculate_peaks()
            elif isinstance(self.structure, list):
                return [f.calculate_peaks() for f in self.structure]
            else:
                raise Exception("Unknown structure")


if __name__ == "__main__":
    # print(spglib.get_symmetry_from_database(27))

    from xrpd_toolbox.utils.utils import normalise

    cif_filepath = "/workspaces/XRPD-Toolbox/cifs/si.cif"

    # elements = ["Si", "Si", "Si"]

    # atom_ff = get_x_ray_form_factor_parameters(elements)

    # q_space = np.linspace(0.01, 10, 10000)

    # ff = calculate_form_factor(elements, q_space)

    beam_energy = 15
    wavelength = beam_energy_to_wavelength(beam_energy)

    si_structure = Structure.load_from_cif(cif_filepath)

    print(si_structure.volume)

    print(si_structure.spacegroup)
    print(si_structure.spacegroup_number)

    hkl, two_theta_degrees, intensity = si_structure.calculate_peaks(
        wavelength=wavelength
    )

    intensity = normalise(intensity) * 4863.94

    for miller, tth, i in zip(hkl, two_theta_degrees, intensity, strict=True):
        print(f"{miller} {tth:.2f} {i:.2f}")

        if tth > 75:
            break
