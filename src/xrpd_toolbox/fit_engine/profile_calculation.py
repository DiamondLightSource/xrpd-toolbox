from collections.abc import Collection, Sequence
from functools import cached_property
from pathlib import Path
from typing import Literal, TypeAlias

import matplotlib.pyplot as plt
import numpy as np

# from numba import njit
from pydantic import Field, computed_field, model_validator

from xrpd_toolbox.core import (
    DataType,
    Parameter,
    ScatteringData,
    SerialisableNDArray,
)
from xrpd_toolbox.fit_engine.atom import Atoms
from xrpd_toolbox.fit_engine.background import (
    Background,
    BackgroundType,
)
from xrpd_toolbox.fit_engine.constants import (
    ELEMENT_ATOMIC_NUMBER,
)
from xrpd_toolbox.fit_engine.fit_statistics import calculate_chi_squared
from xrpd_toolbox.fit_engine.fitting_core import (
    Model,
    RefinementBaseModel,
    refine_model,
)
from xrpd_toolbox.fit_engine.form_factors import X_RAY_FORM_FACTORS
from xrpd_toolbox.fit_engine.lattice import (
    Lattice,
    crystal_lattice_factory,
)
from xrpd_toolbox.fit_engine.peak_shape_functions import (
    FCJPseudoVoigt,
    IntrumentResolutionFunction,
)
from xrpd_toolbox.fit_engine.peaks import (
    PeakType,
    calculate_profile,
    peak_factory,
)
from xrpd_toolbox.fit_engine.symmetry import (
    SpaceGroup,
    format_space_group_name,
    get_symmetry_tables,
)
from xrpd_toolbox.plotting import FittedDataPlot
from xrpd_toolbox.utils.cif_reader import read_cif
from xrpd_toolbox.utils.unit_conversion import (
    beam_energy_to_wavelength,
    q_space_to_s,
    q_space_to_theta,
)

CrystalType: TypeAlias = Literal["powder", "single-crystal"]


ITC_TABLES = get_symmetry_tables()


def merge_peaks(
    two_theta, intensity, hkl, tol=1e-5
) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Vectorised merging of reflections with same 2θ (within tolerance).
    Sums intensities (equivalent to multiplicity handling).
    """

    # Sort by 2θ
    sort_idx = np.argsort(two_theta)
    tth = two_theta[sort_idx]
    inten = intensity[sort_idx]
    hkl_sorted = hkl[sort_idx]

    # Find group boundaries
    dt = np.diff(tth)
    new_group = np.concatenate(([True], dt > tol))

    # Indices where groups start
    group_starts = np.flatnonzero(new_group)

    # Sum intensities per group
    merged_intensity = np.add.reduceat(inten, group_starts)

    # Representative 2θ (take first in each group)
    merged_tth = tth[group_starts]

    # HKL groups (still needs object handling)
    group_indices = np.split(np.arange(len(tth)), group_starts[1:])
    merged_hkl = [hkl_sorted[idx] for idx in group_indices]

    return merged_tth, merged_intensity, merged_hkl


# @njit()
def absorption_correction(
    theta: np.ndarray,
    mu: float,
    radius: float,
) -> np.ndarray:
    """
    Debye–Scherrer absorption correction (capillary geometry).

    Parameters
    ----------
    theta : np.ndarray
        Bragg angle in radians
    mu : float
        Linear absorption coefficient (1 / length)
    radius : float
        Capillary radius (same length units as mu^-1)

    Returns
    -------
    np.ndarray
        Absorption correction factor
    """
    x = 2.0 * mu * radius / np.sin(theta)
    return (1.0 - np.exp(-x)) / x


def calculate_form_factor(elements: Collection[str], s: np.ndarray) -> np.ndarray:
    params = np.asarray([X_RAY_FORM_FACTORS[el] for el in elements])

    if params.shape[0] != len(elements):
        raise ValueError("Element form factor parameters not found for all elements")

    a = params[:, 0:5]  # (n_atoms, 5)
    c = params[:, 5]  # (n_atoms,)
    b = params[:, 6:11]  # (n_atoms, 5)

    s2 = s**2  # (s,)

    exp_term = np.exp(-s2[:, None, None] * b[None, :, :])

    ff = np.sum(a[None, :, :] * exp_term, axis=2) + c[None, :]

    return ff  # (n_atoms, s)


# @njit()
def calculate_debye_waller_factor(b_iso: np.ndarray, s: np.ndarray):
    """s is scattering vector in radians"""
    # (n_hkl, n_atoms)
    return np.exp(-np.outer(s**2, b_iso))


def calculate_structure_factor(
    hkl: np.ndarray,
    positions: np.ndarray,
    occupancy: np.ndarray,
    b_iso: np.ndarray,
    elements: np.ndarray,
    q: np.ndarray,
) -> np.ndarray:
    phase = 2j * np.pi * (hkl @ positions.T)  # (n_hkl, n_atoms)

    s = q_space_to_s(q)
    ff = calculate_form_factor(elements, s)  # (n_hkl, n_atoms)
    dw = calculate_debye_waller_factor(b_iso, s)  # (n_hkl, n_atoms)

    f_hkl = np.sum(
        occupancy * ff * dw * np.exp(phase),
        axis=1,
    )

    ###this is an alternate way which may be faster for larger arrays
    # f_hkl = np.einsum(
    #     "a,ha,ha,ha->h",
    #     occupancy,
    #     ff,
    #     dw,
    #     np.exp(phase),
    #     optimize=True,
    # )

    return f_hkl


# @njit
# @timeit
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
    equiv = np.abs(equiv)

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

    return hkl @ reciprocal_lattice_matrix


def q_magnitude(q_vectors: np.ndarray) -> np.ndarray:
    """also known as G* in single crystal"""

    return np.linalg.norm(q_vectors, axis=1)


# @njit
def lorentz_polarisation(
    theta: np.ndarray,
    polarisation: float | int = 0.0,
    azimuth: float | int | None = None,
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

    return (1 + np.cos(2 * theta) ** 2) / (np.sin(theta) ** 2 * np.cos(theta))

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


# @njit
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


def lattice_to_metric(a, b, c, alpha, beta, gamma):
    alpha, beta, gamma = np.radians([alpha, beta, gamma])

    g = np.array(
        [
            [a * a, a * b * np.cos(gamma), a * c * np.cos(beta)],
            [a * b * np.cos(gamma), b * b, b * c * np.cos(alpha)],
            [a * c * np.cos(beta), b * c * np.cos(alpha), c * c],
        ]
    )

    g_star = np.linalg.inv(g)
    return g, g_star


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


def generate_h_k_l(hkl_max: np.ndarray) -> np.ndarray:
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

    hkl = generate_h_k_l(np.array([hmax, kmax, lmax]))

    q_vec = hkl @ reciprocal_lattice_matrix
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

    return np.vstack([a_star, b_star, c_star])


def allowed_reflections(
    hkl: np.ndarray,
    rotations: np.ndarray,
    translations: np.ndarray,
    tol: float = 1e-8,
) -> np.ndarray:
    hkl_rot = np.einsum("rij,nj->rni", rotations, hkl, optimize=True)
    delta = hkl_rot - hkl[None]
    invariant = np.all(np.isclose(delta, np.rint(delta), atol=tol), axis=2)
    phase = 2 * np.pi * (hkl @ translations.T)
    real = (np.cos(phase).T * invariant).sum(axis=0)
    imag = (np.sin(phase).T * invariant).sum(axis=0)
    extinct = np.isclose(real, 0.0, atol=tol) & np.isclose(imag, 0.0, atol=tol)
    return ~extinct


def d_spacing(hkl: np.ndarray, g_star: np.ndarray) -> np.ndarray:
    """calculate d spacing from reciprocal metric tensor and hkls"""

    g_hkl = np.einsum("mi,ij,nj->n", hkl, g_star, hkl, optimize=True)
    return 1.0 / np.sqrt(g_hkl)


def plot_form_factors(elements: Sequence[str], q_space: np.ndarray | None = None):
    if q_space is None:
        q_space = np.linspace(0.01, 25, 1000)

    scatter_vec = q_space / (4 * np.pi)

    form_factors = calculate_form_factor(elements, scatter_vec).T

    plt.figure(figsize=(16, 10))

    for element, ff in zip(elements, form_factors, strict=True):
        plt.plot(q_space, ff, label=element)
    plt.legend()
    plt.show()


def apply_symmetry_operations_to_atoms(
    asymmetric_atoms: Atoms, rotations: np.ndarray, translations: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Expand asymmetric unit → full unit cell (vectorised)."""

    # base_pos = np.array([a.xyz for a in asymmetric_atoms])
    # base_occ = np.array([a.occupancy for a in asymmetric_atoms])
    # base_b = np.array([a.b_iso for a in asymmetric_atoms])
    # base_el = np.array([a.element for a in asymmetric_atoms])

    positions = (
        np.einsum("sij,aj->sai", rotations, asymmetric_atoms.xyz, optimize=True)
        + translations[:, None, :]
    )

    positions = positions % 1.0
    positions = np.round(positions, 10)

    positions = positions.reshape(-1, 3)
    occupancies = np.tile(asymmetric_atoms.occupancies, len(rotations))
    b_iso = np.tile(asymmetric_atoms.b_iso, len(rotations))
    elements = np.tile(asymmetric_atoms.elements, len(rotations))

    # key = np.round(positions * 1e8).astype(int)
    _, idx = np.unique(positions, axis=0, return_index=True)

    positions = positions[idx]
    occupancies = occupancies[idx]
    b_iso = b_iso[idx]
    elements = elements[idx]

    return positions, occupancies, b_iso, elements


# pydantic models to store the data


class Radiation(RefinementBaseModel):
    """This defines the type of radiation that is diffracting"""

    radiation: DataType
    wavelength: float | int | None = None
    energy: float | int | None = None

    @model_validator(mode="after")
    def check_wavelength_or_energy(self):
        if (
            self.radiation in ["xray", "lab-xray", "cw-neutron"]
            and self.energy is not None
            and self.wavelength is None
        ):
            self.wavelength = beam_energy_to_wavelength(self.energy)

        return self


class Structure(RefinementBaseModel):
    """This should contain everything needed to calculate the structure factor,
    peak intensities and peak position of the diffraction peeaks
    lattice: Lattice
    atoms: Atoms
    spacegroup_symbol: str = "P1"
    symmetry_operations: np.ndarray
    """

    spacegroup: str = "P1"
    lattice: Lattice
    atoms: Atoms
    name: str | None = None
    source: str | None = None
    # symmetry_operations: npt.NDArray[np.str_] | None = None

    @classmethod
    def load_from_cif(cls, cif_filepath: str | Path) -> "Structure":
        return cif_to_structure(cif_filepath)

    @cached_property
    def spacegroup_class(self) -> SpaceGroup:
        sg = ITC_TABLES[self.spacegroup]
        return sg

    @cached_property
    def spacegroup_number(self) -> int:
        sg = ITC_TABLES[self.spacegroup]
        return int(sg["number"])

    # @cached_property
    @property
    def element_array(self):
        # element_array = np.array([a.element for a in self.atoms])  # (N_atoms, 3)
        return self.atoms.elements

    @property
    def positions(self):
        """Turns the list of Atoms into a vectorised numpy array
        for fast computation later"""
        # positions = np.stack([a.xyz for a in self.atoms])  # (N_atoms, 3)
        return self.atoms.xyz

    @property
    def occupancies(self):
        # occ_array = np.array([a.occupancy for a in self.atoms])  # (N_atoms,)
        return self.atoms.occupancies

    @property
    def b_iso_array(self):
        # b_iso_array = np.array([a.b_iso for a in self.atoms])  # (N_atoms,)
        return self.atoms.b_iso

    # @cached_property
    @property
    def atomic_numbers(self):
        atomic_numbers = np.array(
            [ELEMENT_ATOMIC_NUMBER[el] for el in self.atoms.elements]
        )  # (N_atoms,)
        return atomic_numbers

    @cached_property
    def centering(self):
        return list(self.spacegroup)[0]

    def reciprocal_lattice_matrix(self):
        return reciprocal_lattice_matrix(
            a=float(self.lattice.a),
            b=float(self.lattice.b),
            c=float(self.lattice.c),
            alpha=float(self.lattice.alpha_radians),
            beta=float(self.lattice.beta_radians),
            gamma=float(self.lattice.gamma_radians),
        )

    @property
    def volume(self):
        return unit_cell_volume(
            a=float(self.lattice.a),
            b=float(self.lattice.b),
            c=float(self.lattice.c),
            alpha=float(self.lattice.alpha_radians),
            beta=float(self.lattice.beta_radians),
            gamma=float(self.lattice.gamma_radians),
            degrees=True,
        )

    # TODO: Make this read from the cif file if provided
    def get_symmetry_operations(self):
        # if self.symmetry_operations is None:
        sg = ITC_TABLES[self.spacegroup]
        sym_ops = sg.get_rotations_and_translations()

        # else:
        #     sym_ops = self.symmetry_operations

        return sym_ops

    def generate_structure_hkls(
        self,
        reciprocal_lattice_matrix: np.ndarray,
        rotations: np.ndarray,
        translations: np.ndarray,
        q_max: float,
    ) -> np.ndarray:
        """HKL generator with systematic absence filtering
        based on symmetry operations."""

        # generate hkls
        hkl = generate_hkl(reciprocal_lattice_matrix, q_max=q_max)
        # remove systematic absences

        mask = allowed_reflections(hkl, rotations, translations)
        return hkl[mask]

    # @timeit
    def calculate_reflections(
        self,
        wavelength: float | int = 1.5406,
        mode: CrystalType = "powder",
        radiation: DataType = "xray",
        multiplicity_method: bool = False,
    ) -> tuple[list, np.ndarray, np.ndarray, np.ndarray]:
        """calculates peaks"""

        reciprocal_lattice_matrix = self.reciprocal_lattice_matrix()
        rotations, translations = self.get_symmetry_operations()

        # g, g_star = lattice_to_metric(
        #     self.lattice.a,
        #     self.lattice.b,
        #     self.lattice.c,
        #     self.lattice.alpha,
        #     self.lattice.beta,
        #     self.lattice.gamma,
        # )

        q_max = 4 * np.pi / wavelength

        hkl = self.generate_structure_hkls(
            reciprocal_lattice_matrix, rotations, translations, q_max
        )

        # hkl, multiplicity = hkl_laue_reduction(hkl, rotations)

        # mask = allowed_reflections_simple(hkl, self.centering)
        # hkl = hkl[mask]

        # d_space = d_spacing(hkl, g_star)
        # Bragg angle
        # theta = np.arcsin(wavelength / (2 * d_space))
        # s = np.sin(theta) / wavelength

        if mode == "powder":
            if multiplicity_method:
                hkl, multiplicity = hkl_laue_reduction(hkl, rotations)
                # some pxrd refs are equal
            else:
                multiplicity = np.ones(len(hkl))
        else:
            multiplicity = np.ones(len(hkl))

        q_vec = q_vectors_from_hkl(hkl, reciprocal_lattice_matrix)
        q = q_magnitude(q_vec)

        # print(q.shape, hkl.shape)
        assert q.shape[0] == hkl.shape[0]

        sort_index = np.argsort(q)
        q = q[sort_index]
        q_vec = q_vec[sort_index]
        hkl = hkl[sort_index]
        multiplicity = multiplicity[sort_index]

        theta = q_space_to_theta(q, wavelength=wavelength)  # in radians still
        two_theta = 2 * theta
        peak_centres_tthdeg = np.degrees(two_theta)

        # positions = self.positions
        # occupancies = self.occupancies
        # b_iso = self.b_iso_array
        # elements = self.element_array

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

        f_abs = np.abs(structure_factor)
        structure_factor_squared = f_abs**2  # |F|**2

        intensity = structure_factor_squared

        if multiplicity_method:
            intensity = intensity * multiplicity

        intensity = intensity / (self.volume**2)

        # intensity = np.ones_like(intensity)  # normalise to max 1

        abs_corr = absorption_correction(theta, 0.1, 0.1)
        intensity = intensity * abs_corr

        if mode == "powder":
            lp_factor = lorentz_polarisation(theta, 0.9)  # 0 for unpolarized
            intensity = intensity * lp_factor

        peak_centres_tthdeg, intensity, hkl = merge_peaks(
            peak_centres_tthdeg, intensity, hkl
        )

        return hkl, f_abs, peak_centres_tthdeg, intensity

    def to_peaks(
        self,
        wavelength: float | int = 1.5406,
        mode: CrystalType = "powder",
        peak_type: str = "gaussian",
        radiation: DataType = "xray",
    ):
        hkl, f_abs, peak_centres_two_theta_degrees, intensity = (
            self.calculate_reflections(wavelength=wavelength, mode=mode)
        )

        peak_func = peak_factory(peak_type)

        peaks = []

        for inten, tth in zip(intensity, peak_centres_two_theta_degrees, strict=True):
            peaks.append(peak_func(amplitude=inten, centre=tth))

        return peaks

    def calculate_profile(
        self,
        x: np.ndarray,
        wavelength: float | int = 1.5406,
        backround: int | float | np.ndarray | Background = 0,
        mode: CrystalType = "powder",
        peak_type: str = "gaussian",
        phase_scale: int | float = 1,
        wdt: int | float = 5,
        radiation: DataType = "xray",
    ) -> np.ndarray:
        hkl, f_abs, two_theta_degrees, intensity = self.calculate_reflections(
            wavelength=wavelength, mode=mode, radiation=radiation
        )

        peak_func = peak_factory(peak_type)

        peaks = []

        for inten, tth in zip(intensity, two_theta_degrees, strict=True):
            peaks.append(peak_func(amplitude=inten, centre=tth))

        profile = PeakProfile(
            x=x, peaks=peaks, background=backround, phase_scale=phase_scale, wdt=wdt
        )

        return profile.calculate_profile()

    def plot_unit_cell(
        self,
    ):
        """
        Plot full unit cell (symmetry-expanded).
        """

        # Apply symmetry
        rotations, translations = self.get_symmetry_operations()

        positions, occupancies, b_iso, elements = apply_symmetry_operations_to_atoms(
            self.atoms,
            rotations,
            translations,
        )

        for el, pos in zip(elements, positions, strict=True):
            print(f"{el}: {pos}")

        shifts = np.array(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1],
            ],
            dtype=float,
        )

        n_atoms = positions.shape[0]
        n_shifts = shifts.shape[0]

        positions = positions[:, None, :] + shifts[None, :, :]
        positions = positions.reshape(n_atoms * n_shifts, 3)

        elements = np.repeat(elements, len(shifts))
        occupancies = np.repeat(occupancies, len(shifts))

        mask = (
            (positions[:, 0] >= 0)
            & (positions[:, 0] <= 1)
            & (positions[:, 1] >= 0)
            & (positions[:, 1] <= 1)
            & (positions[:, 2] >= 0)
            & (positions[:, 2] <= 1)
        )

        positions = positions[mask]
        elements = elements[mask]
        occupancies = occupancies[mask]

        for el, pos in zip(elements, positions, strict=True):
            print(f"{el}: {pos}")

        coords = positions
        xlabel, ylabel, zlabel = "a", "b", "c"

        # Colors
        unique_elements = np.unique(elements)
        cmap = plt.cm.get_cmap("tab10", len(unique_elements))

        element_to_color = {el: cmap(i) for i, el in enumerate(unique_elements)}

        colors = np.array([element_to_color[el] for el in elements])

        # Sizes
        sizes = 100 * np.array([ELEMENT_ATOMIC_NUMBER[el] for el in elements])

        # Plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.set_aspect("equal")
        ax.set_aspect("equal")

        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],  # type: ignore
            c=colors,
            s=sizes,  # type: ignore
            depthshade=True,
            # label=unique_elements,
        )

        # Formatting
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)

        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
        max_range = (maxs - mins).max()
        mid = (maxs + mins) / 2

        ax.set_xlim(mid[0] - max_range / 2, mid[0] + max_range / 2)
        ax.set_ylim(mid[1] - max_range / 2, mid[1] + max_range / 2)
        ax.set_zlim(mid[2] - max_range / 2, mid[2] + max_range / 2)

        plt.xlim(0, 1)
        plt.ylim(0, 1)
        ax.set_zlim(0, 1)
        # plt.legend()
        plt.title(f"Unit Cell ({self.spacegroup})")
        plt.tight_layout()
        plt.show()


def cif_to_structure(cif_filepath: str | Path) -> Structure:
    spacegroup_symbol, lattice, atoms, symmetry_operations, name = read_cif(
        str(cif_filepath)
    )

    spacegroup = format_space_group_name(spacegroup_symbol)

    sg = ITC_TABLES[spacegroup]
    crystal_class = sg["crystal_class"]

    latticecls = crystal_lattice_factory(crystal_class)
    lattice = latticecls.model_validate(lattice)

    structure = Structure(
        spacegroup=spacegroup,
        lattice=lattice,
        atoms=atoms,
        source=str(cif_filepath),
        name=name,
        # symmetry_operations=symmetry_operations,
    )

    return structure


##### more complex pydantic models to do whole profiles
class PeakProfile(RefinementBaseModel):
    phase_scale: int | float | Parameter = Parameter(value=1e-5, bounds=[0, np.inf])
    x: SerialisableNDArray
    peaks: list[PeakType]
    background: np.ndarray | int | float | Background = 0.0
    wdt: int | float = 5

    @property
    def sorted_peaks(self):
        return sorted(self.peaks, key=lambda p: p.centre)

    # @timeit
    def calculate_profile(self):
        """calculates the profile with the parameters stored within self"""

        return calculate_profile(
            x=self.x,
            peaks=self.sorted_peaks,
            background=self.background,
            phase_scale=float(self.phase_scale),
            wdt=self.wdt,
        )


class ReitveldRefinement(Model[ScatteringData]):
    phase_scale: int | float | Parameter = Parameter(value=1e-2, bounds=[0, np.inf])
    structure: Structure | list[Structure]
    zero_offset: int | float | Parameter = Parameter(value=0, bounds=[-10, 10])
    irf: IntrumentResolutionFunction = Field(default=FCJPseudoVoigt())
    background: np.ndarray | float | int | BackgroundType | Parameter = Parameter(
        value=0
    )
    intensity_modifiers: Parameter | None = None
    position_modifiers: Parameter | None = None
    shape_modifiers: Parameter | None = None
    calculated_intensity: SerialisableNDArray | None = Field(
        default=None, repr=False
    )  # this gets created the first time calc profile is run

    """How can we divide down a reitveld refinement -
    peaks pos/intensity, peak width function, background, detecor parameters
    """

    def load_cif(self, cif_filepath: str | Path) -> Structure | list[Structure]:
        new_structure = Structure.load_from_cif(cif_filepath)

        if isinstance(self.structure, Structure):
            self.structure = [self.structure, new_structure]
        elif isinstance(self.structure, list):
            self.structure.append(new_structure)
        else:
            self.structure = new_structure

        return self.structure

    def save_cif(self, output_path: str | Path):
        """Saves the structure to a CIF file."""
        assert self.structure is not None
        raise NotImplementedError("CIF writing not implemented yet")

    def calculate_peaks(self):
        if isinstance(self.structure, Structure):
            return self.structure.calculate_reflections()
        elif isinstance(self.structure, Collection):
            raise NotImplementedError("Multiple phase calculation not implemented yet")
        else:
            raise Exception("Unknown structure")

    def calculate_profile(self) -> np.ndarray:
        if not isinstance(self.structure, Structure):
            raise NotImplementedError("Not yet implemented for mullti-phase patterns")

        hkl, f_abs, peak_centres_tthdeg, intensity = (
            self.structure.calculate_reflections(float(self.data.wavelength))
        )

        zero_offset_x = self.data.x - float(self.zero_offset)

        calculated_intensity = self.irf.calculate_profile(
            x=zero_offset_x,
            peak_centres=peak_centres_tthdeg,
            peak_intensities=intensity,
        )

        if isinstance(self.background, Background):
            background = self.background.calculate(x=self.data.x)
        else:
            background = self.background

        self.calculated_intensity = (
            calculated_intensity * float(self.phase_scale)
        ) + background

        return self.calculated_intensity

    @computed_field
    @property
    def chi_squared(self) -> float:
        if self.calculated_intensity is not None:
            chi_squared = calculate_chi_squared(
                self.calculated_intensity, self.data.y, self.data.e
            )

            return chi_squared

        else:
            return np.inf

    def calculate_residual(self) -> np.ndarray:
        # http://pd.chem.ucl.ac.uk/pdnn/refine1/practice.htm

        self.calculated_intensity = self.calculate_profile()

        _ = self.chi_squared

        return self.data.y - self.calculated_intensity

    def plot(self):
        # if self.calculated_intensity is None:
        self.calculated_intensity = self.calculate_profile()

        if isinstance(self.background, Background):
            background = self.background.calculate(self.data.x)
        else:
            background = float(self.background)

        plot_data = FittedDataPlot(
            data=self.data,
            calc=self.calculated_intensity,
            diff=self.data.y - self.calculated_intensity,
            background=background,
        )
        plot_data.plot()


if __name__ == "__main__":
    output_name = "/workspaces/outputs/test.toml"

    def test_refine_silicon():
        cif_filepath = "/workspaces/XRPD-Toolbox/cifs/Si.cif"
        si_structure = Structure.load_from_cif(cif_filepath)

        beam_energy = 15
        wavelength = beam_energy_to_wavelength(beam_energy)

        data = ScatteringData.from_xye(
            "/workspaces/outputs/step_scan/1410696.nxs_summed_mythen3.xye",
            #  "/workspaces/outputs/1429744_summed_mythen3.xye",
            x_unit="tth",
            data_type="xray",
            wavelength=Parameter(value=wavelength, refine=False),
        )

        from xrpd_toolbox.fit_engine.background import (
            # ChebyshevBackground,
            LinearInterpolationBackground,
        )

        background = LinearInterpolationBackground.estimate(data.x, data.y)

        model = ReitveldRefinement(
            data=data, background=background, structure=si_structure
        )

        assert isinstance(model.background, Background)
        model.background.refine_none()

        print(model.get_refinement_parameters())

        model.irf.refine_none()

        updated, model, result = refine_model(model, plot=True)

        model.save(output_name)

        return model

    def test_load_refinement_and_refine():
        loaded_refinement = ReitveldRefinement.load(output_name)

        loaded_refinement.calculate_profile()
        loaded_refinement.plot()

        # refine_model(loaded_refinement)

    model = test_refine_silicon()
    # test_load_refinement_and_refine()

    two_theta = np.linspace(1, 70, 1000)

    width = FCJPseudoVoigt().calculate_peak_widths(two_theta)[0]

    plt.plot(two_theta, width)
    plt.show()
