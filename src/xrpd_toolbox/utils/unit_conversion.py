import math
from collections.abc import Iterable
from typing import Literal

import numpy as np

# constants
ev_to_j = 1.602176634e-19  # electron volt to joule factor
h_planck = 6.62607015e-34  # Planck's constant
c_speed_of_light = 299792458.0  # m/s


def beam_energy_to_wavelength(
    beam_energy: float | int,
    unit: Literal["keV", "eV", "kev", "ev"] = "kev",
) -> float:
    """

    Calculates wavelength (Angstrom) from beam energy in kev.

    To allow convertion of tth to Q space, using the energy of the beam. beam energy is
    converted to wavlength because it's better

    """
    if unit.lower() == "kev":
        beam_energy_ev = beam_energy * 1000
    else:
        beam_energy_ev = beam_energy

    beam_energy_j = beam_energy_ev * ev_to_j
    wavelength_m = (h_planck * c_speed_of_light) / (beam_energy_j)
    wavelength = wavelength_m * 1e10

    return wavelength


def wavelength_to_beam_energy(
    wavelength: float | int,
    unit: Literal["keV", "eV", "kev", "ev"] = "kev",
) -> float:
    """
    Calculates beam energy from wavelength (Angstrom).

    Converts wavelength to energy using E = hc / λ.
    Returns energy in keV or eV depending on `unit`.
    """

    # convert Å → m
    wavelength_m = wavelength * 1e-10

    # energy in joules
    beam_energy_j = (h_planck * c_speed_of_light) / wavelength_m

    # convert J → eV
    beam_energy_ev = beam_energy_j / ev_to_j

    if unit.lower() == "kev":
        return beam_energy_ev / 1000
    else:
        return beam_energy_ev


def two_theta_to_q(
    tth: Iterable[int | float] | int | float, wavelength: float
) -> np.ndarray:
    """

    Converts a 2th angle to Q using the wavelength. Simple.

    Whatever unit wavelength is in, will be the unit of Q.

    https://www.ill.eu/fileadmin/user_upload/ILL/3_Users/Support_labs_infrastructure/Software-tools/DIF_tools/neutrons.html

    """

    tth_array = np.array(tth, dtype=float)
    q_space = (4 * math.pi / wavelength) * np.sin(np.deg2rad(tth_array) / 2)

    return q_space


def d_to_two_theta(
    d_spacing: np.ndarray | float | int, wavelength: float | int, degrees: bool = True
) -> np.ndarray:
    """convertts d to tth - simples
    sinθ=2dnλ​
    """

    sintheta = wavelength / (2 * d_spacing)
    theta = np.arcsin(sintheta)
    two_theta = 2 * theta

    if degrees:
        two_theta = np.degrees(two_theta)

    return two_theta


def q_space_to_d(q: np.ndarray | float | int):
    return 2 * np.pi / q


def q2_to_d(q2: np.ndarray | float | int):
    """q2 = scattering vector squared to d-space"""

    d = 2 * np.pi / np.sqrt(q2)

    return d


def d_spacing_to_scattering_vector(d_spacing: np.ndarray) -> np.ndarray:
    scattering_vector = 1 / (2 * d_spacing)
    return scattering_vector


def d_to_tof(d: float | np.ndarray, difa: float, difc: float, tzero: float = 0.0):
    """This is for TOF neutron data"""

    tof = difc * d + difa * d**2 + tzero

    return tof
