from _collections_abc import Iterable
from typing import Literal

import numpy as np
import pint

from xrpd_toolbox.utils.settings import SettingsBase

ureg = pint.UnitRegistry()


class Wavelength(SettingsBase):
    value: float
    unit: str

    def to(self, target_unit: str) -> float:
        quantity = self.value * ureg(self.unit)
        return quantity.to(target_unit).magnitude


def beam_energy_to_wavelength(
    beam_energy: float | int, unit: Literal["kev", "ev"] = "kev"
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

    ev_to_j = 1.602176634e-19  # electron volt to joule factor
    h_planck = 6.62607015e-34  # h_planck plancks constant
    c_speed_of_light = 299792458.0  # m/s
    beam_energy_j = beam_energy_ev * ev_to_j
    wavelength_m = (h_planck * c_speed_of_light) / (beam_energy_j)
    wavelength = wavelength_m * 1e10

    return wavelength


def tth_to_q(tth: Iterable[int | float] | int | float, wavelength: float) -> np.ndarray:
    """

    Converts a 2th angle to Q using the wavelength. Simple.

    Whatever unit wavelength is in, will be the unit of Q.

    https://www.ill.eu/fileadmin/user_upload/ILL/3_Users/Support_labs_infrastructure/Software-tools/DIF_tools/neutrons.html

    """

    tth_array = np.array(tth, dtype=float)

    pi = 3.141592653589
    q_space = (4 * pi / wavelength) * np.sin(np.deg2rad(tth_array) / 2)

    return q_space
