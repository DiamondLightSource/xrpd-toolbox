from collections.abc import Collection

import numpy as np
from pydantic import BaseModel


class Peak(BaseModel):
    centre: int | float
    amplitude: int | float
    fwhm: int | float


def gaussian(
    x: np.ndarray, cen: int | float, amp: int | float, fwhm: int | float
) -> np.ndarray:
    """1-d gaussian: gaussian(x, amp, cen, fwhm)"""

    return (amp / (np.sqrt(2 * np.pi) * fwhm)) * np.exp(
        -((x - cen) ** 2) / (2 * fwhm**2)
    )


def multi_gaussian(
    x: np.ndarray,
    peaks: Collection[Peak],
    background: int | float | np.ndarray,
    phase_scale: int | float = 1,
):
    """wdt (range) of calculated profile of a single Bragg reflection in units of FWHM
    (typically 4 for Gaussian and 20-30 for Lorentzian, 4-5 for TOF).

    peaks: list of (cen, amp, fwhm)

    background: scalar or array, if array must be same shape as x
    """

    intensity = np.zeros_like(x) + background

    for peak in peaks:
        start_idx = np.searchsorted(x, peak.centre - peak.fwhm)
        end_idx = np.searchsorted(x, peak.centre + peak.fwhm, side="right")

        xi = x[start_idx:end_idx]
        peak = gaussian(xi, peak.centre, peak.amplitude, peak.fwhm) * phase_scale

        intensity[start_idx:end_idx] += peak

    return intensity
