from collections.abc import Collection

import numpy as np
from pydantic import BaseModel
from scipy.optimize import curve_fit


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


def fit_peaks(x: np.ndarray, y: np.ndarray, x_pos: Collection[int | float]):
    amps, x_positions, fwhms = [], [], []

    for x_guess in x_pos:
        try:
            width_guess = 0.03
            # Estimate amplitude from nearest data point
            idx = np.argmin(np.abs(x - x_guess))
            amp_guess = y[idx] * np.sqrt(2 * np.pi) * width_guess
            width_guess = 0.03

            p0 = [x_guess, amp_guess, width_guess]

            start_idx = np.searchsorted(x, x_guess - 1)
            end_idx = np.searchsorted(x, x_guess + 1, side="right")

            x_fit = x[start_idx:end_idx]
            y_fit = y[start_idx:end_idx]

            if len(y_fit) == 0:
                x_positions.append(np.nan)
                amps.append(np.nan)
                fwhms.append(np.nan)
                continue

            popt, _ = curve_fit(gaussian, x_fit, y_fit, p0=p0, maxfev=10000)

            x_positions.append(popt[0])  # cen
            amps.append(popt[1])  # amp
            fwhms.append(popt[2])  # fwhm (actually sigma in your formula)

        except RuntimeError:
            x_positions.append(np.nan)
            amps.append(np.nan)
            fwhms.append(np.nan)

    return (
        np.array(amps),
        np.array(x_positions),
        np.array(fwhms),
    )
