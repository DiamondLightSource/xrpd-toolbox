from collections.abc import Collection

import numpy as np
import peakutils
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


def fit_peaks(
    x: np.ndarray, y: np.ndarray, initial_x_pos: Collection[int | float]
) -> list[Peak]:
    fitted_peaks = []

    for x_guess in initial_x_pos:
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
                fitted_peaks.append(Peak(centre=np.nan, amplitude=np.nan, fwhm=np.nan))
                continue

            popt, _ = curve_fit(gaussian, x_fit, y_fit, p0=p0, maxfev=10000)  # type: ignore
            fitted_peaks.append(Peak(centre=popt[0], amplitude=popt[1], fwhm=popt[2]))

        except RuntimeError:
            fitted_peaks.append(Peak(centre=np.nan, amplitude=np.nan, fwhm=np.nan))

    return fitted_peaks


def find_and_fit_peaks(x: np.ndarray, y: np.ndarray) -> list[Peak]:
    """function to get the centre peaks given without guessing"""

    y_smoothed = np.convolve(
        y, np.ones(5), mode="same"
    )  # smooth the data to reduce noise

    threshold = np.amax(y_smoothed) / 20
    indexes = peakutils.indexes(y_smoothed, thres=threshold, min_dist=30)  # type: ignore

    initial_x_pos = x[indexes]
    fitted_peaks = fit_peaks(x, y_smoothed, initial_x_pos=initial_x_pos)

    return fitted_peaks


def closest_indices(arr1, arr2):
    """
    For each value in arr1, find the index of the closest value in arr2.
    Returns an array of indices with the same shape as arr1.
    """
    arr1 = np.asarray(arr1)
    arr2 = np.asarray(arr2)
    # Broadcast arr1 and arr2 to compute pairwise differences
    diffs = np.abs(arr1[..., np.newaxis] - arr2)
    # Find the index of the minimum difference along the last axis (arr2)
    idx = np.argmin(diffs, axis=-1)
    return idx
