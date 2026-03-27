import math
from collections.abc import Collection
from typing import Literal

import numpy as np
import peakutils
from pydantic import BaseModel, Field, model_validator
from scipy.optimize import curve_fit
from scipy.special import erf


def gaussian_sigma_to_fwhm(sigma: float | int) -> float:
    return float(sigma) * 2 * np.sqrt(2 * np.log(2))


def gaussian_fwhm_to_sigma(fwhm: float | int) -> float:
    return float(fwhm) / (2 * np.sqrt(2 * np.log(2)))


def lorentzian_gamma_to_fwhm(gamma: float | int) -> float:
    return 2 * float(gamma)


def lorentzian_fwhm_to_gamma(fwhm: float | int) -> float:
    return float(fwhm) / 2


def gaussian(
    x: np.ndarray,
    amplitude: float | int,
    centre: float | int,
    fwhm: float | int,
    background: float | int | np.ndarray = 0,
    normalised: bool = True,
) -> np.ndarray:
    """
    Gaussian peak function.

    Parameters
    ----------
    x : array-like
        Input coordinate(s).
    amplitude : float | int
        Amplitude parameter:
        - If normalised=True: total area under the curve.
        - If normalised=False: peak height.
    centre : float | int
        Peak center.
    fwhm : float | int
        Full-Wdth at half maximum of the peak - the peak width (must be > 0).
    background : float | int | array-like, optional
        Additive background (constant or array matching `x`). Default is 0.
    normalised : bool, optional
        If True (default), returns an area-normalised Gaussian.
        If False, returns a Gaussian with peak height A.

    Returns
    -------
    NDArray[np.float64]
        Evaluated Gaussian function.
    """

    sigma = gaussian_fwhm_to_sigma(fwhm)

    if normalised:
        prefactor = amplitude / (sigma * np.sqrt(2 * np.pi))
    else:
        prefactor = amplitude

    return prefactor * np.exp(-((x - centre) ** 2) / (2 * sigma**2)) + background


def lorentzian(
    x: np.ndarray,
    amplitude: float | int,
    centre: float | int,
    fwhm: float | int,
    background: float | int | np.ndarray = 0,
    normalised: bool = True,
) -> np.ndarray:
    gamma = float(fwhm) / 2

    if normalised:
        prefactor = amplitude / np.pi
        core = gamma / ((x - centre) ** 2 + gamma**2)
    else:
        prefactor = amplitude
        core = gamma**2 / ((x - centre) ** 2 + gamma**2)

    return prefactor * core + background


def pseudo_voigt(
    x: np.ndarray,
    amplitude: float | int,
    centre: float | int,
    fwhm: float | int,
    eta: float | int,
    background: float | int | np.ndarray = 0,
    normalised: bool = True,
) -> np.ndarray:
    """
    Pseudo-Voigt peak function with a single FWHM parameter.

    Parameters
    ----------
    x : array-like
        Input coordinate(s).
    amplitude : float | int
        Amplitude parameter:
        - If normalised=True: total area under the curve.
        - If normalised=False: approximate peak height.
    centre : float | int
        Peak center.
    fwhm : float | int
        Full width at half maximum (must be > 0).
    eta : float | int
        Mixing parameter:
        - 0 → pure Gaussian
        - 1 → pure Lorentzian
    background : float | int | array-like, optional
        Additive background. Default is 0.
    normalised : bool, optional
        If True (default), area-normalised.
        If False, amplitude ≈ peak height.

    Returns
    -------
    NDArray[np.float64]
        Evaluated pseudo-Voigt function.
    """

    fwhm = float(fwhm)
    eta = float(eta)

    gauss = gaussian(
        x=x,
        amplitude=amplitude,
        centre=centre,
        fwhm=fwhm,
        background=0,
        normalised=normalised,
    )
    lorentz = lorentzian(
        x=x,
        amplitude=amplitude,
        centre=centre,
        fwhm=fwhm,
        background=0,
        normalised=normalised,
    )

    return amplitude * (eta * lorentz + (1 - eta) * gauss) + background


def smooth_tophat(
    x: np.ndarray,
    amplitude: float | int,
    centre: float | int,
    fwhm: float | int,
    eta: float | int,
    background: float | int | np.ndarray = 0,
    normalised: bool = True,
):
    """
    Gaussian-smoothed top-hat (fast, analytic normalization).

    Parameters
    ----------
    amplitude : float
        Area (if normalized=True) or height (if normalized=False)
    centre : float
        Centre of the plateau
    fwhm : float
        Width of the plateau
    eta : float
        Edge smoothing parameter (0–1)
    """

    # map eta → sigma (smoothing width)
    sigma = max(eta * fwhm / 2, 1e-12)

    half_width = fwhm / 2

    left = (x - (centre - half_width)) / (np.sqrt(2) * sigma)
    right = (x - (centre + half_width)) / (np.sqrt(2) * sigma)

    profile = 0.5 * (erf(left) - erf(right))

    if normalised:
        # analytic area = fwhm
        scale = amplitude / fwhm
    else:
        scale = amplitude

    return scale * profile + background


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


class Peak(BaseModel):
    amplitude: float | int = Field(gt=0)
    centre: float | int
    fwhm: float | int = Field(gt=0)

    peak_type: Literal["gaussian", "lorentzian", "pseudo-voigt", "top-hat"] = "gaussian"

    eta: float | int | None = None  # used for pseudo-voigt/tophat - mixing param

    background: float | int = 0
    normalised: bool = True  # if normalised the
    # integral under peak is equal to amplitude. ie number of counts in peak

    @model_validator(mode="after")
    def validate_parameters(self):
        # Allow NaNs to bypass strict validation

        if self.peak_type in ["pseudo-voigt", "top-hat"]:
            if self.eta is None:
                raise ValueError("eta must be provided for pseudo-voigt")
            if not (0 <= self.eta <= 1):
                raise ValueError("eta must be between 0 and 1")
        else:
            if self.eta is not None:
                raise ValueError("eta should only be set for pseudo-voigt")

        return self

    def calculate(self, x: np.ndarray) -> np.ndarray:
        if self.peak_type == "gaussian":
            return gaussian(
                x=x,
                amplitude=self.amplitude,
                centre=self.centre,
                fwhm=self.fwhm,
                background=self.background,
                normalised=self.normalised,
            )
        elif self.peak_type == "lorentzian":
            return lorentzian(
                x=x,
                amplitude=self.amplitude,
                centre=self.centre,
                fwhm=self.fwhm,
                background=self.background,
                normalised=self.normalised,
            )
        elif self.peak_type == "pseudo-voigt":
            assert self.eta is not None

            return pseudo_voigt(
                x=x,
                amplitude=self.amplitude,
                centre=self.centre,
                fwhm=self.fwhm,
                eta=self.eta,
                background=self.background,
                normalised=self.normalised,
            )
        elif self.peak_type == "top-hat":
            assert self.eta is not None
            return smooth_tophat(
                x=x,
                amplitude=self.amplitude,
                centre=self.centre,
                fwhm=self.fwhm,
                eta=self.eta,
                background=self.background,
                normalised=self.normalised,
            )
        else:
            raise ValueError(f"{self.peak_type} is not an allowed peak type")


def caglioti_fwhm(
    two_theta: np.ndarray | float,
    u,
    v: float,
    w: float,
) -> np.ndarray | float:
    """
    Compute FWHM using the Caglioti function.

    Parameters
    ----------
    two_theta : float or array
        2θ in degrees
    u, v, w : float
        Caglioti parameters

    """
    theta = np.radians(two_theta / 2.0)
    tan_theta = np.tan(theta)

    return np.sqrt(u * tan_theta**2 + v * tan_theta + w)


def estimate_fwhm(
    x_values: np.ndarray,
    y_values: np.ndarray,
    peak_index: int,
) -> float:
    """
    Estimate the full width at half maximum (FWHM) of a peak
    using neighbouring points.

    Parameters
    ----------
    x_values : NDArray[np.float64]
        Coordinate values.
    y_values : NDArray[np.float64]
        Signal/intensity values.
    peak_index : int
        Index of the peak maximum.

    Returns
    -------
    float
        Estimated FWHM.
    """
    # --- fallback based on sampling spacing ---
    if len(x_values) > 1:
        average_spacing = float(np.mean(np.diff(x_values)))
    else:
        average_spacing = 1.0

    fallback_fwhm = average_spacing * 2

    # need neighbours on both sides
    if peak_index <= 0 or peak_index >= len(x_values) - 1:
        return fallback_fwhm

    peak_x = x_values[peak_index]
    peak_y = y_values[peak_index]

    if peak_y <= 0 or not np.isfinite(peak_y):
        return fallback_fwhm

    estimated_sigmas: list[float] = []

    # --- right neighbour ---
    right_x = x_values[peak_index + 1]
    right_y = y_values[peak_index + 1]

    if 0 < right_y < peak_y:
        try:
            sigma = abs(right_x - peak_x) / np.sqrt(2 * np.log(peak_y / right_y))
            estimated_sigmas.append(sigma)
        except Exception:
            pass

    # --- left neighbour ---
    left_x = x_values[peak_index - 1]
    left_y = y_values[peak_index - 1]

    if 0 < left_y < peak_y:
        try:
            sigma = abs(left_x - peak_x) / np.sqrt(2 * np.log(peak_y / left_y))
            estimated_sigmas.append(sigma)
        except Exception:
            pass

    if not estimated_sigmas:
        return fallback_fwhm

    mean_sigma = float(np.mean(estimated_sigmas))

    # convert sigma to FWHM
    fwhm = mean_sigma * 2 * np.sqrt(2 * np.log(2))

    if not np.isfinite(fwhm) or fwhm <= 0:
        return fallback_fwhm

    return fwhm


def fit_peaks(
    x: np.ndarray, y: np.ndarray, initial_x_pos: Collection[int | float]
) -> list[Peak]:
    fitted_peaks = []

    for x_guess in initial_x_pos:
        try:
            peak_index = int(np.argmin(np.abs(x - x_guess)))
            width_guess = estimate_fwhm(x, y, peak_index)
            # Estimate amplitude from nearest data point
            idx = np.argmin(np.abs(x - x_guess))
            amp_guess = y[idx] * np.sqrt(2 * np.pi) * width_guess

            p0 = [
                amp_guess,
                x_guess,
                width_guess,
            ]

            start_idx = np.searchsorted(x, x_guess - 1)
            end_idx = np.searchsorted(x, x_guess + 1, side="right")

            x_fit = x[start_idx:end_idx]
            y_fit = y[start_idx:end_idx]

            if len(y_fit) == 0:
                fitted_peaks.append(
                    Peak(amplitude=math.nan, centre=math.nan, fwhm=math.nan)
                )
                continue

            popt, _ = curve_fit(gaussian, x_fit, y_fit, p0=p0, maxfev=10000)  # type: ignore

            fitted_peaks.append(Peak(amplitude=popt[0], centre=popt[1], fwhm=popt[2]))

        except RuntimeError:
            fitted_peaks.append(
                Peak(amplitude=math.nan, centre=math.nan, fwhm=math.nan)
            )

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
