"""Peak shape definitions, peak fitting utilities, and profile calculation.

This module defines analytical peak functions, Pydantic peak models, and
utility routines for XRPD profile generation and peak detection.
"""

from __future__ import annotations

import math
from abc import abstractmethod
from collections.abc import Collection, Sequence
from typing import Annotated, Literal, TypeAlias

import numpy as np
import peakutils
from numba import njit
from pydantic import Field
from scipy.optimize import curve_fit
from scipy.special import erf

from xrpd_toolbox.core import Parameter
from xrpd_toolbox.fit_engine.background import Background
from xrpd_toolbox.fit_engine.fitting_core import RefinementBaseModel

IMPLEMENTED_PEAK_FUNCTONS: TypeAlias = Literal[
    "gaussian", "lorentzian", "pseudo_voigt", "tophat"
]


@njit()
def gaussian_sigma_to_fwhm(sigma: float | int) -> float:
    """Convert a Gaussian standard deviation to full width at half maximum."""
    return float(sigma) * 2 * np.sqrt(2 * np.log(2))


@njit()
def gaussian_fwhm_to_sigma(fwhm: float | int) -> float:
    """Convert a Gaussian full width at half maximum to standard deviation."""
    return float(fwhm) / (2 * np.sqrt(2 * np.log(2)))


@njit()
def lorentzian_gamma_to_fwhm(gamma: float | int) -> float:
    """Convert Lorentzian half width at half maximum to FWHM."""
    return 2 * float(gamma)


@njit()
def lorentzian_fwhm_to_gamma(fwhm: float | int) -> float:
    """Convert Lorentzian FWHM to half width at half maximum."""
    return float(fwhm) / 2


@njit()
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


@njit()
def lorentzian(
    x: np.ndarray,
    amplitude: float | int,
    centre: float | int,
    fwhm: float | int,
    background: float | int | np.ndarray = 0,
    normalised: bool = True,
) -> np.ndarray:
    """Lorentzian peak function.

    Parameters
    ----------
    x : array-like
        Input coordinate(s).
    amplitude : float | int
        Amplitude parameter:
        - If normalised=True: total area under the curve.
        - If normalised=False: peak height.
    centre : float | int
        Peak centre.
    fwhm : float | int
        Full width at half maximum.
    background : float | int | array-like, optional
        Additive background term. Default is 0.
    normalised : bool, optional
        If True, returns an area-normalised Lorentzian.

    Returns
    -------
    NDArray[np.float64]
        Evaluated Lorentzian function.
    """
    gamma = float(fwhm) / 2

    if normalised:
        prefactor = amplitude / np.pi
        core = gamma / ((x - centre) ** 2 + gamma**2)
    else:
        prefactor = amplitude
        core = gamma**2 / ((x - centre) ** 2 + gamma**2)

    return prefactor * core + background


@njit()
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

    pv = eta * lorentz + (1.0 - eta) * gauss
    pv = pv + background
    return pv


def smooth_tophat(
    x: np.ndarray,
    amplitude: float | int,
    centre: float | int,
    fwhm: float | int,
    epsilon: float | int,
    background: float | int | np.ndarray = 0,
    normalised: bool = True,
):
    """Return a Gaussian-smoothed top-hat profile.

    Parameters
    ----------
    x : array-like
        Input coordinate(s).
    amplitude : float | int
        Plateau area if normalised=True, otherwise plateau height.
    centre : float | int
        Centre of the plateau.
    fwhm : float | int
        Width of the plateau.
    epsilon : float | int
        Smoothing parameter for edge softening.
    background : float | int | array-like, optional
        Additive background. Default is 0.
    normalised : bool, optional
        If True, returns an area-normalised shape.

    Returns
    -------
    NDArray[np.float64]
        Evaluated smooth top-hat function.
    """

    sigma = max(epsilon * fwhm / 2.0, 1e-12)
    half_width = fwhm / 2.0

    left = (x - (centre - half_width)) / (np.sqrt(2) * sigma)
    right = (x - (centre + half_width)) / (np.sqrt(2) * sigma)

    profile = 0.5 * (erf(left) - erf(right))

    if normalised:
        # area is already fwhm * 1.0 (plateau height = 1)
        # so amplitude is converted to height
        scale = amplitude / fwhm
    else:
        scale = amplitude

    return scale * profile + background


def closest_indices(arr1: np.ndarray, arr2: np.ndarray):
    """Find the closest index in `arr2` for each value in `arr1`.

    Parameters
    ----------
    arr1 : array-like
        Values whose closest matches are sought.
    arr2 : array-like
        Reference values for matching.

    Returns
    -------
    numpy.ndarray
        Indices into `arr2` corresponding to the closest values.
    """
    arr1 = np.asarray(arr1)
    arr2 = np.asarray(arr2)
    # Broadcast arr1 and arr2 to compute pairwise differences
    diffs = np.abs(arr1[..., np.newaxis] - arr2)
    # Find the index of the minimum difference along the last axis (arr2)
    idx = np.argmin(diffs, axis=-1)
    return idx


def peak_factory(peak_type: str):
    """Return the peak model class for a given peak type name.

    Parameters
    ----------
    peak_type : str
        Supported values: 'gaussian', 'lorentzian', 'pseudo_voigt', 'tophat'.

    Returns
    -------
    type[Peak]
        The peak model class corresponding to `peak_type`.

    Raises
    ------
    ValueError
        If `peak_type` is not in the implemented peak types.
    """
    match peak_type:
        case "gaussian":
            return GaussianPeak
        case "lorentzian":
            return LorentzianPeak
        case "pseudo_voigt":
            return PseudoVoigtPeak
        case "tophat":
            return TopHatPeak
        case _:
            raise ValueError(
                f"peak_type must be one of the following {IMPLEMENTED_PEAK_FUNCTONS}"
            )


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
    """Fit Gaussian peaks to observed data using initial position guesses.

    Parameters
    ----------
    x : NDArray[np.float64]
        Coordinate values.
    y : NDArray[np.float64]
        Observed intensity values.
    initial_x_pos : Collection[int | float]
        Initial guesses for the peak positions.

    Returns
    -------
    list[Peak]
        Fitted Gaussian peak models.
    """
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
                    GaussianPeak(amplitude=math.nan, centre=math.nan, fwhm=math.nan)
                )
                continue

            popt, _ = curve_fit(gaussian, x_fit, y_fit, p0=p0, maxfev=10000)  # type: ignore

            fitted_peaks.append(
                GaussianPeak(amplitude=popt[0], centre=popt[1], fwhm=popt[2])
            )

        except RuntimeError:
            fitted_peaks.append(
                GaussianPeak(amplitude=math.nan, centre=math.nan, fwhm=math.nan)
            )

    return fitted_peaks


def find_and_fit_peaks(x: np.ndarray, y: np.ndarray, smoothing: int = 5) -> list[Peak]:
    """Detect peaks in a signal and fit Gaussian models automatically.

    Parameters
    ----------
    x : NDArray[np.float64]
        Coordinate values.
    y : NDArray[np.float64]
        Observed signal intensities.
    smoothing : int, optional
        Width of the moving-average filter used to reduce noise.

    Returns
    -------
    list[Peak]
        Gaussian peak models fit to the detected peak positions.
    """

    y_smoothed = np.convolve(
        y, np.ones(smoothing), mode="same"
    )  # smooth the data to reduce noise

    threshold = np.amax(y_smoothed) / smoothing

    indexes = peakutils.indexes(y_smoothed, thres=threshold, min_dist=3)  # type: ignore

    initial_x_pos = x[indexes]
    fitted_peaks = fit_peaks(x, y_smoothed, initial_x_pos=initial_x_pos)

    return fitted_peaks


class Peak(RefinementBaseModel):
    """Abstract base class for parameterised peak models.

    Peak subclasses provide analytic peak shapes and support refinement
    through the shared parameter interface.
    """

    amplitude: Parameter | float = Field(gt=0)
    centre: Parameter | float = Field(gt=0)
    fwhm: Parameter | float = Field(gt=0, default=Parameter(value=0.02))
    normalised: bool = True  # if normalised the

    @abstractmethod
    def calculate(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the peak shape on a coordinate array."""
        raise NotImplementedError("Must implement calculate method in peak subclass")


class GaussianPeak(Peak):
    """Gaussian peak model class."""

    peak_type: Literal["gaussian"] = "gaussian"

    def calculate(self, x: np.ndarray) -> np.ndarray:
        return gaussian(
            x,
            float(self.amplitude),
            float(self.centre),
            float(self.fwhm),
            normalised=self.normalised,
        )


class LorentzianPeak(Peak):
    """Lorentzian peak model class."""

    peak_type: Literal["lorentzian"] = "lorentzian"

    def calculate(self, x: np.ndarray) -> np.ndarray:
        return lorentzian(
            x,
            float(self.amplitude),
            float(self.centre),
            float(self.fwhm),
            normalised=self.normalised,
        )


class PseudoVoigtPeak(Peak):
    """Pseudo-Voigt peak model class combining Gaussian and Lorentzian shapes."""

    peak_type: Literal["pseudo_voigt"] = "pseudo_voigt"

    eta: Parameter | float | int = Field(
        ge=0, le=1, default=0.5
    )  # used for pseudo-voigt - mixing param

    def calculate(self, x: np.ndarray) -> np.ndarray:
        return pseudo_voigt(
            x,
            float(self.amplitude),
            float(self.centre),
            float(self.fwhm),
            float(self.eta),
            normalised=self.normalised,
        )


class TopHatPeak(Peak):
    """Smoothed top-hat peak model class."""

    peak_type: Literal["tophat"] = "tophat"

    epsilon: Parameter | float | int = Field(
        ge=0, le=1, default=0.1
    )  # used for tophat - smoothing

    def calculate(self, x: np.ndarray) -> np.ndarray:
        return smooth_tophat(
            x,
            float(self.amplitude),
            float(self.centre),
            float(self.fwhm),
            float(self.epsilon),
            normalised=self.normalised,
        )


# would it be useful to have a peak that can be any of of the other peaks
# possibly useful if you don't know what type of peak shape you actually want?
# class MutatablePeak(Peak):
#     peak_type: Literal["mutatable"] = "mutatable"


def calculate_profile(
    x: np.ndarray,
    peaks: Sequence[Peak],
    background: int | float | np.ndarray | Background = 0,
    phase_scale: int | float = 1,
    wdt: int | float = 5,
):
    """Calculate the combined peak profile and optional background.

    Parameters
    ----------
    x : NDArray[np.float64]
        Coordinate values at which to evaluate the profile.
    peaks : Sequence[Peak]
        Sequence of peak models to sum.
    background : int | float | np.ndarray | Background, optional
        Baseline to add to the summed peak profile. Default is 0.
    phase_scale : int | float, optional
        Multiplicative scale factor applied to the peak intensity.
    wdt : int | float, optional
        Range in units of peak FWHM used when evaluating each peak.

    Returns
    -------
    numpy.ndarray
        Summed intensity profile for the input coordinates.
    """

    if isinstance(background, np.ndarray):
        assert len(x) == len(background)

    intensity = np.zeros_like(x)

    for peak in peaks:
        start_idx = np.searchsorted(x, float(peak.centre) - (wdt * float(peak.fwhm)))
        end_idx = np.searchsorted(
            x, float(peak.centre) + (wdt * float(peak.fwhm)), side="right"
        )

        if end_idx <= start_idx:
            continue

        xi = x[start_idx:end_idx]
        peak_intensity = peak.calculate(xi)
        intensity[start_idx:end_idx] += peak_intensity

    if isinstance(background, Background):
        background = background.calculate(x)

    intensity = (intensity * phase_scale) + background

    return intensity


@njit(parallel=True)
def calculate_profile_parallel(
    x: np.ndarray,
    peaks: Sequence[Peak],
    background: int | float | np.ndarray | Background = 0,
    phase_scale: int | float = 1,
    wdt: int | float = 5,
):
    """Placeholder for a parallel profile calculation implementation.

    Parameters
    ----------
    x : NDArray[np.float64]
        Coordinate values.
    peaks : Sequence[Peak]
        Sequence of peak models.
    background : int | float | np.ndarray | Background, optional
        Background model or baseline.
    phase_scale : int | float, optional
        Multiplicative intensity scale.
    wdt : int | float, optional
        Evaluation window in units of FWHM.

    Returns
    -------
    numpy.ndarray
        Combined intensity profile.

    Notes
    -----
    This implementation is currently not completed and raises
    NotImplementedError by design.
    """

    raise NotImplementedError("Not implemented well yet")

    if isinstance(background, np.ndarray):
        assert len(x) == len(background)

    intensity = np.zeros_like(x)

    for peak_index in range(len(peaks)):
        peak = peaks[peak_index]

        assert peak.background == 0

        start_idx = np.searchsorted(x, peak.centre - (wdt * peak.fwhm))
        end_idx = np.searchsorted(x, peak.centre + (wdt * peak.fwhm), side="right")

        xi = x[start_idx:end_idx]
        peak_intensity = peak.calculate(xi)

        intensity[start_idx:end_idx] += peak_intensity

    intensity = (intensity * phase_scale) + background

    return intensity


PeakType = Annotated[
    GaussianPeak | LorentzianPeak | PseudoVoigtPeak | TopHatPeak,
    Field(discriminator="peak_type"),
]


if __name__ == "__main__":
    gauss = PseudoVoigtPeak(amplitude=1, centre=1, fwhm=1)

    print(gauss)

    gauss.refine_none()

    print(gauss)

    gauss.refine_all()

    print(gauss)
