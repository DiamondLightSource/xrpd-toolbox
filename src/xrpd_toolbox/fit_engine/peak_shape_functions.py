from abc import abstractmethod
from typing import Annotated, Literal

import numpy as np
from numba import njit
from pydantic import Field

from xrpd_toolbox.core import Parameter
from xrpd_toolbox.fit_engine.fitting_core import RefinementBaseModel


def caglioti_fwhm(
    two_theta: np.ndarray | float,  # in degrees
    u: float | int,
    v: float | int,
    w: float | int,
) -> np.ndarray:
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


@njit
def extended_caglioti_fwhm(
    two_theta: np.ndarray | float,
    u: float,
    v: float,
    w: float,
    x: float,
    y: float,
) -> np.ndarray:
    theta = np.radians(two_theta / 2.0)
    tan_theta = np.tan(theta)
    sec_theta = 1.0 / np.cos(theta)

    return np.sqrt(
        u * tan_theta**2 + v * tan_theta + w + x * sec_theta + y * np.cos(theta)
    )


@njit
def lorentzian_fwhm(
    two_theta: np.ndarray | float,
    x: float,
    y: float,
) -> np.ndarray:
    theta = np.radians(two_theta / 2.0)

    return x / np.cos(theta) + y * np.tan(theta)


@njit
def tchz_fwhm(
    two_theta: np.ndarray,
    u: float,
    v: float,
    w: float,
    x: float,
    y: float,
):
    hg = caglioti_fwhm(two_theta, u, v, w)
    hl = lorentzian_fwhm(two_theta, x, y)

    return hg, hl


class IRF(RefinementBaseModel):
    wdt: float | int = 5.0  # window for peak calculation. 5 = 5 * fwhm

    @abstractmethod
    def calculate_peak_widths(self, peak_centres: np.ndarray):
        raise NotImplementedError(
            "calculate_peak_widths method must be implemented for subclasses"
        )

    @abstractmethod
    def calculate_profile(
        self, x: np.ndarray, peak_centres: np.ndarray, peak_intensities: np.ndarray
    ) -> np.ndarray:
        """Calculates the profile of peaks with the stored values,
        and for the given peak_centres/intensities"""
        raise NotImplementedError(
            "calculate_profile method must be implemented for subclasses"
        )


class Cagloti(IRF):
    irf_type: Literal["Cagloti"] = "Cagloti"

    u: float | Parameter = Field(default=Parameter(value=1))
    v: float | Parameter = Field(default=Parameter(value=1))
    w: float | Parameter = Field(default=Parameter(value=1))

    def calculate(self, peak_centres: np.ndarray) -> np.ndarray:
        return caglioti_fwhm(
            two_theta=peak_centres, u=float(self.u), v=float(self.v), w=float(self.w)
        )


class ExtendedCaglioti(IRF):
    irf_type: Literal["ExtendedCaglioti"] = "ExtendedCaglioti"

    u: float | Parameter = Field(default=Parameter(value=1e-4))
    v: float | Parameter = Field(default=Parameter(value=0))
    w: float | Parameter = Field(default=Parameter(value=1e-4))
    x: float | Parameter = Field(default=Parameter(value=1e-3))
    y: float | Parameter = Field(default=Parameter(value=0.0))

    def calculate(self, peak_centres: np.ndarray) -> np.ndarray:
        return extended_caglioti_fwhm(
            peak_centres,
            float(self.u),
            float(self.v),
            float(self.w),
            float(self.x),
            float(self.y),
        )


@njit
def tchz_combine_fwhm(hg, hl):
    return (
        hg**5
        + 2.69269 * hg**4 * hl
        + 2.42843 * hg**3 * hl**2
        + 4.47163 * hg**2 * hl**3
        + 0.07842 * hg * hl**4
        + hl**5
    ) ** (1 / 5)


@njit
def tchz_eta(hg, hl, h):
    r = hl / h
    return 1.36603 * r - 0.47719 * r**2 + 0.11116 * r**3


# # TODO: Add modified pearson
# class ModifiedPearson(IRF):


class TCHZ(IRF):
    """Thompson–Cox–Hastings pseudo-Voigt"""

    irf_type: Literal["TCHZ"] = "TCHZ"

    u: float | Parameter = Field(default=Parameter(value=0.002))
    v: float | Parameter = Field(default=Parameter(value=0.0, refine=False))
    w: float | Parameter = Field(default=Parameter(value=0.0006))
    x: float | Parameter = Field(default=Parameter(value=0.003))
    y: float | Parameter = Field(default=Parameter(value=0.01))

    def calculate_peak_widths(self, peak_centres: np.ndarray):
        hg = caglioti_fwhm(
            peak_centres,
            float(self.u),
            float(self.v),
            float(self.w),
        )

        hl = lorentzian_fwhm(
            peak_centres,
            float(self.x),
            float(self.y),
        )

        fwhm = tchz_combine_fwhm(hg, hl)
        eta = tchz_eta(hg, hl, fwhm)

        return fwhm, eta

    def calculate_shape(self, x: np.ndarray, centre: float, fwhm: float, eta: float):
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        gamma = fwhm / 2.0

        dx = x - centre

        gauss = np.exp(-(dx**2) / (2 * sigma**2))
        lorentz = gamma**2 / (dx**2 + gamma**2)

        return eta * lorentz + (1 - eta) * gauss

    def calculate_profile(
        self,
        x: np.ndarray,
        peak_centres: np.ndarray,
        peak_intensities: np.ndarray,
    ) -> np.ndarray:
        y = np.zeros_like(x, dtype=float)

        fwhm, eta = self.calculate_peak_widths(peak_centres)

        for i, centre in enumerate(peak_centres):
            w = fwhm[i]

            start = np.searchsorted(x, centre - self.wdt * w)
            end = np.searchsorted(x, centre + self.wdt * w, side="right")

            if end <= start:
                continue

            xi = x[start:end]

            peak = self.calculate_shape(
                xi,
                centre,
                w,
                eta[i],
            )

            y[start:end] += peak_intensities[i] * peak

        return y


class ModifiedPearsonVII(IRF):
    """Modified Pearson VII peak shape function"""

    irf_type: Literal["MPV"] = "MPV"

    u: float | Parameter = Field(default=Parameter(value=1))  # shape parameter m
    v: float | Parameter = Field(
        default=Parameter(value=-1e-1, refine=False)
    )  # optional fixed offset
    w: float | Parameter = Field(
        default=Parameter(value=5e-3)
    )  # instrument broadening term

    def calculate_peak_widths(self, peak_centres: np.ndarray):
        # Replace with your own Caglioti/Lorentzian model if needed
        # Here assumed to directly produce FWHM per peak centre
        fwhm = caglioti_fwhm(
            peak_centres,
            float(self.u),
            float(self.v),
            float(self.w),
        )
        return fwhm

    def calculate_shape(self, x: np.ndarray, centre: float, fwhm: float, m: float):
        dx = x - centre

        # avoid invalid m values
        m = max(float(m), 1e-6)

        denom = 2 * np.sqrt(2 ** (1.0 / m) - 1)
        gamma = fwhm / denom

        return (1.0 + (dx**2) / (gamma**2)) ** (-m)

    def calculate_profile(
        self,
        x: np.ndarray,
        peak_centres: np.ndarray,
        peak_intensities: np.ndarray,
    ) -> np.ndarray:
        y = np.zeros_like(x, dtype=float)

        fwhm = self.calculate_peak_widths(peak_centres)
        m = float(self.u)

        for i, centre in enumerate(peak_centres):
            w = fwhm[i]

            start = np.searchsorted(x, centre - self.wdt * w)
            end = np.searchsorted(x, centre + self.wdt * w, side="right")

            if end <= start:
                continue

            xi = x[start:end]

            peak = self.calculate_shape(
                xi,
                centre,
                w,
                m,
            )

            y[start:end] += peak_intensities[i] * peak

        return y


class FCJPseudoVoigt(IRF):
    """
    Finger–Cox–Jephcoat asymmetric pseudo-Voigt (practical approximation)

    Uses:
    - pseudo-Voigt core
    - exponential tailing asymmetry (axial divergence model)
    """

    irf_type: Literal["FCJ_PV"] = "FCJ_PV"

    u: float | Parameter = Field(default=Parameter(value=0.002))
    v: float | Parameter = Field(default=Parameter(value=0.0, refine=False))
    w: float | Parameter = Field(default=Parameter(value=0.0006))
    x: float | Parameter = Field(default=Parameter(value=0.003))
    y: float | Parameter = Field(default=Parameter(value=0.01))

    a: float | Parameter = Field(default=Parameter(value=0.0))  # FCJ asymmetry

    def calculate_peak_widths(self, peak_centres: np.ndarray):
        hg = caglioti_fwhm(peak_centres, float(self.u), float(self.v), float(self.w))
        hl = lorentzian_fwhm(peak_centres, float(self.x), float(self.y))

        fwhm = tchz_combine_fwhm(hg, hl)
        eta = tchz_eta(hg, hl, fwhm)

        return fwhm, eta

    def calculate_shape(self, x, centre, fwhm, eta, alpha):
        dx = x - centre

        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        gamma = fwhm / 2.0

        # symmetric components
        gauss = np.exp(-(dx**2) / (2 * sigma**2))
        lorentz = gamma**2 / (dx**2 + gamma**2)

        pv = eta * lorentz + (1 - eta) * gauss

        # FCJ asymmetry: exponential right-hand tailing
        # (common diffraction approximation)
        asym = np.exp(-alpha * np.clip(dx, 0, None))

        return pv * asym

    def calculate_profile(
        self,
        x: np.ndarray,
        peak_centres: np.ndarray,
        peak_intensities: np.ndarray,
    ) -> np.ndarray:
        y = np.zeros_like(x, dtype=float)

        fwhm, eta = self.calculate_peak_widths(peak_centres)

        alpha = float(self.a)

        for i, centre in enumerate(peak_centres):
            w = fwhm[i]

            start = np.searchsorted(x, centre - self.wdt * w)
            end = np.searchsorted(x, centre + self.wdt * w, side="right")

            if end <= start:
                continue

            xi = x[start:end]

            peak = self.calculate_shape(
                xi,
                centre,
                w,
                eta[i],
                alpha,
            )

            y[start:end] += peak_intensities[i] * peak

        return y


IntrumentResolutionFunction = Annotated[
    Cagloti | ExtendedCaglioti | TCHZ | ModifiedPearsonVII | FCJPseudoVoigt,
    Field(discriminator="irf_type"),
]
