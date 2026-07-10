"""Pair distribution function (PDF) calculator.
Converts powder X-ray diffraction (2-theta vs intensity) into the reduced
PDF G(r). See `compute_pdf` for the full pipeline in one place.
"""

from __future__ import annotations

import warnings
from enum import StrEnum
from pathlib import Path
from typing import Literal, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pydantic import Field, ValidationInfo, field_validator, model_validator
from scipy.interpolate import BSpline, UnivariateSpline
from scipy.optimize import curve_fit, least_squares
from scipy.signal import find_peaks

from xrpd_toolbox.core import (
    ScatteringData,
    SerialisableNDArray,
    XRPDBaseModel,
    XYEData,
)
from xrpd_toolbox.fit_engine.form_factors import (
    calculate_compton_for_element,
    calculate_form_factor_for_element,
)
from xrpd_toolbox.fit_engine.peaks import GaussianPeak, gaussian
from xrpd_toolbox.logger import logger
from xrpd_toolbox.utils.chemical_formula import ChemicalFormula
from xrpd_toolbox.utils.unit_conversion import q_space_to_s, two_theta_to_q
from xrpd_toolbox.utils.utils import load_xy, lorentz_polarisation

WindowType = Literal["soper_lorch", "lorch", "cosine", "none"]
BackgroundType = Literal[
    "constant", "polynomial", "chebyshev", "linear", "bspline", "cosine"
]
NormalisationMethod = Literal["krogh_moe", "eggert", "billinge", "warren"]

# NormalisationMethod summary (see normalise_intensity for the maths):
#   krogh_moe: closed-form Krogh-Moe/Norman sum-rule integral; needs number_density.
#   warren:    joint linear least-squares fit of scale + background to self-scattering.
#   eggert:    nonlinear least-squares fit of scale + a constant offset (Eggert, 2012).
#   billinge:  krogh_moe fit, then a PDFgetX3-style additive low-r drift correction.


def gr_baseline(r_values: np.ndarray, rho0: float) -> np.ndarray:
    """Physical low-r baseline: G(r) = -4*pi*r*rho0 (no pairs below r_min)."""
    return -4.0 * np.pi * r_values * rho0


class ExportFormat(StrEnum):
    GR = "gr"  # Reduced PDF G(r)
    SQ = "sq"  # Total structure function S(Q)
    IQ = "iq"  # Coherent intensity in electron units I(Q)
    FQ = "fq"  # Reduced structure function F(Q) = Q[S(Q)-1]


class PDFNormalisationError(RuntimeError):
    """Raised when a normalisation scale factor cannot be determined."""


class PDFConfig(XRPDBaseModel):
    """All parameters controlling the PDF calculation."""

    composition: ChemicalFormula
    sample_name: str = Field(default="pdf")
    data: ScatteringData
    number_density: float = Field(gt=0, description="Atomic number density (atoms/A^3)")

    q_min: float = Field(default=0.5, gt=0)
    q_max: float = Field(default=30.0, gt=0)
    q_step: float | None = Field(default=None, gt=0)

    polarisation_factor: bool = True
    is_synchrotron: bool = True
    polarisation_p: float = Field(default=0.99, ge=0.0, le=1.0)
    background_file: Path | None = None
    background_scale: float = 1.0

    absorption_correction: bool = True
    mu_r: float | None = Field(
        default=None,
        gt=0,
        description="Linear absorption coefficient x cylindrical sample radius (mu*R).",
    )

    fluorescence_correction: bool = Field(
        default=False,
        description="Subtract an isotropic fluorescence floor before absorption/pol.",
    )
    fluorescence_level: float | None = Field(
        default=None,
        ge=0.0,
        description="Fixed fluorescence floor (raw counts); auto-estimated if None.",
    )
    fluorescence_percentile: float = Field(
        default=1.0,
        ge=0.0,
        le=50.0,
        description="Percentile of raw intensity used to auto-estimate the floor.",
    )

    norm_poly_degree: int = Field(default=3, ge=0)
    norm_q_min: float | None = None
    background_type: BackgroundType = "chebyshev"
    normalisation_method: NormalisationMethod = "krogh_moe"
    r_poly: float = Field(
        default=1.5,
        gt=0,
        description="billinge only: r cutoff (A) for the ad-hoc low-r drift correction.",
    )

    compute_full_covariance: bool = Field(
        default=True,
        description=(
            "Attempt full (n_r, n_r) covariance propagation for G(r); "
            "skipped automatically above covariance_max_points."
        ),
    )
    covariance_max_points: int = Field(
        default=1000,
        gt=0,
        description="Full covariance is skipped if the Q- or r-grid exceeds this size.",
    )

    auto_q_max: bool = Field(
        default=False,
        description="Auto-select q_max from where I(Q) SNR drops below the threshold.",
    )
    auto_q_max_snr_threshold: float = Field(default=1.5, gt=0)
    auto_q_max_search_min: float = Field(default=5.0, gt=0)

    r_min: float = Field(default=0.0, ge=0.0)
    r_max: float = Field(default=30.0, gt=0)
    r_step: float = Field(default=0.01, gt=0)

    termination_window: WindowType | None = "lorch"
    soper_lorch_power: int | float | None = Field(default=2)

    qdamp: float = Field(
        default=0.030,
        ge=0.0,
        description="r-space PDF envelope decay: G(r) *= exp(-(qdamp*r)^2/2).",
    )
    qbroad: float = Field(
        default=0.0,
        ge=0.0,
        description="r-dependent peak broadening: sigma(r) = qbroad*r^2. 0 disables it.",
    )

    use_real_space_constraint: bool = True
    real_space_constraint_iterations: int = Field(default=10, ge=0)
    r_constraint_max: float | None = Field(default=None, gt=0)
    r_constraint_search_min: float = Field(default=1.2, gt=0)
    r_constraint_search_max: float = Field(default=3.5, gt=0)

    export_formats: list[ExportFormat] = Field(
        default_factory=lambda: [ExportFormat.GR]
    )
    output_dir: Path = Path(".")

    @staticmethod
    def _require_greater(
        value: float, info: ValidationInfo, other: str, msg: str
    ) -> float:
        other_value = info.data.get(other)
        if other_value is not None and value <= other_value:
            raise ValueError(msg)
        return value

    @field_validator("q_max")
    @classmethod
    def _q_max_above_q_min(cls, value: float, info: ValidationInfo) -> float:
        return cls._require_greater(
            value, info, "q_min", "q_max must be greater than q_min"
        )

    @field_validator("r_max")
    @classmethod
    def _r_max_above_r_min(cls, value: float, info: ValidationInfo) -> float:
        return cls._require_greater(
            value, info, "r_min", "r_max must be greater than r_min"
        )

    @field_validator("r_constraint_search_max")
    @classmethod
    def _search_max_above_min(cls, value: float, info: ValidationInfo) -> float:
        return cls._require_greater(
            value,
            info,
            "r_constraint_search_min",
            "r_constraint_search_max must exceed r_constraint_search_min",
        )

    @model_validator(mode="after")
    def _apply_defaults(self) -> PDFConfig:
        if self.norm_q_min is None:
            self.norm_q_min = max(self.q_min, self.q_max - 10.0)
        if self.norm_q_min >= self.q_max:
            raise ValueError("norm_q_min must be below q_max")
        if self.absorption_correction and self.mu_r is None:
            raise ValueError("mu_r is required when absorption_correction=True")
        return self


class PDFResult(XRPDBaseModel):
    """Computed PDF results; each stage carries its own propagated uncertainty.
    Treats scale_factor/background as fixed, so error bars are a lower bound.
    """

    iq: ScatteringData  # I(Q) in electron units, x_unit="q"
    sq: XYEData  # S(Q) total structure function, x_unit="q"
    fq: XYEData  # F(Q) = Q[S(Q)-1] (A^-1), x_unit="q"
    gr: XYEData  # G(r) (A^-2), x_unit="r"
    gr_covariance: SerialisableNDArray | None = None  # full (n_r, n_r) Cov[G(r)]
    scale_factor: float
    background: SerialisableNDArray  # electron units, same convention as I(Q)
    number_density: float
    sample_name: str

    @property
    def r(self) -> np.ndarray:
        return self.gr.x

    @property
    def q(self) -> np.ndarray:
        return self.iq.x

    @property
    def baseline(self) -> np.ndarray:
        """Physical low-r baseline -4*pi*r*rho0."""
        return gr_baseline(self.gr.x, self.number_density)

    @property
    def rdf(self) -> np.ndarray:
        """g(r) = 1 - G(r)/baseline; baseline is negative, so g(r)=0 in the no-pairs region."""
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.divide(
                self.gr.y,
                self.baseline,
                out=np.full_like(self.gr.y, np.nan),
                where=~np.isclose(self.baseline, 0.0),
            )
        return 1.0 - ratio

    def save_results(
        self, export_formats: list[ExportFormat] | list[str], output_dir: str | Path
    ) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        headers = {
            ExportFormat.GR: (
                "G(r) - reduced PDF\n# r (A)   G(r) (A^-2)   sigma_G(r)",
                self.gr,
            ),
            ExportFormat.SQ: (
                "S(Q) - total structure function\n# Q (A^-1)   S(Q)   sigma_S(Q)",
                self.sq,
            ),
            ExportFormat.IQ: (
                "I(Q) - Intensity (e.u.)\n# Q (A^-1)   I(Q) (e.u.)   sigma_I(Q)",
                self.iq,
            ),
            ExportFormat.FQ: (
                "F(Q) = Q[S(Q)-1]\n# Q (A^-1)   F(Q) (A^-1)   sigma_F(Q)",
                self.fq,
            ),
        }
        for fmt in export_formats:
            fmt = ExportFormat(fmt)
            header, data = headers[fmt]
            data.save_to_xye(output_dir / f"{self.sample_name}.{fmt}", header=header)

    @staticmethod
    def style_axis(
        ax: Axes, xlabel: str, ylabel: str, title: str, legend: bool = False
    ) -> None:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        if legend:
            ax.legend(fontsize=9)

    def plot(
        self,
        save_filepath: str | Path | None = None,
        ref_filepath: str | Path | None = None,
    ) -> None:
        """Diagnostic 2x2 plot of I(Q), S(Q), F(Q) and G(r) with 1-sigma bands."""
        eline = 0.1

        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        fig.suptitle("Total Scattering Data", fontsize=13)

        ax_iq, ax_sq, ax_fq, ax_gr = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

        ax_iq.errorbar(self.iq.x, self.iq.y, self.iq.e, elinewidth=eline)
        self.style_axis(
            ax_iq, "Q (A^-1)", "I(Q) (e.u.)", "Normalised Scattering Intensity"
        )

        ax_sq.errorbar(self.sq.x, self.sq.y, self.sq.e, elinewidth=eline)
        ax_sq.axhline(1.0, color="k", lw=0.6, ls="--", label="S(Q) = 1")
        self.style_axis(
            ax_sq, "Q (A^-1)", "S(Q)", "Total structure function", legend=True
        )

        ax_fq.errorbar(self.fq.x, self.fq.y, self.fq.e, elinewidth=eline)
        ax_fq.axhline(0.0, color="k", lw=0.6, ls="--")
        self.style_axis(
            ax_fq, "Q (A^-1)", "F(Q) = Q[S(Q)-1] (A^-1)", "Reduced structure function"
        )

        if ref_filepath is not None and Path(ref_filepath).exists():
            ref_data_x, ref_data_y = load_xy(Path(ref_filepath))
            ax_gr.plot(
                ref_data_x,
                ref_data_y,
                color="darkorange",
                lw=1.0,
                ls="--",
                label="Reference G(r)",
            )

        plot_r = self.r[self.r < 5]
        plot_baseline = self.baseline[self.r < 5]
        ax_gr.errorbar(self.gr.x, self.gr.y, self.gr.e, elinewidth=eline)
        ax_gr.plot(
            plot_r, plot_baseline, "k--", lw=0.8, label=r"$-4\pi r\rho_0$", zorder=2
        )
        self.style_axis(
            ax_gr,
            "r (A)",
            "G(r) (A^-2)",
            "Pair distribution function G(r)",
            legend=True,
        )
        ax_gr.set_xlim(0, float(np.amax(self.gr.x)))

        fig.tight_layout()
        if save_filepath is not None:
            fig.savefig(save_filepath, dpi=150)
        plt.show()


# --------------------------------------------------------------------------
# Coordination number analysis: peak finding, gaussian fitting, integration
# --------------------------------------------------------------------------


class PeakFitResult(GaussianPeak):
    peak_r: float
    sigma: float
    r_left: float
    r_right: float
    coordination_number: float


def gaussian_peak(
    r: np.ndarray, amplitude: float, center: float, sigma: float
) -> np.ndarray:
    return gaussian(x=r, amplitude=amplitude, centre=center, sigma=sigma)


def find_rdf_peak_positions(
    r: np.ndarray,
    g_r: np.ndarray,
    n_peaks: int = 3,
    r_search_min: float = 1.5,
    r_search_max: float = 8.0,
    prominence: float = 0.02,
) -> np.ndarray:
    """Locate up to n_peaks local maxima of g(r) within [r_search_min, r_search_max]."""
    mask = (r >= r_search_min) & (r <= r_search_max) & np.isfinite(g_r)
    r_window, g_window = r[mask], g_r[mask]
    peak_idx, _ = find_peaks(g_window, prominence=prominence)
    if len(peak_idx) < n_peaks:
        warnings.warn(
            f"found only {len(peak_idx)} of {n_peaks} requested peaks", stacklevel=2
        )
    return r_window[peak_idx[:n_peaks]]


def find_peak_bounds(
    r: np.ndarray,
    g_r: np.ndarray,
    peak_r: float,
    r_search_min: float,
    r_search_max: float,
) -> tuple[float, float]:
    """Bracket a peak between its neighbouring valleys (or +/-0.5 A if none found)."""
    mask = (r >= r_search_min) & (r <= r_search_max) & np.isfinite(g_r)
    r_window, g_window = r[mask], g_r[mask]
    valley_idx, _ = find_peaks(-g_window, prominence=0.01)
    valley_r = r_window[valley_idx]
    left = valley_r[valley_r < peak_r]
    right = valley_r[valley_r > peak_r]
    r_left = left[-1] if len(left) else peak_r - 0.5
    r_right = right[0] if len(right) else peak_r + 0.5
    return float(r_left), float(r_right)


def fit_gaussian_to_peak(
    r: np.ndarray, g_r: np.ndarray, peak_r: float, r_left: float, r_right: float
) -> tuple[float, float, float]:
    """Fit a single Gaussian to g(r) within [r_left, r_right]."""
    mask = (r >= r_left) & (r <= r_right) & np.isfinite(g_r)
    r_fit, g_fit = r[mask], g_r[mask]
    initial_guess = [
        float(np.nanmax(g_fit)),
        peak_r,
        max((r_right - r_left) / 4.0, 0.01),
    ]
    bounds = ([0.0, r_left, 0.005], [np.inf, r_right, r_right - r_left])
    params, _ = curve_fit(
        gaussian_peak, r_fit, g_fit, p0=initial_guess, bounds=bounds, maxfev=5000
    )
    amplitude, center, sigma = params
    return float(amplitude), float(center), float(sigma)


def coordination_number_from_gaussian(
    amplitude: float,
    center: float,
    sigma: float,
    rho0: float,
    r_left: float,
    r_right: float,
    n_points: int = 2000,
) -> float:
    """Coordination number = integral of 4*pi*rho0*r^2*g(r) over the fitted peak."""
    r_dense = np.linspace(r_left, r_right, n_points)
    g_dense = gaussian_peak(r_dense, amplitude, center, sigma)
    return float(np.trapezoid(4.0 * np.pi * rho0 * r_dense**2 * g_dense, r_dense))


def analyse_rdf_coordination(
    result: PDFResult,
    n_peaks: int = 3,
    r_search_min: float = 1.5,
    r_search_max: float = 8.0,
    prominence: float = 0.02,
) -> list[PeakFitResult]:
    """Find, fit, and integrate the first n_peaks coordination shells of g(r)."""
    r, g_r, rho0 = (
        np.asarray(result.gr.x),
        np.asarray(result.rdf),
        result.number_density,
    )

    peak_positions = find_rdf_peak_positions(
        r, g_r, n_peaks, r_search_min, r_search_max, prominence
    )
    peak_fits: list[PeakFitResult] = []
    for peak_r in peak_positions:
        r_left, r_right = find_peak_bounds(r, g_r, peak_r, r_search_min, r_search_max)
        try:
            amplitude, center, sigma = fit_gaussian_to_peak(
                r, g_r, peak_r, r_left, r_right
            )
        except RuntimeError:
            warnings.warn(f"gaussian fit failed near r={peak_r:.2f}", stacklevel=2)
            continue
        coordination_number = coordination_number_from_gaussian(
            amplitude, center, sigma, rho0, r_left, r_right
        )
        peak_fits.append(
            PeakFitResult(
                peak_r=float(peak_r),
                amplitude=amplitude,
                centre=center,
                sigma=sigma,
                r_left=r_left,
                r_right=r_right,
                coordination_number=coordination_number,
            )
        )
    return peak_fits


def plot_rdf_coordination(
    result: PDFResult,
    peak_fits: list[PeakFitResult],
    save_filepath: str | Path | None = None,
) -> Figure:
    """Plot g(r) with each fitted coordination shell shaded and annotated."""
    r, g_r = np.asarray(result.gr.x), np.asarray(result.rdf)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(r, g_r, color="steelblue", lw=1.0, label="g(r)")

    colors = ["tomato", "seagreen", "darkorange", "orchid"]
    r_max_plot = 8.0
    for i, fit in enumerate(peak_fits):
        color = colors[i % len(colors)]
        r_dense = np.linspace(fit.r_left, fit.r_right, 500)
        g_fit_dense = gaussian_peak(
            r_dense, float(fit.amplitude), float(fit.centre), float(fit.sigma)
        )
        ax.plot(
            r_dense,
            g_fit_dense,
            color=color,
            lw=1.5,
            ls="--",
            label=f"fit {i + 1}: r={fit.centre:.2f} A, cn={fit.coordination_number:.2f}",
        )
        ax.fill_between(r_dense, 0.0, g_fit_dense, color=color, alpha=0.3)
        ax.annotate(
            f"cn={fit.coordination_number:.2f}",
            xy=(float(fit.centre), float(fit.amplitude)),
            xytext=(float(fit.centre), float(fit.amplitude) * 1.15),
            ha="center",
            fontsize=9,
            color=color,
        )
        r_max_plot = max(r_max_plot, fit.r_right + 1.0)

    ax.axhline(0.0, color="k", lw=0.6, ls="--")
    ax.set_xlabel("r (A)")
    ax.set_ylabel("g(r)")
    ax.set_xlim(0, r_max_plot)
    ax.set_title("RDF peak fits and coordination numbers")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_filepath is not None:
        fig.savefig(save_filepath, dpi=150)
    plt.show()
    return fig


def build_scattering_factors(
    composition: ChemicalFormula, q_values: SerialisableNDArray
) -> tuple[
    SerialisableNDArray, SerialisableNDArray, SerialisableNDArray, SerialisableNDArray
]:
    """Composition-averaged <f>, <f>^2, <f^2> and Compton scattering on q."""
    scattering_s_values = q_space_to_s(q_values)
    weights = composition.atomic_fraction

    f_mean = np.zeros_like(q_values)
    f_sq_mean = np.zeros_like(q_values)
    compton = np.zeros_like(q_values)

    for weight, element in zip(weights, composition.elements, strict=True):
        f_element = calculate_form_factor_for_element(element, scattering_s_values)
        f_mean += weight * f_element
        f_sq_mean += weight * f_element**2
        compton += weight * calculate_compton_for_element(element, scattering_s_values)

    return f_mean, f_mean**2, f_sq_mean, compton


def cylindrical_absorption_correction(
    two_theta_deg: SerialisableNDArray,
    mu_r: float,
    n_grid: int = 41,
    n_theta_coarse: int = 121,
) -> np.ndarray:
    """Transmission-weighted absorption correction 1/<T> for a cylindrical sample.
    Integrates over the cylinder cross-section (mu_r=mu*R) on a coarse 2theta
    grid, then interpolates. Returns: corrected = raw * result.
    """
    lin = np.linspace(-1.0, 1.0, n_grid)
    grid_x, grid_y = np.meshgrid(lin, lin)
    inside_disc = grid_x**2 + grid_y**2 < 1.0
    disc_x, disc_y = grid_x[inside_disc], grid_y[inside_disc]
    disc_radius_sq = disc_x**2 + disc_y**2

    two_theta_coarse = np.linspace(
        float(np.amin(two_theta_deg)), float(np.amax(two_theta_deg)), n_theta_coarse
    )
    theta_rad = np.deg2rad(two_theta_coarse)
    incident_path_projection = disc_x  # incident beam fixed along +x

    transmission = np.empty_like(two_theta_coarse)
    for theta_index, theta in enumerate(theta_rad):
        d_out = (np.cos(theta), np.sin(theta))
        scattered_path_projection = disc_x * d_out[0] + disc_y * d_out[1]

        path_in = incident_path_projection + np.sqrt(
            np.maximum(incident_path_projection**2 - (disc_radius_sq - 1.0), 0.0)
        )
        path_out = -scattered_path_projection + np.sqrt(
            np.maximum(scattered_path_projection**2 - (disc_radius_sq - 1.0), 0.0)
        )
        transmission[theta_index] = np.mean(np.exp(-mu_r * (path_in + path_out)))

    t_interp = np.interp(two_theta_deg, two_theta_coarse, transmission)
    return 1.0 / np.maximum(t_interp, 1e-12)


def estimate_fluorescence_level(
    intensity: np.ndarray, percentile: float = 1.0
) -> float:
    """Robust floor estimate for fluorescence: a low percentile of raw counts.
    Fluorescence emission is (nearly) angle-independent, so it sets a
    roughly flat floor under the Bragg peaks and diffuse scattering.
    """
    return float(np.percentile(intensity, percentile))


def top_hat_bin_average(
    q_raw: np.ndarray,
    intensity: np.ndarray,
    sigma: np.ndarray,
    q_values: np.ndarray,
    q_step_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average raw points inside +/-q_step/2 of each grid point (top-hat rebin).
    Equivalent to convolving with a top-hat kernel of width q_step_size
    before sampling; shrinks sigma by sqrt(n). counts==0 flags empty bins.
    """
    n_bins = len(q_values)
    bin_edges = np.concatenate(
        [[q_values[0] - 0.5 * q_step_size], q_values + 0.5 * q_step_size]
    )
    bin_index = np.searchsorted(bin_edges, q_raw, side="right") - 1
    in_range = (bin_index >= 0) & (bin_index < n_bins)
    bin_index_valid = bin_index[in_range]

    counts = np.bincount(bin_index_valid, minlength=n_bins)[:n_bins]
    sum_intensity = np.bincount(
        bin_index_valid, weights=intensity[in_range], minlength=n_bins
    )[:n_bins]
    sum_sigma_sq = np.bincount(
        bin_index_valid, weights=sigma[in_range] ** 2, minlength=n_bins
    )[:n_bins]

    safe_counts = np.maximum(counts, 1)
    mean_intensity = sum_intensity / safe_counts
    mean_sigma = np.sqrt(sum_sigma_sq) / safe_counts
    return mean_intensity, mean_sigma, counts


def auto_select_q_max(
    q_values: SerialisableNDArray,
    intensity: SerialisableNDArray,
    sigma: SerialisableNDArray,
    snr_threshold: float = 1.5,
    q_search_min: float = 5.0,
    window_points: int = 25,
) -> float:
    """Pick q_max as the highest Q where local I(Q) SNR stays above snr_threshold."""
    from scipy.ndimage import uniform_filter1d

    order = np.argsort(q_values)
    q_s, i_s, sig_s = q_values[order], intensity[order], sigma[order]

    local_mean = uniform_filter1d(i_s, window_points)
    local_std = np.sqrt(
        np.maximum(uniform_filter1d((i_s - local_mean) ** 2, window_points), 0.0)
    )
    local_sigma = uniform_filter1d(sig_s, window_points)
    snr = local_std / np.maximum(local_sigma, 1e-12)

    valid = q_s >= q_search_min
    above = snr >= snr_threshold
    sustained = uniform_filter1d(above.astype(float), max(window_points // 2, 1)) > 0.9

    candidates = np.where(valid & sustained)[0]
    return float(q_s[candidates[-1]]) if candidates.size else float(q_search_min)


def r_dependent_broadening_matrix(
    r_values: SerialisableNDArray, qbroad: float, n_sigma_cutoff: float = 4.0
) -> np.ndarray:
    """PDFgui/PDFgetX3-style r-dependent Gaussian broadening kernel, sigma(r)=qbroad*r^2."""
    num_r_points = len(r_values)
    if qbroad <= 0.0:
        return np.eye(num_r_points)

    sigma_r = np.maximum(qbroad * r_values**2, 1e-6)
    diff = r_values[:, None] - r_values[None, :]
    kernel = np.exp(-0.5 * (diff / sigma_r[:, None]) ** 2)
    kernel[np.abs(diff) > n_sigma_cutoff * sigma_r[:, None]] = 0.0
    return kernel / np.maximum(kernel.sum(axis=1, keepdims=True), 1e-30)


# --------------------------------------------------------------------------
# Normalisation: fit scale_factor and an electron-unit background such that
#   I_eu(Q) = scale_factor * I_raw(Q) - background(Q)  ~=  <f^2(Q)> + Compton(Q)
# See the NormalisationMethod summary near the top of the file.
# --------------------------------------------------------------------------


def _clip_background_extrapolation(
    q_values: SerialisableNDArray,
    background: SerialisableNDArray,
    q_min_fit: float,
    q_max_fit: float,
) -> np.ndarray:
    """Flatten a background polynomial outside its fit window to prevent divergence."""
    clipped = background.copy()
    clipped[q_values < q_min_fit] = float(np.interp(q_min_fit, q_values, background))
    clipped[q_values > q_max_fit] = float(np.interp(q_max_fit, q_values, background))
    return clipped


def _bspline_basis(
    x: np.ndarray, x_min: float, x_max: float, num_basis: int, spline_degree: int
) -> np.ndarray:
    """Clamped uniform B-spline design matrix with num_basis basis functions on [x_min, x_max]."""
    degree = max(0, min(spline_degree, num_basis - 1))
    num_interior = max(num_basis - degree - 1, 0)
    interior_knots = np.linspace(x_min, x_max, num_interior + 2)[1:-1]
    knots = np.concatenate(
        [np.full(degree + 1, x_min), interior_knots, np.full(degree + 1, x_max)]
    )
    return BSpline.design_matrix(
        np.clip(x, x_min, x_max), knots, degree, extrapolate=False
    ).toarray()


def _background_basis(
    q_values: SerialisableNDArray,
    q_min_fit: float,
    q_max_fit: float,
    degree: int,
    background_type: BackgroundType,
) -> np.ndarray:
    """Background design matrix on q_values, normalised to x in [-1, 1] for conditioning."""
    midpoint = 0.5 * (q_max_fit + q_min_fit)
    half_range = max(0.5 * (q_max_fit - q_min_fit), 1e-12)
    normalised_q = (q_values - midpoint) / half_range

    if background_type == "constant":
        return np.ones((len(normalised_q), 1))
    if background_type == "chebyshev":
        return np.polynomial.chebyshev.chebvander(normalised_q, degree)
    if background_type == "polynomial":
        return np.vander(normalised_q, degree + 1, increasing=True)
    if background_type == "cosine":
        orders = np.arange(degree + 1)
        angle = 0.5 * np.pi * (normalised_q[:, None] + 1.0)
        return np.cos(orders[None, :] * angle)
    if background_type in ("linear", "bspline"):
        spline_degree = 1 if background_type == "linear" else 3
        return _bspline_basis(normalised_q, -1.0, 1.0, degree + 1, spline_degree)
    raise ValueError(f"Unknown background_type '{background_type}'.")


def _warn_if_ill_conditioned(design: np.ndarray, label: str) -> None:
    """Warn if a least-squares design matrix is poorly conditioned."""
    condition_number = np.linalg.cond(design)
    if condition_number > 1e10:
        warnings.warn(
            f"{label} design matrix is poorly conditioned (cond={condition_number:.2e}).",
            stacklevel=2,
        )


def _trapz_weights(grid: np.ndarray) -> np.ndarray:
    """Trapezoidal quadrature weights for a 1-D grid."""
    weights = np.zeros_like(grid, dtype=float)
    weights[1:-1] = (grid[2:] - grid[:-2]) / 2.0
    weights[0] = (grid[1] - grid[0]) / 2.0
    weights[-1] = (grid[-1] - grid[-2]) / 2.0
    return weights


def _sine_transform_to_r(
    q_values: np.ndarray, columns: np.ndarray, r_values: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Q -> r sine transform: f(r) = (2/pi) * integral f(Q) sin(Qr) dQ."""
    single = columns.ndim == 1
    cols = columns[:, None] if single else columns
    transformed = (2.0 / np.pi) * (
        np.sin(np.outer(r_values, q_values)) @ (cols * weights[:, None])
    )
    return transformed[:, 0] if single else transformed


def _fit_background(
    q_values: np.ndarray,
    residual: np.ndarray,
    fit_mask: np.ndarray,
    poly_degree: int,
    background_type: BackgroundType,
) -> np.ndarray:
    """Least-squares background (electron units) matching `residual` over fit_mask."""
    q_min_fit, q_max_fit = (
        float(q_values[fit_mask].min()),
        float(q_values[fit_mask].max()),
    )
    weights = q_values[fit_mask]
    basis = _background_basis(
        q_values[fit_mask], q_min_fit, q_max_fit, poly_degree, background_type
    )
    coeffs, *_ = np.linalg.lstsq(
        basis * weights[:, None], residual[fit_mask] * weights, rcond=None
    )
    _warn_if_ill_conditioned(basis * weights[:, None], "Background fitting")

    full_basis = _background_basis(
        q_values, q_min_fit, q_max_fit, poly_degree, background_type
    )
    return _clip_background_extrapolation(
        q_values, full_basis @ coeffs, q_min_fit, q_max_fit
    )


def _billinge_low_r_correction(
    q_values: np.ndarray,
    intensity_q: np.ndarray,
    target: np.ndarray,
    f_mean_sq: np.ndarray,
    scale_factor: float,
    background: np.ndarray,
    q_max_fit: float,
    r_poly: float,
    r_step: float,
    background_type: BackgroundType,
    window: np.ndarray,
) -> np.ndarray:
    """PDFgetX3-style additive correction cancelling low-r drift in G(r) below r_poly.
    Degree is Nyquist-limited (floor(q_max_fit*r_poly/pi)) so it can only remove
    slow drift, not genuine structure. scale_factor is untouched.
    """
    degree = max(int(np.floor(q_max_fit * r_poly / np.pi)), 0)
    r_low = np.arange(0.0, r_poly, r_step)
    if r_low.size <= degree + 1:
        raise PDFNormalisationError("r_poly too small for the Nyquist-derived degree")

    q_weights = _trapz_weights(q_values)
    s_minus_1 = (scale_factor * intensity_q - background - target) / f_mean_sq
    current_d_r = _sine_transform_to_r(
        q_values, q_values * s_minus_1 * window, r_low, q_weights
    )

    basis = _background_basis(
        q_values, float(np.amin(q_values)), q_max_fit, degree, background_type
    )
    design_columns = (q_values / f_mean_sq)[:, None] * basis * window[:, None]
    design = _sine_transform_to_r(q_values, design_columns, r_low, q_weights)

    coeffs, *_ = np.linalg.lstsq(design, current_d_r, rcond=None)
    _warn_if_ill_conditioned(design, "billinge low-r correction")
    return background + basis @ coeffs


def normalise_intensity(
    q_values: SerialisableNDArray,
    intensity_q: SerialisableNDArray,
    f_sq_mean: SerialisableNDArray,
    f_mean_sq: SerialisableNDArray,
    compton: SerialisableNDArray,
    q_min_fit: float,
    q_max_fit: float,
    poly_degree: int,
    background_type: BackgroundType,
    method: NormalisationMethod,
    rho: float | None,
    r_poly: float,
    r_step: float,
    window: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Fit scale_factor and an electron-unit background(Q) for the given method:
    I_eu(Q) = scale_factor * I_raw(Q) - background(Q), matched to <f^2>+Compton.
    See the NormalisationMethod summary near the top of the file.
    """
    q_max_fit = min(q_max_fit, float(np.amax(q_values)))
    q_min_fit = max(q_min_fit, float(np.amin(q_values)))
    fit_mask = (q_values >= q_min_fit) & (q_values <= q_max_fit)
    if fit_mask.sum() < poly_degree + 2:
        fit_mask = np.ones_like(q_values, dtype=bool)
    target = f_sq_mean + compton

    if method in ("krogh_moe", "billinge"):
        if rho is None:
            raise ValueError(f"method='{method}' requires number_density")
        weights = _trapz_weights(q_values[fit_mask])
        sum_rule_weight = weights * q_values[fit_mask] ** 2
        scale_factor = (
            np.sum(sum_rule_weight * target[fit_mask]) - 2.0 * np.pi**2 * rho
        ) / np.sum(sum_rule_weight * intensity_q[fit_mask])
        background = (
            _fit_background(
                q_values,
                scale_factor * intensity_q - target,
                fit_mask,
                poly_degree,
                background_type,
            )
            if poly_degree > 0
            else np.zeros_like(q_values)
        )

    elif method == "warren":
        # Joint linear fit of [scale_factor, background_coeffs] in one least-squares solve.
        basis = _background_basis(
            q_values[fit_mask], q_min_fit, q_max_fit, poly_degree, background_type
        )
        design = np.column_stack([intensity_q[fit_mask], -basis])
        weights = q_values[fit_mask]
        solution, *_ = np.linalg.lstsq(
            design * weights[:, None], target[fit_mask] * weights, rcond=None
        )
        _warn_if_ill_conditioned(design * weights[:, None], "warren fit")
        scale_factor = float(solution[0])
        full_basis = _background_basis(
            q_values, q_min_fit, q_max_fit, poly_degree, background_type
        )
        background = _clip_background_extrapolation(
            q_values, full_basis @ solution[1:], q_min_fit, q_max_fit
        )

    elif method == "eggert":
        # Nonlinear least-squares fit of scale + a constant multiple-scattering offset.
        def residuals(params: np.ndarray) -> np.ndarray:
            scale, offset = params
            return q_values[fit_mask] * (
                scale * intensity_q[fit_mask] - offset - target[fit_mask]
            )

        initial_scale = 1.0 / max(float(np.mean(intensity_q[fit_mask])), 1e-12)
        fit = least_squares(residuals, x0=[initial_scale, 0.0])
        scale_factor, offset = fit.x
        background = np.full_like(q_values, offset)

    else:
        raise ValueError(f"Unknown normalisation method '{method}'.")

    if scale_factor <= 0:
        raise PDFNormalisationError(
            f"'{method}' normalisation gave a non-positive scale_factor"
        )

    if method == "billinge":
        background = _billinge_low_r_correction(
            q_values,
            intensity_q,
            target,
            f_mean_sq,
            scale_factor,
            background,
            q_max_fit,
            r_poly,
            r_step,
            background_type,
            window,
        )

    return float(scale_factor), background


# --------------------------------------------------------------------------
# Fourier transforms and termination windows
# --------------------------------------------------------------------------


def make_termination_window(
    q_values: SerialisableNDArray,
    q_max: float,
    window_type: WindowType | None,
    soper_lorch_power: int | float | None = 2,
) -> np.ndarray:
    """Multiplicative window W(Q) applied to F(Q) to reduce Fourier termination ripples."""
    if window_type is None or window_type == "none":
        return np.ones_like(q_values)
    if window_type == "cosine":
        return 0.5 * (1.0 + np.cos(np.pi * q_values / q_max))
    if window_type in ("lorch", "soper_lorch"):
        power = 1 if window_type == "lorch" else soper_lorch_power
        if power is None:
            raise ValueError(f"window_type='{window_type}' needs soper_lorch_power set")
        window = np.ones_like(q_values)
        nonzero = q_values > 0.0
        arg = np.pi * q_values[nonzero] / q_max
        window[nonzero] = (np.sin(arg) / arg) ** power
        return window
    raise ValueError(f"Unknown termination_window '{window_type}'.")


def sine_transform_fq_to_gr(
    q_values: SerialisableNDArray,
    f_q: SerialisableNDArray,
    r_values: SerialisableNDArray,
    q_step_size: float,
    sin_qr: np.ndarray | None = None,
) -> np.ndarray:
    """Forward sine transform: G(r) = (2/pi) * integral F(Q) sin(Qr) dQ."""
    if sin_qr is None:
        sin_qr = np.sin(np.outer(r_values, q_values))
    return (2.0 / np.pi) * q_step_size * (sin_qr @ f_q)


def sine_transform_gr_to_fq(
    r_values: SerialisableNDArray,
    g_r: SerialisableNDArray,
    q_values: SerialisableNDArray,
    r_weights: np.ndarray | None = None,
    sin_qr: np.ndarray | None = None,
) -> np.ndarray:
    """Inverse (back) sine transform: F(Q) = integral G(r) sin(Qr) dr."""
    if sin_qr is None:
        sin_qr = np.sin(np.outer(q_values, r_values))
    if r_weights is None:
        return np.trapezoid(sin_qr * g_r[np.newaxis, :], r_values, axis=1)  # type: ignore
    return sin_qr @ (g_r * r_weights)


def sine_transform_sigma(
    q_values: SerialisableNDArray,
    sigma_f_q: SerialisableNDArray,
    r_values: SerialisableNDArray,
    q_step_size: float,
    sin_qr_sq: np.ndarray | None = None,
) -> np.ndarray:
    """Propagate independent per-Q uncertainty in F(Q) through the sine transform to G(r)."""
    if sin_qr_sq is None:
        sin_qr_sq = np.sin(np.outer(r_values, q_values)) ** 2
    return (2.0 / np.pi) * q_step_size * np.sqrt(sin_qr_sq @ sigma_f_q**2)


def linear_interp_operator(x_new: np.ndarray, x_old: np.ndarray) -> np.ndarray:
    """Dense matrix L such that L @ y_old == np.interp(x_new, x_old, y_old); x_old sorted."""
    n_old = len(x_old)
    idx = np.clip(np.searchsorted(x_old, x_new), 1, n_old - 1)
    x0, x1 = x_old[idx - 1], x_old[idx]
    frac = np.clip(
        np.divide(x_new - x0, x1 - x0, out=np.zeros_like(x_new), where=(x1 - x0) != 0),
        0.0,
        1.0,
    )
    interp_matrix = np.zeros((len(x_new), n_old))
    rows = np.arange(len(x_new))
    interp_matrix[rows, idx - 1] = 1.0 - frac
    interp_matrix[rows, idx] = frac
    interp_matrix[(x_new < x_old[0]) | (x_new > x_old[-1]), :] = 0.0
    return interp_matrix


# --------------------------------------------------------------------------
# Real-space constraint: Toby-Egami back-Fourier method (Toby & Egami, 1992)
# --------------------------------------------------------------------------


def _auto_r_constraint(
    r_values: SerialisableNDArray,
    g_values: SerialisableNDArray,
    rho0: float,
    r_search_min: float = 1.2,
    r_search_max: float = 3.5,
) -> float:
    """First upward crossing of G(r) through the physical baseline in the search window."""
    deviation = g_values - gr_baseline(r_values, rho0)
    idx = (r_values >= r_search_min) & (r_values <= r_search_max)
    if not np.any(idx):
        return r_search_min

    r_sub, d_sub = r_values[idx], deviation[idx]
    crossings = np.where((d_sub[:-1] < 0.0) & (d_sub[1:] >= 0.0))[0]
    if len(crossings) == 0:
        return float(r_search_min)

    i = crossings[0]
    r0, r1, d0, d1 = (
        float(r_sub[i]),
        float(r_sub[i + 1]),
        float(d_sub[i]),
        float(d_sub[i + 1]),
    )
    return float(np.clip(r0 - d0 * (r1 - r0) / (d1 - d0), r_search_min, r_search_max))


def apply_real_space_constraint(
    r_values: SerialisableNDArray,
    g_values: SerialisableNDArray,
    f_q: SerialisableNDArray,
    q_values: SerialisableNDArray,
    q_step_size: float,
    rho0: float,
    r_max_constraint: float,
    n_iterations: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Iteratively force G(r) = -4*pi*r*rho0 below r_max_constraint (Toby & Egami, 1992)."""
    g_values, f_q = g_values.copy(), f_q.copy()
    g_physical_full = gr_baseline(r_values, rho0)
    mask_low = r_values <= r_max_constraint
    mask_phys = ~mask_low

    sin_qr_gr_to_fq = np.sin(np.outer(q_values, r_values))
    sin_qr_fq_to_gr_pos = np.sin(np.outer(r_values, q_values))[1:, :]
    r_values_pos = r_values[1:]
    r_weights = _trapz_weights(r_values)

    for _ in range(n_iterations):
        delta_g = np.where(mask_low, g_values - g_physical_full, 0.0)
        rms = float(np.sqrt(np.mean(delta_g[mask_low] ** 2)))
        g_peak = (
            float(np.max(np.abs(g_values[mask_phys]))) if np.any(mask_phys) else 1.0
        )
        if rms / max(g_peak, 1e-12) < 1e-5:
            break

        f_q -= sine_transform_gr_to_fq(
            r_values, delta_g, q_values, r_weights=r_weights, sin_qr=sin_qr_gr_to_fq
        )
        g_pos = sine_transform_fq_to_gr(
            q_values, f_q, r_values_pos, q_step_size, sin_qr=sin_qr_fq_to_gr_pos
        )
        g_values = np.concatenate([[0.0], g_pos])

    return r_values, g_values, f_q


def _propagate_uncertainty(
    q_values: SerialisableNDArray,
    sigma_i_q: np.ndarray,
    scale_factor: float,
    inv_f_mean_sq: SerialisableNDArray,
    below_q_min_mask: np.ndarray,
    window: np.ndarray,
    r_pos: SerialisableNDArray,
    q_step_size: float,
    envelope: np.ndarray,
    broadening_kernel: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, None]:
    """Independent per-Q uncertainty propagation; no cross-correlation, no dense (n,n) matrices."""
    sigma_i_eu = scale_factor * sigma_i_q
    sigma_s_q = sigma_i_eu * inv_f_mean_sq
    sigma_s_q[below_q_min_mask] = 0.0
    sigma_f_mod = q_values * sigma_s_q * window

    sigma_g_full = np.concatenate(
        [[0.0], sine_transform_sigma(q_values, sigma_f_mod, r_pos, q_step_size)]
    )
    sigma_g_full = sigma_g_full * envelope
    if broadening_kernel is not None:
        sigma_g_full = np.sqrt(
            np.maximum((broadening_kernel**2) @ sigma_g_full**2, 0.0)
        )

    return sigma_i_eu, sigma_s_q, q_values * sigma_s_q, sigma_g_full, None


# --------------------------------------------------------------------------
# Pipeline stages: load -> correct -> grids -> resample -> normalise -> transform
# --------------------------------------------------------------------------


def _load_and_correct_intensity(
    xy_path: Path, config: PDFConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load raw .xy data and apply background subtraction, absorption, and polarisation."""
    logger.info(f"compute_pdf: loading {xy_path}")
    two_theta_deg, intensity = load_xy(xy_path)

    if config.data.e is not None and len(config.data.e) == len(intensity):
        sigma = np.asarray(config.data.e, dtype=np.float64).copy()
    else:
        sigma = np.sqrt(np.maximum(intensity, 1.0))

    if config.background_file is not None:
        two_theta_bg, intensity_bg = load_xy(config.background_file)
        intensity_bg_interp = np.interp(
            two_theta_deg, two_theta_bg, intensity_bg, left=0.0, right=0.0
        )
        intensity = intensity - config.background_scale * intensity_bg_interp

    if config.fluorescence_correction:
        fluorescence_level = config.fluorescence_level
        if fluorescence_level is None:
            fluorescence_level = estimate_fluorescence_level(
                intensity, config.fluorescence_percentile
            )
        logger.info(f"fluorescence correction: floor={fluorescence_level:.6g}")
        # Fluorescence is isotropic re-emission, not coherent diffraction, so it
        # is removed before the angle-dependent absorption/polarisation steps.
        intensity = intensity - fluorescence_level

    intensity = np.maximum(intensity, 0.0)

    if config.absorption_correction:
        absorption = cylindrical_absorption_correction(
            two_theta_deg, float(config.mu_r)
        )  # type: ignore[arg-type]
        intensity = intensity * absorption
        sigma = sigma * absorption

    if config.polarisation_factor:
        polarisation = np.maximum(
            lorentz_polarisation(
                two_theta_deg, config.is_synchrotron, config.polarisation_p
            ),
            1e-12,
        )
        intensity = intensity / polarisation
        sigma = sigma / polarisation

    return two_theta_deg, intensity, sigma


def _build_q_grid(
    config: PDFConfig,
    two_theta_deg: np.ndarray,
    intensity: np.ndarray,
    sigma: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
    """Convert 2theta -> Q, optionally auto-select q_max, and build the uniform Q grid."""
    q_raw = two_theta_to_q(two_theta_deg, float(config.data.wavelength))
    q_max_data, q_min_data = float(q_raw.max()), float(q_raw.min())

    if config.auto_q_max:
        order = np.argsort(q_raw)
        q_max_auto = auto_select_q_max(
            q_raw[order],
            intensity[order],
            sigma[order],
            snr_threshold=config.auto_q_max_snr_threshold,
            q_search_min=config.auto_q_max_search_min,
        )
        q_max_use = min(config.q_max, q_max_data, q_max_auto)
    else:
        q_max_use = min(config.q_max, q_max_data)

    q_step_size = config.q_step or min(float(np.median(np.diff(q_raw))), 0.05)
    q_values = np.arange(
        config.q_min, q_max_use + 0.5 * q_step_size, q_step_size, dtype=np.float64
    )
    logger.info(
        f"Q-grid: [{config.q_min}, {q_max_use:.3f}] step {q_step_size:.5f}, n_q={len(q_values)}"
    )
    return q_raw, q_values, q_step_size, q_min_data, q_max_data, q_max_use


def _build_r_grid(config: PDFConfig) -> np.ndarray:
    """Positive part of the uniform r-grid (r=0 is prepended once transformed)."""
    return np.arange(
        config.r_step,
        config.r_max + 0.5 * config.r_step,
        config.r_step,
        dtype=np.float64,
    )


def _resample_onto_q_grid(
    q_raw: np.ndarray,
    intensity: np.ndarray,
    sigma: np.ndarray,
    q_values: np.ndarray,
    q_min_data: float,
    q_max_data: float,
    q_step_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Top-hat box-average onto the uniform Q grid; cubic spline fills empty bins.
    Box-averaging (rather than point-sampling) avoids aliasing wherever the
    raw 2theta->Q spacing is finer than q_step_size (see the module notes).
    """
    valid = intensity > 0
    if valid.sum() < 4:
        raise ValueError("Fewer than 4 valid data points; cannot fit a spline.")
    order = np.argsort(q_raw[valid])
    q_raw_valid = q_raw[valid][order]
    intensity_valid = intensity[valid][order]
    sigma_valid = sigma[valid][order]

    intensity_spline = UnivariateSpline(q_raw_valid, intensity_valid, s=0, k=3, ext=1)
    intensity_q_spline = np.maximum(intensity_spline(q_values), 0.0)
    sigma_i_q_spline = np.interp(
        q_values, q_raw_valid, sigma_valid, left=0.0, right=0.0
    )

    intensity_q_binned, sigma_i_q_binned, bin_counts = top_hat_bin_average(
        q_raw_valid, intensity_valid, sigma_valid, q_values, q_step_size
    )

    empty_bin = bin_counts == 0
    intensity_q = np.where(empty_bin, intensity_q_spline, intensity_q_binned)
    sigma_i_q = np.where(empty_bin, sigma_i_q_spline, sigma_i_q_binned)

    intensity_q = np.maximum(intensity_q, 0.0)
    out_of_range = (q_values < q_min_data) | (q_values > q_max_data)
    intensity_q[out_of_range] = 0.0
    sigma_i_q[out_of_range] = 0.0
    return intensity_q, sigma_i_q, q_raw_valid, sigma_valid


def calculate_structure_functions(
    q_values: np.ndarray,
    intensity_q: np.ndarray,
    scale_factor: float,
    background: np.ndarray,
    f_sq_mean: np.ndarray,
    f_mean_sq: np.ndarray,
    compton: np.ndarray,
    q_min: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """I(Q) [electron units] -> S(Q) -> F(Q) = Q[S(Q)-1]. Returns (i_eu, s_q, f_q, inv_f_mean_sq, below_q_min_mask)."""
    i_eu = scale_factor * intensity_q - background

    # S(Q) = [I_eu - I_Compton - (<f^2> - <f>^2)] / <f>^2; the Laue term is zero for one element.
    with np.errstate(divide="ignore", invalid="ignore"):
        s_q = (i_eu - compton - (f_sq_mean - f_mean_sq)) / np.maximum(f_mean_sq, 1e-30)
    s_q = np.where(np.isfinite(s_q), s_q, 1.0)

    inv_f_mean_sq = 1.0 / np.maximum(f_mean_sq, 1e-30)
    below_q_min_mask = q_values < q_min
    s_q[below_q_min_mask] = 1.0

    f_q = q_values * (s_q - 1.0)
    f_q[q_values == 0.0] = 0.0
    return i_eu, s_q, f_q, inv_f_mean_sq, below_q_min_mask


class _RealSpacePDF(NamedTuple):
    r_full: np.ndarray
    g_full: np.ndarray
    f_mod: np.ndarray
    window: np.ndarray
    real_space_constraint_applied: bool
    r_constraint_cutoff: float | None
    envelope: np.ndarray
    broadening_kernel: np.ndarray | None


def _transform_to_gr(
    q_values: np.ndarray,
    f_q: np.ndarray,
    r_pos: np.ndarray,
    q_step_size: float,
    config: PDFConfig,
    q_max_use: float,
    window: np.ndarray,
) -> _RealSpacePDF:
    """Window F(Q), sine-transform to G(r), apply the Toby-Egami low-r constraint, then qdamp/qbroad."""
    f_mod = f_q * window
    r_full = np.concatenate([[0.0], r_pos])
    g_full = np.concatenate(
        [[0.0], sine_transform_fq_to_gr(q_values, f_mod, r_pos, q_step_size)]
    )

    real_space_constraint_applied = False
    r_constraint_cutoff: float | None = None
    if config.use_real_space_constraint and config.real_space_constraint_iterations > 0:
        r_constraint_cutoff = config.r_constraint_max or _auto_r_constraint(
            r_full,
            g_full,
            config.number_density,
            r_search_min=config.r_constraint_search_min,
            r_search_max=config.r_constraint_search_max,
        )
        r_full, g_full, f_mod = apply_real_space_constraint(
            r_full,
            g_full,
            f_mod,
            q_values,
            q_step_size,
            config.number_density,
            r_max_constraint=r_constraint_cutoff,
            n_iterations=config.real_space_constraint_iterations,
        )
        real_space_constraint_applied = True

    envelope = np.exp(-0.5 * (config.qdamp * r_full) ** 2)
    g_full = g_full * envelope

    broadening_kernel = None
    if config.qbroad > 0.0:
        broadening_kernel = r_dependent_broadening_matrix(r_full, config.qbroad)
        g_full = broadening_kernel @ g_full

    return _RealSpacePDF(
        r_full,
        g_full,
        f_mod,
        window,
        real_space_constraint_applied,
        r_constraint_cutoff,
        envelope,
        broadening_kernel,
    )


def compute_pdf(xy_path: Path, config: PDFConfig) -> PDFResult:
    """Full PDF pipeline: load -> correct -> build grids -> normalise -> S(Q) -> F(Q) -> G(r)."""
    two_theta_deg, intensity, sigma = _load_and_correct_intensity(xy_path, config)
    q_raw, q_values, q_step_size, q_min_data, q_max_data, q_max_use = _build_q_grid(
        config, two_theta_deg, intensity, sigma
    )
    r_pos = _build_r_grid(config)
    intensity_q, sigma_i_q, _, _ = _resample_onto_q_grid(
        q_raw, intensity, sigma, q_values, q_min_data, q_max_data, q_step_size
    )
    f_mean, f_mean_sq, f_sq_mean, compton = build_scattering_factors(
        config.composition, q_values
    )

    window = make_termination_window(
        q_values, q_max_use, config.termination_window, config.soper_lorch_power
    )

    scale_factor, background = normalise_intensity(
        q_values,
        intensity_q,
        f_sq_mean,
        f_mean_sq,
        compton,
        q_min_fit=float(config.norm_q_min),  # type: ignore[arg-type]
        q_max_fit=q_max_use,
        poly_degree=config.norm_poly_degree,
        background_type=config.background_type,
        method=config.normalisation_method,
        rho=config.number_density,
        r_poly=config.r_poly,
        r_step=config.r_step,
        window=window,
    )
    logger.info(
        f"normalisation: {config.normalisation_method} scale_factor: {scale_factor:.6g} "
        f"background range [{background.min():.4g}, {background.max():.4g}]"
    )

    i_eu, s_q, f_q, inv_f_mean_sq, below_q_min_mask = calculate_structure_functions(
        q_values,
        intensity_q,
        scale_factor,
        background,
        f_sq_mean,
        f_mean_sq,
        compton,
        config.q_min,
    )
    gr = _transform_to_gr(q_values, f_q, r_pos, q_step_size, config, q_max_use, window)

    sigma_i_eu, sigma_s_q, sigma_f_q, sigma_g_full, gr_covariance = (
        _propagate_uncertainty(
            q_values,
            sigma_i_q,
            scale_factor,
            inv_f_mean_sq,
            below_q_min_mask,
            gr.window,
            r_pos,
            q_step_size,
            gr.envelope,
            gr.broadening_kernel,
        )
    )

    iq_result = ScatteringData(
        x=q_values,
        y=i_eu,
        e=sigma_i_eu,
        x_unit="q",
        y_unit="I(Q) (e.u.)",
        data_type=config.data.data_type,
        wavelength=config.data.wavelength,
        source=str(xy_path),
    )
    sq_result = XYEData(x=q_values, y=s_q, e=sigma_s_q, x_unit="q", y_unit="S(Q)")
    fq_result = XYEData(
        x=q_values, y=f_q, e=sigma_f_q, x_unit="q", y_unit="F(Q) = Q[S(Q)-1] (A^-1)"
    )
    gr_result = XYEData(
        x=gr.r_full, y=gr.g_full, e=sigma_g_full, x_unit="r", y_unit="G(r) (A^-2)"
    )

    return PDFResult(
        iq=iq_result,
        sq=sq_result,
        fq=fq_result,
        gr=gr_result,
        gr_covariance=gr_covariance,
        scale_factor=scale_factor,
        background=background,
        number_density=config.number_density,
        sample_name=config.sample_name,
    )


def run_pdf(
    xy_path: Path | str,
    formula: str | dict | ChemicalFormula,
    wavelength: float,
    number_density: float,
    sample_name: str | None = None,
    q_min: float = 0.5,
    q_max: float = 30.0,
    r_max: float = 30.0,
    r_step: float = 0.01,
    qdamp: float = 0.030,
    qbroad: float = 0.0,
    termination_window: WindowType = "soper_lorch",
    soper_lorch_power: int | float = 2,
    use_real_space_constraint: bool = True,
    real_space_constraint_iterations: int = 10,
    r_constraint_max: float | None = None,
    norm_poly_degree: int = 5,
    norm_q_min: float | None = None,
    background_type: BackgroundType = "chebyshev",
    normalisation_method: NormalisationMethod = "krogh_moe",
    r_poly: float = 1.5,
    is_synchrotron: bool = True,
    polarisation_p: float = 0.99,
    background_file: Path | str | None = None,
    background_scale: float = 1.0,
    absorption_correction: bool = False,
    mu_r: float | None = None,
    fluorescence_correction: bool = False,
    fluorescence_level: float | None = None,
    fluorescence_percentile: float = 1.0,
    auto_q_max: bool = False,
    auto_q_max_snr_threshold: float = 1.5,
    export_formats: list[ExportFormat] | list[str] | None = None,
    output_dir: Path | str | None = None,
) -> PDFResult:
    """Compute G(r) from a raw .xy file and optionally write output files.
    composition is e.g. {"Si": 1} or {"Pb": 1, "Ti": 1, "O": 3}; number_density
    is atoms per cell divided by cell volume (atoms/A^3).
    """
    if isinstance(formula, str):
        composition = ChemicalFormula(formula=formula)
    elif isinstance(formula, dict):
        composition = ChemicalFormula.load_from_composition(formula)
    elif isinstance(formula, ChemicalFormula):
        composition = formula
    else:
        raise ValueError("composition must be str or dict")

    export_formats = export_formats or [ExportFormat.GR]
    output_dir = (
        Path(output_dir) if output_dir is not None else Path(xy_path).parent.resolve()
    )
    sample_name = sample_name or Path(xy_path).stem

    data = ScatteringData.from_xye(
        filepath=xy_path, x_unit="tth", data_type="xray", wavelength=wavelength
    )

    config = PDFConfig(
        composition=composition,
        data=data,
        sample_name=sample_name,
        number_density=number_density,
        q_min=q_min,
        q_max=q_max,
        r_max=r_max,
        r_step=r_step,
        qdamp=qdamp,
        qbroad=qbroad,
        termination_window=termination_window,
        soper_lorch_power=soper_lorch_power,
        use_real_space_constraint=use_real_space_constraint,
        real_space_constraint_iterations=real_space_constraint_iterations,
        r_constraint_max=r_constraint_max,
        norm_poly_degree=norm_poly_degree,
        norm_q_min=norm_q_min,
        background_type=background_type,
        normalisation_method=normalisation_method,
        r_poly=r_poly,
        is_synchrotron=is_synchrotron,
        polarisation_p=polarisation_p,
        background_file=Path(background_file) if background_file else None,
        background_scale=background_scale,
        absorption_correction=absorption_correction,
        mu_r=mu_r,
        fluorescence_correction=fluorescence_correction,
        fluorescence_level=fluorescence_level,
        fluorescence_percentile=fluorescence_percentile,
        auto_q_max=auto_q_max,
        auto_q_max_snr_threshold=auto_q_max_snr_threshold,
        export_formats=[ExportFormat(fmt) for fmt in export_formats],
        output_dir=Path(output_dir),
    )

    result = compute_pdf(Path(xy_path), config)
    result.save_results(
        export_formats=config.export_formats, output_dir=config.output_dir
    )
    return result


if __name__ == "__main__":
    xy_file = Path("/workspaces/xrpd-toolbox/tests/data/Si_pe2_i15_1.xy")

    # Crystalline Si: 8 atoms per conventional cubic unit cell, a = 5.4309 A.
    rho_si = 8.0 / 5.4309**3

    result = run_pdf(
        xy_path=xy_file,
        formula="Si",
        sample_name="Si_pdf",
        wavelength=0.16,
        number_density=rho_si,
        q_min=0.8,
        q_max=24.0,
        r_max=30.0,
        r_step=0.02,
        qdamp=0.003,
        is_synchrotron=True,
        polarisation_p=0.99,
        termination_window="soper_lorch",
        soper_lorch_power=2,
        background_type="chebyshev",
        normalisation_method="krogh_moe",
        use_real_space_constraint=True,
        real_space_constraint_iterations=10,
        r_constraint_max=2,
        norm_poly_degree=3,
        auto_q_max=False,
        export_formats=["gr", "sq", "fq", "iq"],
    )

    result.plot(
        save_filepath="./pdf.png",
        ref_filepath=Path("/workspaces/xrpd-toolbox/tests/data/Si_pe2_i15_1.gr"),
    )

    coordination_fits = analyse_rdf_coordination(result, n_peaks=3)
    for fit_index, coordination_fit in enumerate(coordination_fits, start=1):
        logger.info(
            f"peak {fit_index}: r={coordination_fit.centre:.3f} A, "
            f"cn={coordination_fit.coordination_number:.3f}"
        )
    plot_rdf_coordination(result, coordination_fits, save_filepath="./cn.png")
