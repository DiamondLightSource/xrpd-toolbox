"""Pair distribution function (PDF) calculator.

Converts powder X-ray diffraction data (2-theta vs intensity) into the
reduced PDF G(r) = 4*pi*r*[rho(r) - rho_0], obtained via the sine Fourier
transform of the reduced structure function F(Q) = Q*[S(Q) - 1]:

    G(r) = (2/pi) * integral_0^Qmax F(Q) * W(Q) * sin(Q*r) dQ

For r below the first interatomic distance, no atom pairs exist and
G(r) = -4*pi*r*rho_0 exactly. This physical constraint is enforced
iteratively by the Toby-Egami back-Fourier method.

The absolute scale factor scale_factor is fixed by the Krogh-Moe/Norman condition
that the measured intensity converges to the self-scattering at high Q.

References: Egami & Billinge (2003); Toby & Egami, Acta Cryst. A48 (1992);
Juhas et al., J. Appl. Cryst. 46 (2013); Norman, Acta Cryst. 10 (1957);
Krogh-Moe, Acta Cryst. 9 (1956); Waasmaier & Kirfel, Acta Cryst. A51 (1995);
Lorch, J. Phys. C 2 (1969).
"""

# assumes reduced pdf convention: g_of_r = 4 * pi * rho * r * (pcf - 1)
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
from scipy.optimize import curve_fit
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
from xrpd_toolbox.utils.utils import load_xy

WindowType = Literal["soper_lorch", "lorch", "cosine", "none"]
BackgroundType = Literal[
    "constant", "polynomial", "chebyshev", "linear", "bspline", "cosine"
]
NormalisationMethod = Literal["krogh_moe", "eggert", "billinge", "warren"]


def gr_baseline(r_values: np.ndarray, rho0: float) -> np.ndarray:
    """Physical low-r baseline: G(r) = -4*pi*r*rho0 (no pairs below r_min)."""
    return -4.0 * np.pi * r_values * rho0


class ExportFormat(StrEnum):
    GR = "gr"  # Reduced PDF G(r)
    SQ = "sq"  # Total structure function S(Q)
    IQ = "iq"  # Coherent intensity in electron units I(Q)
    FQ = "fq"  # Reduced structure function F(Q) = Q[S(Q)-1]


class PDFNormalisationError(RuntimeError):
    """Raised when the Krogh-Moe/Norman scale factor cannot be determined."""


class PDFConfig(XRPDBaseModel):
    """All parameters controlling the PDF calculation."""

    composition: ChemicalFormula
    sample_name: str = Field(default="pdf")
    data: ScatteringData
    # wavelength: float = Field(gt=0, description="X-ray wavelength (Å)")
    number_density: float = Field(gt=0, description="Atomic number density (atoms/Å³)")

    q_min: float = Field(default=0.5, gt=0)
    q_max: float = Field(default=30.0, gt=0)
    q_step: float | None = Field(default=None, gt=0)

    polarisation_factor: bool = True
    is_synchrotron: bool = True
    polarisation_p: float = Field(default=0.99, ge=0.0, le=1.0)
    background_file: Path | None = None
    background_scale: float = 1.0
    known_scale_factor: float | None = Field(
        default=None,
        gt=0.0,
        description=(
            "Use a user-supplied scale factor instead of fitting it. "
            "When provided, only the background is refined."
        ),
    )

    absorption_correction: bool = True
    mu_r: float | None = Field(
        default=None,
        gt=0,
        description=(
            "Linear absorption coefficient times cylindrical sample radius "
            "(dimensionless, mu*R). Not computed from composition/energy "
            "here -- measure via transmission, or supply from tabulated "
            "mass attenuation coefficients. Required if "
            "absorption_correction=True."
        ),
    )

    norm_poly_degree: int = Field(default=3, ge=0)
    norm_q_min: float | None = None
    background_type: BackgroundType = "chebyshev"
    normalisation_method: NormalisationMethod = "krogh_moe"

    compute_full_covariance: bool = Field(
        default=True,
        description=(
            "Attempt full (n_r, n_r) covariance propagation for G(r) "
            "(normalisation_method='krogh_moe' only). Automatically skipped "
            "-- falling back to cheap independent-per-Q uncertainty -- if "
            "the problem size exceeds covariance_max_points, to avoid "
            "the O(n^2) memory / O(n^3) compute blowup of dense linear "
            "algebra at full data resolution."
        ),
    )
    covariance_max_points: int = Field(
        default=1000,
        gt=0,
        description=(
            "Full covariance is skipped if either the Q-grid or r-grid "
            "point count exceeds this. 800 points means an (800, 800) "
            "matrix (~5 MB) and O(800^3) matmuls (~0.5 GFLOP each) -- "
            "safe on essentially any machine. Real full-resolution data "
            "easily reaches n_q/n_r in the thousands, where the same "
            "matrices are hundreds of MB each and matmuls are O(10-100) "
            "GFLOP; raise this deliberately, and expect multi-GB memory "
            "use and real wall-clock cost, not a hang."
        ),
    )

    auto_q_max: bool = Field(
        default=False,
        description=(
            "Automatically choose q_max from where the local signal-to-"
            "noise ratio of I(Q) drops below auto_q_max_snr_threshold, "
            "instead of using the fixed q_max value."
        ),
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
        description=(
            "r-space PDF envelope decay from finite Q-resolution: "
            "G(r) *= exp(-(qdamp*r)^2/2). Matches the standard PDFgetX3/"
            "PDFgui convention (an r-space envelope, not a Q-space damping "
            "of F(Q))."
        ),
    )
    qbroad: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "r-dependent peak broadening from finite Q-resolution: each "
            "point in G(r) is smeared by a Gaussian of width "
            "sigma(r) = qbroad*r^2, matching the standard PDFgetX3/PDFgui "
            "convention. 0 disables broadening."
        ),
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
    """Computed PDF results; each stage carries its own propagated uncertainty
    via XYEData/ScatteringData rather than bare arrays.

    Uncertainty covers counting stats -> polarisation -> Q-grid interpolation
    -> normalisation -> S(Q)/F(Q) -> the sine transform to G(r), but treats
    scale_factor/background as fixed (their own fit uncertainty isn't
    propagated) -- error bars here are a lower bound, not PDFgetX3's full covariance.
    """

    iq: ScatteringData  # I(Q) in electron units, x_unit="q"
    sq: XYEData  # S(Q) total structure function, x_unit="q"
    fq: XYEData  # F(Q) = Q[S(Q)-1] (Å⁻¹), x_unit="q"
    gr: XYEData  # G(r) (Å⁻²), x_unit="r"
    gr_covariance: SerialisableNDArray | None = None  # full (n_r, n_r) Cov[G(r)]
    scale_factor: float
    background: SerialisableNDArray
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
        """g(r) = 1 - G(r)/baseline, equivalent to g(r) = 1 + G(r)/(4*pi*r*rho0).
        Sign flipped vs. the naive form because baseline is negative: in the
        constrained no-pairs region G(r) == baseline, so g(r) == 0.
        """
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

        for fmt in export_formats:
            if fmt == ExportFormat.GR:
                header = "G(r) — reduced PDF\n# r (Å)   G(r) (Å⁻²)   sigma_G(r)"
                self.gr.save_to_xye(
                    output_dir / f"{self.sample_name}.{fmt}", header=header
                )

            if fmt == ExportFormat.SQ:
                header = (
                    "S(Q) — total structure function\n# Q (Å⁻¹)   S(Q)   sigma_S(Q)"
                )
                self.sq.save_to_xye(
                    output_dir / f"{self.sample_name}.{fmt}", header=header
                )

            if fmt == ExportFormat.IQ:
                header = "I(Q) — Intensity (e.u.)\n# Q (Å⁻¹)   I(Q) (e.u.)   sigma_I(Q)"
                self.iq.save_to_xye(
                    output_dir / f"{self.sample_name}.{fmt}", header=header
                )

            if fmt == ExportFormat.FQ:
                header = "F(Q) = Q[S(Q)-1]\n# Q (Å⁻¹)   F(Q) (Å⁻¹)   sigma_F(Q)"
                self.fq.save_to_xye(
                    output_dir / f"{self.sample_name}.{fmt}", header=header
                )

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
    ) -> None:
        """Diagnostic 2x2 plot of I(Q), S(Q), F(Q) and G(r), with 1-sigma
        uncertainty bands where available.
        """
        eline = 0.1

        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        fig.suptitle("Total Scattering Data", fontsize=13)

        ax_iq, ax_sq, ax_fq, ax_gr = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

        ax_iq.errorbar(self.iq.x, self.iq.y, self.iq.e, elinewidth=eline)
        self.style_axis(
            ax_iq,
            "Q (Å⁻¹)",
            "I(Q) (e.u.)",
            "Normalised Scattering Intensity (electron units)",
        )

        ax_sq.errorbar(self.sq.x, self.sq.y, self.sq.e, elinewidth=eline)
        ax_sq.axhline(1.0, color="k", lw=0.6, ls="--", label="S(Q) = 1")
        self.style_axis(
            ax_sq, "Q (Å⁻¹)", "S(Q)", "Total structure function", legend=True
        )

        ax_fq.errorbar(self.fq.x, self.fq.y, self.fq.e, elinewidth=eline)
        ax_fq.axhline(0.0, color="k", lw=0.6, ls="--")
        self.style_axis(
            ax_fq, "Q (Å⁻¹)", "F(Q) = Q[S(Q)−1] (Å⁻¹)", "Reduced structure function"
        )

        plot_r = self.r[self.r < 5]
        plot_baseline = self.baseline[self.r < 5]
        ax_gr.errorbar(self.gr.x, self.gr.y, self.gr.e, elinewidth=eline)
        ax_gr.plot(
            plot_r, plot_baseline, "k--", lw=0.8, label=r"$-4\pi r\rho_0$", zorder=2
        )

        self.style_axis(
            ax_gr, "r (Å)", "G(r) (Å⁻²)", "Pair distribution function G(r)", legend=True
        )
        ax_gr.set_xlim(0, float(np.amax(self.gr.x)))

        fig.tight_layout()
        if save_filepath is not None:
            fig.savefig(save_filepath, dpi=150)
        plt.show()


# Coordination number analysis: peak finding, gaussian fitting, integration
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
    mask = (r >= r_search_min) & (r <= r_search_max) & np.isfinite(g_r)
    r_window = r[mask]
    g_window = g_r[mask]
    peak_idx, _ = find_peaks(g_window, prominence=prominence)
    if len(peak_idx) < n_peaks:
        warnings.warn(
            f"found only {len(peak_idx)} of {n_peaks} requested peaks", stacklevel=2
        )
    peak_idx = peak_idx[:n_peaks]
    return r_window[peak_idx]


def find_peak_bounds(
    r: np.ndarray,
    g_r: np.ndarray,
    peak_r: float,
    r_search_min: float,
    r_search_max: float,
) -> tuple[float, float]:
    mask = (r >= r_search_min) & (r <= r_search_max) & np.isfinite(g_r)
    r_window = r[mask]
    g_window = g_r[mask]
    valley_idx, _ = find_peaks(-g_window, prominence=0.01)
    valley_r = r_window[valley_idx]
    left_candidates = valley_r[valley_r < peak_r]
    right_candidates = valley_r[valley_r > peak_r]
    r_left = left_candidates[-1] if len(left_candidates) else peak_r - 0.5
    r_right = right_candidates[0] if len(right_candidates) else peak_r + 0.5
    return float(r_left), float(r_right)


def fit_gaussian_to_peak(
    r: np.ndarray, g_r: np.ndarray, peak_r: float, r_left: float, r_right: float
) -> tuple[float, float, float]:
    mask = (r >= r_left) & (r <= r_right) & np.isfinite(g_r)
    r_fit = r[mask]
    g_fit = g_r[mask]
    amplitude_guess = float(np.nanmax(g_fit))
    sigma_guess = max((r_right - r_left) / 4.0, 0.01)
    initial_guess = [amplitude_guess, peak_r, sigma_guess]
    lower_bounds = [0.0, r_left, 0.005]
    upper_bounds = [np.inf, r_right, r_right - r_left]
    params, _ = curve_fit(
        gaussian_peak,
        r_fit,
        g_fit,
        p0=initial_guess,
        bounds=(lower_bounds, upper_bounds),
        maxfev=5000,
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
    r_dense = np.linspace(r_left, r_right, n_points)
    g_dense = gaussian_peak(r_dense, amplitude, center, sigma)
    integrand = 4.0 * np.pi * rho0 * r_dense**2 * g_dense
    return float(np.trapezoid(integrand, r_dense))


def analyse_rdf_coordination(
    result: PDFResult,
    n_peaks: int = 3,
    r_search_min: float = 1.5,
    r_search_max: float = 8.0,
    prominence: float = 0.02,
) -> list[PeakFitResult]:
    r = np.asarray(result.gr.x)
    g_r = np.asarray(result.rdf)
    rho0 = result.number_density

    peak_positions = find_rdf_peak_positions(
        r,
        g_r,
        n_peaks=n_peaks,
        r_search_min=r_search_min,
        r_search_max=r_search_max,
        prominence=prominence,
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
    r = np.asarray(result.gr.x)
    g_r = np.asarray(result.rdf)

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
            label=(
                f"fit {i + 1}: r={fit.centre:.2f} Å, cn={fit.coordination_number:.2f}"
            ),
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
    ax.set_xlabel("r (Å)")
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

    f_mean_sq = f_mean**2
    return f_mean, f_mean_sq, f_sq_mean, compton


# Instrumental corrections
def polarisation_correction(
    two_theta_deg: SerialisableNDArray, synchrotron: bool, polarisation_p: float
) -> SerialisableNDArray:
    """Polarisation factor P(2*theta) for synchrotron or lab X-rays."""
    cos2t = np.cos(np.deg2rad(two_theta_deg))
    if synchrotron:
        return (1.0 - polarisation_p) + polarisation_p * cos2t**2
    return (1.0 + cos2t**2) / 2.0


def cylindrical_absorption_correction(
    two_theta_deg: SerialisableNDArray,
    mu_r: float,
    n_grid: int = 41,
    n_theta_coarse: int = 121,
) -> np.ndarray:
    """Transmission-weighted absorption correction A(2theta) = 1/<T> for a
    cylindrical (Debye-Scherrer/capillary) sample, via direct numerical
    integration over the cylinder cross-section rather than a tabulated fit
    -- exact to grid resolution. mu_r = mu * R (only their product matters).
    Evaluated on a coarse 2theta grid and interpolated for speed. Returns
    the multiplicative correction: corrected = raw / T(2theta).
    """
    # Uniform grid over the unit disc (cylinder cross-section, radius 1;
    # path lengths are computed in units of R and then scaled by mu_r).
    lin = np.linspace(-1.0, 1.0, n_grid)
    grid_x, grid_y = np.meshgrid(lin, lin)
    inside_disc = grid_x**2 + grid_y**2 < 1.0
    disc_x = grid_x[inside_disc]
    disc_y = grid_y[inside_disc]
    disc_radius_sq = disc_x**2 + disc_y**2

    two_theta_coarse = np.linspace(
        float(np.amin(two_theta_deg)), float(np.amax(two_theta_deg)), n_theta_coarse
    )
    theta_rad = np.deg2rad(two_theta_coarse)

    # incident beam direction fixed along +x
    incident_path_projection = disc_x  # P . (1,0)

    transmission = np.empty_like(two_theta_coarse)
    for theta_index, theta in enumerate(theta_rad):
        d_out = (np.cos(theta), np.sin(theta))
        scattered_path_projection = disc_x * d_out[0] + disc_y * d_out[1]

        # path length in, from the cylinder boundary to (x,y), along d_in
        disc = incident_path_projection**2 - (disc_radius_sq - 1.0)
        path_in = incident_path_projection + np.sqrt(np.maximum(disc, 0.0))

        # path length out, from (x,y) to the boundary, along d_out
        disc_out = scattered_path_projection**2 - (disc_radius_sq - 1.0)
        path_out = -scattered_path_projection + np.sqrt(np.maximum(disc_out, 0.0))

        path_length = mu_r * (path_in + path_out)
        transmission[theta_index] = np.mean(np.exp(-path_length))

    t_interp = np.interp(two_theta_deg, two_theta_coarse, transmission)
    return 1.0 / np.maximum(t_interp, 1e-12)


def auto_select_q_max(
    q_values: SerialisableNDArray,
    intensity: SerialisableNDArray,
    sigma: SerialisableNDArray,
    snr_threshold: float = 1.5,
    q_search_min: float = 5.0,
    window_points: int = 25,
) -> float:
    """Pick q_max from where the local SNR of I(Q) drops below snr_threshold
    and stays there. Local "signal" is the rolling std of I(Q) about its
    rolling mean; scans from high Q down and returns the highest Q where the
    ratio holds above threshold for at least window_points/2 consecutive points.
    """
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
    hold = max(window_points // 2, 1)
    sustained = uniform_filter1d(above.astype(float), hold) > 0.9

    candidates = np.where(valid & sustained)[0]
    if candidates.size == 0:
        return float(q_search_min)
    return float(q_s[candidates[-1]])


def r_dependent_broadening_matrix(
    r_values: SerialisableNDArray, qbroad: float, n_sigma_cutoff: float = 4.0
) -> np.ndarray:
    """Linear operator implementing PDFgui/PDFgetX3-style qbroad: each
    point in G(r) is smeared by a Gaussian of width sigma(r) = qbroad*r^2.
    Returns a dense (n_r, n_r) row-normalised kernel matrix K such that
    G_broadened = K @ G (and, being linear, Cov_broadened = K @ Cov @ K.T).
    qbroad=0 returns the identity.
    """
    num_r_points = len(r_values)
    if qbroad <= 0.0:
        return np.eye(num_r_points)

    sigma_r = np.maximum(qbroad * r_values**2, 1e-6)
    # (num_r_points, num_r_points): row i, col j -> r_values[i] - r_values[j]
    diff = r_values[:, None] - r_values[None, :]
    kernel = np.exp(-0.5 * (diff / sigma_r[:, None]) ** 2)
    kernel[np.abs(diff) > n_sigma_cutoff * sigma_r[:, None]] = 0.0
    row_sums = kernel.sum(axis=1, keepdims=True)
    return kernel / np.maximum(row_sums, 1e-30)


# Krogh-Moe / Norman normalisation
def _clip_background_extrapolation(
    q_values: SerialisableNDArray,
    background: SerialisableNDArray,
    q_min_fit: float,
    q_max_fit: float,
) -> np.ndarray:
    """Flatten the background polynomial outside the fit window.

    A polynomial fitted only over the high-Q normalisation window can
    diverge when evaluated at low Q; flat extrapolation prevents this from
    contaminating S(Q) and G(r) outside the fit window.
    """
    clipped = background.copy()
    low_value = float(np.interp(q_min_fit, q_values, background))
    high_value = float(np.interp(q_max_fit, q_values, background))
    clipped[q_values < q_min_fit] = low_value
    clipped[q_values > q_max_fit] = high_value
    return clipped


def _bspline_basis(
    x: np.ndarray, x_min: float, x_max: float, num_basis: int, spline_degree: int
) -> np.ndarray:
    """Clamped uniform B-spline design matrix with num_basis basis functions
    on [x_min, x_max] -- degree 1 gives piecewise-linear interpolation
    between num_basis control points (each fitted coefficient *is* the
    control point's y-value); degree 3 gives a smooth cubic spline through
    the same idea. Points outside [x_min, x_max] are clamped to the
    boundary, matching the flat extrapolation applied everywhere else.
    """
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
    """Background design matrix on q_values, normalised to x in [-1, 1] --
    raw powers of Q span many orders of magnitude and make the design matrix
    near-singular otherwise. "chebyshev" (default) is the most numerically
    stable global polynomial; "polynomial" is a plain power basis; "constant"
    is a single DC offset, most robust when data is noisy or the window is
    narrow. "linear" and "bspline" are piecewise (degree-1 / cubic) B-splines
    on degree+1 uniformly spaced knots -- locally adaptive, so they track a
    bump or dip a single global polynomial can't without the ringing a
    high-degree polynomial needs to do the same. "cosine" is a truncated
    Fourier cosine series: a well-conditioned, bounded global alternative to
    chebyshev, often a better fit than a polynomial for a broad hump/dip
    (e.g. an air-scatter bump or amorphous halo under Bragg peaks).

    Caution for method="krogh_moe"/"eggert"/"warren" (background fitted
    jointly with scale_factor): any basis flexible enough to track a smooth
    curve -- "bspline" at degree >= 2, same as "polynomial"/"chebyshev" --
    can alias with the smoothly Q-decaying self-scattering term and destabilise
    scale_factor itself (large condition number, sometimes a wildly wrong
    scale_factor). "linear" stays well-conditioned across degree in testing,
    and "cosine" is a solid, more flexible second choice; prefer both over
    "bspline"/high-degree "polynomial" here. This risk doesn't apply to
    method="billinge", where scale_factor is already fixed before the
    background-only correction runs -- "bspline" is fine there.
    """
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
        angle = (
            0.5 * np.pi * (normalised_q[:, None] + 1.0)
        )  # x in [-1,1] -> angle in [0,pi]
        return np.cos(orders[None, :] * angle)
    if background_type in ("linear", "bspline"):
        spline_degree = 1 if background_type == "linear" else 3
        return _bspline_basis(normalised_q, -1.0, 1.0, degree + 1, spline_degree)
    raise ValueError(f"Unknown background_type '{background_type}'.")


def _fit_scale_factor_given_background(
    q_values: np.ndarray,
    intensity_q: np.ndarray,
    self_scattering_q: np.ndarray,
    background_q: np.ndarray,
) -> float:
    """Fit scale_factor with background fixed using the Norman weight scheme."""
    weights = q_values
    weighted = weights * weights
    numerator = np.sum(weighted * self_scattering_q * (intensity_q - background_q))
    denominator = np.sum(weighted * self_scattering_q**2)
    if denominator <= 0:
        raise PDFNormalisationError(
            "cannot determine scale_factor: self-scattering denominator is non-positive"
        )
    scale_factor = float(numerator / denominator)
    if scale_factor <= 2e-12:
        raise PDFNormalisationError("scale_factor from Norman fit is non-positive")
    return scale_factor


def _fit_background_given_scale_factor(
    q_values: SerialisableNDArray,
    intensity_q: SerialisableNDArray,
    self_scattering: SerialisableNDArray,
    scale_factor: float,
    q_min_fit: float,
    q_max_fit: float,
    degree: int,
    background_type: BackgroundType,
) -> np.ndarray:
    """Fit a smooth background with scale_factor held fixed."""
    mask = (q_values >= q_min_fit) & (q_values <= q_max_fit)
    n_background_coeffs = 1 if background_type == "constant" else degree + 1
    min_points = n_background_coeffs + 2
    if mask.sum() < min_points:
        mask = np.ones(len(q_values), dtype=bool)
    if mask.sum() < min_points:
        raise PDFNormalisationError(
            "not enough data points for the requested background"
        )

    q_fit = q_values[mask]
    intensity_fit = intensity_q[mask]
    self_scattering_fit = self_scattering[mask]
    weights = q_fit
    basis = _background_basis(q_fit, q_min_fit, q_max_fit, degree, background_type)
    target = (intensity_fit - scale_factor * self_scattering_fit) * weights
    design = basis * weights[:, None]
    coeffs, *_ = np.linalg.lstsq(design, target, rcond=None)
    _warn_if_ill_conditioned(design, "Background fitting")

    full_basis = _background_basis(
        q_values, q_min_fit, q_max_fit, degree, background_type
    )
    background_full = full_basis @ coeffs
    return _clip_background_extrapolation(
        q_values, background_full, q_min_fit, q_max_fit
    )


def _warn_if_ill_conditioned(design: np.ndarray, label: str) -> None:
    """Warn if a least-squares design matrix is poorly conditioned."""
    condition_number = np.linalg.cond(design)
    if condition_number > 1e10:
        warnings.warn(
            f"{label} design matrix is poorly conditioned "
            f"(cond={condition_number:.2e}); results may be unreliable.",
            stacklevel=2,
        )


def _fit_norman_scale_factor(
    q_values: SerialisableNDArray,
    intensity_q: SerialisableNDArray,
    f_sq_mean: SerialisableNDArray,
    compton: SerialisableNDArray,
    q_min_fit: float,
    q_max_fit: float,
    degree: int,
    background_type: BackgroundType,
) -> tuple[float, np.ndarray]:
    """Compute scale_factor first, then fit background with that scale fixed."""
    mask = (q_values >= q_min_fit) & (q_values <= q_max_fit)
    n_background_coeffs = 1 if background_type == "constant" else degree + 1
    min_points = n_background_coeffs + 2
    if mask.sum() < min_points:
        mask = np.ones(len(q_values), dtype=bool)
    if mask.sum() < min_points:
        raise PDFNormalisationError(
            "not enough data points for the requested background"
        )

    q_fit = q_values[mask]
    i_fit = intensity_q[mask]
    self_scattering_fit = f_sq_mean[mask] + compton[mask]
    scale_factor = _fit_scale_factor_given_background(
        q_fit, i_fit, self_scattering_fit, np.zeros_like(q_fit)
    )
    background = _fit_background_given_scale_factor(
        q_values,
        intensity_q,
        f_sq_mean + compton,
        scale_factor,
        q_min_fit,
        q_max_fit,
        degree,
        background_type,
    )

    if scale_factor <= 2e-12:
        raise PDFNormalisationError("bounded normalisation fit did not converge")

    return scale_factor, background


# Shared low-r helpers (Eggert / Billinge / Warren)
def _trapz_weights(grid: np.ndarray) -> np.ndarray:
    """Trapezoidal quadrature weights for a grid."""
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


# Warren: closed-form sum-rule scale_factor, no background (unchanged from before)
def _warren_scale_factor(
    q_values: SerialisableNDArray,
    intensity_q: SerialisableNDArray,
    f_sq_mean: SerialisableNDArray,
    f_mean_sq: SerialisableNDArray,
    compton: SerialisableNDArray,
    rho: float,
) -> tuple[float, np.ndarray]:
    """scale_factor from lim(r->0) D(r)/r = -4*pi*rho, i.e. Warren's sum rule."""
    weights = _trapz_weights(q_values)
    scale = q_values**2 / f_mean_sq
    numerator = -2.0 * np.pi**2 * rho + np.sum(weights * scale * (f_sq_mean + compton))
    denominator = np.sum(weights * scale * intensity_q)
    if denominator <= 0:
        raise PDFNormalisationError("Warren sum-rule denominator is non-positive")
    scale_factor = numerator / denominator
    if scale_factor <= 2e-12:
        raise PDFNormalisationError(
            "Warren normalisation gave a non-positive scale_factor"
        )
    return scale_factor, np.zeros_like(q_values)


# Eggert: iterative low-r Fourier-filter correction (unchanged, still
# experimental -- see convergence caveats from earlier testing)
def _eggert_scale_factor(
    q_values: SerialisableNDArray,
    intensity_q: SerialisableNDArray,
    f_sq_mean: SerialisableNDArray,
    f_mean_sq: SerialisableNDArray,
    compton: SerialisableNDArray,
    rho: float,
    r_low: np.ndarray,
    max_iter: int,
    tol: float,
    damping: float,
    window: np.ndarray,
    known_scale_factor: float | None = None,
) -> tuple[float, np.ndarray]:
    """Repeatedly subtract the low-r D(r) error, Fourier-filtered into Q-space.

    window is the same termination-window * qdamp product applied to F(Q)
    elsewhere in the pipeline (see make_termination_window). Without it,
    sharp Bragg content produces termination ripple the correction cannot
    distinguish from real background contamination.
    """
    weights_q = _trapz_weights(q_values)
    weights_r = _trapz_weights(r_low)
    if known_scale_factor is not None:
        scale_factor = known_scale_factor
    else:
        scale_factor, _ = _warren_scale_factor(
            q_values, intensity_q, f_sq_mean, f_mean_sq, compton, rho
        )
    background = np.zeros_like(q_values)
    q_over_f_mean_sq = q_values / f_mean_sq
    safe_q = np.where(q_values > 0, q_values, np.inf)

    for _ in range(max_iter):
        s_minus_1_q = q_over_f_mean_sq * (
            scale_factor * intensity_q - background - compton - f_sq_mean
        )
        d_r = _sine_transform_to_r(q_values, s_minus_1_q * window, r_low, weights_q)
        delta = d_r - (-4.0 * np.pi * rho * r_low)
        if np.sqrt(np.mean(delta**2)) < tol:
            break
        delta_qs = np.sin(np.outer(q_values, r_low)) @ (delta * weights_r)
        background = background + damping * f_mean_sq * (delta_qs / safe_q) * window
    else:
        warnings.warn("Eggert iterative normalisation did not converge", stacklevel=2)

    if scale_factor <= 2e-12:
        raise PDFNormalisationError(
            "Eggert normalisation gave a non-positive scale_factor"
        )
    return scale_factor, background


# Billinge (PDFgetX3 rpoly): background-only refinement on top of an
# already-determined scale_factor. No density required, scale_factor is never touched --
# this is what makes it stable, per Billinge & Farrow (2013).
def _billinge_refine_background(
    q_values: SerialisableNDArray,
    intensity_q: SerialisableNDArray,
    f_sq_mean: SerialisableNDArray,
    f_mean_sq: SerialisableNDArray,
    compton: SerialisableNDArray,
    scale_factor: float,
    background: np.ndarray,
    q_min_fit: float,
    q_max_fit: float,
    r_poly: float,
    r_step: float,
    background_type: BackgroundType,
    window: np.ndarray,
) -> np.ndarray:
    """Ad-hoc additive polynomial correction flattening D(r) to zero below
    r_poly (degree n = floor(Qmaxinst * r_poly / pi), per PDFgetX3). The
    basis is normalised over the *full* transformed Q range, not the narrow
    Krogh-Moe fit window -- normalising over the fit window instead makes
    every basis column collinear over most of the domain and blows up G(r).
    window (matching the rest of the pipeline) is applied before
    transforming, or unwindowed Bragg ripple gets chased instead of removed.
    """
    degree = max(int(np.floor(q_max_fit * r_poly / np.pi)), 0)
    r_low = np.arange(0.0, r_poly, r_step)
    if r_low.size <= degree + 1:
        raise PDFNormalisationError("r_poly too small for the Nyquist-derived degree")

    weights = _trapz_weights(q_values)
    q_over_f_mean_sq = q_values / f_mean_sq

    # D(r) implied by the current scale_factor/background -- this is the "error"
    # the correction polynomial needs to cancel, not a physical target.
    current_s_minus_1 = q_over_f_mean_sq * (
        scale_factor * intensity_q - background - compton - f_sq_mean
    )
    current_d_r = _sine_transform_to_r(
        q_values, current_s_minus_1 * window, r_low, weights
    )

    basis_q_min = float(np.amin(q_values))
    basis = _background_basis(q_values, basis_q_min, q_max_fit, degree, background_type)
    design = _sine_transform_to_r(
        q_values, q_over_f_mean_sq[:, None] * basis * window[:, None], r_low, weights
    )

    coeffs, *_ = np.linalg.lstsq(design, current_d_r, rcond=None)
    _warn_if_ill_conditioned(design, "Billinge rpoly")
    return background + basis @ coeffs


# Krogh-Moe / Norman normalisation (dispatcher: method selects the residual)
def normalise_intensity(
    q_values: SerialisableNDArray,
    intensity_q: SerialisableNDArray,
    f_sq_mean: SerialisableNDArray,
    compton: SerialisableNDArray,
    q_min_fit: float,
    q_max_fit: float,
    poly_degree: int,
    background_type: BackgroundType = "chebyshev",
    method: NormalisationMethod = "krogh_moe",
    f_mean_sq: SerialisableNDArray | None = None,
    rho: float | None = None,
    r_min: float = 0.0,
    r_max_unphysical: float = 1.5,
    r_step: float = 0.01,
    r_poly: float | None = None,
    eggert_max_iter: int = 20,
    eggert_tol: float = 1e-6,
    eggert_damping: float = 0.5,
    termination_window: WindowType | None = "lorch",
    soper_lorch_power: int | float | None = 2,
    qdamp: float = 0.0,
    known_scale_factor: float | None = None,
) -> tuple[float, np.ndarray]:
    """Fit scale_factor (+ background) via the chosen normalisation convention.

    "krogh_moe" (default): safe for Bragg-peak (crystalline) data, never touches r-space

    "warren": closed-form sum-rule scale_factor, no background; needs rho; uses the full
    Q-range integral directly, so avoid on strongly Bragg-peaked data.

    "eggert": iterative low-r Fourier-filter correction; needs rho; experimental.

    "billinge": krogh_moe fit, then a PDFgetX3-style additive polynomial correction

    (degree from Nyquist, target zero) to the background only; scale_factor untouched,
    rho not required; r_poly (default r_max_unphysical) sets the correction's r bound.

    termination_window/soper_lorch_power/qdamp (for "eggert"/"billinge" only): these
    transform Q-space into r-space, so on crystalline data they need the same window/
    damping already applied to F(Q) elsewhere -- pass config.termination_window/qdamp
    rather than the defaults, or Bragg content gets mistaken for background.
    """
    q_max_fit = min(q_max_fit, float(np.amax(q_values)))
    q_min_fit = max(q_min_fit, float(np.amin(q_values)))
    window = make_termination_window(
        q_values, q_max_fit, termination_window, soper_lorch_power
    ) * np.exp(-0.5 * (qdamp * q_values) ** 2)

    def _run_krogh_moe(
        degree: int, current_type: BackgroundType
    ) -> tuple[float, np.ndarray]:
        last_error: Exception | None = None
        while True:
            try:
                return _fit_norman_scale_factor(
                    q_values,
                    intensity_q,
                    f_sq_mean,
                    compton,
                    q_min_fit,
                    q_max_fit,
                    degree,
                    current_type,
                )
            except PDFNormalisationError as exc:
                last_error = exc
                if degree > 0:
                    warnings.warn(
                        f"Normalisation failed at degree={degree} ({exc}); "
                        f"retrying with degree={degree - 1}.",
                        stacklevel=2,
                    )
                    degree -= 1
                elif current_type != "constant":
                    warnings.warn(
                        f"Normalisation failed ({exc}); falling back to a "
                        "constant background.",
                        stacklevel=2,
                    )
                    current_type = "constant"
                else:
                    break
        raise PDFNormalisationError(
            "Krogh-Moe/Norman normalisation failed even with a constant "
            "background. Check q_max, the normalisation window (norm_q_min), "
            "and the input composition."
        ) from last_error

    if method == "krogh_moe":
        if known_scale_factor is not None:
            background = _fit_background_given_scale_factor(
                q_values,
                intensity_q,
                f_sq_mean + compton,
                known_scale_factor,
                q_min_fit,
                q_max_fit,
                poly_degree,
                background_type,
            )
            return known_scale_factor, background
        return _run_krogh_moe(poly_degree, background_type)

    if method == "billinge":
        if f_mean_sq is None:
            raise ValueError("method='billinge' requires f_mean_sq")
        if known_scale_factor is not None:
            background = _fit_background_given_scale_factor(
                q_values,
                intensity_q,
                f_sq_mean + compton,
                known_scale_factor,
                q_min_fit,
                q_max_fit,
                poly_degree,
                background_type,
            )
            background = _billinge_refine_background(
                q_values,
                intensity_q,
                f_sq_mean,
                f_mean_sq,
                compton,
                known_scale_factor,
                background,
                q_min_fit,
                q_max_fit,
                r_poly if r_poly is not None else r_max_unphysical,
                r_step,
                background_type,
                window,
            )
            return known_scale_factor, background
        scale_factor, background = _run_krogh_moe(poly_degree, background_type)
        background = _billinge_refine_background(
            q_values,
            intensity_q,
            f_sq_mean,
            f_mean_sq,
            compton,
            scale_factor,
            background,
            q_min_fit,
            q_max_fit,
            r_poly if r_poly is not None else r_max_unphysical,
            r_step,
            background_type,
            window,
        )
        return scale_factor, background

    if f_mean_sq is None or rho is None:
        raise ValueError(f"method='{method}' requires f_mean_sq and rho")

    r_low = np.arange(r_min, r_max_unphysical, r_step)
    if r_low.size == 0:
        raise PDFNormalisationError(
            "r_max_unphysical must exceed r_min by at least r_step"
        )

    if method == "warren":
        if known_scale_factor is not None:
            return known_scale_factor, np.zeros_like(q_values)
        return _warren_scale_factor(
            q_values, intensity_q, f_sq_mean, f_mean_sq, compton, rho
        )

    if method == "eggert":
        return _eggert_scale_factor(
            q_values,
            intensity_q,
            f_sq_mean,
            f_mean_sq,
            compton,
            rho,
            r_low,
            eggert_max_iter,
            eggert_tol,
            eggert_damping,
            window,
            known_scale_factor,
        )

    raise ValueError(f"Unknown method '{method}'.")


# Termination window functions
def make_termination_window(
    q_values: SerialisableNDArray,
    q_max: float,
    window_type: WindowType | None,
    soper_lorch_power: int | float | None = 2,
) -> np.ndarray:
    """Multiplicative window W(Q) applied to F(Q) to reduce Fourier
    termination ripples. "lorch" gives the strongest ripple suppression at
    the cost of peak broadening; "cosine" is intermediate; "none" applies
    no window.
    soper_lorch_power
    Exponent used for the Super-Lorch window. A value of 2 reproduces
    the commonly used Super-Lorch window. A value of 1 is identical to
    the standard Lorch window.
    """

    if window_type is None or window_type == "none":
        return np.ones_like(q_values)
    if window_type == "cosine":
        return 0.5 * (1.0 + np.cos(np.pi * q_values / q_max))
    if window_type in ("lorch", "soper_lorch"):
        power = 1 if window_type == "lorch" else soper_lorch_power
        if power is None:
            raise ValueError(
                f"If window_type is '{window_type}'. Then soper_lorch_power must be int"
            )
        window = np.ones_like(q_values)
        nonzero = q_values > 0.0
        arg = np.pi * q_values[nonzero] / q_max
        window[nonzero] = (np.sin(arg) / arg) ** power
        return window
    raise ValueError(
        f"Unknown termination_window '{window_type}'. Choose 'lorch', "
        "'cosine', 'soper_lorch' or 'none'."
    )


# Fourier transforms
def sine_transform_fq_to_gr(
    q_values: SerialisableNDArray,
    f_q: SerialisableNDArray,
    r_values: SerialisableNDArray,
    q_step_size: float,
) -> np.ndarray:
    """Forward sine transform G(r) = (2/pi) * integral F(Q) sin(Qr) dQ,
    via direct matrix product (avoids FFT grid constraints).
    """
    sin_qr = np.sin(np.outer(r_values, q_values))
    return (2.0 / np.pi) * q_step_size * (sin_qr @ f_q)


def sine_transform_gr_to_fq(
    r_values: SerialisableNDArray,
    g_r: SerialisableNDArray,
    q_values: SerialisableNDArray,
) -> np.ndarray:
    """Inverse (back) sine transform F(Q) = integral G(r) sin(Qr) dr."""
    sin_qr = np.sin(np.outer(q_values, r_values))
    return np.trapezoid(sin_qr * g_r[np.newaxis, :], r_values, axis=1)  # type: ignore - it will be an array of floats


def sine_transform_sigma(
    q_values: SerialisableNDArray,
    sigma_f_q: SerialisableNDArray,
    r_values: SerialisableNDArray,
    q_step_size: float,
) -> np.ndarray:
    """Propagate independent per-Q uncertainty in F(Q) through the linear
    sine transform to G(r): sigma_G(r)^2 = (2/pi*dq)^2 * sum_Q sin(Qr)^2 *
    sigma_F(Q)^2. Assumes uncorrelated errors between Q points; does not
    include scale_factor/background fit uncertainty (see PDFResult docstring).
    """
    sin_qr_sq = np.sin(np.outer(r_values, q_values)) ** 2
    return (2.0 / np.pi) * q_step_size * np.sqrt(sin_qr_sq @ sigma_f_q**2)


def linear_interp_operator(x_new: np.ndarray, x_old: np.ndarray) -> np.ndarray:
    """Dense (n_new, n_old) matrix L such that L @ y_old == np.interp(x_new,
    x_old, y_old). Used to propagate covariance through resampling:
    Cov_new = L @ Cov_old @ L.T. x_old must be sorted ascending.
    """
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
    # points outside the raw data range get zero weight, matching np.interp
    # 'left'/'right' fill behaviour used elsewhere in this module
    out_of_range = (x_new < x_old[0]) | (x_new > x_old[-1])
    interp_matrix[out_of_range, :] = 0.0
    return interp_matrix


# Real-space constraint: Toby-Egami / PDFgetX3 method
def _auto_r_constraint(
    r_values: SerialisableNDArray,
    g_values: SerialisableNDArray,
    rho0: float,
    r_search_min: float = 1.2,
    r_search_max: float = 3.5,
) -> float:
    """First upward crossing of G(r) through the physical baseline in
    [r_search_min, r_search_max], marking the onset of the first
    coordination shell. Falls back to r_search_min if no crossing exists.
    """
    baseline = gr_baseline(r_values, rho0)
    deviation = g_values - baseline

    idx = (r_values >= r_search_min) & (r_values <= r_search_max)
    if not np.any(idx):
        return r_search_min

    r_sub = r_values[idx]
    d_sub = deviation[idx]

    crossings = np.where((d_sub[:-1] < 0.0) & (d_sub[1:] >= 0.0))[0]
    if len(crossings) == 0:
        return float(r_search_min)

    crossing_index = crossings[0]
    r0, r1 = float(r_sub[crossing_index]), float(r_sub[crossing_index + 1])
    d0, d1 = float(d_sub[crossing_index]), float(d_sub[crossing_index + 1])
    r_cross = r0 - d0 * (r1 - r0) / (d1 - d0)
    return float(np.clip(r_cross, r_search_min, r_search_max))


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
    """Iterative Toby-Egami back-Fourier correction (Toby & Egami, 1992).

    Below r_max_constraint, G(r) must equal -4*pi*r*rho0 (no atom pairs).
    Any deviation there is unphysical (normalisation error, residual
    background, or termination ripple); each cycle back-transforms the
    deviation to Q-space and subtracts it from F(Q), then re-transforms.
    """
    g_values = g_values.copy()
    f_q = f_q.copy()

    g_physical_full = gr_baseline(r_values, rho0)
    mask_low = r_values <= r_max_constraint
    mask_phys = ~mask_low

    for _ in range(n_iterations):
        delta_g = np.where(mask_low, g_values - g_physical_full, 0.0)

        rms = float(np.sqrt(np.mean(delta_g[mask_low] ** 2)))
        g_peak = (
            float(np.max(np.abs(g_values[mask_phys]))) if np.any(mask_phys) else 1.0
        )
        if rms / max(g_peak, 1e-12) < 1e-5:
            break

        delta_f = sine_transform_gr_to_fq(r_values, delta_g, q_values)
        f_q -= delta_f

        r_pos = r_values[1:]
        g_pos = sine_transform_fq_to_gr(q_values, f_q, r_pos, q_step_size)
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
    """Uncertainty propagation: independent per-Q
    errors, no cross-correlation tracked. Never allocates an (n_q, n_q) or
    (n_r, n_r) dense matrix
    """
    sigma_i_eu = sigma_i_q / scale_factor
    sigma_s_q = sigma_i_eu * inv_f_mean_sq
    sigma_s_q[below_q_min_mask] = 0.0
    sigma_f_q = q_values * sigma_s_q
    sigma_f_mod = sigma_f_q * window

    sigma_g_full = np.concatenate(
        [[0.0], sine_transform_sigma(q_values, sigma_f_mod, r_pos, q_step_size)]
    )
    sigma_g_full = sigma_g_full * envelope
    if broadening_kernel is not None:
        sigma_g_full = np.sqrt(
            np.maximum((broadening_kernel**2) @ sigma_g_full**2, 0.0)
        )

    return sigma_i_eu, sigma_s_q, sigma_f_q, sigma_g_full, None


def _load_and_correct_intensity(
    xy_path: Path, config: PDFConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Step 1: load raw .xy data; subtract background; absorption/polarisation-correct.

    Returns (two_theta_deg, intensity, sigma).
    """
    logger.info(f"compute_pdf: loading {xy_path}")
    two_theta_deg, intensity = load_xy(xy_path)
    logger.debug(
        f"loaded {len(two_theta_deg)} raw points, 2theta range "
        f"[{two_theta_deg.min():.4f}, {two_theta_deg.max():.4f}] deg"
    )

    if config.data.e is not None and len(config.data.e) == len(intensity):
        sigma = np.asarray(config.data.e, dtype=np.float64).copy()
        logger.debug("using supplied experimental uncertainties (data.e)")
    else:
        sigma = np.sqrt(np.maximum(intensity, 1.0))
        logger.debug("data.e not usable; assuming Poisson counting statistics")

    if config.background_file is not None:
        two_theta_bg, intensity_bg = load_xy(config.background_file)
        intensity_bg_interp = np.interp(
            two_theta_deg, two_theta_bg, intensity_bg, left=0.0, right=0.0
        )
        intensity = intensity - config.background_scale * intensity_bg_interp
        logger.info(
            f"subtracted background from {config.background_file} "
            f"(scale={config.background_scale})"
        )

    intensity = np.maximum(intensity, 0.0)

    if config.absorption_correction:
        absorption = cylindrical_absorption_correction(
            two_theta_deg,
            float(config.mu_r),  # type: ignore[arg-type]
        )
        intensity = intensity * absorption
        sigma = sigma * absorption
        logger.info(
            f"applied cylindrical absorption correction (mu_r={config.mu_r}), "
            f"correction factor range [{absorption.min():.3f}, {absorption.max():.3f}]"
        )

    if config.polarisation_factor:
        polarisation = polarisation_correction(
            two_theta_deg, config.is_synchrotron, config.polarisation_p
        )
        polarisation = np.maximum(polarisation, 1e-12)
        intensity = intensity / polarisation
        sigma = sigma / polarisation
        logger.debug(
            f"applied polarisation correction (synchrotron={config.is_synchrotron}, "
            f"p={config.polarisation_p})"
        )

    return two_theta_deg, intensity, sigma


def _build_q_grid(
    config: PDFConfig,
    two_theta_deg: np.ndarray,
    intensity: np.ndarray,
    sigma: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
    """Step 2: 2theta -> Q, optionally auto-select q_max, build the uniform Q grid.

    Returns (q_raw, q_values, q_step_size, q_min_data, q_max_data, q_max_use).
    """
    q_raw = two_theta_to_q(two_theta_deg, float(config.data.wavelength))
    q_max_data = float(q_raw.max())
    q_min_data = float(q_raw.min())
    logger.debug(f"Q range in raw data: [{q_min_data:.4f}, {q_max_data:.4f}] A^-1")

    if config.auto_q_max:
        raw_order = np.argsort(q_raw)
        q_max_auto = auto_select_q_max(
            q_raw[raw_order],
            intensity[raw_order],
            sigma[raw_order],
            snr_threshold=config.auto_q_max_snr_threshold,
            q_search_min=config.auto_q_max_search_min,
        )
        q_max_use = min(config.q_max, q_max_data, q_max_auto)
        logger.info(
            f"auto_q_max selected {q_max_auto:.3f} A^-1 at SNR threshold "
            f"{config.auto_q_max_snr_threshold} (requested q_max={config.q_max}, "
            f"using {q_max_use:.3f})"
        )
    else:
        q_max_use = min(config.q_max, q_max_data)

    q_step_size = (
        config.q_step
        if config.q_step is not None
        else min(float(np.median(np.diff(q_raw))), 0.05)
    )
    q_values = np.arange(
        config.q_min, q_max_use + 0.5 * q_step_size, q_step_size, dtype=np.float64
    )
    logger.info(
        f"Q-grid: [{config.q_min}, {q_max_use:.3f}] step {q_step_size:.5f}, "
        f"n_q={len(q_values)}"
    )
    return q_raw, q_values, q_step_size, q_min_data, q_max_data, q_max_use


def _build_r_grid(config: PDFConfig) -> np.ndarray:
    """Positive part of the uniform r-grid (r=0 prepended once transformed);
    built early so the covariance feasibility check can see n_r first.
    """
    r_pos = np.arange(
        config.r_step,
        config.r_max + 0.5 * config.r_step,
        config.r_step,
        dtype=np.float64,
    )
    logger.info(
        f"r-grid: [0, {config.r_max}] step {config.r_step}, n_r={len(r_pos) + 1}"
    )
    return r_pos


def _resample_onto_q_grid(
    q_raw: np.ndarray,
    intensity: np.ndarray,
    sigma: np.ndarray,
    q_values: np.ndarray,
    q_min_data: float,
    q_max_data: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cubic-spline resample intensity onto the uniform Q grid; linear-interp
    sigma as the cheap, always-available marginal uncertainty (O(n)).

    Returns (intensity_q, sigma_i_q, q_raw_valid, sigma_valid).
    """
    valid = intensity > 0
    if valid.sum() < 4:
        raise ValueError("Fewer than 4 valid data points; cannot fit a spline.")
    q_raw_valid = q_raw[valid]
    valid_order = np.argsort(q_raw_valid)
    q_raw_valid = q_raw_valid[valid_order]
    intensity_valid = intensity[valid][valid_order]
    sigma_valid = sigma[valid][valid_order]

    intensity_spline = UnivariateSpline(q_raw_valid, intensity_valid, s=0, k=3, ext=1)
    intensity_q = np.maximum(intensity_spline(q_values), 0.0)
    intensity_q[(q_values < q_min_data) | (q_values > q_max_data)] = 0.0

    sigma_i_q = np.interp(q_values, q_raw_valid, sigma_valid, left=0.0, right=0.0)
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
    """I(Q) (electron units) -> S(Q) -> F(Q) = Q[S(Q)-1].

    Returns (i_eu, s_q, f_q, inv_f_mean_sq, below_q_min_mask).
    """
    i_eu = (intensity_q - background) / scale_factor

    # S(Q) = [I_eu - I_Compton - (<f^2> - <f>^2)] / <f>^2
    # (<f^2> - <f>^2) is the Laue monotonic diffuse term, zero for a
    # single-element sample. S(Q) -> 1 at large Q.
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
    """Bundled so it can be passed as one argument into uncertainty propagation."""

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
) -> _RealSpacePDF:
    """Steps 7-10: window F(Q), sine-transform to G(r), apply the iterative
    Toby-Egami low-r constraint, then the qdamp envelope and qbroad broadening.
    """
    window = make_termination_window(
        q_values, q_max_use, config.termination_window, config.soper_lorch_power
    )
    f_mod = f_q * window

    g_pos = sine_transform_fq_to_gr(q_values, f_mod, r_pos, q_step_size)
    r_full = np.concatenate([[0.0], r_pos])
    g_full = np.concatenate([[0.0], g_pos])

    real_space_constraint_applied = False
    r_constraint_cutoff: float | None = None
    if config.use_real_space_constraint and config.real_space_constraint_iterations > 0:
        r_constraint_cutoff = (
            float(config.r_constraint_max)
            if config.r_constraint_max is not None
            else _auto_r_constraint(
                r_full,
                g_full,
                config.number_density,
                r_search_min=config.r_constraint_search_min,
                r_search_max=config.r_constraint_search_max,
            )
        )
        logger.debug(f"real-space constraint: r_cut={r_constraint_cutoff:.3f}")
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

    # qdamp: r-space envelope decay (standard PDFgetX3/PDFgui convention --
    # NOT a Q-space damping of F(Q), which instead broadens peak widths).
    envelope = np.exp(-0.5 * (config.qdamp * r_full) ** 2)
    g_full = g_full * envelope

    # qbroad: r-dependent peak broadening, sigma(r) = qbroad * r^2. Skipped
    # entirely (no (n_r, n_r) matrix built) when qbroad<=0, which is the
    # default -- most runs never pay for this.
    broadening_kernel = None
    if config.qbroad > 0.0:
        broadening_kernel = r_dependent_broadening_matrix(r_full, config.qbroad)
        g_full = broadening_kernel @ g_full
        logger.debug(
            f"applied qbroad={config.qbroad} broadening kernel, "
            f"shape {broadening_kernel.shape}"
        )

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
    """Full PDF pipeline: load -> correct -> normalise -> S(Q) -> F(Q) -> G(r).

    Each stage below is a separate function named for what it does; read
    this function top to bottom for the pipeline order.
    """

    two_theta_deg, intensity, sigma = _load_and_correct_intensity(xy_path, config)
    q_raw, q_values, q_step_size, q_min_data, q_max_data, q_max_use = _build_q_grid(
        config, two_theta_deg, intensity, sigma
    )
    r_pos = _build_r_grid(config)

    intensity_q, sigma_i_q, q_raw_valid, sigma_valid = _resample_onto_q_grid(
        q_raw, intensity, sigma, q_values, q_min_data, q_max_data
    )

    f_mean, f_mean_sq, f_sq_mean, compton = build_scattering_factors(
        config.composition, q_values
    )

    scale_factor, background = normalise_intensity(
        q_values,
        intensity_q,
        f_sq_mean=f_sq_mean,
        compton=compton,
        q_min_fit=float(config.norm_q_min),  # type: ignore[arg-type]
        q_max_fit=q_max_use,
        poly_degree=config.norm_poly_degree,
        background_type=config.background_type,
        method=config.normalisation_method,
        f_mean_sq=f_mean_sq,
        rho=config.number_density,
        r_min=config.r_min,
        r_step=config.r_step,
        termination_window=config.termination_window,
        soper_lorch_power=config.soper_lorch_power,
        qdamp=0.0,  # qdamp is now applied in r-space, below -- see PDFConfig.qdamp
        known_scale_factor=config.known_scale_factor,
    )
    logger.info(
        f"normalisation: {config.normalisation_method} "
        f"scale_factor: {scale_factor:.6g} "
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
    gr = _transform_to_gr(q_values, f_q, r_pos, q_step_size, config, q_max_use)

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
        x=q_values, y=f_q, e=sigma_f_q, x_unit="q", y_unit="F(Q) = Q[S(Q)-1] (Å⁻¹)"
    )
    gr_result = XYEData(
        x=gr.r_full, y=gr.g_full, e=sigma_g_full, x_unit="r", y_unit="G(r) (Å⁻²)"
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
    is_synchrotron: bool = True,
    polarisation_p: float = 0.99,
    background_file: Path | str | None = None,
    background_scale: float = 1.0,
    absorption_correction: bool = False,
    mu_r: float | None = None,
    auto_q_max: bool = False,
    auto_q_max_snr_threshold: float = 1.5,
    known_scale_factor: float | None = None,
    export_formats: list[ExportFormat] | list[str] | None = None,
    output_dir: Path | str | None = None,
) -> PDFResult:
    """Compute G(r) from a raw .xy diffraction file and optionally write
    output files. composition is e.g. {"Si": 1} or {"Pb": 1, "Ti": 1, "O": 3}.
    number_density is atoms per cell divided by cell volume (atoms/Å³).
    known_scale_factor may be supplied to bypass scale factor refinement.
    """

    if isinstance(formula, str):
        composition = ChemicalFormula(formula=formula)
    elif isinstance(formula, dict):
        composition = ChemicalFormula.load_from_composition(formula)
    elif isinstance(formula, ChemicalFormula):
        composition = formula
    else:
        raise ValueError("composition must be str or dict")

    if export_formats is None:
        export_formats = [ExportFormat.GR]

    if output_dir is None:
        output_dir = Path(xy_path).parent.resolve()

    if sample_name is None:
        sample_name = Path(xy_path).stem

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
        is_synchrotron=is_synchrotron,
        polarisation_p=polarisation_p,
        background_file=Path(background_file) if background_file else None,
        background_scale=background_scale,
        known_scale_factor=known_scale_factor,
        absorption_correction=absorption_correction,
        mu_r=mu_r,
        auto_q_max=auto_q_max,
        auto_q_max_snr_threshold=auto_q_max_snr_threshold,
        export_formats=[ExportFormat(fmt) for fmt in export_formats],
        output_dir=Path(output_dir),
    )
    result: PDFResult = compute_pdf(Path(xy_path), config)
    result.save_results(
        export_formats=config.export_formats, output_dir=config.output_dir
    )

    return result


#  crystalline silicon at i15-1
if __name__ == "__main__":
    xy_file = Path("/workspaces/xrpd-toolbox/tests/data/Si_pe2_i15_1.xy")

    # Crystalline Si: 8 atoms per conventional cubic unit cell, a = 5.4309 Å
    rho_si = 8.0 / 5.4309**3

    # First Si-Si nearest-neighbour distance = 2.352 Å;
    # the constraint boundary must stay below this.
    result = run_pdf(
        xy_path=xy_file,
        formula="Si",
        sample_name="Si_pdf",
        wavelength=0.161669,
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
        background_type="polynomial",
        normalisation_method="krogh_moe",
        use_real_space_constraint=True,
        real_space_constraint_iterations=10,
        r_constraint_max=2,
        norm_poly_degree=3,
        auto_q_max=False,
        export_formats=["gr", "sq", "fq", "iq"],
    )

    result.plot(save_filepath="./pdf.png")

    coordination_fits = analyse_rdf_coordination(result, n_peaks=3)
    for fit_index, coordination_fit in enumerate(coordination_fits, start=1):
        logger.info(
            f"peak {fit_index}: r={coordination_fit.centre:.3f} A, "
            f"cn={coordination_fit.coordination_number:.3f}"
        )
    plot_rdf_coordination(result, coordination_fits, save_filepath="./cn.png")
