"""Pair distribution function (PDF) calculator.

Converts powder X-ray diffraction data (2-theta vs intensity) into the
reduced PDF G(r) = 4*pi*r*[rho(r) - rho_0], obtained via the sine Fourier
transform of the reduced structure function F(Q) = Q*[S(Q) - 1]:

    G(r) = (2/pi) * integral_0^Qmax F(Q) * W(Q) * sin(Q*r) dQ

For r below the first interatomic distance, no atom pairs exist and
G(r) = -4*pi*r*rho_0 exactly. This physical constraint is enforced
iteratively by the Toby-Egami back-Fourier method.

The absolute scale factor alpha is fixed by the Krogh-Moe/Norman condition
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
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from pydantic import Field, field_validator, model_validator
from scipy.interpolate import UnivariateSpline
from scipy.optimize import lsq_linear

from xrpd_toolbox.core import (
    ScatteringData,
    SerialisableNDArray,
    XRPDBaseModel,
)
from xrpd_toolbox.fit_engine.form_factors import (
    calculate_compton_for_element,
    calculate_form_factor_for_element,
)
from xrpd_toolbox.utils.chemical_formula import ChemicalFormula
from xrpd_toolbox.utils.unit_conversion import q_space_to_s, two_theta_to_q

# ---------------------------------------------------------------------------
# Type aliases and enums
# ---------------------------------------------------------------------------

WindowType = Literal["super-lorch", "lorch", "cosine", "none"]
BackgroundType = Literal["constant", "polynomial", "chebyshev"]


class ExportFormat(StrEnum):
    GR = "gr"  # Reduced PDF G(r)
    SQ = "sq"  # Total structure function S(Q)
    IQ = "iq"  # Coherent intensity in electron units I(Q)
    FQ = "fq"  # Reduced structure function F(Q) = Q[S(Q)-1]


class PDFNormalisationError(RuntimeError):
    """Raised when the Krogh-Moe/Norman scale factor cannot be determined."""


class PDFConfig(XRPDBaseModel):
    """All parameters controlling the PDF calculation.

    composition / wavelength / number_density: required physical inputs.
    q_min, q_max, q_step: Fourier transform Q-range and grid spacing (Å⁻¹).
    polarisation_factor, is_synchrotron, polarisation_p: instrument
        polarisation correction.
    background_file, background_scale: optional background subtraction.
    norm_poly_degree, norm_q_min: Krogh-Moe/Norman high-Q normalisation
        window and the degree of the smooth background fitted jointly
        with the scale factor.
    background_type: basis for that smooth background — "chebyshev"
        (default, best conditioned), "polynomial" (plain power basis),
        or "constant" (single offset, most robust for noisy/narrow data).
    r_min, r_max, r_step: real-space output grid (Å).
    termination_window, qdamp: Fourier termination window and Q-resolution
        damping applied before transforming to r-space.
    use_real_space_constraint, real_space_constraint_iterations,
        r_constraint_max, r_constraint_search_min/max: Toby-Egami
        constraint that enforces G(r) = -4*pi*r*rho0 below the first
        interatomic distance. r_constraint_max is auto-detected within
        [r_constraint_search_min, r_constraint_search_max] if not given.
    """

    composition: ChemicalFormula
    sample_name: str = Field(default="pdf")
    data: ScatteringData
    # wavelength: float = Field(gt=0, description="X-ray wavelength (Å)")
    number_density: float = Field(gt=0, description="Atomic number density (atoms/Å³)")

    q_min: float = Field(default=0.5, gt=0)
    q_max: float = Field(default=30.0, gt=0)
    q_step: float | None = Field(default=None, gt=0)

    polarisation_factor: bool = True
    is_synchrotron: bool = False
    polarisation_p: float = Field(default=0.99, ge=0.0, le=1.0)
    background_file: Path | None = None
    background_scale: float = 1.0

    norm_poly_degree: int = Field(default=3, ge=0)
    norm_q_min: float | None = None
    background_type: BackgroundType = "chebyshev"

    r_min: float = Field(default=0.0, ge=0.0)
    r_max: float = Field(default=30.0, gt=0)
    r_step: float = Field(default=0.01, gt=0)

    termination_window: WindowType | None = "lorch"
    super_lorch_power: int | float | None = Field(default=2)

    qdamp: float = Field(default=0.030, ge=0.0)

    use_real_space_constraint: bool = True
    real_space_constraint_iterations: int = Field(default=10, ge=0)
    r_constraint_max: float | None = Field(default=None, gt=0)
    r_constraint_search_min: float = Field(default=1.2, gt=0)
    r_constraint_search_max: float = Field(default=3.5, gt=0)

    export_formats: list[ExportFormat] = Field(
        default_factory=lambda: [ExportFormat.GR]
    )
    output_dir: Path = Path(".")

    @field_validator("q_max")
    @classmethod
    def _q_max_above_q_min(cls, value: float, info) -> float:
        q_min = info.data.get("q_min")
        if q_min is not None and value <= q_min:
            raise ValueError("q_max must be greater than q_min")
        return value

    @field_validator("r_max")
    @classmethod
    def _r_max_above_r_min(cls, value: float, info) -> float:
        r_min = info.data.get("r_min")
        if r_min is not None and value <= r_min:
            raise ValueError("r_max must be greater than r_min")
        return value

    @field_validator("r_constraint_search_max")
    @classmethod
    def _search_max_above_min(cls, value: float, info) -> float:
        search_min = info.data.get("r_constraint_search_min")
        if search_min is not None and value <= search_min:
            raise ValueError(
                "r_constraint_search_max must exceed r_constraint_search_min"
            )
        return value

    @model_validator(mode="after")
    def _apply_defaults(self) -> PDFConfig:
        if self.norm_q_min is None:
            self.norm_q_min = max(self.q_min, self.q_max - 10.0)
        if self.norm_q_min >= self.q_max:
            raise ValueError("norm_q_min must be below q_max")
        return self


class PDFResult(XRPDBaseModel):
    """Computed PDF results on uniform Q and r grids."""

    q: SerialisableNDArray  # Q grid (Å⁻¹)
    iq_corrected: SerialisableNDArray  # I(Q) in electron units
    sq: SerialisableNDArray  # S(Q) total structure function
    fq: SerialisableNDArray  # F(Q) = Q[S(Q) - 1] (Å⁻¹)
    r: SerialisableNDArray  # r grid (Å)
    gr: SerialisableNDArray  # G(r) (Å⁻²)
    number_density: float
    sample_name: str

    model_config = {"arbitrary_types_allowed": True}

    @property
    def baseline(self) -> np.ndarray:
        """Physical low-r baseline -4*pi*r*rho0."""
        return gr_baseline(self.r, self.number_density)

    @property
    def rdf(self) -> np.ndarray:
        """Pair correlation function g(r) = 1 + G(r)/baseline."""
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.divide(
                self.gr,
                self.baseline,
                out=np.full_like(self.gr, np.nan),
                where=~np.isclose(self.baseline, 0.0),
            )
        return 1.0 + ratio

    def save_results(
        self, export_formats: list[ExportFormat] | list[str], output_dir: str | Path
    ):

        output_dir = Path(output_dir)

        for fmt in export_formats:
            if fmt == ExportFormat.GR:
                save_two_column(
                    output_dir / f"{self.sample_name}.gr",
                    self.r,
                    self.gr,
                    header="G(r) — reduced PDF\n# r (Å)   G(r) (Å⁻²)",
                )
            elif fmt == ExportFormat.SQ:
                save_two_column(
                    output_dir / f"{self.sample_name}.sq",
                    self.q,
                    self.sq,
                    header="S(Q) — total structure function\n# Q (Å⁻¹)   S(Q)",
                )
            elif fmt == ExportFormat.IQ:
                save_two_column(
                    output_dir / f"{self.sample_name}.iq",
                    self.q,
                    self.iq_corrected,
                    header="I(Q) — Intensity (e.u.)\n# Q (Å⁻¹)   I(Q) (e.u.)",
                )
            elif fmt == ExportFormat.FQ:
                save_two_column(
                    output_dir / f"{self.sample_name}.fq",
                    self.q,
                    self.fq,
                    header="F(Q) = Q[S(Q)-1]\n# Q (Å⁻¹)   F(Q) (Å⁻¹)",
                )

    def plot(
        self,
        save_filepath: str | Path | None = None,
        ref_file: str | Path | None = None,
    ) -> None:
        """Diagnostic 2x2 plot of I(Q), S(Q), F(Q) and G(r)."""

        ref_path = Path(ref_file) if ref_file is not None else None

        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        fig.suptitle("PDF pipeline diagnostics", fontsize=13)

        ax = axes[0, 0]
        ax.plot(self.q, self.iq_corrected, lw=0.8, color="steelblue")
        ax.set_xlabel("Q (Å⁻¹)")
        ax.set_ylabel("I(Q) (e.u.)")
        ax.set_title("Normalised Scattering Intensity (electron units)")
        ax.grid(alpha=0.3)

        ax = axes[0, 1]
        ax.plot(self.q, self.sq, lw=0.8, color="steelblue")
        ax.axhline(1.0, color="k", lw=0.6, ls="--", label="S(Q) = 1")
        ax.set_xlabel("Q (Å⁻¹)")
        ax.set_ylabel("S(Q)")
        ax.set_title("Total structure function")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        ax = axes[1, 0]
        ax.plot(self.q, self.fq, lw=0.8, color="steelblue")
        ax.axhline(0.0, color="k", lw=0.6, ls="--")
        ax.set_xlabel("Q (Å⁻¹)")
        ax.set_ylabel("F(Q) = Q[S(Q)−1] (Å⁻¹)")
        ax.set_title("Reduced structure function")
        ax.grid(alpha=0.3)

        plot_r = self.r[self.r < 5]
        plot_baseline = self.baseline[self.r < 5]

        ax = axes[1, 1]
        ax.plot(self.r, self.gr, lw=0.9, color="steelblue", label="Computed G(r)")
        ax.plot(
            plot_r, plot_baseline, "k--", lw=0.8, label=r"$-4\pi r\rho_0$", zorder=2
        )

        if ref_path is not None and ref_path.exists():
            r_ref, g_ref = load_xy(ref_path)
            ax.plot(
                r_ref,
                g_ref,
                color="tomato",
                lw=0.9,
                ls="--",
                alpha=0.8,
                label="Reference G(r)",
            )

        ax.set_xlabel("r (Å)")
        ax.set_ylabel("G(r) (Å⁻²)")
        ax.set_title("Pair distribution function G(r)")
        ax.set_xlim(0, float(np.amax(self.r)))
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        fig.tight_layout()

        if save_filepath is not None:
            fig.savefig(save_filepath, dpi=150)
        plt.show()


def gr_baseline(r: SerialisableNDArray, rho0: float) -> np.ndarray:
    """Physical low-r baseline: G(r) = -4*pi*r*rho0 (no pairs below r_min)."""
    return -4.0 * np.pi * r * rho0


def build_scattering_factors(
    composition: ChemicalFormula,
    q: SerialisableNDArray,
) -> tuple[
    SerialisableNDArray, SerialisableNDArray, SerialisableNDArray, SerialisableNDArray
]:
    """Composition-averaged <f>, <f>^2, <f^2> and Compton scattering on q."""
    s = q_space_to_s(q)
    weights = composition.atomic_fraction

    f_mean = np.zeros_like(q)
    f_sq_mean = np.zeros_like(q)
    compton = np.zeros_like(q)

    for weight, element in zip(weights, composition.elements, strict=True):
        f_element = calculate_form_factor_for_element(element, s)
        f_mean += weight * f_element
        f_sq_mean += weight * f_element**2
        compton += weight * calculate_compton_for_element(element, s)

    f_mean_sq = f_mean**2
    return f_mean, f_mean_sq, f_sq_mean, compton


# Instrumental corrections
def polarisation_correction(
    two_theta_deg: SerialisableNDArray,
    synchrotron: bool,
    p: float,
) -> SerialisableNDArray:
    """Polarisation factor P(2*theta) for synchrotron or lab X-rays."""
    cos2t = np.cos(np.deg2rad(two_theta_deg))
    if synchrotron:
        return (1.0 - p) + p * cos2t**2
    return (1.0 + cos2t**2) / 2.0


# I/O utilities
def load_xy(path: Path) -> tuple[SerialisableNDArray, SerialisableNDArray]:
    """Load a two-column ASCII file, ignoring #/!/; comments. Returns (x, y)."""
    data = np.loadtxt(path, comments=["#", "!", ";"])
    if data.ndim == 1:
        raise ValueError(f"Expected two-column data in {path}; got one column.")
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Expected at least two columns in {path}.")
    x, y = data[:, 0], data[:, 1]
    order = np.argsort(x)
    return x[order].astype(np.float64), y[order].astype(np.float64)


def save_two_column(
    path: Path, x: SerialisableNDArray, y: SerialisableNDArray, header: str = ""
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.column_stack([x, y]), header=header, fmt="%.8e")


# ---------------------------------------------------------------------------
# Krogh-Moe / Norman normalisation
# ---------------------------------------------------------------------------
def _clip_background_extrapolation(
    q: SerialisableNDArray,
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
    low_value = float(np.interp(q_min_fit, q, background))
    high_value = float(np.interp(q_max_fit, q, background))
    clipped[q < q_min_fit] = low_value
    clipped[q > q_max_fit] = high_value
    return clipped


def _background_basis(
    q_values: SerialisableNDArray,
    q_min_fit: float,
    q_max_fit: float,
    degree: int,
    background_type: BackgroundType,
) -> np.ndarray:
    """Background design matrix on q_values, normalised to x in [-1, 1].

    Normalising Q before building the basis is essential for conditioning:
    raw powers of Q (Q, Q^2, Q^3, ...) span many orders of magnitude over a
    typical fit window and make the design matrix near-singular, which is
    enough on its own to produce wild background/alpha values. "chebyshev"
    uses the orthogonal Chebyshev basis on the normalised window (the most
    numerically stable choice and the default); "polynomial" uses a plain
    (normalised) power basis; "constant" uses a single DC offset, which is
    the most robust choice when the data is noisy or the window is narrow.
    """
    midpoint = 0.5 * (q_max_fit + q_min_fit)
    half_range = max(0.5 * (q_max_fit - q_min_fit), 1e-12)
    x = (q_values - midpoint) / half_range

    if background_type == "constant":
        return np.ones((len(x), 1))
    if background_type == "chebyshev":
        return np.polynomial.chebyshev.chebvander(x, degree)
    if background_type == "polynomial":
        return np.vander(x, degree + 1, increasing=True)
    raise ValueError(f"Unknown background_type '{background_type}'.")


def _fit_norman_alpha(
    q: SerialisableNDArray,
    intensity_q: SerialisableNDArray,
    f_sq_mean: SerialisableNDArray,
    compton: SerialisableNDArray,
    q_min_fit: float,
    q_max_fit: float,
    degree: int,
    background_type: BackgroundType,
) -> tuple[float, np.ndarray]:
    """One attempt at the joint Krogh-Moe/Norman fit for a fixed degree."""
    mask = (q >= q_min_fit) & (q <= q_max_fit)
    n_background_coeffs = 1 if background_type == "constant" else degree + 1
    min_points = n_background_coeffs + 2
    if mask.sum() < min_points:
        mask = np.ones(len(q), dtype=bool)
    if mask.sum() < min_points:
        raise PDFNormalisationError(
            "not enough data points for the requested background"
        )

    q_fit = q[mask]
    i_fit = intensity_q[mask]
    self_scattering_fit = f_sq_mean[mask] + compton[mask]
    background_basis = _background_basis(
        q_fit, q_min_fit, q_max_fit, degree, background_type
    )

    # Alpha and the background are refined together in a single bounded,
    # Q-weighted least-squares solve (weight ~ Q approximates the Q^2
    # weighting of the Norman criterion while downweighting noisy low-Q
    # data). Bounding alpha >= 0 makes a non-physical negative scale
    # factor structurally impossible, instead of an after-the-fact check.
    weights = q_fit
    design = np.column_stack([self_scattering_fit, background_basis]) * weights[:, None]
    target = i_fit * weights

    n_coeffs = design.shape[1]
    lower = np.full(n_coeffs, -np.inf)
    lower[0] = 1e-12
    upper = np.full(n_coeffs, np.inf)

    fit = lsq_linear(design, target, bounds=(lower, upper))
    alpha = float(fit.x[0])
    background_coeffs = fit.x[1:]

    if not fit.success or alpha <= 2e-12:
        raise PDFNormalisationError("bounded normalisation fit did not converge")

    condition_number = np.linalg.cond(design)
    if condition_number > 1e10:
        warnings.warn(
            f"Normalisation design matrix is poorly conditioned "
            f"(cond={condition_number:.2e}); results may be unreliable.",
            stacklevel=2,
        )

    full_basis = _background_basis(q, q_min_fit, q_max_fit, degree, background_type)
    background_full = full_basis @ background_coeffs
    background_full = _clip_background_extrapolation(
        q, background_full, q_min_fit, q_max_fit
    )
    return alpha, background_full


def krogh_moe_normalise(
    q: SerialisableNDArray,
    intensity_q: SerialisableNDArray,
    f_sq_mean: SerialisableNDArray,
    compton: SerialisableNDArray,
    q_min_fit: float,
    q_max_fit: float,
    poly_degree: int,
    background_type: BackgroundType = "chebyshev",
) -> tuple[float, np.ndarray]:
    """Jointly fit the scale factor alpha and a smooth background over the
    high-Q window [q_min_fit, q_max_fit] via one bounded least-squares solve:

        I_meas(Q) ~= alpha * [<f^2(Q)> + I_Compton(Q)] + background(Q)

    alpha is bounded to be strictly positive, so it cannot come out
    negative. If the requested background degree cannot be fit, the degree
    is reduced and, as a last resort, the background falls back to a
    single constant offset before raising.
    """
    q_max_fit = min(q_max_fit, float(np.amax(q)))
    q_min_fit = max(q_min_fit, float(np.amin(q)))

    degree = poly_degree
    current_type = background_type
    last_error: Exception | None = None
    while True:
        try:
            return _fit_norman_alpha(
                q,
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


# Termination window functions
def make_termination_window(
    q: SerialisableNDArray,
    q_max: float,
    window_type: WindowType | None,
    super_lorch_power: int | float | None = 2,
) -> np.ndarray:
    """Multiplicative window W(Q) applied to F(Q) to reduce Fourier
    termination ripples. "lorch" gives the strongest ripple suppression at
    the cost of peak broadening; "cosine" is intermediate; "none" applies
    no window.
    super_lorch_power
    Exponent used for the Super-Lorch window. A value of 2 reproduces
    the commonly used Super-Lorch window. A value of 1 is identical to
    the standard Lorch window.
    """

    if window_type == "lorch":
        window = np.ones_like(q)
        nonzero = q > 0.0
        arg = np.pi * q[nonzero] / q_max
        window[nonzero] = np.sin(arg) / arg
        return window
    elif window_type == "cosine":
        return 0.5 * (1.0 + np.cos(np.pi * q / q_max))
    elif window_type == "super-lorch" and super_lorch_power is not None:
        window = np.ones_like(q)
        nonzero = q > 0.0
        arg = np.pi * q[nonzero] / q_max
        sinc = np.sin(arg) / arg
        window[nonzero] = sinc**super_lorch_power
        return window
    elif window_type is None or window_type == "none":
        return np.ones_like(q)
    elif window_type == "super-lorch" and super_lorch_power is None:
        raise ValueError(
            f"If window_type is '{window_type}'. Then super_lorch_power must be int"
        )
    else:
        raise ValueError(
            f"Unknown termination_window '{window_type}'. Choose 'lorch', "
            "'cosine', 'super-lorch' or 'none'."
        )


# Fourier transforms
def sine_transform_fq_to_gr(
    q: SerialisableNDArray,
    f_q: SerialisableNDArray,
    r: SerialisableNDArray,
    dq: float,
) -> np.ndarray:
    """Forward sine transform G(r) = (2/pi) * integral F(Q) sin(Qr) dQ,
    via direct matrix product (avoids FFT grid constraints).
    """
    sin_qr = np.sin(np.outer(r, q))
    return (2.0 / np.pi) * dq * (sin_qr @ f_q)


def sine_transform_gr_to_fq(
    r: SerialisableNDArray,
    g_r: SerialisableNDArray,
    q: SerialisableNDArray,
) -> np.ndarray:
    """Inverse (back) sine transform F(Q) = integral G(r) sin(Qr) dr."""
    sin_qr = np.sin(np.outer(q, r))
    return np.trapezoid(sin_qr * g_r[np.newaxis, :], r, axis=1)  # type: ignore - it will be an array of floats


# Real-space constraint: Toby-Egami / PDFgetX3 method
def _auto_r_constraint(
    r: SerialisableNDArray,
    g: SerialisableNDArray,
    rho0: float,
    r_search_min: float = 1.2,
    r_search_max: float = 3.5,
) -> float:
    """First upward crossing of G(r) through the physical baseline in
    [r_search_min, r_search_max], marking the onset of the first
    coordination shell. Falls back to r_search_min if no crossing exists.
    """
    baseline = gr_baseline(r, rho0)
    deviation = g - baseline

    idx = (r >= r_search_min) & (r <= r_search_max)
    if not np.any(idx):
        return r_search_min

    r_sub = r[idx]
    d_sub = deviation[idx]

    crossings = np.where((d_sub[:-1] < 0.0) & (d_sub[1:] >= 0.0))[0]
    if len(crossings) == 0:
        return float(r_search_min)

    i = crossings[0]
    r0, r1 = float(r_sub[i]), float(r_sub[i + 1])
    d0, d1 = float(d_sub[i]), float(d_sub[i + 1])
    r_cross = r0 - d0 * (r1 - r0) / (d1 - d0)
    return float(np.clip(r_cross, r_search_min, r_search_max))


def apply_real_space_constraint(
    r: SerialisableNDArray,
    g: SerialisableNDArray,
    f_q: SerialisableNDArray,
    q: SerialisableNDArray,
    dq: float,
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
    g = g.copy()
    f_q = f_q.copy()

    g_physical_full = gr_baseline(r, rho0)
    mask_low = r <= r_max_constraint
    mask_phys = ~mask_low

    for _ in range(n_iterations):
        delta_g = np.where(mask_low, g - g_physical_full, 0.0)

        rms = float(np.sqrt(np.mean(delta_g[mask_low] ** 2)))
        g_peak = float(np.max(np.abs(g[mask_phys]))) if np.any(mask_phys) else 1.0
        if rms / max(g_peak, 1e-12) < 1e-5:
            break

        delta_f = sine_transform_gr_to_fq(r, delta_g, q)
        f_q -= delta_f

        r_pos = r[1:]
        g_pos = sine_transform_fq_to_gr(q, f_q, r_pos, dq)
        g = np.concatenate([[0.0], g_pos])

    return r, g, f_q


def compute_pdf(xy_path: Path, config: PDFConfig) -> PDFResult:
    """Full PDF pipeline: load -> correct -> normalise -> S(Q) -> F(Q) -> G(r).

    1. Load raw .xy data; subtract background; polarisation-correct.
    2. Convert 2-theta -> Q; spline-resample onto a uniform Q grid.
    3. Compute Waasmaier-Kirfel form factors and Compton scattering.
    4. Krogh-Moe/Norman normalisation: determine alpha and background.
    5. Coherent elastic intensity in electron units.
    6. Total structure function S(Q); reduced structure function F(Q).
    7. Apply termination window and Q-resolution damping.
    8. Sine Fourier transform -> initial G(r).
    9. Iterative real-space (Toby-Egami) constraint at low r.
    """
    tth, intensity = load_xy(xy_path)

    if config.background_file is not None:
        tth_bg, i_bg = load_xy(config.background_file)
        i_bg_interp = np.interp(tth, tth_bg, i_bg, left=0.0, right=0.0)
        intensity = intensity - config.background_scale * i_bg_interp

    intensity = np.maximum(intensity, 0.0)

    if config.polarisation_factor:
        pol = polarisation_correction(tth, config.is_synchrotron, config.polarisation_p)
        intensity = intensity / np.maximum(pol, 1e-12)

    q_raw = two_theta_to_q(tth, float(config.data.wavelength))
    q_max_data = float(q_raw.max())
    q_min_data = float(q_raw.min())
    q_max_use = min(config.q_max, q_max_data)

    dq = (
        config.q_step
        if config.q_step is not None
        else min(float(np.median(np.diff(q_raw))), 0.05)
    )
    q = np.arange(config.q_min, q_max_use + 0.5 * dq, dq, dtype=np.float64)

    valid = intensity > 0
    if valid.sum() < 4:
        raise ValueError("Fewer than 4 valid data points; cannot fit a spline.")
    spline = UnivariateSpline(q_raw[valid], intensity[valid], s=0, k=3, ext=1)
    i_q = np.maximum(spline(q), 0.0)
    i_q[(q < q_min_data) | (q > q_max_data)] = 0.0

    f_mean, f_mean_sq, f_sq_mean, compton = build_scattering_factors(
        config.composition, q
    )

    alpha, background = krogh_moe_normalise(
        q,
        i_q,
        f_sq_mean,
        compton,
        q_min_fit=float(config.norm_q_min),  # type: ignore[arg-type]
        q_max_fit=q_max_use,
        poly_degree=config.norm_poly_degree,
        background_type=config.background_type,
    )

    i_eu = (i_q - background) / alpha

    # S(Q) = [I_eu - I_Compton - (<f^2> - <f>^2)] / <f>^2
    # (<f^2> - <f>^2) is the Laue monotonic diffuse term, zero for a
    # single-element sample. S(Q) -> 1 at large Q.
    with np.errstate(divide="ignore", invalid="ignore"):
        s_q = (i_eu - compton - (f_sq_mean - f_mean_sq)) / np.maximum(f_mean_sq, 1e-30)
    s_q = np.where(np.isfinite(s_q), s_q, 1.0)

    # No reliable data below q_min: S(Q) = 1 there is the correct null
    # assumption and introduces no spurious low-r features.
    s_q[q < config.q_min] = 1.0

    f_q = q * (s_q - 1.0)
    f_q[q == 0.0] = 0.0

    window = make_termination_window(
        q, q_max_use, config.termination_window, config.super_lorch_power
    )
    damping = np.exp(-0.5 * (config.qdamp * q) ** 2)
    f_mod = f_q * window * damping

    r_pos = np.arange(
        config.r_step,
        config.r_max + 0.5 * config.r_step,
        config.r_step,
        dtype=np.float64,
    )
    g_pos = sine_transform_fq_to_gr(q, f_mod, r_pos, dq)
    r_full = np.concatenate([[0.0], r_pos])
    g_full = np.concatenate([[0.0], g_pos])

    if config.use_real_space_constraint and config.real_space_constraint_iterations > 0:
        if config.r_constraint_max is not None:
            r_cut = float(config.r_constraint_max)
        else:
            r_cut = _auto_r_constraint(
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
            q,
            dq,
            config.number_density,
            r_max_constraint=r_cut,
            n_iterations=config.real_space_constraint_iterations,
        )

    return PDFResult(
        q=q,
        iq_corrected=i_eu,
        sq=s_q,
        fq=f_q,  # the un-windowed F(Q)
        r=r_full,
        gr=g_full,
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
    termination_window: WindowType = "super-lorch",
    super_lorch_power: int | float = 2,
    use_real_space_constraint: bool = True,
    real_space_constraint_iterations: int = 10,
    r_constraint_max: float | None = None,
    norm_poly_degree: int = 3,
    norm_q_min: float | None = None,
    background_type: BackgroundType = "chebyshev",
    is_synchrotron: bool = True,
    polarisation_p: float = 0.99,
    background_file: Path | str | None = None,
    background_scale: float = 1.0,
    export_formats: list[ExportFormat] | list[str] | None = None,
    output_dir: Path | str | None = None,
) -> PDFResult:
    """Compute G(r) from a raw .xy diffraction file and optionally write
    output files. composition is e.g. {"Si": 1} or {"Pb": 1, "Ti": 1, "O": 3}.
    number_density is atoms per cell divided by cell volume (atoms/Å³).
    """

    if isinstance(formula, str):
        isinstance(formula, str)
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
        sample_name = Path(xy_file).stem

    data = ScatteringData.from_xye(
        filepath=xy_path, x_unit="tth", data_type="xray", wavelength=wavelength
    )

    cfg = PDFConfig(
        composition=composition,
        data=data,
        sample_name=sample_name,
        number_density=number_density,
        q_min=q_min,
        q_max=q_max,
        r_max=r_max,
        r_step=r_step,
        qdamp=qdamp,
        termination_window=termination_window,
        super_lorch_power=super_lorch_power,
        use_real_space_constraint=use_real_space_constraint,
        real_space_constraint_iterations=real_space_constraint_iterations,
        r_constraint_max=r_constraint_max,
        norm_poly_degree=norm_poly_degree,
        norm_q_min=norm_q_min,
        background_type=background_type,
        is_synchrotron=is_synchrotron,
        polarisation_p=polarisation_p,
        background_file=Path(background_file) if background_file else None,
        background_scale=background_scale,
        export_formats=[ExportFormat(f) for f in export_formats],
        output_dir=Path(output_dir),
    )
    result = compute_pdf(Path(xy_path), cfg)
    result.save_results(export_formats=cfg.export_formats, output_dir=cfg.output_dir)

    return result


#  crystalline silicon at i15-1
if __name__ == "__main__":
    xy_file = Path("/workspaces/xrpd-toolbox/tests/data/Si_pe2_i15_1.xy")
    ref_file = Path("/workspaces/xrpd-toolbox/tests/data/Si_pe2_i15_1.gr")

    # Crystalline Si: 8 atoms per conventional cubic unit cell, a = 5.4309 Å
    rho_si = 8.0 / 5.4309**3

    # si = ChemicalFormula.load_from_composition({"Si": 1})

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
        r_step=0.01,
        qdamp=0.003,
        is_synchrotron=True,
        polarisation_p=0.99,
        termination_window="super-lorch",
        super_lorch_power=2,
        use_real_space_constraint=True,
        real_space_constraint_iterations=10,
        r_constraint_max=2.1,
        norm_poly_degree=3,
        export_formats=["gr", "sq", "fq", "iq"],
    )

    result.plot(ref_file=ref_file)
