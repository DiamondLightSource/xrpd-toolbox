"""
Pair Distribution Function (PDF) Calculator
============================================
Converts powder X-ray diffraction data (.xy, 2θ vs intensity) into the
reduced pair distribution function G(r).

Physical definitions
--------------------
The reduced PDF is defined as (Egami & Billinge, 2003):

    G(r) = 4πr[ρ(r) - ρ₀]

where ρ(r) is the local atomic number density and ρ₀ is the average
atomic number density.  G(r) is obtained via the sine Fourier transform
of the reduced structure function F(Q):

    F(Q) = Q[S(Q) - 1]

    G(r) = (2/π) ∫₀^{Q_max}  F(Q) · W(Q) · sin(Qr) dQ

where W(Q) is an optional termination window function.

Physical boundary condition (real-space constraint)
----------------------------------------------------
For r less than the shortest interatomic distance r_min, no atom pairs
exist, so the local density equals the average:  ρ(r) = ρ₀, giving:

    G(r) = -4πrρ₀    for r < r_min

This is enforced iteratively by the Toby–Egami back-Fourier method
(Toby & Egami, 1992; Juhás et al., 2013):

  1. Compute deviation  ΔG(r) = G(r) - (-4πrρ₀)  in the unphysical region.
  2. Back-transform ΔG(r) to Q-space to obtain ΔF(Q).
  3. Subtract ΔF(Q) from F(Q) and re-transform.
  4. Repeat to convergence.

This is the only physically rigorous method to suppress Fourier
termination ripples and normalisation artefacts at low r.

Normalisation
-------------
The absolute scale factor α is determined by the Krogh-Moe / Norman
method: in the high-Q limit the measured coherent intensity (in electron
units) must converge to the self-scattering:

    I_eu(Q) → ⟨f²(Q)⟩ + I_Compton(Q)   as Q → Q_max

Using Q²-weighted integrals (Norman, 1957) to down-weight the noisier
low-Q region:

    α = ∫ [⟨f²⟩ + I_Compton] Q² dQ  /  ∫ [I_meas - background] Q² dQ

where the background is a low-order polynomial in Q fitted simultaneously.

References
----------
Egami T. & Billinge S.J.L. (2003). Underneath the Bragg Peaks.
  Pergamon, Oxford.
Norman N. (1957). Acta Cryst. 10, 370-373.
Krogh-Moe J. (1956). Acta Cryst. 9, 951-953.
Toby B.H. & Egami T. (1992). Acta Cryst. A48, 336-346.
Juhás P., Davis T., Farrow C.L. & Billinge S.J.L. (2013).
  J. Appl. Cryst. 46, 560-566.
Waasmaier D. & Kirfel A. (1995). Acta Cryst. A51, 416-431.
Lorch E. (1969). J. Phys. C: Solid State Phys. 2, 229-232.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator
from scipy.interpolate import UnivariateSpline
from scipy.signal import find_peaks

from xrpd_toolbox.constants import ELEMENT_ATOMIC_NUMBER
from xrpd_toolbox.core import FloatArray
from xrpd_toolbox.fit_engine.form_factors import (
    X_RAY_FORM_FACTORS,
    calculate_form_factor_for_element,
)
from xrpd_toolbox.utils.unit_conversion import two_theta_to_q

# ---------------------------------------------------------------------------
# Waasmaier–Kirfel atomic form factor coefficients
# f(s) = Σᵢ aᵢ exp(-bᵢ s²) + c,   s = sinθ/λ = Q / (4π)
# Source: Waasmaier & Kirfel (1995) Acta Cryst. A51, 416-431.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Configuration and result data models
# ---------------------------------------------------------------------------
WINDOW_TYPES = Literal["lorch", "cosine", "none"]
BACKGROUND_TYPES = Literal["constant", "polynomial", "chebyshev"]


class ExportFormat(StrEnum):
    GR = "gr"  # Reduced PDF  G(r)
    SQ = "sq"  # Total structure function  S(Q)
    IQ = "iq"  # Coherent intensity in electron units  I(Q)
    FQ = "fq"  # Reduced structure function  F(Q) = Q[S(Q)-1]


class CompositionEntry(BaseModel):
    element: str
    count: float = Field(gt=0)

    @field_validator("element")
    @classmethod
    def element_must_be_known(cls, v: str) -> str:
        if v not in X_RAY_FORM_FACTORS.keys():
            raise ValueError(
                f"Element '{v}' not in Waasmaier-Kirfel table. "
                f"Available: {sorted(X_RAY_FORM_FACTORS.keys())}"
            )
        return v


class PDFConfig(BaseModel):
    """
    All parameters controlling the PDF calculation.

    Required
    --------
    composition     : list of CompositionEntry (element, stoichiometric count)
    wavelength      : X-ray wavelength in Å
    number_density  : average atomic number density ρ₀ in atoms/Å³

    Q-space
    -------
    q_min           : lower Q limit used for the Fourier transform (Å⁻¹).
                      Data below q_min are excluded; F(Q) is set to zero there.
                      Typical: 0.5–1.0 Å⁻¹.
    q_max           : upper Q limit (Å⁻¹); clipped to the data maximum.
    q_step          : uniform Q grid spacing (Å⁻¹); auto-derived from data
                      if None (recommended).

    Instrumental corrections
    ------------------------
    polarisation_factor : apply polarisation correction to raw intensity
    is_synchrotron      : True → synchrotron polarisation model
    polarisation_p      : polarisation fraction p (synchrotron only; 0.95–0.99)
    background_file     : path to background .xy file (e.g. empty capillary)
    background_scale    : scale factor applied to the background before subtraction

    Normalisation (Krogh-Moe / Norman)
    -----------------------------------
    norm_poly_degree : degree of the polynomial background fitted jointly with α
                       in the high-Q normalisation window (typical: 3–5)
    norm_q_min       : lower bound of the high-Q normalisation window (Å⁻¹).
                       Defaults to max(q_min, q_max - 10).

    Real-space output
    -----------------
    r_min            : start of r grid in Å (keep at 0.0)
    r_max            : maximum r in Å
    r_step           : r grid spacing in Å (0.01 Å is typical)

    Termination and damping
    -----------------------
    termination_window : "lorch"  – sinc window W(Q) = sin(πQ/Q_max)/(πQ/Q_max)
                                    (Lorch 1969); broadens peaks slightly but
                                    strongly suppresses termination ripples.
                         "cosine" – Hann window W(Q) = ½[1+cos(πQ/Q_max)];
                                    moderate ripple suppression.
                         "none"   – no window (maximum real-space resolution
                                    but strong termination ripples).
    qdamp            : Gaussian Q-resolution parameter σ_Q (Å).  Models
                       instrument Q-resolution; broadens G(r) peaks with a
                       width that grows linearly with r.  Typical: 0.01–0.05 Å.

    Real-space constraint
    ---------------------
    use_real_space_constraint          : enable Toby–Egami constraint
    real_space_constraint_iterations   : number of back-Fourier cycles (5–20)
    r_constraint_max : upper limit of the unphysical region (Å).
                       Must be set below the first interatomic distance.
                       If None, auto-detected from the first crossing of
                       G(r) above the -4πrρ₀ baseline.
    """

    composition: list[CompositionEntry]
    wavelength: float = Field(gt=0, description="X-ray wavelength (Å)")
    number_density: float = Field(gt=0, description="Atomic number density (atoms/Å³)")

    q_min: float = Field(default=0.5, gt=0)
    q_max: float = Field(default=30.0, gt=0)
    q_step: float | None = None

    polarisation_factor: bool = True
    is_synchrotron: bool = False
    polarisation_p: float = Field(default=0.99, ge=0.0, le=1.0)
    background_file: Path | None = None
    background_scale: float = 1.0

    norm_poly_degree: int = Field(default=3, ge=0)
    norm_q_min: float | None = None

    r_min: float = 0.0
    r_max: float = 30.0
    r_step: float = Field(default=0.01, gt=0)

    termination_window: Literal["lorch", "cosine", "none"] | None = "cosine"
    qdamp: float = Field(default=0.030, ge=0.0)

    use_real_space_constraint: bool = True
    real_space_constraint_iterations: int = Field(default=10, ge=0)
    r_constraint_max: float | None = None

    export_formats: list[ExportFormat] = Field(
        default_factory=lambda: [ExportFormat.GR]
    )
    output_dir: Path = Path(".")
    output_stem: str = "pdf_output"

    @model_validator(mode="after")
    def _set_defaults(self) -> PDFConfig:
        if self.norm_q_min is None:
            val = max(self.q_min, self.q_max - 10.0)
            object.__setattr__(self, "norm_q_min", val)
        return self

    model_config = {"arbitrary_types_allowed": True}


class PDFResult(BaseModel):
    """Computed PDF results on uniform grids."""

    q: FloatArray  # Q grid (Å⁻¹)
    iq_corrected: FloatArray  # I(Q) in electron units
    sq: FloatArray  # S(Q) total structure function
    fq: FloatArray  # F(Q) = Q[S(Q) - 1] (Å⁻¹)
    r: FloatArray  # r grid (Å)
    gr: FloatArray  # G(r) (Å⁻²)

    model_config = {"arbitrary_types_allowed": True}


def _compton_hubbell(element: str, s_arr: FloatArray) -> FloatArray:
    """
    Compton (inelastic) scattering per atom using the Cromer–Mann
    approximation:  I_C(s) ≈ Z - f²(s)/Z

    This is the analytic approximation; for publication work, tabulated
    Hubbell coefficients should be used if available.  The approximation
    is adequate for light to medium-weight elements (Z ≤ 40).
    """
    z = ELEMENT_ATOMIC_NUMBER[element]
    f = calculate_form_factor_for_element(element, s_arr)

    print(f)
    print(z)

    return np.maximum(0.0, z - f**2 / z)


def build_scattering_factors(
    composition: list[CompositionEntry],
    q: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """
    Compute composition-averaged scattering quantities on the Q grid.

    Returns
    -------
    f_mean    : ⟨f(Q)⟩   — composition-weighted mean form factor
    f_mean_sq : ⟨f(Q)⟩²  — square of the mean
    f_sq_mean : ⟨f²(Q)⟩  — mean of the squares (includes Laue term)
    compton   : ⟨I_C(Q)⟩  — Compton scattering per average atom
    """
    s = q / (4.0 * np.pi)
    total = sum(e.count for e in composition)
    weights = np.array([e.count / total for e in composition])

    f_mean = np.zeros_like(q)
    f_sq_mean = np.zeros_like(q)
    compton = np.zeros_like(q)

    for w, entry in zip(weights, composition):
        f_el = calculate_form_factor_for_element(entry.element, s)
        f_mean += w * f_el
        f_sq_mean += w * f_el**2
        compton += w * _compton_hubbell(entry.element, s)

    f_mean_sq = f_mean**2
    return f_mean, f_mean_sq, f_sq_mean, compton


# ---------------------------------------------------------------------------
# Instrumental corrections
# ---------------------------------------------------------------------------


def polarisation_correction(
    two_theta_deg: FloatArray,
    synchrotron: bool,
    p: float,
) -> FloatArray:
    """
    Polarisation factor P(2θ) for total scattering data.

    For synchrotron radiation with horizontal polarisation fraction p:
        P = (1 - p) + p cos²(2θ)

    For conventional (unpolarised) laboratory X-rays:
        P = (1 + cos²(2θ)) / 2

    Note: the Lorentz factor is *not* applied because total scattering
    analysis uses the full continuous diffraction pattern, not just
    integrated Bragg peak intensities.
    """
    cos2t = np.cos(np.deg2rad(two_theta_deg))
    if synchrotron:
        return (1.0 - p) + p * cos2t**2
    return (1.0 + cos2t**2) / 2.0


def two_theta_to_q(two_theta_deg: FloatArray, wavelength: float) -> FloatArray:
    """Q = 4π sinθ / λ  (Å⁻¹)."""
    return 4.0 * np.pi * np.sin(np.deg2rad(two_theta_deg / 2.0)) / wavelength


# ---------------------------------------------------------------------------
# I/O utilities
# ---------------------------------------------------------------------------


def load_xy(path: Path) -> tuple[FloatArray, FloatArray]:
    """
    Load a two-column ASCII file (whitespace or comma separated).
    Comment lines starting with #, !, or ; are ignored.
    Returns (x, y) sorted by x.
    """
    data = np.loadtxt(path, comments=["#", "!", ";"])
    if data.ndim == 1:
        raise ValueError(f"Expected two-column data in {path}; got one column.")
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Expected at least two columns in {path}.")
    x, y = data[:, 0], data[:, 1]
    order = np.argsort(x)
    return x[order].astype(np.float64), y[order].astype(np.float64)


def save_two_column(path: Path, x: FloatArray, y: FloatArray, header: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.column_stack([x, y]), header=header, fmt="%.8e")


# ---------------------------------------------------------------------------
# Krogh-Moe / Norman normalisation
# ---------------------------------------------------------------------------


def krogh_moe_normalise(
    q: FloatArray,
    intensity_q: FloatArray,
    f_sq_mean: FloatArray,
    compton: FloatArray,
    q_min_fit: float,
    q_max_fit: float,
    poly_degree: int,
) -> tuple[float, FloatArray]:
    """
    Determine the absolute scale factor α and a smooth additive background
    polynomial by imposing the Krogh-Moe / Norman condition over the
    high-Q normalisation window [q_min_fit, q_max_fit]:

        I_meas(Q) ≈ α · [⟨f²(Q)⟩ + I_Compton(Q)] + P(Q)

    where P(Q) is a polynomial of degree `poly_degree` in Q (not Q², to
    allow for asymmetric backgrounds arising from air scatter, fluorescence
    tails, and other sources that do not respect even symmetry in Q).

    The scale factor α is determined from the Q²-weighted Norman integral:

        α = ∫_{fit} [⟨f²⟩ + I_C] Q² dQ
            ─────────────────────────────
            ∫_{fit} [I_meas - P(Q)] Q² dQ

    Iterations:
      1. Simultaneous least-squares fit of α and polynomial coefficients.
      2. Recompute the Norman ratio; update α.
      3. Refit polynomial with fixed α.
      4. Repeat until |Δα/α| < 1×10⁻⁵ (typically < 20 iterations).

    This two-step procedure avoids the conflation of scale and background
    that occurs when both are varied simultaneously in a single χ² fit
    (which can yield unphysical negative α values for noisy data).

    Parameters
    ----------
    q            : Q grid (Å⁻¹)
    intensity_q  : resampled, polarisation-corrected intensity on q
    f_sq_mean    : ⟨f²(Q)⟩ on q
    compton      : Compton scattering per atom on q
    q_min_fit    : lower bound of normalisation window (Å⁻¹)
    q_max_fit    : upper bound of normalisation window (Å⁻¹)
    poly_degree  : degree of background polynomial

    Returns
    -------
    alpha      : absolute scale factor (dimensionless)
    background : polynomial background evaluated on the full q grid
    """
    mask = (q >= q_min_fit) & (q <= q_max_fit)
    if mask.sum() < poly_degree + 3:
        # Fallback: use the whole Q range
        mask = np.ones(len(q), dtype=bool)

    q_fit = q[mask]
    I_fit = intensity_q[mask]
    self_sc_fit = f_sq_mean[mask] + compton[mask]

    # Design matrix: [self-scattering column | polynomial columns 1, Q, Q², …]
    poly_matrix = np.column_stack([q_fit**k for k in range(poly_degree + 1)])
    A = np.column_stack([self_sc_fit, poly_matrix])
    coeffs, *_ = np.linalg.lstsq(A, I_fit, rcond=None)

    alpha = float(coeffs[0])
    poly_coeffs = coeffs[1:]  # shape: (poly_degree + 1,)

    if alpha <= 0:
        # Protect against ill-conditioned initial fit
        alpha = float(
            np.trapezoid(self_sc_fit * q_fit**2, q_fit)
            / np.trapezoid(I_fit * q_fit**2, q_fit)
        )

    # Polynomial basis on the full Q grid
    poly_basis_full = np.column_stack([q**k for k in range(poly_degree + 1)])
    poly_basis_fit = np.column_stack([q_fit**k for k in range(poly_degree + 1)])

    # Iterative α refinement using the Q²-weighted Norman integral
    for _ in range(50):
        background = poly_basis_full @ poly_coeffs

        # Norman ratio: integral of self-scattering / integral of corrected data
        # Both weighted by Q² to suppress the noisy low-Q region
        numerator = np.trapezoid(self_sc_fit * q_fit**2, q_fit)
        denominator = np.trapezoid(
            (I_fit - poly_basis_fit @ poly_coeffs) * q_fit**2, q_fit
        )
        if abs(denominator) < 1e-30:
            break

        alpha_new = numerator / denominator

        # Refit background polynomial with the updated α
        res = I_fit - alpha_new * self_sc_fit
        poly_coeffs, *_ = np.linalg.lstsq(poly_basis_fit, res, rcond=None)

        if abs(alpha_new - alpha) / max(abs(alpha), 1e-30) < 1e-5:
            alpha = alpha_new
            break
        alpha = alpha_new

    background = poly_basis_full @ poly_coeffs
    return alpha, background


# ---------------------------------------------------------------------------
# Termination window functions
# ---------------------------------------------------------------------------


def make_termination_window(
    q: FloatArray,
    q_max: float,
    window_type: str,
) -> FloatArray:
    """
    Multiplicative window W(Q) applied to F(Q) before Fourier transformation
    to reduce Fourier termination ripples arising from the finite Q_max cutoff.

    The window acts as a low-pass filter in r-space:  a sharper window
    (Lorch) broadens peaks more but suppresses ripples more aggressively.

    "lorch"  : sinc window (Lorch 1969):
                   W(Q) = sin(πQ/Q_max) / (πQ/Q_max)
               Equivalent to convolving G(r) with a step function of
               half-width Δr = π/Q_max.  Provides the strongest ripple
               suppression at the cost of peak broadening.

    "cosine" : Hann window:
                   W(Q) = ½[1 + cos(πQ/Q_max)]
               Intermediate ripple suppression with less broadening than
               the Lorch window.

    "none"   : W(Q) = 1  (no window; maximum real-space resolution but
               strong termination ripples if Q_max is finite).

    All windows satisfy W(0) = 1 and W(Q_max) = 0.
    """
    if window_type == "lorch":
        w = np.ones_like(q)
        nz = q > 0.0
        arg = np.pi * q[nz] / q_max
        w[nz] = np.sin(arg) / arg
        return w
    elif window_type == "cosine":
        return 0.5 * (1.0 + np.cos(np.pi * q / q_max))
    elif window_type == "none":
        return np.ones_like(q)
    else:
        raise ValueError(
            f"Unknown termination_window '{window_type}'. "
            "Choose 'lorch', 'cosine', or 'none'."
        )


# ---------------------------------------------------------------------------
# Fourier transforms
# ---------------------------------------------------------------------------


def sine_transform_fq_to_gr(
    q: FloatArray,
    f_q: FloatArray,
    r: FloatArray,
    dq: float,
) -> FloatArray:
    """
    Forward sine Fourier transform:

        G(r) = (2/π) ∫₀^{Q_max} F(Q) sin(Qr) dQ

    Implemented as a direct matrix product — accurate at the Q-point
    densities used in total scattering and avoids FFT grid constraints.

    Parameters
    ----------
    q   : Q grid (Å⁻¹), shape (N_q,)
    f_q : F(Q) on q,    shape (N_q,)
    r   : r grid (Å),   shape (N_r,)  — must not include r = 0
    dq  : uniform Q step (Å⁻¹)

    Returns
    -------
    g_r : G(r) at each r point,  shape (N_r,)
    """
    # sin_qr[i, j] = sin(r[i] * q[j]),  shape (N_r, N_q)
    sin_qr = np.sin(np.outer(r, q))
    return (2.0 / np.pi) * dq * (sin_qr @ f_q)


def sine_transform_gr_to_fq(
    r: FloatArray,
    g_r: FloatArray,
    q: FloatArray,
) -> FloatArray:
    """
    Inverse (back) sine Fourier transform:

        F(Q) = ∫₀^{r_max} G(r) sin(Qr) dr

    Used in the real-space constraint iteration.  Integration via the
    trapezoidal rule to handle the non-uniform first interval [0, r_step].

    Parameters
    ----------
    r   : r grid (Å), shape (N_r,) — must include r = 0 as first element
    g_r : G(r) on r,  shape (N_r,)
    q   : Q grid (Å⁻¹), shape (N_q,)

    Returns
    -------
    f_q : shape (N_q,)
    """
    # sin_qr[j, i] = sin(q[j] * r[i]),  shape (N_q, N_r)
    sin_qr = np.sin(np.outer(q, r))
    # Trapezoidal integration over r for each Q point
    return np.trapezoid(sin_qr * g_r[np.newaxis, :], r, axis=1)


# ---------------------------------------------------------------------------
# Real-space constraint: Toby–Egami / PDFgetX3 method
# ---------------------------------------------------------------------------


def _auto_r_constraint(
    r: FloatArray,
    g: FloatArray,
    rho0: float,
    r_search_min: float = 1.2,
    r_search_max: float = 3.5,
) -> float:
    """
    Automatically find r_constraint_max as the first r in [r_search_min, r_search_max]
    where G(r) crosses upward through the physical baseline -4πrρ₀.

    This crossing marks the onset of the first coordination shell.
    Everything below it is the unphysical region where the constraint applies.

    Falls back to r_search_min if no crossing is found.
    """
    baseline = -4.0 * np.pi * r * rho0
    deviation = g - baseline

    idx = (r >= r_search_min) & (r <= r_search_max)
    if not np.any(idx):
        return r_search_min

    r_sub = r[idx]
    d_sub = deviation[idx]

    # Find upward zero crossings (negative → positive)
    crossings = np.where((d_sub[:-1] < 0.0) & (d_sub[1:] >= 0.0))[0]
    if len(crossings) == 0:
        return float(r_search_min)

    i = crossings[0]
    r0, r1 = float(r_sub[i]), float(r_sub[i + 1])
    d0, d1 = float(d_sub[i]), float(d_sub[i + 1])
    # Linear interpolation to exact crossing
    r_cross = r0 - d0 * (r1 - r0) / (d1 - d0)
    return float(np.clip(r_cross, r_search_min, r_search_max))


def apply_real_space_constraint(
    r: FloatArray,
    g: FloatArray,
    f_q: FloatArray,
    q: FloatArray,
    dq: float,
    rho0: float,
    r_max_constraint: float,
    n_iterations: int,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """
    Iterative real-space constraint correction (Toby & Egami, 1992).

    Physical basis
    --------------
    Below the shortest interatomic distance r_min, the pair density equals
    the average density (no atom pairs exist):

        G(r) = -4πrρ₀   for r < r_min

    Any deviation  ΔG(r) = G(r) - (-4πrρ₀)  in r ∈ [0, r_max_constraint]
    is unphysical, arising from:
      - Normalisation errors (wrong α)
      - Imperfect Compton background subtraction
      - Fourier termination ripples

    Algorithm (one cycle)
    ---------------------
    1. Form  ΔG(r) = G(r) + 4πrρ₀  for r ≤ r_max_constraint, else 0.
    2. Back-transform:  ΔF(Q) = ∫₀^{r_max} ΔG(r) sin(Qr) dr
       using the trapezoidal rule (handles the r=0 boundary correctly).
    3. Correct:  F(Q) ← F(Q) − ΔF(Q)
    4. Re-transform:  G(r) = (2/π) ∫₀^{Q_max} F(Q) sin(Qr) dQ
    5. Check convergence; repeat.

    Notes
    -----
    - The window and damping envelope have already been folded into f_q
      (F_mod in the main pipeline) before this routine is called.
      The back-transform therefore operates on the windowed F(Q), which
      is consistent: the correction ΔF is subtracted in the same
      windowed space, so no window re-application is needed.
    - G(0) = 0 exactly (no atom pairs at r = 0); this is enforced by
      including r = 0 in the r grid as the first point.
    - Convergence is monitored as the RMS of ΔG(r) in the constrained
      region relative to the peak amplitude of G(r) outside it.

    Parameters
    ----------
    r                : full r grid including r = 0 (Å), shape (N_r,)
    g                : G(r) on r,  shape (N_r,)
    f_q              : F_mod(Q) on q,  shape (N_q,)
    q                : Q grid (Å⁻¹),  shape (N_q,)
    dq               : Q grid spacing (Å⁻¹)
    rho0             : average atomic number density (atoms/Å³)
    r_max_constraint : upper limit of the unphysical r region (Å)
    n_iterations     : maximum number of correction cycles

    Returns
    -------
    r, g, f_q  (updated in-place copies)
    """
    g = g.copy()
    f_q = f_q.copy()

    # Physical baseline  -4πrρ₀  on the full r grid
    g_physical_full = -4.0 * np.pi * r * rho0

    # Boolean mask: constrained (unphysical) region
    mask_low = r <= r_max_constraint
    # Complement: the physical region containing real peaks
    mask_phys = ~mask_low

    r_step_grid = r[1] - r[0]  # uniform spacing (used for display only)

    for _it in range(n_iterations):
        # 1. Deviation from the physical baseline in the constrained region
        delta_g = np.where(mask_low, g - g_physical_full, 0.0)

        # Convergence criterion: RMS(ΔG) / max|G(r)| in the physical region
        rms = float(np.sqrt(np.mean(delta_g[mask_low] ** 2)))
        g_peak = float(np.max(np.abs(g[mask_phys]))) if np.any(mask_phys) else 1.0
        if rms / max(g_peak, 1e-12) < 1e-5:
            break

        # 2. Back-Fourier transform ΔG(r) → ΔF(Q)
        #    Using trapezoidal rule so the r=0 boundary is handled correctly.
        #    Note: delta_g[0] = g[0] - 0 = 0 + 4π·0·ρ₀ = 0, so the r=0
        #    contribution is zero as required.
        delta_f = sine_transform_gr_to_fq(r, delta_g, q)

        # 3. Subtract the unphysical contribution from F(Q)
        f_q -= delta_f

        # 4. Re-transform F(Q) → G(r).
        #    Compute only for r > 0; G(0) = 0 exactly.
        r_pos = r[1:]
        g_pos = sine_transform_fq_to_gr(q, f_q, r_pos, dq)
        g = np.concatenate([[0.0], g_pos])

    return r, g, f_q


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def compute_pdf(xy_path: Path, config: PDFConfig) -> PDFResult:
    """
    Full PDF pipeline: load → correct → normalise → S(Q) → F(Q) → G(r).

    Steps
    -----
    1.  Load raw .xy data; subtract background; polarisation-correct.
    2.  Convert 2θ → Q; spline-resample onto a uniform Q grid.
    3.  Compute Waasmaier–Kirfel form factors and Compton scattering.
    4.  Krogh-Moe / Norman normalisation: determine α and background polynomial.
    5.  Coherent elastic intensity in electron units:  I_eu = (I_meas - bg) / α
    6.  Total structure function:  S(Q) = [I_eu - I_C - (⟨f²⟩ - ⟨f⟩²)] / ⟨f⟩²
    7.  Reduced structure function:  F(Q) = Q[S(Q) - 1]
    8.  Apply termination window W(Q) and Q-resolution damping exp(-½σ_Q²Q²).
    9.  Sine Fourier transform → initial G(r).
    10. Iterative real-space constraint (Toby–Egami) to enforce G(r) = -4πrρ₀
        for r below the first interatomic distance.
    """

    # ------------------------------------------------------------------
    # 1.  Load and correct raw data
    # ------------------------------------------------------------------
    tth, intensity = load_xy(xy_path)

    if config.background_file is not None:
        tth_bg, i_bg = load_xy(config.background_file)
        i_bg_interp = np.interp(tth, tth_bg, i_bg, left=0.0, right=0.0)
        intensity = intensity - config.background_scale * i_bg_interp

    # Intensities must remain non-negative after background subtraction
    intensity = np.maximum(intensity, 0.0)

    if config.polarisation_factor:
        pol = polarisation_correction(tth, config.is_synchrotron, config.polarisation_p)
        intensity = intensity / np.maximum(pol, 1e-12)

    # ------------------------------------------------------------------
    # 2.  Convert 2θ → Q; resample onto uniform Q grid
    # ------------------------------------------------------------------
    q_raw = two_theta_to_q(tth, config.wavelength)
    q_max_data = float(q_raw.max())
    q_min_data = float(q_raw.min())
    q_max_use = min(config.q_max, q_max_data)

    # Q step: use the median measured step (robust to gaps), capped at 0.05 Å⁻¹
    dq = (
        config.q_step
        if config.q_step is not None
        else min(float(np.median(np.diff(q_raw))), 0.05)
    )
    q = np.arange(config.q_min, q_max_use + 0.5 * dq, dq, dtype=np.float64)

    # Spline interpolation; s=0 → exact interpolant, ext=1 → extrapolate to zero
    valid = (intensity > 0) & (q_raw >= q_min_data) & (q_raw <= q_max_data)
    if valid.sum() < 4:
        raise ValueError("Fewer than 4 valid data points; cannot fit a spline.")
    spl = UnivariateSpline(q_raw[valid], intensity[valid], s=0, k=3, ext=1)
    i_q = np.maximum(spl(q), 0.0)

    # Zero out any values outside the measured Q range (avoids extrapolation artefacts)
    i_q[(q < q_min_data) | (q > q_max_data)] = 0.0

    # ------------------------------------------------------------------
    # 3.  Scattering factors on the Q grid
    # ------------------------------------------------------------------
    f_mean, f_mean_sq, f_sq_mean, compton = build_scattering_factors(
        config.composition, q
    )

    # ------------------------------------------------------------------
    # 4.  Krogh-Moe / Norman normalisation
    # ------------------------------------------------------------------
    alpha, background = krogh_moe_normalise(
        q,
        i_q,
        f_sq_mean,
        compton,
        q_min_fit=float(config.norm_q_min),  # type: ignore[arg-type]
        q_max_fit=q_max_use,
        poly_degree=config.norm_poly_degree,
    )

    # ------------------------------------------------------------------
    # 5.  Coherent elastic intensity in electron units
    #     I_eu(Q) = [I_meas(Q) - background(Q)] / α
    # ------------------------------------------------------------------
    i_eu = (i_q - background) / alpha

    # ------------------------------------------------------------------
    # 6.  Total structure function S(Q)
    #
    #     S(Q) = [I_eu(Q) - I_Compton(Q) - (⟨f²⟩ - ⟨f⟩²)] / ⟨f⟩²
    #
    #     The term (⟨f²⟩ - ⟨f⟩²) = Laue monotonic diffuse scattering.
    #     For a single-element sample this is exactly zero.
    #
    #     S(Q) → 1 at large Q; deviations from 1 carry structural information.
    #
    #     Low-Q regime: data below q_min cannot be reliably collected in most
    #     total-scattering experiments (primary beam, low-angle optics).
    #     Setting S(Q) = 1 (i.e. F(Q) = 0) below q_min is the correct null
    #     assumption: it introduces no spurious features in G(r).
    # ------------------------------------------------------------------
    with np.errstate(divide="ignore", invalid="ignore"):
        s_q = (i_eu - compton - (f_sq_mean - f_mean_sq)) / np.maximum(f_mean_sq, 1e-30)

    s_q = np.where(np.isfinite(s_q), s_q, 1.0)

    # Enforce S(Q) = 1 for Q < q_min (no measured data; contributes F=0)
    s_q[q < config.q_min] = 1.0

    # ------------------------------------------------------------------
    # 7.  Reduced structure function  F(Q) = Q[S(Q) - 1]
    #
    #     Boundary conditions:
    #       F(0) = 0  exactly (Q = 0 ⟹ F = 0·[S-1] = 0)
    #       F(Q_max) → 0  via the termination window
    # ------------------------------------------------------------------
    f_q_raw = q * (s_q - 1.0)
    f_q_raw[q == 0.0] = 0.0

    # ------------------------------------------------------------------
    # 8.  Termination window  W(Q)  and Q-resolution damping
    #
    #     F_mod(Q) = F(Q) · W(Q) · exp(-½ σ_Q² Q²)
    #
    #     The Gaussian damping exp(-½ σ_Q² Q²) models finite Q-resolution.
    #     Its effect in real space is to convolve G(r) with a Gaussian
    #     whose width grows as σ_r(r) = σ_Q · r (PDFfit2 convention).
    # ------------------------------------------------------------------
    W = make_termination_window(q, q_max_use, config.termination_window)
    D = np.exp(-0.5 * (config.qdamp * q) ** 2)
    f_mod = f_q_raw * W * D

    # ------------------------------------------------------------------
    # 9.  Sine Fourier transform  F_mod(Q) → G(r)
    #
    #     r grid starts at r_step (not 0) because sin(0) = 0 trivially;
    #     we prepend G(0) = 0 analytically.
    # ------------------------------------------------------------------
    r_pos = np.arange(
        config.r_step,
        config.r_max + 0.5 * config.r_step,
        config.r_step,
        dtype=np.float64,
    )
    g_pos = sine_transform_fq_to_gr(q, f_mod, r_pos, dq)

    r_full = np.concatenate([[0.0], r_pos])
    g_full = np.concatenate([[0.0], g_pos])

    # ------------------------------------------------------------------
    # 10. Real-space constraint  (Toby–Egami method)
    #
    #     Determine the upper limit of the unphysical region.
    #     For crystalline Si: first Si–Si bond = 2.352 Å → use 2.1 Å.
    # ------------------------------------------------------------------
    if config.use_real_space_constraint and config.real_space_constraint_iterations > 0:
        if config.r_constraint_max is not None:
            r_cut = float(config.r_constraint_max)
        else:
            r_cut = _auto_r_constraint(r_full, g_full, config.number_density)

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
        fq=f_q_raw,  # return the un-windowed F(Q) — the scientifically meaningful quantity
        r=r_full,
        gr=g_full,
    )


# ---------------------------------------------------------------------------
# Convenience public API
# ---------------------------------------------------------------------------


def run_pdf(
    xy_path: Path | str,
    composition: dict[str, float],
    wavelength: float,
    number_density: float,
    *,
    q_min: float = 0.5,
    q_max: float = 30.0,
    r_max: float = 30.0,
    r_step: float = 0.01,
    qdamp: float = 0.030,
    termination_window: str = "cosine",
    use_real_space_constraint: bool = True,
    real_space_constraint_iterations: int = 10,
    r_constraint_max: float | None = None,
    norm_poly_degree: int = 3,
    norm_q_min: float | None = None,
    is_synchrotron: bool = False,
    polarisation_p: float = 0.99,
    background_file: Path | str | None = None,
    background_scale: float = 1.0,
    export_formats: list[ExportFormat] | list[str] | None = None,
    output_dir: Path | str = ".",
    output_stem: str = "pdf_output",
) -> PDFResult:
    """
    Compute G(r) from a raw .xy diffraction file and optionally write output files.

    Parameters
    ----------
    xy_path         : path to the input .xy file (2θ in degrees, intensity)
    composition     : chemical composition as {element: stoichiometric count},
                      e.g. {"Si": 1} or {"Pb": 1, "Ti": 1, "O": 3}
    wavelength      : X-ray wavelength in Å
    number_density  : average atomic number density ρ₀ in atoms/Å³
                      (= n_atoms_per_unit_cell / V_unit_cell)

    All other parameters are forwarded to PDFConfig (see its docstring).

    Returns
    -------
    PDFResult with fields: q, iq_corrected, sq, fq, r, gr
    """
    comp_entries = [
        CompositionEntry(element=el, count=float(cnt))
        for el, cnt in composition.items()
    ]
    if export_formats is None:
        export_formats = [ExportFormat.GR]

    cfg = PDFConfig(
        composition=comp_entries,
        wavelength=wavelength,
        number_density=number_density,
        q_min=q_min,
        q_max=q_max,
        r_max=r_max,
        r_step=r_step,
        qdamp=qdamp,
        termination_window=termination_window,
        use_real_space_constraint=use_real_space_constraint,
        real_space_constraint_iterations=real_space_constraint_iterations,
        r_constraint_max=r_constraint_max,
        norm_poly_degree=norm_poly_degree,
        norm_q_min=norm_q_min,
        is_synchrotron=is_synchrotron,
        polarisation_p=polarisation_p,
        background_file=Path(background_file) if background_file else None,
        background_scale=background_scale,
        export_formats=[ExportFormat(f) for f in export_formats],
        output_dir=Path(output_dir),
        output_stem=output_stem,
    )
    result = compute_pdf(Path(xy_path), cfg)

    for fmt in cfg.export_formats:
        if fmt == ExportFormat.GR:
            save_two_column(
                cfg.output_dir / f"{cfg.output_stem}.gr",
                result.r,
                result.gr,
                header=(
                    "G(r) — reduced pair distribution function\n"
                    "# Computed with pdf_calculator.py\n"
                    "# r (Å)   G(r) (Å⁻²)"
                ),
            )
        elif fmt == ExportFormat.SQ:
            save_two_column(
                cfg.output_dir / f"{cfg.output_stem}.sq",
                result.q,
                result.sq,
                header=("S(Q) — total structure function\n# Q (Å⁻¹)   S(Q)"),
            )
        elif fmt == ExportFormat.IQ:
            save_two_column(
                cfg.output_dir / f"{cfg.output_stem}.iq",
                result.q,
                result.iq_corrected,
                header=(
                    "I(Q) — coherent elastic intensity in electron units\n"
                    "# Q (Å⁻¹)   I(Q) (e.u.)"
                ),
            )
        elif fmt == ExportFormat.FQ:
            save_two_column(
                cfg.output_dir / f"{cfg.output_stem}.fq",
                result.q,
                result.fq,
                header=(
                    "F(Q) = Q[S(Q)-1] — reduced structure function\n"
                    "# Q (Å⁻¹)   F(Q) (Å⁻¹)"
                ),
            )
    return result


# ---------------------------------------------------------------------------
# Validation against a reference G(r)
# ---------------------------------------------------------------------------


def validate_against_reference(
    computed: PDFResult,
    ref_path: Path,
    r_min_compare: float = 1.5,
    n_peaks: int = 5,
    tol_position_ang: float = 0.02,
    tol_height_ratio: float = 0.05,
) -> bool:
    """
    Compare the computed G(r) against a reference file by matching the
    dominant peaks by position and relative height.

    Parameters
    ----------
    computed          : PDFResult from compute_pdf / run_pdf
    ref_path          : path to reference two-column G(r) file
    r_min_compare     : minimum r for peak search (Å); exclude the low-r region
    n_peaks           : number of tallest peaks to compare
    tol_position_ang  : position tolerance (Å)
    tol_height_ratio  : relative height tolerance (dimensionless)

    Returns
    -------
    True if all peak positions and relative heights agree within tolerances.
    """
    r_ref, g_ref = load_xy(ref_path)

    def _top_peaks(
        r_arr: FloatArray, g_arr: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        idx = (r_arr > r_min_compare) & (r_arr < r_arr.max() - 1.0)
        peaks, _ = find_peaks(g_arr[idx], height=0.1, distance=10)
        if len(peaks) == 0:
            return np.array([]), np.array([])
        top = np.argsort(g_arr[idx][peaks])[::-1][:n_peaks]
        sel = peaks[top]
        return r_arr[idx][sel], g_arr[idx][sel]

    r_c, h_c = _top_peaks(computed.r, computed.gr)
    r_r, h_r = _top_peaks(r_ref, g_ref)

    if len(r_c) < 3 or len(r_r) < 3:
        print("Insufficient peaks for comparison (need ≥ 3).")
        return False

    # Sort by position for a consistent comparison
    sc, sr = np.argsort(r_c), np.argsort(r_r)
    r_c, h_c = r_c[sc], h_c[sc]
    r_r, h_r = r_r[sr], h_r[sr]

    pos_ok = bool(np.all(np.abs(r_c - r_r) < tol_position_ang))
    h_c_norm = h_c / h_c[0]
    h_r_norm = h_r / h_r[0]
    h_ok = bool(np.all(np.abs(h_c_norm - h_r_norm) < tol_height_ratio))

    print("Peak positions (Å):")
    print(f"  computed  : {np.round(r_c, 3)}")
    print(f"  reference : {np.round(r_r, 3)}")
    print(f"  Δr (Å)   : {np.round(np.abs(r_c - r_r), 4)}")
    print("Relative heights (normalised to first peak):")
    print(f"  computed  : {np.round(h_c_norm, 3)}")
    print(f"  reference : {np.round(h_r_norm, 3)}")
    print(
        f"Position check : {'PASS' if pos_ok else 'FAIL'} "
        f"(tolerance {tol_position_ang} Å)"
    )
    print(
        f"Height check   : {'PASS' if h_ok else 'FAIL'} (tolerance {tol_height_ratio})"
    )
    return pos_ok and h_ok


# ---------------------------------------------------------------------------
# Example usage:  crystalline silicon at I15-1, Diamond Light Source
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    xy_file = Path("/workspaces/xrpd-toolbox/tests/data/Si_pe2_i15_1.xy")
    ref_file = Path("/workspaces/xrpd-toolbox/tests/data/Si_pe2_i15_1.gr")

    # Crystalline Si: 8 atoms per conventional cubic unit cell,
    # a = 5.4309 Å  →  ρ₀ = 8 / 5.4309³ = 0.04996 atoms/Å³
    rho_si = 8.0 / 5.4309**3

    # First Si–Si nearest-neighbour distance = a√3/4 = 2.352 Å.
    # Set the constraint boundary well below this: 2.1 Å is safe.
    result = run_pdf(
        xy_path=xy_file,
        composition={"Si": 1},
        wavelength=0.16,  # I15-1 wavelength (Å), check your logbook
        number_density=rho_si,
        q_min=0.8,
        q_max=24.0,
        r_max=20.0,
        r_step=0.01,
        qdamp=0.003,
        is_synchrotron=True,
        polarisation_p=0.99,
        termination_window="cosine",
        use_real_space_constraint=True,
        real_space_constraint_iterations=10,
        r_constraint_max=2.1,
        norm_poly_degree=3,
        export_formats=["gr", "sq", "fq", "iq"],
        output_dir=".",
        output_stem="Si_pdf",
    )

    if ref_file.exists():
        print("\n--- Validation against reference ---")
        passed = validate_against_reference(result, ref_file)
        print(f"\nOverall: {'PASSED' if passed else 'FAILED'}\n")

    # ---- Diagnostic plot ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("PDF pipeline diagnostics — crystalline Si", fontsize=13)

    ax = axes[0, 0]
    ax.plot(result.q, result.iq_corrected, lw=0.8, color="steelblue")
    ax.set_xlabel("Q (Å⁻¹)")
    ax.set_ylabel("I(Q) (e.u.)")
    ax.set_title("Coherent intensity (electron units)")
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(result.q, result.sq, lw=0.8, color="steelblue")
    ax.axhline(1.0, color="k", lw=0.6, ls="--", label="S(Q) = 1")
    ax.set_xlabel("Q (Å⁻¹)")
    ax.set_ylabel("S(Q)")
    ax.set_title("Total structure function")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(result.q, result.fq, lw=0.8, color="steelblue")
    ax.axhline(0.0, color="k", lw=0.6, ls="--")
    ax.set_xlabel("Q (Å⁻¹)")
    ax.set_ylabel("F(Q) = Q[S(Q)−1] (Å⁻¹)")
    ax.set_title("Reduced structure function")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(result.r, result.gr, lw=0.9, color="steelblue", label="Computed G(r)")
    baseline = -4.0 * np.pi * result.r * rho_si
    ax.plot(result.r, baseline, "k--", lw=0.8, label=r"$-4\pi r\rho_0$", zorder=2)
    if ref_file.exists():
        r_ref, g_ref = load_xy(ref_file)
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
    ax.set_xlim(0, 15)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("Si_pdf_diagnostics.png", dpi=150)
    plt.show()
