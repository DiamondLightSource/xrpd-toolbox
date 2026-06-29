from __future__ import annotations

import math
from enum import StrEnum
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field, field_validator, model_validator
from scipy.interpolate import UnivariateSpline
from scipy.signal import argrelextrema, find_peaks

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
FloatArray = npt.NDArray[np.float64]

# ---------------------------------------------------------------------------
# Waasmaier–Kirfel atomic form factor coefficients (complete table)
# ---------------------------------------------------------------------------
WK_COEFFICIENTS: dict[str, tuple[list[float], list[float], float]] = {
    "H": (
        [0.413048, 0.294953, 0.187491, 0.080701, 0.023736],
        [15.569946, 32.398468, 5.711404, 61.889874, 1.334118],
        0.000049,
    ),
    "He": (
        [0.732354, 0.513796, 0.156638, 0.058372, 0.025765],
        [11.553918, 4.595831, 1.624735, 26.014978, 0.124741],
        0.000000,
    ),
    "Li": (
        [0.974637, 0.158472, 0.811855, 0.262416, 0.790108],
        [4.334946, 0.342451, 97.102966, 201.363831, 1.409234],
        0.002542,
    ),
    "Be": (
        [1.533712, 0.638283, 0.601052, 0.106139, 1.118414],
        [42.662079, 0.595420, 99.106499, 0.151767, 1.843093],
        0.002511,
    ),
    "B": (
        [2.085185, 1.064580, 1.062788, 0.140515, 0.641784],
        [23.494068, 1.137894, 61.238975, 0.114886, 0.399036],
        0.003823,
    ),
    "C": (
        [2.657506, 1.078079, 1.490909, 0.865927, 0.213326],
        [14.780758, 0.776775, 42.086843, 0.239535, 0.000004],
        0.238811,
    ),
    "N": (
        [11.893780, 3.277479, 1.858092, 0.858927, 0.912985],
        [0.000158, 10.232723, 30.344690, 0.656065, 0.217287],
        -11.804902,
    ),
    "O": (
        [2.960427, 2.508818, 0.637853, 0.722838, 1.142756],
        [14.182259, 5.936858, 0.112726, 34.958481, 0.390240],
        0.027014,
    ),
    "F": (
        [3.511943, 2.772244, 0.678385, 0.915159, 1.089261],
        [10.687859, 4.380466, 0.093982, 27.255203, 0.313066],
        0.032557,
    ),
    "Ne": (
        [4.183749, 2.905726, 0.520513, 1.135641, 1.228065],
        [8.175457, 3.252536, 0.063295, 21.813910, 0.224952],
        0.025576,
    ),
    "Na": (
        [4.910127, 3.081816, 1.262067, 1.098938, 0.560991],
        [3.281434, 9.119178, 0.102763, 132.013947, 0.405878],
        0.079712,
    ),
    "Mg": (
        [4.708971, 1.194814, 1.558157, 1.170413, 3.239403],
        [4.875207, 108.506081, 0.111516, 48.292408, 1.928171],
        0.126842,
    ),
    "Al": (
        [4.730796, 2.313951, 1.541980, 1.117564, 3.154754],
        [3.628931, 43.051167, 0.095960, 108.932388, 1.555918],
        0.139509,
    ),
    "Si": (
        [5.275329, 3.191038, 1.511514, 1.356849, 2.519114],
        [2.631338, 33.730728, 0.081119, 86.288642, 1.170087],
        0.145073,
    ),
    "P": (
        [1.950541, 4.146930, 1.494560, 1.522042, 5.729711],
        [0.908139, 27.044952, 0.071280, 67.520187, 1.981173],
        0.155233,
    ),
    "S": (
        [6.372157, 5.154568, 1.473732, 1.635073, 1.209372],
        [1.514347, 22.092527, 0.061373, 55.445176, 0.646925],
        0.154722,
    ),
    "Cl": (
        [1.446071, 6.870609, 6.151801, 1.750347, 0.634168],
        [0.052357, 1.193165, 18.343416, 46.398396, 0.401005],
        0.146773,
    ),
    "Ar": (
        [7.188004, 6.638454, 0.454180, 1.929593, 1.523654],
        [0.956221, 15.339877, 15.339862, 39.043823, 0.062409],
        0.265954,
    ),
    "K": (
        [8.163991, 7.146945, 1.070140, 0.877316, 1.486434],
        [12.816323, 0.808945, 210.327011, 39.597652, 0.052821],
        0.253614,
    ),
    "Ca": (
        [8.593655, 1.477324, 1.436254, 1.182839, 7.113258],
        [10.460644, 0.041891, 81.390381, 169.847839, 0.688098],
        0.196255,
    ),
    "Ti": (
        [9.818524, 1.522646, 1.703101, 1.768774, 7.082555],
        [8.001879, 0.029763, 39.885422, 120.157997, 0.532405],
        0.102473,
    ),
    "V": (
        [10.473575, 1.547881, 1.986381, 1.865616, 7.056250],
        [7.081940, 0.026040, 31.909672, 108.022842, 0.474882],
        0.067744,
    ),
    "Cr": (
        [10.598877, 1.567499, 2.234024, 2.015768, 7.036000],
        [6.323477, 0.021812, 26.204033, 95.526306, 0.427343],
        0.027014,
    ),
    "Mn": (
        [11.281740, 1.591878, 2.468200, 2.207132, 7.034750],
        [5.681989, 0.018955, 21.678535, 89.517914, 0.382661],
        0.027014,
    ),
    "Fe": (
        [11.769545, 7.357573, 3.208969, 2.388290, 1.005285],
        [4.714004, 0.307200, 15.951768, 84.032900, 0.033270],
        0.017148,
    ),
    "Co": (
        [12.284521, 7.340904, 4.003531, 2.349787, 1.006645],
        [4.279139, 0.269530, 13.536359, 71.169850, 0.027000],
        0.016100,
    ),
    "Ni": (
        [12.837788, 7.292040, 4.442437, 2.380560, 1.034032],
        [3.878467, 0.250900, 12.176150, 66.342049, 0.026040],
        0.011800,
    ),
    "Cu": (
        [13.337960, 7.167402, 5.615674, 1.673951, 1.191458],
        [3.583370, 0.247078, 11.396707, 64.812271, 0.001530],
        0.139610,
    ),
    "Zn": (
        [14.074323, 7.031723, 5.162269, 2.410088, 1.304343],
        [3.265688, 0.233350, 10.316071, 58.709356, 0.001029],
        0.128870,
    ),
    "Ge": (
        [15.237342, 6.700755, 4.359001, 2.962069, 1.718136],
        [3.036902, 0.210519, 8.934454, 47.775999, 0.001622],
        0.110780,
    ),
    "Sr": (
        [17.566077, 9.658680, 7.449760, 1.469445, 1.600781],
        [2.769048, 0.202088, 14.099012, 0.116621, 73.013885],
        0.119207,
    ),
    "Y": (
        [17.776040, 10.294938, 5.726292, 3.265587, 1.912487],
        [2.460860, 0.110255, 15.142515, 109.406509, 0.000000],
        0.000000,
    ),
    "Zr": (
        [17.876765, 10.948008, 5.417932, 3.657471, 2.069869],
        [2.418295, 0.103710, 15.802070, 99.353020, 0.001890],
        0.167030,
    ),
    "Ba": (
        [19.967192, 11.692680, 10.650861, 3.234045, 1.555918],
        [0.435606, 4.828548, 23.271650, 0.000101, 51.651099],
        0.021290,
    ),
    "La": (
        [20.578000, 11.372669, 11.823250, 3.286620, 2.000000],
        [0.408853, 4.662201, 23.244834, 0.000000, 47.227188],
        0.000000,
    ),
    "Pb": (
        [31.061739, 13.063700, 18.441900, 5.969600, 2.467900],
        [0.690200, 2.357600, 8.618000, 47.257999, 0.128700],
        0.128700,
    ),
    "Bi": (
        [33.368900, 12.951000, 16.587799, 6.469200, 2.405600],
        [0.704000, 2.923800, 8.793700, 48.009300, 0.091400],
        0.091400,
    ),
}

_ATOMIC_NUMBER: dict[str, int] = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ge": 32,
    "Sr": 38,
    "Y": 39,
    "Zr": 40,
    "Ba": 56,
    "La": 57,
    "Pb": 82,
    "Bi": 83,
}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class ExportFormat(StrEnum):
    GR = "gr"
    SQ = "sq"
    IQ = "iq"
    FQ = "fq"


class CompositionEntry(BaseModel):
    element: str
    count: float = Field(gt=0)

    @field_validator("element")
    @classmethod
    def element_must_be_known(cls, v: str) -> str:
        if v not in WK_COEFFICIENTS:
            raise ValueError(f"Unknown element: {v}")
        return v


class PDFConfig(BaseModel):
    composition: list[CompositionEntry]
    wavelength: float = Field(gt=0)
    number_density: float = Field(gt=0, description="Atomic number density (atoms/Å³)")

    q_min: float = 0.5
    q_max: float = 30.0
    q_step: float | None = None

    polarisation_factor: bool = True
    is_synchrotron: bool = False
    polarisation_p: float = 0.99
    background_file: Path | None = None
    background_scale: float = 1.0

    rpoly_degree: int = 3  # degree for Q‑space background polynomial
    rpoly_q_min: float | None = None  # start of high‑Q fit region (default q_max-10)

    r_min: float = 0.0
    r_max: float = 30.0
    r_step: float = 0.01

    qdamp: float = 0.030
    termination_window: Literal["lorch", "cosine", "none"] = (
        "cosine"  # "lorch", "cosine", or "none"
    )
    use_rpoly_correction: bool = True
    rpoly_correction_degree: int = 2  # degree of low‑r polynomial correction
    rpoly_iterations: int = 2  # number of r‑poly correction cycles

    export_formats: list[ExportFormat] = Field(
        default_factory=lambda: [ExportFormat.GR]
    )
    output_dir: Path = Path(".")
    output_stem: str = "pdf_output"

    @model_validator(mode="after")
    def set_defaults(self) -> PDFConfig:
        if self.rpoly_q_min is None:
            object.__setattr__(self, "rpoly_q_min", max(self.q_min, self.q_max - 10.0))
        return self

    model_config = {"arbitrary_types_allowed": True}


class PDFResult(BaseModel):
    q: FloatArray
    iq_corrected: FloatArray
    sq: FloatArray
    fq: FloatArray
    r: FloatArray
    gr: FloatArray

    model_config = {"arbitrary_types_allowed": True}


# ---------------------------------------------------------------------------
# Physical helper functions
# ---------------------------------------------------------------------------
def atomic_form_factor(element: str, s: float) -> float:
    a_list, b_list, c = WK_COEFFICIENTS[element]
    return c + sum(
        a * math.exp(-b * s * s) for a, b in zip(a_list, b_list, strict=True)
    )


def mean_form_factor(composition: list[CompositionEntry], s: float) -> float:
    total = sum(e.count for e in composition)
    return sum(
        (e.count / total) * atomic_form_factor(e.element, s) for e in composition
    )


def mean_form_factor_sq(composition: list[CompositionEntry], s: float) -> float:
    return mean_form_factor(composition, s) ** 2


def mean_sq_form_factor(composition: list[CompositionEntry], s: float) -> float:
    total = sum(e.count for e in composition)
    return sum(
        (e.count / total) * atomic_form_factor(e.element, s) ** 2 for e in composition
    )


def compton_per_atom(composition: list[CompositionEntry], s: float) -> float:
    total = sum(e.count for e in composition)
    return sum(
        (e.count / total) * _compton_cromer_mann(e.element, s) for e in composition
    )


def _compton_cromer_mann(element: str, s: float) -> float:
    z = _ATOMIC_NUMBER.get(element, 0)
    if z == 0:
        raise ValueError(f"Unknown element: {element}")
    f = atomic_form_factor(element, s)
    return max(0.0, z - f * f / z)


def polarisation_factor(
    two_theta_deg: FloatArray, synchrotron: bool, p: float
) -> FloatArray:
    """Polarisation correction for total scattering (no Lorentz factor)."""
    cos_2t = np.cos(np.deg2rad(two_theta_deg))
    if synchrotron:
        return (1.0 - p) + p * cos_2t**2
    return (1.0 + cos_2t**2) / 2.0


def two_theta_to_q(two_theta_deg: FloatArray, wavelength: float) -> FloatArray:
    return 4.0 * np.pi * np.sin(np.deg2rad(two_theta_deg / 2.0)) / wavelength


def load_xy(path: Path) -> tuple[FloatArray, FloatArray]:
    data = np.loadtxt(path, comments=["#", "!", ";"])
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Invalid .xy file: {path}")
    tth, intensity = data[:, 0], data[:, 1]
    order = np.argsort(tth)
    return tth[order].astype(np.float64), intensity[order].astype(np.float64)


def save_two_column(path: Path, x: FloatArray, y: FloatArray, header: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.column_stack([x, y]), header=header, fmt="%.8e")


# ---------------------------------------------------------------------------
# Form factor arrays
# ---------------------------------------------------------------------------
def build_form_factor_arrays(
    composition: list[CompositionEntry],
    q_arr: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    s_arr = q_arr / (4.0 * np.pi)
    n = len(q_arr)
    f_mean = np.empty(n, dtype=np.float64)
    f_mean_sq = np.empty(n, dtype=np.float64)
    f_sq_mean = np.empty(n, dtype=np.float64)
    compton = np.empty(n, dtype=np.float64)
    for i, s in enumerate(s_arr):
        f_mean[i] = mean_form_factor(composition, s)
        f_mean_sq[i] = f_mean[i] ** 2
        f_sq_mean[i] = mean_sq_form_factor(composition, s)
        compton[i] = compton_per_atom(composition, s)
    return f_mean, f_mean_sq, f_sq_mean, compton


# ---------------------------------------------------------------------------
# Normalisation: fit α + polynomial background over the high‑Q region,
# then iteratively refine α until I_eu matches the self‑scattering.
# ---------------------------------------------------------------------------
def fit_normalization(
    q_arr: FloatArray,
    intensity_q: FloatArray,
    f_sq_mean: FloatArray,
    compton: FloatArray,
    q_min_fit: float,
    q_max_fit: float,
    degree: int,
) -> tuple[float, FloatArray]:
    mask = (q_arr >= q_min_fit) & (q_arr <= q_max_fit)
    if mask.sum() < degree + 3:
        # fallback to whole Q range
        mask = np.ones_like(q_arr, dtype=bool)

    q_fit = q_arr[mask]
    i_fit = intensity_q[mask]
    self_sc = f_sq_mean[mask] + compton[mask]

    # Initial fit: I = α * self_sc + polynomial(Q)
    a = np.column_stack([self_sc] + [q_fit**k for k in range(degree + 1)])
    coeffs, *_ = np.linalg.lstsq(a, i_fit, rcond=None)
    alpha = float(coeffs[0])
    poly_coeffs = coeffs[1:]

    # Iterative refinement: force I_eu to equal self_sc on average in the fit range
    for _ in range(10):
        background = np.zeros_like(q_arr)
        for k, c in enumerate(poly_coeffs):
            background += c * q_arr**k
        i_eu = (intensity_q - background) / alpha
        ratio = np.mean(i_eu[mask] / self_sc)
        alpha *= ratio
        # Refit polynomial with fixed alpha
        res = i_fit - alpha * self_sc
        a_poly = a[:, 1:]  # polynomial columns only
        poly_coeffs, *_ = np.linalg.lstsq(a_poly, res, rcond=None)
        if abs(ratio - 1.0) < 0.001:
            break

    background = np.zeros_like(q_arr)
    for k, c in enumerate(poly_coeffs):
        background += c * q_arr**k
    return alpha, background


# ---------------------------------------------------------------------------
# Termination window + damping
# ---------------------------------------------------------------------------
def apply_termination_window(
    f: FloatArray,
    q: FloatArray,
    q_max: float,
    window_type: str,
    qdamp: float,
) -> FloatArray:
    fw = f.copy()
    if window_type == "lorch":
        fw *= np.sinc(q / q_max)
    elif window_type == "cosine":
        fw *= 0.5 * (1.0 + np.cos(np.pi * q / q_max))
    # "none" does nothing
    if qdamp > 0.0:
        fw *= np.exp(-0.5 * (q * qdamp) ** 2)
    return fw


# ---------------------------------------------------------------------------
# Find first minimum of G(r) after 0.5 Å
# ---------------------------------------------------------------------------
def find_r_cut(r: FloatArray, g: FloatArray) -> float | None:
    idx_start = int(np.searchsorted(r, 0.5))
    if idx_start >= len(r) - 1:
        return None
    region = g[idx_start:]
    minima = argrelextrema(region, np.less)[0]
    if len(minima) == 0:
        return 1.2
    r_cut = float(r[idx_start + minima[0]])
    return max(r_cut, 0.8)


# ---------------------------------------------------------------------------
# Iterative r‑polynomial correction (PDFgetX3 method)
# ---------------------------------------------------------------------------
def apply_rpoly_correction(
    r_full: FloatArray,
    g_full: FloatArray,
    f_mod: FloatArray,
    q_grid: FloatArray,
    dq: float,
    config: PDFConfig,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Iterative low‑r polynomial correction by back‑Fourier transform."""
    for _ in range(config.rpoly_iterations):
        r_cut = find_r_cut(r_full, g_full)
        if r_cut is None:
            break
        mask_low = r_full <= r_cut
        r_fit = r_full[mask_low]
        g_fit = g_full[mask_low]
        # Include r=0 with G=0 as a heavy anchor
        r_fit_ext = np.concatenate([[0.0], r_fit])
        g_fit_ext = np.concatenate([[0.0], g_fit])
        degree = config.rpoly_correction_degree
        weights = np.ones_like(r_fit_ext)
        weights[0] = 10.0
        coeffs = np.polyfit(r_fit_ext, g_fit_ext, degree, w=weights)
        g_poly = np.polyval(coeffs, r_fit_ext)

        # Back‑transform: ΔF(Q) = ∫₀^{r_cut} r·G_poly(r)·sin(Qr) dr
        integrand = r_fit_ext * g_poly
        sin_qr = np.sin(np.outer(r_fit_ext, q_grid))
        delta_f = np.trapezoid(integrand[:, None] * sin_qr, r_fit_ext, axis=0)

        f_mod = f_mod - delta_f

        # Recompute G(r) from corrected F(Q)
        r_grid = r_full[1:]  # exclude r=0
        sin_qr = np.sin(np.outer(r_grid, q_grid))
        g_r = (2.0 / np.pi) * (sin_qr @ f_mod * dq)
        g_full = np.concatenate([[0.0], g_r])

    return r_full, g_full, f_mod


# ---------------------------------------------------------------------------
# Main PDF pipeline
# ---------------------------------------------------------------------------
def compute_pdf(xy_path: Path, config: PDFConfig) -> PDFResult:
    # 1. Load and correct data
    tth, intensity = load_xy(xy_path)
    if config.background_file is not None:
        tth_bg, intensity_bg = load_xy(config.background_file)
        intensity_bg_interp = np.interp(tth, tth_bg, intensity_bg, left=0.0, right=0.0)
        intensity -= config.background_scale * intensity_bg_interp
    intensity = np.maximum(intensity, 0.0)

    if config.polarisation_factor:
        pol = polarisation_factor(tth, config.is_synchrotron, config.polarisation_p)
        intensity = intensity / np.maximum(pol, 1e-12)

    # 2. Convert to Q and resample
    q_raw = two_theta_to_q(tth, config.wavelength)
    q_max_data = float(q_raw.max())
    q_max_use = min(config.q_max, q_max_data)
    dq = config.q_step or min(float(np.median(np.diff(q_raw))), 0.05)
    q_grid = np.arange(config.q_min, q_max_use + 0.5 * dq, dq, dtype=np.float64)

    valid = (intensity > 0) & (q_raw >= q_raw.min()) & (q_raw <= q_raw.max())
    if valid.sum() < 3:
        raise ValueError("Not enough valid data after corrections.")
    spline = UnivariateSpline(q_raw[valid], intensity[valid], s=0, k=3, ext=1)
    intensity_q = spline(q_grid)
    intensity_q = np.maximum(intensity_q, 0.0)
    intensity_q[(q_grid < q_raw.min()) | (q_grid > q_raw.max())] = 0.0

    # 3. Form factors & Compton
    f_mean, f_mean_sq, f_sq_mean, compton = build_form_factor_arrays(
        config.composition, q_grid
    )

    # 4. Normalisation (high‑Q fit only, with iterative α refinement)
    q_fit_min = config.rpoly_q_min
    q_fit_max = q_max_use
    alpha, background = fit_normalization(
        q_grid,
        intensity_q,
        f_sq_mean,
        compton,
        q_fit_min,
        q_fit_max,
        config.rpoly_degree,
    )

    # 5. Coherent intensity in electron units
    i_eu = (intensity_q - background) / alpha

    # 6. S(Q)
    with np.errstate(divide="ignore", invalid="ignore"):
        s_q = (i_eu - compton - (f_sq_mean - f_mean_sq)) / np.maximum(f_mean_sq, 1e-30)
    s_q = np.where(np.isfinite(s_q), s_q, 0.0)

    # 7. F(Q) = Q * (S(Q) - 1)
    f_q = q_grid * (s_q - 1.0)
    f_q[0] = 0.0

    # 8. Termination window + q‑damping
    f_mod = apply_termination_window(
        f_q,
        q_grid,
        q_max_use,
        config.termination_window,
        config.qdamp,
    )

    # 9. Sine Fourier transform → initial G(r)
    r_grid = np.arange(config.r_step, config.r_max + 0.5 * config.r_step, config.r_step)
    sin_qr = np.sin(np.outer(r_grid, q_grid))
    g_r = (2.0 / np.pi) * (sin_qr @ f_mod * dq)

    r_full = np.concatenate([[0.0], r_grid])
    g_full = np.concatenate([[0.0], g_r])

    # 10. Optional iterative r‑polynomial correction
    if config.use_rpoly_correction and config.rpoly_iterations > 0:
        r_full, g_full, f_mod = apply_rpoly_correction(
            r_full, g_full, f_mod, q_grid, dq, config
        )

    # 11. Very gentle low‑r damping (r < 0.05 Å) to suppress any remaining spike
    idx = (r_full < 0.05) & (r_full > 0)
    if np.any(idx):
        t = r_full[idx] / 0.05
        g_full[idx] *= 0.5 * (1.0 - np.cos(np.pi * t))

    return PDFResult(q=q_grid, iq_corrected=i_eu, sq=s_q, fq=f_q, r=r_full, gr=g_full)


# ---------------------------------------------------------------------------
# Convenience runner – all important parameters are exposed
# ---------------------------------------------------------------------------
def run_pdf(
    xy_path: Path | str,
    composition: dict[str, float],
    wavelength: float,
    number_density: float,  # atoms per Å³ (required)
    *,
    q_min: float = 0.5,
    q_max: float = 30.0,
    r_max: float = 30.0,
    r_step: float = 0.01,
    qdamp: float = 0.030,
    termination_window: str = "cosine",  # "lorch", "cosine", or "none"
    use_rpoly_correction: bool = True,
    rpoly_correction_degree: int = 2,
    rpoly_iterations: int = 2,
    rpoly_degree: int = 3,  # Q‑space background polynomial degree
    rpoly_q_min: float | None = None,  # start of high‑Q fit (default q_max-10)
    is_synchrotron: bool = False,
    polarisation_p: float = 0.99,
    background_file: Path | str | None = None,
    export_formats: list[ExportFormat] | None = None,
    output_dir: Path | str = ".",
    output_stem: str = "pdf_output",
) -> PDFResult:
    comp_entries = [
        CompositionEntry(element=el, count=cnt) for el, cnt in composition.items()
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
        use_rpoly_correction=use_rpoly_correction,
        rpoly_correction_degree=rpoly_correction_degree,
        rpoly_iterations=rpoly_iterations,
        rpoly_degree=rpoly_degree,
        rpoly_q_min=rpoly_q_min,
        is_synchrotron=is_synchrotron,
        polarisation_p=polarisation_p,
        background_file=Path(background_file) if background_file else None,
        export_formats=export_formats,
        output_dir=Path(output_dir),
        output_stem=output_stem,
    )
    result = compute_pdf(Path(xy_path), cfg)

    for fmt in export_formats:
        if fmt == ExportFormat.GR:
            save_two_column(
                cfg.output_dir / f"{cfg.output_stem}.gr",
                result.r,
                result.gr,
                header="G(r) (Å⁻²)",
            )
        elif fmt == ExportFormat.SQ:
            save_two_column(
                cfg.output_dir / f"{cfg.output_stem}.sq",
                result.q,
                result.sq,
                header="S(Q)",
            )
        elif fmt == ExportFormat.IQ:
            save_two_column(
                cfg.output_dir / f"{cfg.output_stem}.iq",
                result.q,
                result.iq_corrected,
                header="I_corr(Q) (e.u.)",
            )
        elif fmt == ExportFormat.FQ:
            save_two_column(
                cfg.output_dir / f"{cfg.output_stem}.fq",
                result.q,
                result.fq,
                header="F(Q) (Å⁻¹)",
            )
    return result


# ---------------------------------------------------------------------------
# Test against reference PDF
# ---------------------------------------------------------------------------
def test_against_reference(
    computed: PDFResult,
    ref_path: Path,
    tol_position: float = 0.02,
    tol_height_ratio: float = 0.05,
) -> bool:
    r_ref, g_ref = load_xy(ref_path)
    idx = (computed.r > 1.5) & (computed.r < computed.r.max() - 1.0)
    peaks_comp, _ = find_peaks(computed.gr[idx], height=0.5, distance=15)
    peaks_comp = peaks_comp[np.argsort(computed.gr[peaks_comp])[::-1]][:5]
    r_comp = computed.r[peaks_comp]
    h_comp = computed.gr[peaks_comp]

    idx_ref = (r_ref > 1.5) & (r_ref < r_ref.max() - 1.0)
    peaks_ref, _ = find_peaks(g_ref[idx_ref], height=0.1, distance=15)
    peaks_ref = peaks_ref[np.argsort(g_ref[peaks_ref])[::-1]][:5]
    r_ref_peaks = r_ref[peaks_ref]
    h_ref_peaks = g_ref[peaks_ref]

    if len(r_comp) < 3 or len(r_ref_peaks) < 3:
        print("Not enough peaks for comparison.")
        return False

    sort_comp = np.argsort(r_comp)
    sort_ref = np.argsort(r_ref_peaks)
    r_comp_sorted = r_comp[sort_comp]
    r_ref_sorted = r_ref_peaks[sort_ref]
    h_comp_sorted = h_comp[sort_comp]
    h_ref_sorted = h_ref_peaks[sort_ref]

    pos_ok = bool(np.all(np.abs(r_comp_sorted - r_ref_sorted) < tol_position))
    h_comp_norm = h_comp_sorted / h_comp_sorted[0]
    h_ref_norm = h_ref_sorted / h_ref_sorted[0]
    h_ok = bool(np.all(np.abs(h_comp_norm - h_ref_norm) < tol_height_ratio))

    print(
        f"Peak positions (sorted): computed {r_comp_sorted}, reference {r_ref_sorted}"
    )
    print(f"Relative heights: computed {h_comp_norm}, reference {h_ref_norm}")
    print(f"Position OK: {pos_ok}, Height ratio OK: {h_ok}")
    return pos_ok and h_ok


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    xy_file = "/workspaces/xrpd-toolbox/tests/data/Si_pe2_i15_1.xy"
    ref_file = Path("/workspaces/xrpd-toolbox/tests/data/Si_pe2_i15_1.gr")

    # Crystalline Si density: 2.33 g/cm³ → 0.04996 atoms/Å³
    rho_si = 0.04996

    result = run_pdf(
        xy_path=xy_file,
        composition={"Si": 1},
        wavelength=0.16,
        number_density=rho_si,
        q_min=0.5,
        q_max=24.0,
        r_max=20.0,
        is_synchrotron=True,
        termination_window="cosine",
        use_rpoly_correction=True,
        rpoly_correction_degree=2,
        rpoly_iterations=2,
        rpoly_degree=3,
        export_formats=["gr", "sq", "fq", "iq"],
        output_dir=".",
        output_stem="Si_pdf",
    )

    if ref_file.exists():
        ok = test_against_reference(result, ref_file)
        print(f"Test {'PASSED' if ok else 'FAILED'}")
    else:
        print("Reference file not found; skipping test.")

    plt.plot(result.r, result.gr, label="Computed")
    if ref_file.exists():
        r_ref, g_ref = load_xy(ref_file)
        plt.plot(r_ref, g_ref, "--", label="Reference")
    plt.xlabel("r (Å)")
    plt.ylabel("G(r) (Å⁻²)")
    plt.xlim(0, 15)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
