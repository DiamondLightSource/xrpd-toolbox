import math

import numpy as np
from pydantic import BaseModel

from xrpd_toolbox.core import XYEData
from xrpd_toolbox.utils.chemical_formula import ChemicalFormula
from xrpd_toolbox.utils.pdfcurl import (
    I15_1_PDFCURL_ENDPOINT,
    EncodedPDFCurlRequest,
    PDFCurlResponse,
    send_post_request,
)


def determine_gr_header(filepath: str):

    with open(filepath) as f:
        lines = f.readlines()

    header = None
    for i, line in enumerate(lines):
        if "# start data" in line:
            header = i + 2
            break

    if header is None:
        raise ValueError(f"Could not find '#L' data header line in {filepath}")

    return header


def _load_gr(pdf_data: XYEData | str) -> XYEData:
    if isinstance(pdf_data, str):
        return XYEData.from_csv(pdf_data, skip_header=determine_gr_header(pdf_data))
    return pdf_data


def gr_background(r: np.ndarray, rho: float):
    """Low-r baseline of the reduced PDF G(r) = 4*pi*r*rho*(g(r) - 1).

    Below the first peak g(r) = 0, so G(r) = -4*pi*rho*r (linear in r).
    """

    background = -4 * math.pi * r * rho
    return background


def radial_distribution(r: np.ndarray, gr: np.ndarray, rho: float) -> np.ndarray:
    """R(r) = r*G(r) + 4*pi*r^2*rho. Integrating R(r) over a peak gives a
    coordination number."""

    return r * gr + 4 * math.pi * (r**2) * rho


BACKGROUND_TOL = 0.1
POSITION_TOL = 0.1
COORDINATION_TOL = 0.5


class GrValidation(BaseModel):
    background_residual: float
    background_ok: bool
    peak_position: float
    expected_peak_position: float
    peak_position_ok: bool
    coordination_number: float
    expected_coordination_number: float | None
    coordination_ok: bool

    @property
    def passed(self) -> bool:
        return self.background_ok and self.peak_position_ok and self.coordination_ok


def validate_gr(
    composition: str,
    density_g_cm3: float,
    pdf_data: XYEData | str,
    expected_peak_position: float | None = None,
    expected_coordination_number: float | None = None,
    background_tol: float = BACKGROUND_TOL,
    position_tol: float = POSITION_TOL,
    coordination_tol: float = COORDINATION_TOL,
) -> GrValidation:
    """Check G(r) against the expected low-r baseline and first peak.

    expected_peak_position defaults to an estimate from ionic radii.
    The coordination check is skipped if expected_coordination_number is None.
    """

    xye_data = _load_gr(pdf_data)
    r = np.asarray(xye_data.x)
    gr = np.asarray(xye_data.y)

    chem = ChemicalFormula(formula=composition)
    rho = chem.rho(density_g_cm3=density_g_cm3)

    if expected_peak_position is None:
        expected_peak_position = float(chem.estimated_min_distance)

    # first peak: the maximum within 30% of the expected position
    window = (r > 0.7 * expected_peak_position) & (r < 1.3 * expected_peak_position)
    if not np.any(window):
        raise ValueError(f"No G(r) points near r={expected_peak_position}")
    peak_r = float(r[window][np.argmax(gr[window])])

    # peak edges: the minima of g(r) - 1 = G / (4 pi rho r) either side of the peak
    left = (r > 0.75 * peak_r) & (r < peak_r)
    right = (r > peak_r) & (r < 1.6 * peak_r)
    if not np.any(left) or not np.any(right):
        raise ValueError("G(r) does not extend either side of the first peak")
    r_low = r[left][np.argmin(gr[left] / r[left])]
    r_high = r[right][np.argmin(gr[right] / r[right])]

    # coordination number: integral of R(r) across the first peak
    peak = (r >= r_low) & (r <= r_high)
    coordination = float(
        np.trapezoid(radial_distribution(r[peak], gr[peak], rho), r[peak])
    )

    # background: RMS deviation from -4*pi*rho*r below the first peak,
    # relative to the size of the baseline over that range
    below_peak = (r > 0) & (r < r_low)
    if not np.any(below_peak):
        raise ValueError(f"No G(r) points below the first peak at r={r_low}")
    background = gr_background(r[below_peak], rho)
    rms = np.sqrt(np.mean((gr[below_peak] - background) ** 2))
    background_residual = float(rms / np.mean(np.abs(background)))

    coordination_ok = expected_coordination_number is None or (
        abs(coordination - expected_coordination_number) <= coordination_tol
    )

    return GrValidation(
        background_residual=background_residual,
        background_ok=background_residual <= background_tol,
        peak_position=peak_r,
        expected_peak_position=expected_peak_position,
        peak_position_ok=abs(peak_r - expected_peak_position) <= position_tol,
        coordination_number=coordination,
        expected_coordination_number=expected_coordination_number,
        coordination_ok=coordination_ok,
    )


class RefinementResult(BaseModel):
    request: EncodedPDFCurlRequest
    gr: XYEData | None = None
    validation: GrValidation | None = None
    score: float = math.inf
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.validation is not None and self.validation.passed


def score_validation(validation: GrValidation) -> float:
    """Sum of each error relative to its tolerance, lower is better.
    All checks pass when every term is <= 1."""

    score = validation.background_residual / BACKGROUND_TOL
    score += (
        abs(validation.peak_position - validation.expected_peak_position) / POSITION_TOL
    )
    if validation.expected_coordination_number is not None:
        score += (
            abs(
                validation.coordination_number - validation.expected_coordination_number
            )
            / COORDINATION_TOL
        )
    return score


def try_pdf_request(
    request: EncodedPDFCurlRequest,
    density_g_cm3: float,
    expected_peak_position: float,
    expected_coordination_number: float | None,
    url: str,
) -> RefinementResult:
    """Send one request to pdfcurl and validate the G(r) it returns.
    Errors are recorded rather than raised so a bad parameter set doesn't
    stop the refinement."""

    try:
        response = send_post_request(url=url, data=request.model_dump(mode="json"))
        pdf_result = PDFCurlResponse.model_validate(response)
        if pdf_result.error or pdf_result.result is None:
            raise ValueError(pdf_result.error or "pdfcurl returned no result")
        gr = pdf_result.result.gr
        if not isinstance(gr, XYEData):
            raise ValueError("pdfcurl did not return G(r) data")

        validation = validate_gr(
            composition=request.composition,
            density_g_cm3=density_g_cm3,
            pdf_data=gr,
            expected_peak_position=expected_peak_position,
            expected_coordination_number=expected_coordination_number,
        )
    except Exception as e:
        return RefinementResult(request=request, error=str(e))

    return RefinementResult(
        request=request,
        gr=gr,
        validation=validation,
        score=score_validation(validation),
    )


def default_search_values(
    request: EncodedPDFCurlRequest, expected_peak_position: float
) -> dict[str, list[float]]:
    """Candidate values for every float parameter in PDFGetConfig except
    wavelength, which is fixed by the measurement. Listed in the order they
    are refined, most influential first."""

    def around(value: float, factors: list[float]) -> list[float]:
        return [round(value * f, 3) for f in factors]

    return {
        # must stay below the first peak or the polynomial correction eats it
        "rpoly": np.arange(0.5, 0.95 * expected_peak_position, 0.1).round(2).tolist(),
        # lower qmax reduces termination ripples but broadens the peaks
        "qmax": around(request.qmax, [0.6, 0.7, 0.8, 0.9, 1.0]),
        "qmaxinst": around(request.qmaxinst, [0.6, 0.7, 0.8, 0.9, 1.0]),
        "qmin": around(request.qmin, [0.5, 0.75, 1.0, 1.5, 2.0]),
        # rmin, rmax and rstep only set the output r-grid, but the validation
        # needs points below the first peak and either side of it
        "rmin": [0.0, round(0.25 * expected_peak_position, 2)],
        "rmax": around(request.rmax, [0.5, 1.0, 1.5]),
        "rstep": [0.01, 0.02, 0.05],
    }


def refine_pdf(
    request: EncodedPDFCurlRequest,
    density_g_cm3: float,
    expected_peak_position: float | None = None,
    expected_coordination_number: float | None = None,
    search_values: dict[str, list[float]] | None = None,
    fixed: list[str] | None = None,
    url: str = I15_1_PDFCURL_ENDPOINT,
) -> RefinementResult:
    """Vary the float parameters of PDFGetConfig until the G(r) from pdfcurl
    passes validate_gr.

    Each parameter in search_values is scanned in turn, with the others held
    at their best value so far. Stops as soon as validation passes, otherwise
    returns the best scoring result.

    Parameters named in fixed (e.g. ["rmin", "rmax", "rstep"]) keep the value
    they have in request.
    """

    if expected_peak_position is None:
        chem = ChemicalFormula(formula=request.composition)
        expected_peak_position = float(chem.estimated_min_distance)

    if search_values is None:
        search_values = default_search_values(request, expected_peak_position)

    best = try_pdf_request(
        request,
        density_g_cm3,
        expected_peak_position,
        expected_coordination_number,
        url,
    )

    for name, values in search_values.items():
        if fixed is not None and name in fixed:
            continue

        for value in values:
            if best.passed:
                return best

            candidate = best.request.model_copy(update={name: value})
            result = try_pdf_request(
                candidate,
                density_g_cm3,
                expected_peak_position,
                expected_coordination_number,
                url,
            )
            print(f"{name}={value}: score={result.score:.3f} {result.error or ''}")

            if result.score < best.score:
                best = result

    return best


if __name__ == "__main__":
    from pathlib import Path

    from xrpd_toolbox.i15_1.eiger_500k import EigerDataLoader
    from xrpd_toolbox.plotting import DataPlot

    eiger_data = EigerDataLoader("/workspaces/outputs/i15-1/i15-1-100603.nxs")

    composition = eiger_data.get_composition()
    density_g_cm3 = eiger_data.get_density()
    wavelength = eiger_data.get_wavelength()

    xy_path = (
        "/workspaces/outputs/i15-1/processed/i15-1-100603/i15-1-100603_fastcs_eiger.xy"
    )

    xye_data = XYEData.from_csv(xy_path)

    pdf_req = EncodedPDFCurlRequest(
        composition=composition,
        wavelength=wavelength,
        name=str(Path(xy_path).stem),
        input_data=xye_data,
        backgroundfile="/dls/i15-1/data/2026/cm44163-4/processed/i15-1-100599/i15-1-100599_fastcs_eiger.xy",
        rmax=30,
        qmax=44,
        qmaxinst=40,
        rpoly=1,
        rmin=1,
        publish=False,
    )

    response = send_post_request(
        url=I15_1_PDFCURL_ENDPOINT, data=pdf_req.model_dump(mode="json")
    )
    pdf_result = PDFCurlResponse.model_validate(response)

    # # K2PdCl6: PdCl6 octahedra, 6 Cl around each Pd at ~2.34 A
    # pdf_result = refine_pdf(
    #     request=pdf_req,
    #     density_g_cm3=density_g_cm3,
    #     expected_peak_position=2.36,
    #     expected_coordination_number=6,
    #     fixed=["rmin", "rmax", "rstep", "rpoly"],
    # )

    # print("passed:", pdf_result.passed)
    # print("best params:", pdf_result.request.model_dump(exclude={"input_data"}))
    # print(result.validation)

    assert pdf_result.result is not None
    assert isinstance(pdf_result.result.gr, XYEData)

    DataPlot(**pdf_result.result.gr.model_dump(), x_label="r(A)", y_label="G(r)").plot()
