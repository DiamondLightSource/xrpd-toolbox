import math

import numpy as np

from xrpd_toolbox.core import XYEData
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
        rmax=35,
        qmax=40,
        qmaxinst=40,
        rpoly=0.9,
        rmin=1,
        publish=False,
        terminationfunctions="lorch",
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
