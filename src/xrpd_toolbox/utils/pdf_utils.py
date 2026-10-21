import math

import numpy as np

from xrpd_toolbox.core import XYEData
from xrpd_toolbox.utils.chemical_formula import ChemicalFormula
from xrpd_toolbox.utils.pdfcurl import EncodedPDFCurlRequest


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


def gr_background(r: np.ndarray, rho: float):

    background = -4 * math.pi * (r**2) * rho
    return background


def validate_gr_density(
    composition: str,
    density_g_cm3: float,
    pdf_data: XYEData | str,
    tol: float = 0.05,
) -> bool:

    if isinstance(pdf_data, str):
        xye_data = XYEData.from_csv(pdf_data, skip_header=determine_gr_header(pdf_data))
    else:
        xye_data = pdf_data

    chem = ChemicalFormula(formula=composition)
    first_peak = chem.estimated_min_distance

    mask = xye_data.x < (first_peak + 0.5)

    gr_x = xye_data.x[mask]
    gr_y = xye_data.y[mask]

    background = gr_background(r=gr_x, rho=chem.rho(density_g_cm3=density_g_cm3))

    diff = gr_y - background

    if np.abs(np.amax(diff)) < tol:
        return True
    else:
        return False


def validate_first_peak_integral(
    composition: str,
    expected_nn: float,
    pdf_data: XYEData | str,
    threshold: float = 0.1,
) -> bool:

    if isinstance(pdf_data, str):
        xye_data = XYEData.from_csv(pdf_data, skip_header=determine_gr_header(pdf_data))
    else:
        xye_data = pdf_data

    print(xye_data)

    chem = ChemicalFormula(formula=composition)

    first_peak = chem.estimated_min_distance

    print(first_peak)

    calc_integral = 3  # dummy number

    if abs(expected_nn - calc_integral) < threshold:
        return True
    else:
        return False


def validate_gr(composition: str, pdf_data: XYEData | str):

    if isinstance(pdf_data, str):
        xye_data = XYEData.from_csv(pdf_data, skip_header=determine_gr_header(pdf_data))
    else:
        xye_data = pdf_data

    density_correct = validate_gr_density(
        composition=composition, density_g_cm3=density_g_cm3, pdf_data=xye_data
    )

    print(density_correct)


if __name__ == "__main__":
    from pathlib import Path

    from xrpd_toolbox.i15_1.eiger_500k import EigerDataLoader
    from xrpd_toolbox.plotting import DataPlot
    from xrpd_toolbox.utils.pdfcurl import (
        EncodedPDFCurlRequest,
        PDFCurlResponse,
        send_post_request,
    )

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
        rmax=30,
        qmax=50,
        qmaxinst=20,
    )

    response = send_post_request(
        url="https://i15-1-pdfget.diamond.ac.uk/data2pdf", data=pdf_req.model_dump()
    )

    print(composition)

    # print(ChemicalFormula(formula="Si").estimated_min_distance)

    chem = ChemicalFormula(formula=composition)

    pairs = chem.get_element_pair_distances()

    print(pairs)

    print(chem.estimated_min_distance)

    pdf_result = PDFCurlResponse.model_validate(response)

    assert pdf_result.result is not None

    gr = pdf_result.result.gr

    assert isinstance(gr, XYEData)

    dataplot = DataPlot(**gr.model_dump(), x_label="r(A)", y_label="Intensity")

    dataplot.plot()

    density_correct = validate_gr_density(
        composition=composition, density_g_cm3=density_g_cm3, pdf_data=gr
    )

    print(density_correct)
