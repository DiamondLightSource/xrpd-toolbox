from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel

I15_1_PDFCURL_ENDPOINT = "https://i15-1-pdfget.diamond.ac.uk/data2pdf"

OUTPUT_TYPES = Literal["iq", "sq", "fq", "gr"]


class PDFGetConfig(BaseModel):
    composition: str
    wavelength: float
    dataformat: Literal["QA", "Qnm", "twotheta"] = "twotheta"
    backgroundfile: str | Path | None = None
    outputtypes: OUTPUT_TYPES | list[OUTPUT_TYPES] = ["iq", "sq", "fq", "gr"]
    qmaxinst: float = 20.0
    qmin: float = 0.0
    qmax: float = 50.0
    rmin: float = 0.0
    rmax: float = 20.0
    rstep: float = 0.01
    rpoly: float = 0.9


class PDFCurlRequest(PDFGetConfig):
    name: str
    input_filepath: str | Path
    output_filepath: list[str | Path] | None = None
    publish: bool = False


def send_post_request(url: str, data: dict) -> httpx.Response:
    """Send a JSON payload to a URL via HTTP POST."""
    response = httpx.post(url, json=data)
    response.raise_for_status()
    return response


def send_xy_to_pdfcurl(
    xy_filepath: str | Path,
    composition: str,
    wavelength: float,
    background_file: str | Path | None = None,
    output_filepath: list[str | Path] | None = None,
    publish: bool = False,
    dataformat: Literal["QA", "Qnm", "twotheta"] = "twotheta",
    outputtypes: OUTPUT_TYPES | list[OUTPUT_TYPES] | None = None,
    qmaxinst: float = 20.0,
    qmin: float = 0.0,
    qmax: float = 50.0,
    rmin: float = 0.0,
    rmax: float = 20.0,
    rstep: float = 0.01,
    rpoly: float = 0.9,
) -> dict:
    """Send an .xy file to pdfcurl. Unspecified options use the pdfcurl defaults."""

    name = Path(xy_filepath).stem

    if outputtypes is None:
        outputtypes = ["iq", "sq", "fq", "gr"]

    pdfcurl_config = PDFCurlRequest(
        name=name,
        input_filepath=xy_filepath,
        output_filepath=output_filepath,
        publish=publish,
        composition=composition,
        wavelength=wavelength,
        dataformat=dataformat,
        backgroundfile=background_file,
        outputtypes=outputtypes,
        qmaxinst=qmaxinst,
        qmin=qmin,
        qmax=qmax,
        rmin=rmin,
        rmax=rmax,
        rstep=rstep,
        rpoly=rpoly,
    )

    response = send_post_request(
        url=I15_1_PDFCURL_ENDPOINT, data=pdfcurl_config.model_dump(mode="json")
    )

    return response.json()
