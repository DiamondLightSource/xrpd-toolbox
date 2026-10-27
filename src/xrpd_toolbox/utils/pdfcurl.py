from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

import httpx
from pydantic import BaseModel, Field

from xrpd_toolbox.core import XYEData

I15_1_PDFCURL_ENDPOINT = "https://i15-1-pdfget.diamond.ac.uk/data2pdf"
TERMINATION_TYPES = Literal["lorch", "step"]

OUTPUT_TYPES = Literal["iq", "sq", "fq", "gr"]


class PDFGetConfig(BaseModel):
    composition: str
    wavelength: float
    dataformat: Literal["QA", "Qnm", "twotheta"] = "twotheta"
    backgroundfile: str | Path | XYEData | None = None
    outputtypes: OUTPUT_TYPES | list[OUTPUT_TYPES] = ["iq", "sq", "fq", "gr"]
    qmaxinst: float = 20.0
    qmin: float = 0.0
    qmax: float = 50.0
    rmin: float = 0.0
    rmax: float = 20.0
    rstep: float = 0.01
    rpoly: float = 0.9
    terminationfunctions: TERMINATION_TYPES | list[TERMINATION_TYPES] | None = None
    twothetazero: float = 0
    bgscale: float = 1.0
    mode: Literal["xray"] = "xray"  #'xray', 'neutron', 'electron', 'sas', 'xdiff'
    force: bool = True


class PDFCurlRequest(PDFGetConfig):
    name: str
    input_filepath: str | Path
    output_filepath: list[str | Path] | None = None
    publish: bool = False
    return_serialised: bool = False


class EncodedPDFCurlRequest(PDFGetConfig):
    name: str
    input_data: XYEData
    publish: bool = False


class PDFCurlResult(BaseModel):
    gr: str | XYEData | None = None
    sq: str | XYEData | None = None
    fq: str | XYEData | None = None
    iq: str | XYEData | None = None


class PDFCurlResponse(BaseModel):
    name: str
    result: PDFCurlResult | None = None
    complete: bool = False
    error: str | None = None
    id: UUID | None = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


def send_post_request(url: str, data: dict) -> dict:
    """Send a JSON payload to a URL via HTTP POST and return the JSON response."""
    response = httpx.post(url, json=data)
    response.raise_for_status()
    return response.json()


def send_xy_to_pdfcurl(
    xy_filepath: str | Path,
    composition: str,
    wavelength: float,
    background_file: str | Path | None = None,
    output_filepath: list[str | Path] | None = None,
    publish: bool = False,
    dataformat: Literal["QA", "Qnm", "twotheta"] = "twotheta",
    outputtypes: OUTPUT_TYPES | list[OUTPUT_TYPES] | None = None,
    qmaxinst: float = 50.0,
    qmin: float = 0.5,
    qmax: float = 50.0,
    rmin: float = 0.0,
    rmax: float = 20.0,
    rstep: float = 0.02,
    rpoly: float = 0.9,
    terminationfunctions: TERMINATION_TYPES | None = None,
    twothetazero: float = 0,
    bgscale: float = 1.0,
    mode: Literal["xray"] = "xray",  #'xray', 'neutron', 'electron', 'sas', 'xdiff'
    force: bool = True,
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
        terminationfunctions=terminationfunctions,
        twothetazero=twothetazero,
        bgscale=bgscale,
        mode=mode,
        force=force,
    )

    response = send_post_request(
        url=I15_1_PDFCURL_ENDPOINT, data=pdfcurl_config.model_dump(mode="json")
    )

    return response
