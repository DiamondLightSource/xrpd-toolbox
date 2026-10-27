import json
import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from xrpd_toolbox.core import XYEData
from xrpd_toolbox.utils.chemical_formula import ChemicalFormula
from xrpd_toolbox.utils.pdf_utils import (
    _load_gr,
    determine_gr_header,
    gr_background,
    radial_distribution,
)
from xrpd_toolbox.utils.pdfcurl import (
    EncodedPDFCurlRequest,
    PDFCurlResponse,
    send_post_request,
    send_xy_to_pdfcurl,
)

COMPOSITION = "K2PdCl6"
DENSITY = 2.7


def synthetic_gr(
    peak_r: float, coordination: float, baseline_scale: float = 1.0
) -> XYEData:
    """G(r) with a single peak whose R(r) integral is coordination."""

    chem = ChemicalFormula(formula=COMPOSITION)
    rho = chem.rho(density_g_cm3=DENSITY)
    r = np.arange(0.01, 4.0, 0.01)
    area = coordination

    sigma = 0.05
    rdf = (
        area
        * np.exp(-((r - peak_r) ** 2) / (2 * sigma**2))
        / (sigma * math.sqrt(2 * math.pi))
    )
    # G = (R - 4 pi r^2 rho) / r, with the baseline optionally mis-scaled
    gr = rdf / r + baseline_scale * gr_background(r, rho)

    return XYEData(title="synthetic", x=r, y=gr)


# ---------------------------------------------------------------------------
# gr_background / radial_distribution
# ---------------------------------------------------------------------------


def test_gr_background_is_the_low_r_line_through_the_origin():
    r = np.array([0.0, 0.5, 1.0, 2.0])
    rho = 0.05

    background = gr_background(r, rho)

    np.testing.assert_allclose(background, -4 * math.pi * rho * r)
    assert background[0] == 0


def test_radial_distribution_is_zero_where_gr_is_the_background():
    # below the first peak g(r) = 0, so R(r) = 0
    r = np.linspace(0.01, 1.0, 50)
    rho = 0.05

    rdf = radial_distribution(r, gr_background(r, rho), rho)

    np.testing.assert_allclose(rdf, 0, atol=1e-12)


def test_radial_distribution_integrates_to_the_coordination_number():
    coordination = 6
    gr_data = synthetic_gr(peak_r=2.34, coordination=coordination)
    rho = ChemicalFormula(formula=COMPOSITION).rho(density_g_cm3=DENSITY)

    rdf = radial_distribution(gr_data.x, gr_data.y, rho)

    assert np.trapezoid(rdf, gr_data.x) == pytest.approx(coordination, rel=1e-3)


# ---------------------------------------------------------------------------
# determine_gr_header / _load_gr
# ---------------------------------------------------------------------------


def _write_gr_file(path: Path) -> Path:
    # as pdfgetx writes a .gr file: a header, then "#### start data", a
    # column-label line, then the data
    path.write_text(
        "[DEFAULT]\n"
        "composition = K2PdCl6\n"
        "\n"
        "#### start data\n"
        "#S 1\n"
        "#L r($\\AA$)  G($\\AA^{-2}$)\n"
        "1.0 0.1\n"
        "2.0 0.2\n"
        "3.0 0.3\n"
    )
    return path


def test_determine_gr_header_skips_to_the_data(tmp_path: Path):
    gr_file = _write_gr_file(tmp_path / "sample.gr")

    header = determine_gr_header(str(gr_file))

    lines = gr_file.read_text().splitlines()
    # the metadata header is skipped and only comments are left before the data
    assert "#### start data" in lines[:header]
    assert all(line.startswith("#") or line[0].isdigit() for line in lines[header:])
    assert "1.0 0.1" in lines[header:]


def test_determine_gr_header_raises_without_start_data(tmp_path: Path):
    gr_file = tmp_path / "sample.gr"
    gr_file.write_text("1.0 0.1\n2.0 0.2\n")

    with pytest.raises(ValueError, match=str(gr_file)):
        determine_gr_header(str(gr_file))


def test_load_gr_reads_a_gr_file(tmp_path: Path):
    gr_file = _write_gr_file(tmp_path / "sample.gr")

    gr_data = _load_gr(str(gr_file))

    assert isinstance(gr_data, XYEData)
    np.testing.assert_array_equal(gr_data.x, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(gr_data.y, [0.1, 0.2, 0.3])
    assert gr_data.e is None


def test_load_gr_passes_xyedata_through():
    gr_data = synthetic_gr(peak_r=2.34, coordination=6)

    assert _load_gr(gr_data) is gr_data


# ---------------------------------------------------------------------------
# pdfcurl
# ---------------------------------------------------------------------------


def test_encoded_pdfcurl_request_serialises_the_data():
    data = XYEData(title="sample", x=np.array([1.0, 2.0]), y=np.array([3.0, 4.0]))
    request = EncodedPDFCurlRequest(
        composition=COMPOSITION,
        wavelength=0.16,
        name="sample",
        input_data=data,
        terminationfunctions="lorch",
    )

    dumped = request.model_dump(mode="json")

    json.dumps(dumped)  # can be posted
    assert dumped["name"] == "sample"
    assert dumped["terminationfunctions"] == "lorch"
    assert dumped["publish"] is False
    roundtrip = EncodedPDFCurlRequest.model_validate(dumped)
    np.testing.assert_array_equal(roundtrip.input_data.x, data.x)
    np.testing.assert_array_equal(roundtrip.input_data.y, data.y)


def test_pdfcurl_response_parses_a_completed_result():
    gr_data = synthetic_gr(peak_r=2.34, coordination=6)

    response = PDFCurlResponse.model_validate(
        {
            "name": "sample",
            "complete": True,
            "result": {"gr": gr_data.model_dump(mode="json")},
        }
    )

    assert response.complete
    assert response.error is None
    assert response.result is not None
    assert isinstance(response.result.gr, XYEData)
    np.testing.assert_allclose(response.result.gr.y, gr_data.y)
    assert response.result.sq is None
    assert response.id is not None


def test_pdfcurl_response_parses_an_error():
    response = PDFCurlResponse.model_validate(
        {"name": "sample", "error": "pdfgetx failed"}
    )

    assert not response.complete
    assert response.result is None
    assert response.error == "pdfgetx failed"


def test_send_post_request_returns_the_json_body():
    response = MagicMock()
    response.json.return_value = {"name": "sample", "complete": True}

    with patch("xrpd_toolbox.utils.pdfcurl.httpx.post", return_value=response) as post:
        result = send_post_request("https://example.com", data={"a": 1})

    post.assert_called_once_with("https://example.com", json={"a": 1})
    response.raise_for_status.assert_called_once()
    assert result == {"name": "sample", "complete": True}


def test_send_xy_to_pdfcurl_sends_the_extra_params(tmp_path: Path):
    xy_file = tmp_path / "sample.xy"
    xy_file.write_text("1 2\n2 3\n")

    with patch(
        "xrpd_toolbox.utils.pdfcurl.send_post_request",
        return_value={"name": "sample"},
    ) as send:
        result = send_xy_to_pdfcurl(
            xy_filepath=xy_file,
            composition=COMPOSITION,
            wavelength=0.16,
            terminationfunctions="lorch",
            twothetazero=0.01,
            bgscale=0.9,
            force=False,
        )

    sent = send.call_args.kwargs["data"]
    assert sent["terminationfunctions"] == "lorch"
    assert sent["twothetazero"] == 0.01
    assert sent["bgscale"] == 0.9
    assert sent["mode"] == "xray"
    assert sent["force"] is False
    assert result == {"name": "sample"}
