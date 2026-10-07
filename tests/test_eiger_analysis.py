"""Tests for xrpd_toolbox.i15_1.eiger_analysis.

These reduce real (synthetic, full-size) scans written to disk and only check
what comes out - the returned paths, the files written and what is sent to
pdfcurl - so they don't depend on how the reduction is done internally.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from eiger_fixtures import build_i15_1_scan, write_i15_1_goniometer_json
from xrpd_toolbox.i15_1 import eiger_analysis
from xrpd_toolbox.i15_1.eiger_pyfai import GONIOMETER_SAVE_NAME

SCAN_TTH = (10.0, 14.0)


@pytest.fixture(autouse=True)
def no_external_services(monkeypatch, tmp_path: Path):
    # nothing is posted to the beamline's data visualisation server
    monkeypatch.setattr("xrpd_toolbox.plotting.requests.post", MagicMock())
    # the beamline's shared calibration under /dls_sw must not leak into tests
    monkeypatch.setattr(
        eiger_analysis, "GEOMETRY_CAL_FILEPATH", tmp_path / "no_beamline_cal.json"
    )


def _save_goniometer_in_processed_dir(directory: Path) -> Path:
    # as do_eiger_goniometer_calibration saves it: timestamped, in flat processed/
    return write_i15_1_goniometer_json(
        directory / "processed" / f"cal_2026-10-01_10-00-00_{GONIOMETER_SAVE_NAME}"
    )


def _assert_is_reduced_xy(xy_filepath: Path):
    tth, intensity = np.loadtxt(xy_filepath, unpack=True)

    assert len(tth) > 100
    assert np.all(np.diff(tth) > 0)
    # the detector spans ~±10° about each arm position
    assert tth[0] < min(SCAN_TTH) and tth[-1] > max(SCAN_TTH)
    assert tth[0] >= 0 and tth[-1] < max(SCAN_TTH) + 15
    assert np.all(np.isfinite(intensity))
    assert np.all(intensity >= 0)
    assert np.median(intensity) > 0


# ---------------------------------------------------------------------------
# do_eiger_data_reduction
# ---------------------------------------------------------------------------


def test_do_eiger_data_reduction_writes_xy_into_processed_dir(tmp_path: Path):
    nexus_filepath = build_i15_1_scan(tmp_path, "scan", tth=SCAN_TTH)
    goniometer_filepath = write_i15_1_goniometer_json(tmp_path / "gonio.json")

    result = eiger_analysis.do_eiger_data_reduction(
        nexus_filepath, goniometer_filepath=goniometer_filepath, publish=False
    )

    expected_path = tmp_path / "processed" / "scan" / "scan_fastcs_eiger.xy"
    assert Path(result) == expected_path
    _assert_is_reduced_xy(expected_path)


def test_do_eiger_data_reduction_respects_explicit_output_xy_filepath(tmp_path: Path):
    nexus_filepath = build_i15_1_scan(tmp_path, "scan", tth=SCAN_TTH)
    goniometer_filepath = write_i15_1_goniometer_json(tmp_path / "gonio.json")
    explicit_output = tmp_path / "elsewhere" / "custom.xy"

    result = eiger_analysis.do_eiger_data_reduction(
        nexus_filepath,
        goniometer_filepath=goniometer_filepath,
        output_xy_filepath=explicit_output,
        publish=False,
    )

    assert Path(result) == explicit_output
    _assert_is_reduced_xy(explicit_output)


def test_do_eiger_data_reduction_uses_calibration_saved_in_processed_dir(
    tmp_path: Path,
):
    # no goniometer given and none in the nexus file: the calibration shared by
    # every scan in the directory, in the flat processed/ folder, is used
    nexus_filepath = build_i15_1_scan(tmp_path, "scan", tth=SCAN_TTH)
    _save_goniometer_in_processed_dir(tmp_path)

    result = eiger_analysis.do_eiger_data_reduction(nexus_filepath, publish=False)

    assert Path(result) == tmp_path / "processed" / "scan" / "scan_fastcs_eiger.xy"
    _assert_is_reduced_xy(Path(result))


def test_do_eiger_data_reduction_raises_without_any_calibration(tmp_path: Path):
    nexus_filepath = build_i15_1_scan(tmp_path, "scan", tth=SCAN_TTH)

    with pytest.raises(FileNotFoundError):
        eiger_analysis.do_eiger_data_reduction(nexus_filepath, publish=False)


# ---------------------------------------------------------------------------
# do_eiger_goniometer_calibration
# ---------------------------------------------------------------------------


def _fake_build_and_save_goniometer(*, output_dir, **kwargs):
    # stands in for the (slow) pyFAI goniometer refinement, saving where asked
    gonio = Path(output_dir) / f"scan_{GONIOMETER_SAVE_NAME}"
    metadata = Path(output_dir) / "scan_calibration_metadata.json"
    gonio.write_text(json.dumps({"fake": "goniometer"}))
    metadata.write_text(json.dumps({"fake": "metadata"}))
    return gonio, metadata


def test_do_eiger_calibration_saves_goniometer_to_processed_dir(tmp_path: Path):
    nexus_filepath = build_i15_1_scan(tmp_path, "scan", tth=SCAN_TTH)

    with patch.object(
        eiger_analysis,
        "build_and_save_goniometer",
        side_effect=_fake_build_and_save_goniometer,
    ):
        gonio, metadata = eiger_analysis.do_eiger_goniometer_calibration(
            nexus_filepath, calibrant_name="Si", plot_fits=False, do_reduction=False
        )

    # shared by every scan in the directory, so flat in processed/
    assert Path(gonio).parent == tmp_path / "processed"
    assert Path(metadata).parent == tmp_path / "processed"
    assert Path(gonio).exists() and Path(metadata).exists()


# ---------------------------------------------------------------------------
# do_eiger_data_reduction_and_send_xy_to_pdfcurl
# ---------------------------------------------------------------------------


def _build_sample_scan(tmp_path: Path) -> Path:
    _save_goniometer_in_processed_dir(tmp_path)
    return build_i15_1_scan(
        tmp_path,
        "scan",
        tth=SCAN_TTH,
        composition="SiO2",
        empty_capillary_filename="empty_capillary.nxs",
    )


def test_pdfcurl_reduction_finds_previously_saved_background_in_processed_dir(
    tmp_path: Path,
):
    nexus_filepath = _build_sample_scan(tmp_path)
    build_i15_1_scan(tmp_path, "empty_capillary", tth=SCAN_TTH)
    existing_bg_xy = (
        tmp_path / "processed" / "empty_capillary" / "empty_capillary_fastcs_eiger.xy"
    )
    existing_bg_xy.parent.mkdir(parents=True)
    existing_bg_xy.write_text("previously reduced background")

    with patch.object(eiger_analysis, "send_xy_to_pdfcurl") as mock_send:
        result = eiger_analysis.do_eiger_data_reduction_and_send_xy_to_pdfcurl(
            nexus_filepath
        )

    expected_xy = tmp_path / "processed" / "scan" / "scan_fastcs_eiger.xy"
    assert Path(result) == expected_xy
    _assert_is_reduced_xy(expected_xy)
    # the saved background is reused as-is, not regenerated
    assert existing_bg_xy.read_text() == "previously reduced background"
    sent = mock_send.call_args.kwargs
    assert Path(sent["xy_filepath"]) == expected_xy
    assert Path(sent["background_file"]) == existing_bg_xy
    assert sent["composition"] == "SiO2"
    assert sent["wavelength"] == pytest.approx(12.398 / 76.69, rel=1e-3)


def test_pdfcurl_reduction_generates_missing_background_into_processed_dir(
    tmp_path: Path,
):
    nexus_filepath = _build_sample_scan(tmp_path)
    build_i15_1_scan(tmp_path, "empty_capillary", tth=SCAN_TTH)

    with patch.object(eiger_analysis, "send_xy_to_pdfcurl") as mock_send:
        result = eiger_analysis.do_eiger_data_reduction_and_send_xy_to_pdfcurl(
            nexus_filepath
        )

    expected_xy = tmp_path / "processed" / "scan" / "scan_fastcs_eiger.xy"
    expected_bg_xy = (
        tmp_path / "processed" / "empty_capillary" / "empty_capillary_fastcs_eiger.xy"
    )
    assert Path(result) == expected_xy
    # the background was reduced from its own nexus file into processed/
    _assert_is_reduced_xy(expected_bg_xy)
    sent = mock_send.call_args.kwargs
    assert Path(sent["xy_filepath"]) == expected_xy
    assert Path(sent["background_file"]) == expected_bg_xy


def test_pdfcurl_reduction_sends_no_background_when_there_is_none(tmp_path: Path):
    # the empty capillary scan was never collected
    nexus_filepath = _build_sample_scan(tmp_path)

    with patch.object(eiger_analysis, "send_xy_to_pdfcurl") as mock_send:
        result = eiger_analysis.do_eiger_data_reduction_and_send_xy_to_pdfcurl(
            nexus_filepath
        )

    _assert_is_reduced_xy(Path(result))
    assert mock_send.call_args.kwargs["background_file"] is None
