"""Tests for xrpd_toolbox.i15_1.eiger_analysis."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from xrpd_toolbox.i15_1 import eiger_analysis
from xrpd_toolbox.i15_1.eiger_500k import (
    apply_mask,
    unique_slices,
)

# ---------------------------------------------------------------------------
# unique_slices
# ---------------------------------------------------------------------------


def test_unique_slices_groups_runs_of_equal_values():
    arr = np.array([1, 1, 2, 2, 2, 3])

    slices = unique_slices(arr)

    assert [arr[s].tolist() for s in slices] == [[1, 1], [2, 2, 2], [3]]


def test_unique_slices_all_values_unique():
    arr = np.array([1.0, 2.0, 3.0])

    slices = unique_slices(arr)

    assert slices == [slice(0, 1), slice(1, 2), slice(2, 3)]


def test_unique_slices_single_value_repeated():
    arr = np.array([5.0, 5.0, 5.0])

    slices = unique_slices(arr)

    assert len(slices) == 1
    assert arr[slices[0]].tolist() == [5.0, 5.0, 5.0]


def test_unique_slices_accepts_plain_list():
    # unique_slices does np.asarray(arr) internally, so list input works too
    slices = unique_slices([1, 1, 2])  # type: ignore[arg-type]

    assert slices == [slice(0, 2), slice(2, 3)]


# ---------------------------------------------------------------------------
# apply_mask
# ---------------------------------------------------------------------------


def test_apply_mask_zeroes_bad_pixels_in_each_frame():
    frames = np.ones((2, 3, 3))
    mask = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 1]])

    masked = apply_mask(frames, mask)

    assert masked.shape == (2, 3, 3)
    # Eiger/pyFAI convention: nonzero in the mask = bad pixel
    assert np.array_equal(masked[0], 1 - mask)
    assert np.array_equal(masked[1], 1 - mask)


def test_apply_mask_with_boolean_mask():
    frames = np.array([[[1.0, 2.0], [3.0, 4.0]]])
    mask = np.array([[True, False], [False, True]])

    masked = apply_mask(frames, mask)

    assert np.array_equal(masked[0], [[0.0, 2.0], [3.0, 0.0]])


def test_apply_mask_removes_saturated_bad_pixels():
    # bad Eiger pixels read as the uint32 max - they must not survive masking
    frames = np.full((1, 2, 2), 10.0)
    frames[0, 0, 0] = np.iinfo(np.uint32).max
    mask = np.array([[1, 0], [0, 0]], dtype=np.uint32)

    masked = apply_mask(frames, mask)

    assert masked.max() == 10.0
    assert masked[0, 0, 0] == 0


# ---------------------------------------------------------------------------
# do_eiger_calibration / do_eiger_data_reduction / ...pdfcurl - all analysis
# for a nexus file must save into, and load from, its "processed" subfolder
# ---------------------------------------------------------------------------


def _fake_eiger_data(**overrides):
    fake = MagicMock()
    fake.get_calibrant.return_value = "Silicon"
    fake.positions = np.array([1.0, 2.0])
    fake.get_summed_normalised_and_masked_frames.return_value = np.zeros((2, 4, 5))
    fake.get_summed_and_masked_frames.return_value = np.zeros((2, 4, 5))
    fake.get_summed_and_normalised_frames.return_value = np.zeros((2, 4, 5))
    fake.get_mask.return_value = None
    fake.wavelength = 1.0
    for key, value in overrides.items():
        setattr(fake, key, value)
    return fake


def test_do_eiger_calibration_saves_goniometer_to_processed_dir(tmp_path: Path):
    nexus_filepath = tmp_path / "scan.nxs"
    nexus_filepath.touch()

    mock_build = MagicMock(return_value=("gonio.json", "meta.json"))
    mock_reduction = MagicMock()

    with (
        patch.object(
            eiger_analysis, "EigerDataLoader", return_value=_fake_eiger_data()
        ),
        patch.object(eiger_analysis, "build_and_save_goniometer", mock_build),
        patch.object(eiger_analysis, "do_eiger_data_reduction", mock_reduction),
    ):
        eiger_analysis.do_eiger_goniometer_calibration(nexus_filepath)

    expected_processed_dir = str(tmp_path / "processed")
    assert mock_build.call_args.kwargs["output_dir"] == expected_processed_dir
    # the calibration scan is reduced straight after the goniometer is built
    mock_reduction.assert_called_once_with(nexus_filepath)
    assert (tmp_path / "processed").is_dir()
    nexus_filepath.unlink(missing_ok=True)


def test_do_eiger_data_reduction_writes_xy_into_processed_dir(tmp_path: Path):
    nexus_filepath = tmp_path / "scan.nxs"
    nexus_filepath.touch()

    def fake_integrate(
        images, positions, goniometer_dir, output_xy_filepath, npt=None, mask=None
    ):
        Path(output_xy_filepath).parent.mkdir(parents=True, exist_ok=True)
        Path(output_xy_filepath).write_text("fake xy data")
        return Path(output_xy_filepath)

    with (
        patch.object(
            eiger_analysis, "EigerDataLoader", return_value=_fake_eiger_data()
        ),
        patch.object(
            eiger_analysis, "integrate_with_goniometer", side_effect=fake_integrate
        ) as mock_integrate,
    ):
        result_path = eiger_analysis.do_eiger_data_reduction(nexus_filepath)

    expected_path = tmp_path / "processed" / "scan" / "scan_fastcs_eiger.xy"
    assert result_path == expected_path
    assert expected_path.exists()
    # the goniometer calibration is shared across every file in the
    # directory, so it must be looked up in the flat processed/ folder, not
    # the per-file processed/scan/ subfolder used for the output xy
    assert mock_integrate.call_args.kwargs["goniometer_dir"] == str(
        tmp_path / "processed"
    )
    nexus_filepath.unlink(missing_ok=True)


def test_do_eiger_data_reduction_respects_explicit_output_xy_filepath(tmp_path: Path):
    nexus_filepath = tmp_path / "scan.nxs"
    nexus_filepath.touch()
    explicit_output = tmp_path / "elsewhere" / "custom.xy"

    def fake_integrate(
        images, positions, goniometer_dir, output_xy_filepath, npt=None, mask=None
    ):
        Path(output_xy_filepath).parent.mkdir(parents=True, exist_ok=True)
        Path(output_xy_filepath).write_text("fake xy data")
        return Path(output_xy_filepath)

    with (
        patch.object(
            eiger_analysis, "EigerDataLoader", return_value=_fake_eiger_data()
        ),
        patch.object(
            eiger_analysis, "integrate_with_goniometer", side_effect=fake_integrate
        ),
    ):
        result_path = eiger_analysis.do_eiger_data_reduction(
            nexus_filepath, explicit_output
        )

    assert result_path == explicit_output
    assert explicit_output.exists()
    nexus_filepath.unlink(missing_ok=True)


def test_pdfcurl_reduction_finds_previously_saved_background_in_processed_dir(
    tmp_path: Path,
):
    nexus_filepath = tmp_path / "scan.nxs"
    nexus_filepath.touch()
    bg_nexus_filepath = tmp_path / "empty_capillary.nxs"
    bg_nexus_filepath.touch()

    bg_processed_dir = tmp_path / "processed" / "empty_capillary"
    bg_processed_dir.mkdir(parents=True)
    existing_bg_xy = bg_processed_dir / "empty_capillary_fastcs_eiger.xy"
    existing_bg_xy.write_text("previously reduced background")

    fake_eiger_data = _fake_eiger_data(
        get_composition=MagicMock(return_value="SiO2"),
        get_wavelength=MagicMock(return_value=0.5),
        get_sample_environment_scan_filepath=MagicMock(
            return_value=str(bg_nexus_filepath)
        ),
    )

    with (
        patch.object(eiger_analysis, "EigerDataLoader", return_value=fake_eiger_data),
        patch.object(eiger_analysis, "do_eiger_data_reduction") as mock_reduction,
        patch.object(eiger_analysis, "send_xy_to_pdfcurl") as mock_send,
    ):
        mock_reduction.return_value = (
            tmp_path / "processed" / "scan" / ("scan_fastcs_eiger.xy")
        )

        eiger_analysis.do_eiger_data_reduction_and_send_xy_to_pdfcurl(nexus_filepath)

    # only called once, for the main scan - the background was already found
    # saved in processed/ so it is not regenerated
    mock_reduction.assert_called_once_with(nexus_filepath, None)
    assert mock_send.call_args.kwargs["background_file"] == str(existing_bg_xy)
    nexus_filepath.unlink(missing_ok=True)


def test_pdfcurl_reduction_generates_missing_background_into_processed_dir(
    tmp_path: Path,
):
    nexus_filepath = tmp_path / "scan.nxs"
    nexus_filepath.touch()
    bg_nexus_filepath = tmp_path / "empty_capillary.nxs"
    bg_nexus_filepath.touch()

    fake_eiger_data = _fake_eiger_data(
        get_composition=MagicMock(return_value="SiO2"),
        get_wavelength=MagicMock(return_value=0.5),
        get_sample_environment_scan_filepath=MagicMock(
            return_value=str(bg_nexus_filepath)
        ),
    )

    expected_bg_xy = (
        tmp_path / "processed" / "empty_capillary" / "empty_capillary_fastcs_eiger.xy"
    )

    with (
        patch.object(eiger_analysis, "EigerDataLoader", return_value=fake_eiger_data),
        patch.object(eiger_analysis, "do_eiger_data_reduction") as mock_reduction,
        patch.object(eiger_analysis, "send_xy_to_pdfcurl") as mock_send,
    ):
        mock_reduction.side_effect = [
            expected_bg_xy,
            tmp_path / "processed" / "scan" / "scan_fastcs_eiger.xy",
        ]

        eiger_analysis.do_eiger_data_reduction_and_send_xy_to_pdfcurl(nexus_filepath)

    assert mock_reduction.call_count == 2
    mock_reduction.assert_any_call(str(bg_nexus_filepath), expected_bg_xy)
    assert mock_send.call_args.kwargs["background_file"] == str(expected_bg_xy)
    nexus_filepath.unlink(missing_ok=True)
