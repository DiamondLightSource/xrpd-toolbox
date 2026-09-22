"""Tests for xrpd_toolbox.i15_1.eiger_analysis.

do_eiger_calibration/do_eiger_data_reduction are tested with the
lower-level EigerDataLoader helpers mocked out so their own orchestration
logic can be verified independently.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from xrpd_toolbox.i15_1 import eiger_analysis
from xrpd_toolbox.i15_1.eiger_500k import (
    apply_mask,
    sum_unique_two_theta_positions_and_normalise,
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
# sum_unique_two_theta_positions_and_normalise
# ---------------------------------------------------------------------------


class FakeEigerData:
    """Minimal stand-in for EigerDataLoader exposing just what
    sum_unique_two_theta_positions_and_normalise needs."""

    def __init__(self, positions, data, i0):
        self.positions = positions
        self._data = data
        self._i0 = i0

    def get_data(self, frames):
        return self._data[frames]

    def get_i0(self):
        return self._i0


def test_sum_unique_two_theta_positions_groups_and_normalises_multiple_positions():
    fake = FakeEigerData(
        positions=np.array([1.0, 2.0, 3.0]),
        data=np.stack([np.full((4, 5), value) for value in (10.0, 20.0, 30.0)]),
        i0=np.array([2.0, 4.0, 5.0]),
    )

    result = sum_unique_two_theta_positions_and_normalise(fake)  # type: ignore[arg-type]

    # one frame per unique position, so summing over axis=0 is a no-op and
    # each frame is normalised by dividing by its own i0
    assert result.shape == (3, 4, 5)
    assert np.allclose(result[0], 10.0 / 2.0)
    assert np.allclose(result[1], 20.0 / 4.0)
    assert np.allclose(result[2], 30.0 / 5.0)


def test_sum_unique_two_theta_positions_single_frame_at_position_one():
    fake = FakeEigerData(
        positions=np.array([1.0]),
        data=np.full((1, 4, 5), 2.0),
        i0=np.array([3.0]),
    )

    result = sum_unique_two_theta_positions_and_normalise(fake)  # type: ignore[arg-type]

    # sum over the frames axis (a no-op for a single frame) then normalised
    # by dividing by the summed i0
    assert result.shape == (1, 4, 5)
    assert np.allclose(result, 2.0 / 3.0)


# ---------------------------------------------------------------------------
# apply_mask
# ---------------------------------------------------------------------------


def test_apply_mask_multiplies_each_frame():
    frames = np.ones((2, 3, 3))
    mask = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 1]])

    masked = apply_mask(frames, mask)

    assert masked.shape == (2, 3, 3)
    assert np.array_equal(masked[0], mask)
    assert np.array_equal(masked[1], mask)


def test_apply_mask_with_boolean_mask():
    frames = np.array([[[1.0, 2.0], [3.0, 4.0]]])
    mask = np.array([[True, False], [False, True]])

    masked = apply_mask(frames, mask)

    assert np.array_equal(masked[0], [[1.0, 0.0], [0.0, 4.0]])


# ---------------------------------------------------------------------------
# do_eiger_calibration / do_eiger_data_reduction / ...pdfcurl - all analysis
# for a nexus file must save into, and load from, its "processed" subfolder
# ---------------------------------------------------------------------------


def _fake_eiger_data(**overrides):
    fake = MagicMock()
    fake.get_calibrant.return_value = "Silicon"
    fake.positions = np.array([1.0, 2.0])
    fake.get_summed_normalised_and_masked_frames.return_value = np.zeros((2, 4, 5))
    fake.get_summed_and_normalised_frames.return_value = np.zeros((2, 4, 5))
    fake.get_mask.return_value = None
    fake.wavelength = 1.0
    for key, value in overrides.items():
        setattr(fake, key, value)
    return fake


def test_do_eiger_calibration_saves_goniometer_to_processed_dir(tmp_path):
    nexus_filepath = tmp_path / "scan.nxs"
    nexus_filepath.touch()

    mock_build = MagicMock(return_value=("gonio.json", "meta.json"))

    with (
        patch.object(
            eiger_analysis, "EigerDataLoader", return_value=_fake_eiger_data()
        ),
        patch.object(eiger_analysis, "build_and_save_goniometer", mock_build),
    ):
        eiger_analysis.do_eiger_calibration(nexus_filepath)

    expected_processed_dir = str(tmp_path / "processed")
    assert mock_build.call_args.kwargs["output_dir"] == expected_processed_dir
    assert (tmp_path / "processed").is_dir()


def test_do_eiger_data_reduction_writes_xy_into_processed_dir(tmp_path):
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


def test_do_eiger_data_reduction_respects_explicit_output_xy_filepath(tmp_path):
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


def test_pdfcurl_reduction_finds_previously_saved_background_in_processed_dir(
    tmp_path,
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


def test_pdfcurl_reduction_generates_missing_background_into_processed_dir(
    tmp_path,
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
