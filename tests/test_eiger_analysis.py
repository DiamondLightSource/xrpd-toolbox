"""Tests for xrpd_toolbox.i15_1.eiger_analysis.

Two of the functions here (sum_unique_two_theta_positions_and_normalise and
run_eiger_analysis) currently contain what look like bugs:

- ``sum_unique_two_theta_positions_and_normalise`` does
  ``assert len(slices) == np.unique(positions)``, comparing an int to an
  array rather than to ``len(np.unique(positions))``. For any input with
  more than one unique tth position this raises ValueError ("truth value of
  an array... is ambiguous") before any real work is done.
- ``run_eiger_analysis`` logs ``analysis_to_run.__name__`` *before* checking
  whether ``analysis_to_run is None``, so the "no analysis plan exists"
  branch can never actually be reached - it raises AttributeError instead.

The tests below document this real, current behaviour rather than papering
over it; do_eiger_calibration/do_eiger_data_reduction are tested with the
lower-level helpers mocked out so their own orchestration logic can be
verified independently of those issues.
"""

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from eiger_fixtures import build_eiger_nexus
from xrpd_toolbox.i15_1 import eiger_analysis as ea
from xrpd_toolbox.i15_1.eiger_500k import EigerDataLoader

# ---------------------------------------------------------------------------
# unique_slices
# ---------------------------------------------------------------------------


def test_unique_slices_groups_runs_of_equal_values():
    arr = np.array([1, 1, 2, 2, 2, 3])

    slices = ea.unique_slices(arr)

    assert [arr[s].tolist() for s in slices] == [[1, 1], [2, 2, 2], [3]]


def test_unique_slices_all_values_unique():
    arr = np.array([1.0, 2.0, 3.0])

    slices = ea.unique_slices(arr)

    assert slices == [slice(0, 1), slice(1, 2), slice(2, 3)]


def test_unique_slices_single_value_repeated():
    arr = np.array([5.0, 5.0, 5.0])

    slices = ea.unique_slices(arr)

    assert len(slices) == 1
    assert arr[slices[0]].tolist() == [5.0, 5.0, 5.0]


def test_unique_slices_accepts_plain_list():
    # unique_slices does np.asarray(arr) internally, so list input works too
    slices = ea.unique_slices([1, 1, 2])  # type: ignore[arg-type]

    assert slices == [slice(0, 2), slice(2, 3)]


# ---------------------------------------------------------------------------
# sum_unique_two_theta_positions_and_normalise
# ---------------------------------------------------------------------------


class FakeEigerData:
    """Minimal stand-in for EigerDataLoader exposing just what
    sum_unique_two_theta_positions_and_normalise needs."""

    def __init__(self, positions, data, durations):
        self.positions = positions
        self._data = data
        self.durations = durations

    def get_data(self, frames):
        return self._data[frames]


def test_sum_unique_two_theta_positions_raises_for_multiple_positions():
    fake = FakeEigerData(
        positions=np.array([1.0, 2.0, 3.0]),
        data=np.zeros((3, 4, 5)),
        durations=np.array([0.1, 0.1, 0.1]),
    )

    with pytest.raises(ValueError, match="ambiguous"):
        ea.sum_unique_two_theta_positions_and_normalise(fake)  # type: ignore[arg-type]


def test_sum_unique_two_theta_positions_single_frame_at_position_one():
    # The buggy `assert len(slices) == np.unique(positions)` only survives
    # when there is exactly one unique position and it equals 1.0 - see
    # module docstring above.
    fake = FakeEigerData(
        positions=np.array([1.0]),
        data=np.full((1, 4, 5), 2.0),
        durations=np.array([3.0]),
    )

    result = ea.sum_unique_two_theta_positions_and_normalise(fake)  # type: ignore[arg-type]

    # sum over the last axis (cols=5) then scaled by duration
    assert result.shape == (1, 1, 4)
    assert np.allclose(result, 2.0 * 5 * 3.0)


# ---------------------------------------------------------------------------
# apply_mask
# ---------------------------------------------------------------------------


def test_apply_mask_multiplies_each_frame():
    frames = np.ones((2, 3, 3))
    mask = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 1]])

    masked = ea.apply_mask(frames, mask)

    assert masked.shape == (2, 3, 3)
    assert np.array_equal(masked[0], mask)
    assert np.array_equal(masked[1], mask)


def test_apply_mask_with_boolean_mask():
    frames = np.array([[[1.0, 2.0], [3.0, 4.0]]])
    mask = np.array([[True, False], [False, True]])

    masked = ea.apply_mask(frames, mask)

    assert np.array_equal(masked[0], [[1.0, 0.0], [0.0, 4.0]])


# ---------------------------------------------------------------------------
# do_eiger_calibration
# ---------------------------------------------------------------------------


def test_do_eiger_calibration_raises_without_calibrant(tmp_path):
    nxs = build_eiger_nexus(tmp_path / "scan.nxs", include_calibrant=False)

    with pytest.raises(Exception, match="Calibraation is not in Nexus file"):
        ea.do_eiger_calibration(nxs)


def test_do_eiger_calibration_orchestrates_helpers(tmp_path):
    nxs = build_eiger_nexus(
        tmp_path / "scan.nxs",
        tth=np.array([1.0]),
        plan_name="calibration_collection",
    )

    dummy_summed = np.zeros((1, 4, 5))
    dummy_mask = np.ones((4, 5), dtype=bool)
    dummy_masked = np.zeros((1, 4, 5))

    mock_sum = MagicMock(return_value=dummy_summed)
    mock_apply_mask = MagicMock(return_value=dummy_masked)
    mock_build = MagicMock(return_value=("gonio.json", "meta.json"))

    with (
        patch.object(EigerDataLoader, "get_calibrant", lambda self: "Si"),
        patch.object(EigerDataLoader, "get_mask", lambda self: dummy_mask),
        patch.object(ea, "sum_unique_two_theta_positions_and_normalise", mock_sum),
        patch.object(ea, "apply_mask", mock_apply_mask),
        patch.object(ea, "build_and_save_goniometer", mock_build),
    ):
        result = ea.do_eiger_calibration(nxs)

    assert result == ("gonio.json", "meta.json")

    mock_apply_mask.assert_called_once()
    assert mock_apply_mask.call_args.kwargs["image_frames"] is dummy_summed
    assert mock_apply_mask.call_args.kwargs["mask"] is dummy_mask

    _, build_kwargs = mock_build.call_args
    assert build_kwargs["nexus_filepath"] == Path(nxs)
    assert build_kwargs["images"] is dummy_masked
    assert build_kwargs["calibrant_name"] == "Si"
    assert build_kwargs["initial_dist_m"] == 0.25
    assert build_kwargs["output_dir"] == str(Path(nxs).parent)
    assert build_kwargs["max_rings"] == [5, 5, 5, 7, 7, 9, 11, 15, 17]
    assert build_kwargs["npt"] == ea.DEFAULT_NPT
    assert np.array_equal(build_kwargs["angles"], [1.0])


# ---------------------------------------------------------------------------
# do_eiger_data_reduction
# ---------------------------------------------------------------------------


def test_do_eiger_data_reduction_orchestrates_helpers(tmp_path):
    nxs = build_eiger_nexus(
        tmp_path / "scan.nxs", tth=np.array([1.0]), plan_name="data_collection"
    )

    dummy_summed = np.zeros((1, 4, 5))
    dummy_mask = np.ones((4, 5), dtype=bool)

    mock_sum = MagicMock(return_value=dummy_summed)
    expected_out = Path(nxs).parent / (Path(nxs).stem + "_eiger.xy")
    mock_integrate = MagicMock(return_value=expected_out)

    with (
        patch.object(EigerDataLoader, "get_mask", lambda self: dummy_mask),
        patch.object(ea, "sum_unique_two_theta_positions_and_normalise", mock_sum),
        patch.object(ea, "integrate_with_goniometer", mock_integrate),
    ):
        result = ea.do_eiger_data_reduction(nxs)

    assert result == expected_out

    _, kwargs = mock_integrate.call_args
    assert kwargs["images"] is dummy_summed
    assert kwargs["goniometer_dir"] == str(Path(nxs).parent)
    assert np.array_equal(kwargs["positions"], [1.0])
    assert kwargs["mask"] is dummy_mask
    assert kwargs["output_xy_filepath"] == expected_out
    assert kwargs["npt"] == ea.DEFAULT_NPT


# ---------------------------------------------------------------------------
# collection_analysis_dict / run_eiger_analysis
# ---------------------------------------------------------------------------


def test_collection_analysis_dict_maps_known_plans():
    assert ea.collection_analysis_dict["data_collection"] is ea.do_eiger_data_reduction
    assert (
        ea.collection_analysis_dict["calibration_collection"] is ea.do_eiger_calibration
    )


def test_run_eiger_analysis_waits_then_dispatches(tmp_path):
    nxs = build_eiger_nexus(tmp_path / "scan.nxs", plan_name="data_collection")

    mock_wait = MagicMock()
    mock_fn = MagicMock()
    mock_fn.__name__ = "do_eiger_data_reduction"

    with (
        patch.object(ea, "wait_for_finished_file", mock_wait),
        patch.object(ea, "collection_analysis_dict", {"data_collection": mock_fn}),
    ):
        ea.run_eiger_analysis(nxs)

    mock_wait.assert_called_once_with(nxs)
    mock_fn.assert_called_once_with(nxs)


def test_run_eiger_analysis_unknown_plan_raises_keyerror(tmp_path):
    nxs = build_eiger_nexus(tmp_path / "scan.nxs", plan_name="some_unregistered_plan")

    with patch.object(ea, "wait_for_finished_file", MagicMock()):
        with pytest.raises(KeyError):
            ea.run_eiger_analysis(nxs)


def test_run_eiger_analysis_none_entry_raises_attributeerror_before_guard(
    tmp_path, caplog
):
    # documents the current (buggy) behaviour: the `analysis_to_run is None`
    # guard is unreachable because `analysis_to_run.__name__` is accessed
    # first, in the logger.info call above it.
    nxs = build_eiger_nexus(tmp_path / "scan.nxs", plan_name="data_collection")

    with (
        patch.object(ea, "wait_for_finished_file", MagicMock()),
        patch.object(ea, "collection_analysis_dict", {"data_collection": None}),
        caplog.at_level(logging.INFO, logger="xrpd_toolbox.i15_1.eiger_analysis"),
    ):
        with pytest.raises(AttributeError):
            ea.run_eiger_analysis(nxs)
