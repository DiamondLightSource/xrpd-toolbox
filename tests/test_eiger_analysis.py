"""Tests for xrpd_toolbox.i15_1.eiger_analysis.

do_eiger_calibration/do_eiger_data_reduction are tested with the
lower-level EigerDataLoader helpers mocked out so their own orchestration
logic can be verified independently.
"""

import numpy as np

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
