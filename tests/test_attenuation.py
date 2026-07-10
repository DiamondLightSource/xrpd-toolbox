import numpy as np
import pytest

from xrpd_toolbox.i15_1.attenuation import _calculate_best_attenuation


def test_calculate_best_attenuation():
    # --- Scalar example ---
    best_atten = _calculate_best_attenuation(
        current_attenuation=0.95,
        max_value_per_frame=50000,
        exposure_time=0.1,
        expected_max_exposure_time=0.5,
    )

    assert pytest.approx(best_atten, rel=1e-1) == 0.695


def test_calculate_best_attenuation_array():
    # --- Array example: several reference frames at different settings ---
    best_atten = _calculate_best_attenuation(
        current_attenuation=[0.95, 0.90, 0.99],
        max_value_per_frame=[50000, 48000, 12000],
        exposure_time=[0.1, 0.1, 0.1],
        expected_max_exposure_time=0.5,
    )

    assert isinstance(best_atten, np.ndarray)

    best_atten_float = np.max(best_atten)  # average over frames

    assert pytest.approx(best_atten_float, rel=1e-1) == 0.7467


def test_calculate_best_attenuation_per_frame():
    # --- Same array example, but get one recommendation per frame ---
    best_atten = _calculate_best_attenuation(
        current_attenuation=[0.95, 0.90, 0.99],
        max_value_per_frame=[50000, 48000, 12000],
        exposure_time=[0.1, 0.1, 0.1],
        expected_max_exposure_time=0.5,
    )

    assert isinstance(best_atten, np.ndarray)

    best_atten_float = np.max(best_atten)  # average over frames

    assert pytest.approx(best_atten_float, rel=1e-1) == 0.7467
