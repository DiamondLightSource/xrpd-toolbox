import numpy as np

from xrpd_toolbox.data_loader import BaseDataLoader
from xrpd_toolbox.utils.utils import (
    h5_to_array,
    wait_for_finished_file,
)

DETECTOR_MAXIMUM_COUNT_RATES = {
    "CdTe_Eiger": 1.7e9
}  # https://media.dectris.com/filer_public/d9/17/d917ef9c-46f6-4c67-9139-ae34d41d53f2/technicalspecifications_eiger2_x_cdte_500k_v190.pdf


SAFETY_MARGIN = 0.9  # 10% safety margin


def _calculate_best_attenuation(
    current_attenuation: np.ndarray | float | list[float],
    max_value_per_frame: np.ndarray | float | list[float],
    exposure_time: np.ndarray | float | list[float],
    expected_max_exposure_time: float,
    max_count_rate: float = 3.8e6,
    safety_margin: float = 0.8,
    frame_max_counts: float | None = None,
) -> float:
    """
    Best attenuation (0 = full flux, 1 = fully blocked) to avoid saturation.
    Inputs may be scalar or array-like; worst case is always used.
    """
    current_attenuation = np.asarray(current_attenuation, dtype=float)
    max_value_per_frame = np.asarray(max_value_per_frame, dtype=float)
    exposure_time = np.asarray(exposure_time, dtype=float)

    if expected_max_exposure_time <= 0:
        raise ValueError("expected_max_exposure_time must be > 0.")
    if np.any(exposure_time <= 0):
        raise ValueError("All exposure_time values must be > 0.")
    if np.any(current_attenuation >= 1):
        raise ValueError("current_attenuation must be < 1 everywhere.")

    current_transmission = 1.0 - current_attenuation
    measured_rate = max_value_per_frame / exposure_time
    full_flux_rate = np.max(measured_rate / current_transmission)  # worst case

    target_rate = max_count_rate * safety_margin
    required_transmission = target_rate / full_flux_rate

    if frame_max_counts is not None:
        target_total_counts = frame_max_counts * safety_margin
        transmission_from_well_depth = target_total_counts / (
            full_flux_rate * expected_max_exposure_time
        )
        required_transmission = min(required_transmission, transmission_from_well_depth)

    required_transmission = max(0.0, min(1.0, required_transmission))
    return 1.0 - required_transmission


def predict_best_attenuation(
    calibration_filepath: str,
    dataset_path: str,
    attenuation_path: str,
    exposure_time_path: str,
    detector: str,
    expected_max_exposure_time: float,
) -> float:
    """Tries to predict the best attenuation for a given detector
    and calibration file, and planned expected_max_exposure_time.

    Will add a 10% safety margin to the predicted value.
    The predicted value is based on the maximum count rate of the detector
    and the maximum value in all pixels per frame in the calibration file."""

    if detector not in DETECTOR_MAXIMUM_COUNT_RATES:
        raise ValueError(
            f"Detector {detector} must be one of these:"
            f" {list(DETECTOR_MAXIMUM_COUNT_RATES.keys())}"
        )

    wait_for_finished_file(calibration_filepath, timeout=600)

    detector_max_rate = DETECTOR_MAXIMUM_COUNT_RATES[detector]

    data = BaseDataLoader(filepath=calibration_filepath, dataset_path=dataset_path)
    max_value_per_frame = data.max_value_per_frame()
    current_attenuation = h5_to_array(calibration_filepath, attenuation_path)
    exposure_time = h5_to_array(calibration_filepath, exposure_time_path)

    calculate_best_attenuation = _calculate_best_attenuation(
        current_attenuation=current_attenuation,
        max_value_per_frame=max_value_per_frame,
        exposure_time=exposure_time,
        expected_max_exposure_time=expected_max_exposure_time,
        max_count_rate=detector_max_rate,
        safety_margin=SAFETY_MARGIN,
    )

    return calculate_best_attenuation


if __name__ == "__main__":
    # --- Scalar example ---
    best_atten = _calculate_best_attenuation(
        current_attenuation=0.95,
        max_value_per_frame=50000,
        exposure_time=0.1,
        expected_max_exposure_time=0.5,
    )
    print(f"Scalar case -> recommended attenuation: {best_atten:.4f}")

    # --- Array example: several reference frames at different settings ---
    atten_arr = _calculate_best_attenuation(
        current_attenuation=[0.95, 0.90, 0.99],
        max_value_per_frame=[50000, 48000, 12000],
        exposure_time=[0.1, 0.1, 0.1],
        expected_max_exposure_time=0.5,
    )
    print(
        f"Array case (worst-case reduced) -> recommended attenuation: {atten_arr:.4f}"
    )

    # --- Same array example, but get one recommendation per frame ---
    atten_per_frame = _calculate_best_attenuation(
        current_attenuation=[0.95, 0.90, 0.99],
        max_value_per_frame=[50000, 48000, 12000],
        exposure_time=[0.1, 0.1, 0.1],
        expected_max_exposure_time=0.5,
    )
    print(f"Per-frame recommended attenuations: {atten_per_frame}")
