import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np

from xrpd_toolbox.i15_1.eiger_500k import EigerDataLoader
from xrpd_toolbox.i15_1.eiger_pyfai import (
    build_and_save_goniometer,
    integrate_with_goniometer,
)
from xrpd_toolbox.utils.pdfcurl import send_xy_to_pdfcurl
from xrpd_toolbox.utils.utils import wait_for_finished_file

logger = logging.getLogger(__name__)

DEFAULT_NPT = 3000


def unique_slices(arr: np.ndarray):
    """Retturns slices at which the all the values for the input array are the same
    assumes that the input array is sorted and only increases/decreases

    if it's not will have to use np.argwhere - but that isn't this functons

    """
    arr = np.asarray(arr)
    _, start_idx = np.unique(arr, return_index=True)
    start_idx = np.sort(start_idx)
    end_idx = np.append(start_idx[1:], len(arr))
    return [slice(s, e) for s, e in zip(start_idx, end_idx, strict=True)]


def sum_unique_two_theta_positions_and_normalise(eiger_data: EigerDataLoader):

    slices_of_data = unique_slices(eiger_data.positions)

    assert len(slices_of_data) == np.unique(eiger_data.positions)

    summed_and_normalised_frames = []

    for slice in slices_of_data:
        frames_with_position = eiger_data.get_data(slice)
        durations_for_frames = eiger_data.durations[slice]

        summed_frames_at_tth_position = np.sum(frames_with_position, axis=-1)

        assert summed_frames_at_tth_position.ndim > 1

        summed_and_normalised_frames_at_tth_position = (
            summed_frames_at_tth_position * durations_for_frames
        )

        summed_and_normalised_frames.append(
            summed_and_normalised_frames_at_tth_position
        )

    summed_and_normalised_frames = np.array(summed_and_normalised_frames)

    return summed_and_normalised_frames


def apply_mask(image_frames: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """applys a mask to all frames in the image"""

    masked_image_frames = [image * mask for image in image_frames]

    masked_image_frames = np.array(masked_image_frames)

    return masked_image_frames


def do_eiger_calibration(nexus_filepath: str | Path):

    distance_in_meters = 0.25  # 250 mm

    eiger_data = EigerDataLoader(nexus_filepath)

    calibrant = eiger_data.get_calibrant()

    if calibrant is None:
        raise Exception("Calibraation is not in Nexus file")

    summed_and_normalised_frames = sum_unique_two_theta_positions_and_normalise(
        eiger_data=eiger_data
    )
    unique_positions = np.unique(eiger_data.positions)
    mask = eiger_data.get_mask()

    summed_normalised_and_masked_frames = apply_mask(
        image_frames=summed_and_normalised_frames, mask=mask
    )

    nexus_filepath = Path(nexus_filepath)

    goniometer_model_json, metadata_json = build_and_save_goniometer(
        nexus_filepath=nexus_filepath,
        images=summed_normalised_and_masked_frames,
        angles=unique_positions,
        wavelength_in_angstrom=eiger_data.wavelength,
        calibrant_name=calibrant,
        initial_dist_m=distance_in_meters,
        output_dir=str(nexus_filepath.parent),
        max_rings=[5, 5, 5, 7, 7, 9, 11, 15, 17],
        pts_per_deg=1.0,
        unit="2th_deg",
        npt=DEFAULT_NPT,
    )

    return goniometer_model_json, metadata_json

    # do_eiger_data_reduction(nexus_filepath) then reduce the data we just collect
    # - do this in workflow?


def do_eiger_data_reduction(nexus_filepath: str | Path) -> Path:
    """This does the eiger data reduction at the end of scan.

    Assumes that the nexus file is a data_collection with N positions

    returns path to xy file
    """

    eiger_data = EigerDataLoader(nexus_filepath)
    nexus_filepath = Path(nexus_filepath)

    summed_and_normalised_frames = sum_unique_two_theta_positions_and_normalise(
        eiger_data
    )

    unique_positions = np.unique(eiger_data.positions)
    mask = eiger_data.get_mask()

    output_xy_filepath = nexus_filepath.parent / (nexus_filepath.stem + "_eiger.xy")
    goniometer_dir = str(nexus_filepath.parent)

    output_xy_filepath = integrate_with_goniometer(
        images=summed_and_normalised_frames,
        goniometer_dir=goniometer_dir,
        positions=unique_positions,
        mask=mask,
        output_xy_filepath=output_xy_filepath,
        npt=DEFAULT_NPT,
    )

    return output_xy_filepath


def do_eiger_data_reduction_and_send_xy_to_pdfcurl(
    nexus_filepath: str | Path,
) -> Path:

    eiger_data = EigerDataLoader(nexus_filepath)
    composition = eiger_data.get_composition()
    wavelength = eiger_data.get_wavelength()

    output_xy_filepath = do_eiger_data_reduction(nexus_filepath)

    response_from_pdfcurl = send_xy_to_pdfcurl(
        xy_filepath=str(output_xy_filepath),
        composition=composition,
        wavelength=wavelength,
    )

    logger.info(response_from_pdfcurl)

    return output_xy_filepath


collection_analysis_dict: dict[str, Callable] = {
    "data_collection": do_eiger_data_reduction,
    "calibration_collection": do_eiger_calibration,
}


def run_eiger_analysis(nexus_filepath: str | Path):

    wait_for_finished_file(nexus_filepath)

    eiger_data = EigerDataLoader(nexus_filepath)
    plan_name = eiger_data.get_plan_name()

    analysis_to_run = collection_analysis_dict[plan_name]

    logger.info(f"{plan_name=}, {analysis_to_run.__name__=}")

    if analysis_to_run is None:
        logger.error(f"No analysis plan exists for bluesky plan: {plan_name}")
        raise RuntimeError(f"No analysis plan exists for bluesky plan: {plan_name}")

    analysis_to_run(nexus_filepath)


if __name__ == "__main__":  # pragma: no cover - manual/interactive smoke test
    nexus_filepath = "/workspaces/outputs/i15-1/i15-1-98478.nxs"
    run_eiger_analysis(nexus_filepath)
