from collections.abc import Callable
from pathlib import Path

import numpy as np

from xrpd_toolbox.i15_1.eiger_500k import EigerDataLoader
from xrpd_toolbox.i15_1.eiger_pyfai import integrate_with_goniometer
from xrpd_toolbox.utils.utils import wait_for_finished_file


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


def do_eiger_calibration(nexus_filepath: str | Path):

    pass

    # do_eiger_data_reduction(nexus_filepath) #then reduce the data we just collected


def do_eiger_data_reduction(nexus_filepath: str | Path):
    """This does the eiger data reduction at the end of scan.

    Assumes that the nexus file is a data_collection with N positions"""

    eiger_data = EigerDataLoader(nexus_filepath)
    nexus_filepath = Path(nexus_filepath)

    slices_of_data = unique_slices(eiger_data.positions)

    summed_and_normalised_frames = []

    for slice in slices_of_data:
        frames_with_position = eiger_data.get_data(slice)
        durations_for_frames = eiger_data.durations[slice]

        summed_and_normalised_frames_at_tth_position = (
            np.sum(frames_with_position, axis=-1) * durations_for_frames
        )

        summed_and_normalised_frames.append(
            summed_and_normalised_frames_at_tth_position
        )

    summed_and_normalised_frames = np.array(summed_and_normalised_frames)

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
    )

    return output_xy_filepath


collection_analysis_dict: dict[str, Callable] = {
    "data_collection": do_eiger_data_reduction,
    "calibration_collection": do_eiger_calibration,
}


def run_eiger_analysis(nexus_filepath: str | Path):

    wait_for_finished_file(nexus_filepath)

    eiger_data = EigerDataLoader(nexus_filepath)
    plan_name = eiger_data.get_plan_name()

    print(plan_name)

    plan_to_run = collection_analysis_dict[plan_name]

    if plan_to_run is not None:
        plan_to_run(nexus_filepath)


if __name__ == "__main__":
    nexus_filepath = "/workspaces/outputs/i15-1/i15-1-98478.nxs"
    run_eiger_analysis(nexus_filepath)
