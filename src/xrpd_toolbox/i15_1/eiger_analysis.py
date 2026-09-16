import logging
from enum import StrEnum
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
DEFAULT_DETECTOR_DISTANCE_M = 0.25  # 250 mm


class CollectionType(StrEnum):
    data_collection = "Data Collection"
    centring = "Centring"
    air = "Air"
    empty = "Empty Capillary"
    calibrant = "Standard Sample"


calibrant_lookup: dict[str, str] = {"Silicon": "Si"}


def do_eiger_calibration(nexus_filepath: str | Path):

    eiger_data = EigerDataLoader(nexus_filepath)

    calibrant_name = eiger_data.get_calibrant()

    if calibrant_name is None:
        raise Exception("Calibration is not in Nexus file")

    calibrant = calibrant_lookup.get(calibrant_name)

    if calibrant is None:
        raise Exception(f"Calibration  {calibrant_name} is not in calibrant_lookup")

    unique_positions = np.unique(eiger_data.positions)

    summed_normalised_and_masked_frames = (
        eiger_data.get_summed_normalised_and_masked_frames()
    )
    nexus_filepath = Path(nexus_filepath)

    goniometer_model_json, metadata_json = build_and_save_goniometer(
        nexus_filepath=nexus_filepath,
        images=summed_normalised_and_masked_frames,
        angles=unique_positions,
        wavelength_in_angstrom=eiger_data.wavelength,
        calibrant_name=calibrant,
        initial_dist_m=DEFAULT_DETECTOR_DISTANCE_M,
        output_dir=str(nexus_filepath.parent),
        max_rings=[5, 5, 5, 7, 7, 9, 11, 15, 17],
        pts_per_deg=1.0,
        unit="2th_deg",
        npt=DEFAULT_NPT,
    )

    # do_eiger_data_reduction(nexus_filepath) then reduce the data we just collect
    # - do this in workflow?

    return goniometer_model_json, metadata_json


def do_eiger_data_reduction(
    nexus_filepath: str | Path, output_xy_filepath: str | Path | None = None
) -> Path:
    """This does the eiger data reduction at the end of scan.

    Assumes that the nexus file is a data_collection with N positions

    returns path to xy file
    """

    eiger_data = EigerDataLoader(nexus_filepath)
    nexus_filepath = Path(nexus_filepath)

    summed_and_normalised_frames = eiger_data.get_summed_and_normalised_frames()

    unique_positions = np.unique(eiger_data.positions)
    mask = eiger_data.get_mask()

    if output_xy_filepath is None:
        output_xy_filepath = (
            nexus_filepath.parent
            / "processed"
            / (nexus_filepath.stem + "_fastcs_eiger.xy")
        )

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
    output_xy_filepath: str | Path | None = None,
) -> Path:

    eiger_data = EigerDataLoader(nexus_filepath)
    composition = eiger_data.get_composition()
    wavelength = eiger_data.get_wavelength()
    sample_environment_filepath = eiger_data.get_sample_environment_scan_filepath()
    background_file_xy = Path(sample_environment_filepath.replace(".nxs", ".xy"))

    if not background_file_xy.exists():
        try:
            background_file_xy = do_eiger_data_reduction(
                sample_environment_filepath, background_file_xy
            )
        except Exception as e:
            logger.error(f"No background xy present, no background nxs present: {e}")
            logger.error("No background used for pdf conversion")
            background_file_xy = None

    output_xy_filepath = do_eiger_data_reduction(nexus_filepath, output_xy_filepath)

    response_from_pdfcurl = send_xy_to_pdfcurl(
        xy_filepath=str(output_xy_filepath),
        composition=composition,
        wavelength=wavelength,
        background_file=str(background_file_xy),
    )

    logger.info(response_from_pdfcurl)

    return output_xy_filepath


def run_eiger_analysis(nexus_filepath: str | Path):

    wait_for_finished_file(nexus_filepath)

    eiger_data = EigerDataLoader(nexus_filepath)
    plan_name = eiger_data.get_plan_name()
    scan_type = eiger_data.get_scan_type()

    if scan_type == CollectionType.centring:
        logger.info(f"Nothing to do for {scan_type}. HeliotrAPI is doing it")

    elif scan_type == CollectionType.air:
        logger.info(f"Running {do_eiger_data_reduction.__name__} for {scan_type}")
        do_eiger_data_reduction(nexus_filepath)

    elif scan_type == CollectionType.empty:
        logger.info(f"Running {do_eiger_data_reduction.__name__} for {scan_type}")
        do_eiger_data_reduction(nexus_filepath)

    elif scan_type == CollectionType.calibrant:
        logger.info(f"Running {do_eiger_calibration.__name__} for {scan_type}")
        do_eiger_calibration(nexus_filepath)

    if scan_type == CollectionType.data_collection:
        # If it's actually a datacollections also send it to pdfcurl too
        logger.info(
            f"Running {do_eiger_data_reduction_and_send_xy_to_pdfcurl.__name__} for {scan_type}"  # noqa
        )
        do_eiger_data_reduction_and_send_xy_to_pdfcurl(nexus_filepath)

    else:
        error = f"No analysis for bluesky plan: {plan_name} & scan type: {scan_type}"
        logger.error(error)
        raise RuntimeError(error)


if __name__ == "__main__":  # pragma: no cover - manual/interactive smoke test
    nexus_filepath = "/workspaces/outputs/i15-1/i15-1-98523.nxs"

    import matplotlib.pyplot as plt

    eiger_data = EigerDataLoader(nexus_filepath)

    print(eiger_data.get_scan_type())

    frames = eiger_data.get_summed_and_normalised_frames()

    for frame in frames:
        plt.imshow(frame)
        plt.show()

    # run_eiger_analysis(nexus_filepath)
