import logging
from enum import StrEnum
from pathlib import Path

from pyFAI.detectors import detector_factory

from xrpd_toolbox.i15_1.eiger_500k import EigerDataLoader
from xrpd_toolbox.i15_1.eiger_pyfai import (
    GONIOMETER_SAVE_NAME,
    build_and_save_goniometer,
    integrate_with_goniometer,
)
from xrpd_toolbox.plotting import DataPlot
from xrpd_toolbox.utils.pdfcurl import send_xy_to_pdfcurl
from xrpd_toolbox.utils.utils import (
    processed_directory_and_filename,
    wait_for_finished_file,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DEFAULT_NPT = 3000
DEFAULT_DETECTOR_DISTANCE_M = 0.25  # 250 mm
# (row, col) at 0° from i15-1-98680 - the detector centre is too far off for
# the rings to be indexed correctly. Update if the detector moves.
DEFAULT_BEAM_CENTRE_PX = (250.0, 470.0)


def high_q_helper(beam_energg: float, tth_angle: float):

    from xrpd_toolbox.utils.unit_conversion import (
        beam_energy_to_wavelength,
        two_theta_to_q,
    )

    wavelength = beam_energy_to_wavelength(beam_energg)
    q = two_theta_to_q(tth_angle, wavelength)
    return q


class CollectionType(StrEnum):
    data_collection = "Data Collection"
    centring = "Centring"
    air = "Air"
    empty = "Empty Capillary"
    calibrant = "Standard Sample"


calibrant_lookup: dict[str, str] = {"Silicon": "Si"}

PYFAI_DETECTOR_NAME = "Eiger2CdTe_500k"


def do_eiger_goniometer_calibration(
    nexus_filepath: str | Path,
    calibrant_name: str | None = None,
    plot_fits: bool = False,
    show_plots: bool = False,
):
    """Calibrate the goniometer from a calibrant scan, then reduce that scan.

    plot_fits saves the fit at each angle to processed/calibration_fits,
    show_plots opens them.
    """

    eiger_data = EigerDataLoader(nexus_filepath)

    if calibrant_name is None:
        calibrant_name = eiger_data.get_calibrant()
        assert calibrant_name is not None

    calibrant = calibrant_lookup.get(calibrant_name) or calibrant_name

    if calibrant is None:
        raise Exception(f"Calibration  {calibrant_name} is not in calibrant_lookup")

    unique_positions = eiger_data.get_unique_tth_positions()

    # not normalised: a bad i0 shouldn't stop a calibration
    summed_and_masked_frames = eiger_data.get_summed_and_masked_frames()
    nexus_filepath = Path(nexus_filepath)

    output_dir, _ = processed_directory_and_filename(
        nexus_filepath, nest_by_filename=False
    )

    goniometer_model_json, metadata_json = build_and_save_goniometer(
        nexus_filepath=nexus_filepath,
        images=summed_and_masked_frames,
        angles=unique_positions,
        wavelength_in_angstrom=eiger_data.wavelength,
        calibrant_name=calibrant,
        initial_dist_m=DEFAULT_DETECTOR_DISTANCE_M,
        output_dir=output_dir,
        max_rings=[3, 5, 5, 5, 7, 7, 9, 11, 15, 17],
        pts_per_deg=1.0,
        unit="2th_deg",
        npt=DEFAULT_NPT,
        detector=detector_factory(PYFAI_DETECTOR_NAME),
        plot_fits=plot_fits,
        show_plots=show_plots,
        initial_beam_centre_px=DEFAULT_BEAM_CENTRE_PX,
    )

    do_eiger_data_reduction(nexus_filepath)

    return goniometer_model_json, metadata_json


def do_eiger_data_reduction(
    nexus_filepath: str | Path,
    output_xy_filepath: str | Path | None = None,
    goniometer_filepath: str | Path | None = None,
) -> Path:
    """Reduce a scan to an .xy file with the saved goniometer."""

    eiger_data = EigerDataLoader(nexus_filepath)
    nexus_filepath = Path(nexus_filepath)

    summed_and_normalised_frames = eiger_data.get_summed_and_normalised_frames()

    unique_positions = eiger_data.get_unique_tth_positions()
    mask = eiger_data.get_mask()

    processed_dir, file_name = processed_directory_and_filename(nexus_filepath)

    if goniometer_filepath is None:
        goniometer_dir, _ = processed_directory_and_filename(
            nexus_filepath, nest_by_filename=False
        )

        goniometer_filepath = Path(goniometer_dir) / GONIOMETER_SAVE_NAME

    if output_xy_filepath is None:
        output_xy_filepath = Path(processed_dir) / (file_name + "_fastcs_eiger.xy")

    output_xy_filepath = integrate_with_goniometer(
        images=summed_and_normalised_frames,
        goniometer_filepath=goniometer_filepath,
        positions=unique_positions,
        mask=mask,
        output_xy_filepath=output_xy_filepath,
        npt=DEFAULT_NPT,
    )

    try:
        data_plot = DataPlot.from_csv(output_xy_filepath)
        data_plot.x_label = "2θ (deg)"
        data_plot.data_type = "pxrd"
        data_plot.publish(beamline="i15-1")
    except Exception as e:
        logger.error(e)

    return output_xy_filepath


def do_eiger_data_reduction_and_send_xy_to_pdfcurl(
    nexus_filepath: str | Path,
    output_xy_filepath: str | Path | None = None,
) -> Path:

    eiger_data = EigerDataLoader(nexus_filepath)
    composition = eiger_data.get_composition()
    wavelength = eiger_data.get_wavelength()
    sample_environment_filepath = eiger_data.get_sample_environment_scan_filepath()
    bg_processed_dir, bg_file_name = processed_directory_and_filename(
        sample_environment_filepath
    )
    background_file_xy = Path(bg_processed_dir) / (bg_file_name + "_fastcs_eiger.xy")

    if not background_file_xy.exists():
        try:
            background_file_xy = do_eiger_data_reduction(
                nexus_filepath=sample_environment_filepath,
                output_xy_filepath=background_file_xy,
            )
        except Exception as e:
            logger.error(f"No background xy present, no background nxs present: {e}")
            logger.error("No background used for pdf conversion")
            background_file_xy = None

    output_xy_filepath = do_eiger_data_reduction(
        nexus_filepath=nexus_filepath, output_xy_filepath=output_xy_filepath
    )

    try:
        response_from_pdfcurl = send_xy_to_pdfcurl(
            xy_filepath=str(output_xy_filepath),
            composition=composition,
            wavelength=wavelength,
            background_file=str(background_file_xy),
        )

        logger.info(response_from_pdfcurl)

    except Exception as e:
        logger.error(e)

    return output_xy_filepath


def run_eiger_analysis(nexus_filepath: str | Path):

    wait_for_finished_file(nexus_filepath)

    eiger_data = EigerDataLoader(nexus_filepath)
    plan_name = eiger_data.get_plan_name()
    scan_type = eiger_data.get_plan_type()

    if scan_type == CollectionType.centring:
        logger.info(f"Nothing to do for {scan_type}. HeliotrAPI is doing it")

    elif scan_type == CollectionType.air:
        logger.info(f"Running {do_eiger_data_reduction.__name__} for {scan_type}")
        do_eiger_data_reduction(nexus_filepath=nexus_filepath)

    elif scan_type == CollectionType.empty:
        logger.info(f"Running {do_eiger_data_reduction.__name__} for {scan_type}")
        do_eiger_data_reduction(nexus_filepath=nexus_filepath)

    elif scan_type == CollectionType.calibrant:
        logger.info(
            f"Running {do_eiger_goniometer_calibration.__name__} for {scan_type}"
        )
        do_eiger_goniometer_calibration(nexus_filepath=nexus_filepath)

    elif scan_type == CollectionType.data_collection:
        # If it's actually a datacollections also send it to pdfcurl too
        logger.info(
            f"Running {do_eiger_data_reduction_and_send_xy_to_pdfcurl.__name__} for {scan_type}"  # noqa
        )
        do_eiger_data_reduction_and_send_xy_to_pdfcurl(nexus_filepath=nexus_filepath)

    else:
        error = f"No analysis for bluesky plan: {plan_name} & scan type: {scan_type}"
        logger.error(error)
        raise RuntimeError(error)


if __name__ == "__main__":
    nexus_filepath = "/workspaces/outputs/i15-1/i15-1-98680.nxs"
    goniometer_filepath = Path(
        "/workspaces/outputs/i15-1/processed/eiger_goniometer_calibration.json"
    )

    # print(high_q_helper(40, 80))
    # print(high_q_helper(76.76, 80))
    # quit()

    eiger_data = EigerDataLoader(nexus_filepath)

    # mask = eiger_data.get_mask(as_nan=False)

    frames = eiger_data.get_summed_and_normalised_frames()

    # for frame, tth in zip(frames, eiger_data.get_unique_tth_positions(), strict=True):
    #     frame[mask] = 0

    #     plt.imshow(frame * mask, cmap="viridis")

    #     np.save(f"/workspaces/outputs/i15-1/processed/i15-1-98700_{tth:.2f}.npy", frame) #noqa

    # plt.savefig(f"/workspaces/outputs/i15-1/processed/i15-1-98700_{tth}.tiff")

    logging.basicConfig(level=logging.INFO)
    gionemeter_cal = do_eiger_goniometer_calibration(
        nexus_filepath, calibrant_name="Si", plot_fits=True, show_plots=True
    )

    # output_xy_filepath = do_eiger_data_reduction_and_send_xy_to_pdfcurl(
    #     nexus_filepath
    # )  # then reduce the data we just collected

    # print(output_xy_filepath)
    # run_eiger_analysis(nexus_filepath)
