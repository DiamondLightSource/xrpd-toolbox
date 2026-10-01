import logging
from enum import StrEnum
from pathlib import Path

from pyFAI.calibrant import get_calibrant
from pyFAI.detectors import detector_factory

from xrpd_toolbox.i15_1.custom_calibrants import load_wb_calibrant
from xrpd_toolbox.i15_1.eiger_500k import EigerDataLoader
from xrpd_toolbox.i15_1.eiger_pyfai import (
    GONIOMETER_SAVE_NAME,
    PYFAI_DETECTOR_NAME,
    _load_goniometer,
    build_and_save_goniometer,
    integrate_with_goniometer,
    mask_edges,
)
from xrpd_toolbox.plotting import DataPlot, FittedDataPlot
from xrpd_toolbox.utils.pdfcurl import send_xy_to_pdfcurl
from xrpd_toolbox.utils.utils import (
    processed_directory_and_filename,
    wait_for_finished_file,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)

DEFAULT_NPT = 3000
DEFAULT_DETECTOR_DISTANCE_M = 0.25  # 250 mm - as defined in cad design. Actually ~0.252
# roughly the centre of of beam in frame where beam is head on
DEFAULT_BEAM_CENTRE_PX = (250.0, 470.0)


def high_q_helper(beam_energy: float, tth_angle: float):

    from xrpd_toolbox.utils.unit_conversion import (
        beam_energy_to_wavelength,
        two_theta_to_q,
    )

    wavelength = beam_energy_to_wavelength(beam_energy)
    q = two_theta_to_q(tth_angle, wavelength)
    return q


class CollectionType(StrEnum):
    data_collection = "Data Collection"
    centring = "Centring"
    air = "Air"
    empty = "Empty Capillary"
    calibrant = "Standard Sample"


calibrant_lookup: dict[str, str] = {"Silicon": "Si", "Tungsten": "W"}


def get_calibation_fit_images(goniometer_model_filepath: str | Path) -> list[Path]:
    """This is a helper to facilitate workflows display the images as an artefact"""

    calibration_fit_folder = Path(goniometer_model_filepath).parent / "calibration_fits"

    return list(calibration_fit_folder.glob("*.png"))


def do_eiger_goniometer_calibration(
    nexus_filepath: str | Path,
    calibrant_name: str | None = None,
    plot_fits: bool = True,
    show_plots: bool = False,
    npt: int | None = None,
    initial_dist_m: float | None = None,
    max_rings: list[int] = [3, 5, 5, 5, 7, 7, 9, 11, 15, 17, 32],  # noqa - we don't modify this within the func
    do_reduction: bool = True,
    use_frames: int | list[int] | slice | None = None,
):
    """Calibrate the goniometer from a calibrant scan, then reduce that scan.

    plot_fits saves the fit at each angle to processed/calibration_fits,
    show_plots opens them in interactive mode
    """

    logger.info(f"Running {do_eiger_goniometer_calibration.__name__}")

    eiger_data = EigerDataLoader(nexus_filepath)

    if calibrant_name is None:
        calibrant_name = eiger_data.get_calibrant()
        assert calibrant_name is not None

    calibrant_name = calibrant_lookup.get(calibrant_name) or calibrant_name

    if calibrant_name is None:
        raise Exception(f"Calibration  {calibrant_name} is not in calibrant_lookup")

    if calibrant_name == "W":
        calibrant = load_wb_calibrant(wavelength=eiger_data.get_wavelength_in_m())
    else:
        calibrant = get_calibrant(
            calibrant_name=calibrant_name, wavelength=eiger_data.get_wavelength_in_m()
        )

    unique_positions = eiger_data.get_unique_tth_positions()

    # not normalised: a bad i0 shouldn't stop a calibration
    summed_and_masked_frames = eiger_data.get_frames(mask=True, normalise=False)

    if use_frames is not None:
        unique_positions = unique_positions[use_frames]
        summed_and_masked_frames = summed_and_masked_frames[use_frames]

    nexus_filepath = Path(nexus_filepath)

    output_dir, _ = processed_directory_and_filename(
        nexus_filepath, nest_by_filename=False
    )

    goniometer_model_json, metadata_json = build_and_save_goniometer(
        nexus_filepath=nexus_filepath,
        images=summed_and_masked_frames,
        angles=unique_positions,
        wavelength_in_angstrom=eiger_data.wavelength,
        calibrant=calibrant,
        initial_dist_m=initial_dist_m or DEFAULT_DETECTOR_DISTANCE_M,
        output_dir=output_dir,
        max_rings=max_rings,
        pts_per_deg=1.0,
        unit="2th_deg",
        npt=npt or DEFAULT_NPT,
        detector=detector_factory(PYFAI_DETECTOR_NAME),
        plot_fits=plot_fits,
        show_plots=show_plots,
        initial_beam_centre_px=DEFAULT_BEAM_CENTRE_PX,
    )

    if do_reduction:
        tth_calibrant_peaks = calibrant.get_peaks()

        do_eiger_data_reduction(
            nexus_filepath,
            known_peak_markers=tth_calibrant_peaks,
            goniometer_filepath=goniometer_model_json,
            data_type="calibration",
        )

    logger.info(
        f"Goniometer saved to: {goniometer_model_json}, metadata saved to: {metadata_json}"  # noqa
    )

    return goniometer_model_json, metadata_json


def get_goniometer_cal_filepath(nexus_filepath: str) -> Path:
    """The most recent goniometer calibration saved in the processed folder."""
    goniometer_dir, _ = processed_directory_and_filename(
        nexus_filepath, nest_by_filename=False
    )

    goniometer_models = list(Path(goniometer_dir).glob(f"*{GONIOMETER_SAVE_NAME}"))
    if not goniometer_models:
        raise FileNotFoundError(f"No goniometer calibration found in {goniometer_dir}")

    # gets newest file based on when it's written
    return max(goniometer_models, key=lambda path: path.stat().st_mtime)


def get_i15_1_polarisation_factor(energy_kev: float):
    # calculated by SHADOW by John Sutter in 2017
    if abs(energy_kev - 40) < 1:
        # if 40kev
        return 0.9177
    elif abs(energy_kev - 65.3) < 1:
        return 0.9393
    elif abs(energy_kev - 76.6) < 1:
        return 0.9455
    else:
        raise AttributeError(f"No known polarisation factor for {energy_kev=} ")


def do_eiger_data_reduction(
    nexus_filepath: str | Path,
    apply_absorption_correction: bool = True,
    apply_azimuthal_mask: bool = True,
    edge_mask_width: tuple[int, int] | None = (10, 10),
    polarization_factor: float | None = None,
    output_xy_filepath: str | Path | None = None,
    goniometer_filepath: str | Path | None = None,
    known_peak_markers: list[float] | None = None,
    data_type: str = "pxrd",
    publish: bool = True,
    save_xye: bool = False,
    save_in_q: bool = False,
) -> Path:
    """Reduce a scan to an .xy file with the saved goniometer."""

    logger.info(f"Running {do_eiger_data_reduction.__name__}")

    eiger_data = EigerDataLoader(nexus_filepath)
    nexus_filepath = Path(nexus_filepath)

    summed_and_normalised_frames = eiger_data.get_frames(mask=True, normalise=True)

    unique_positions = eiger_data.get_unique_tth_positions()
    mask = eiger_data.get_mask()
    energy_kev = eiger_data.energy_kev

    if polarization_factor is None:
        polarization_factor = get_i15_1_polarisation_factor(energy_kev=energy_kev)

    if edge_mask_width is not None and mask is not None:
        edge_mask = mask_edges(
            detector_shape=summed_and_normalised_frames[0].shape,
            mask_width=edge_mask_width,
        )

        mask = edge_mask + mask  # masks the edges of the detector

    processed_dir, file_name = processed_directory_and_filename(nexus_filepath)

    if goniometer_filepath is not None:
        goniometer_model = _load_goniometer(
            goniometer_filepath=Path(goniometer_filepath)
        )
    else:
        goniometer_model = eiger_data.get_goniometer_calibration()

    if goniometer_model is None:
        goniometer_filepath = get_goniometer_cal_filepath(
            nexus_filepath=str(nexus_filepath)
        )
        goniometer_model = _load_goniometer(
            goniometer_filepath=Path(goniometer_filepath)
        )

    if output_xy_filepath is None:
        output_xy_filepath = Path(processed_dir) / (file_name + "_fastcs_eiger.xy")

    output_xy_filepath = integrate_with_goniometer(
        images=summed_and_normalised_frames,
        goniometer=goniometer_model,
        positions=unique_positions,
        mask=mask,
        output_xy_filepath=output_xy_filepath,
        npt=DEFAULT_NPT,
        polarization_factor=polarization_factor,
        apply_absorption_correction=apply_absorption_correction,
        apply_azimuthal_mask=apply_azimuthal_mask,
        save_xye=save_xye,
        save_in_q=save_in_q,
    )

    if publish:
        try:
            data_plot = DataPlot.from_csv(output_xy_filepath)
            data_plot.x_label = "2θ (deg)"
            data_plot.data_type = data_type

            if known_peak_markers is not None:
                data_plot = FittedDataPlot(
                    **data_plot.model_dump(),
                    calc=data_plot.y,
                    markers=list(known_peak_markers),
                )

            data_plot.publish(beamline="i15-1")

        except Exception as e:
            logger.error(e)

    return output_xy_filepath


def _get_background_xy_filepath(eiger_data: EigerDataLoader) -> str | None:
    """Finds the empty capillary background xy in processed/, reducing it from
    its nexus file if it hasn't been reduced yet. None if there's no background."""

    try:
        background_nexus_filepath = eiger_data.get_sample_environment_scan_filepath()
        bg_processed_dir, bg_file_name = processed_directory_and_filename(
            background_nexus_filepath
        )
        background_file_xy = Path(bg_processed_dir) / (
            bg_file_name + "_fastcs_eiger.xy"
        )

        if not Path(background_nexus_filepath).exists():
            raise FileNotFoundError(f"{background_nexus_filepath} does not exist")

        if not background_file_xy.exists():
            do_eiger_data_reduction(
                nexus_filepath=background_nexus_filepath,
                output_xy_filepath=background_file_xy,
            )

        return str(background_file_xy)

    except Exception as e:
        logger.error(f"No background used for pdf conversion: {e}")
        return None


def do_eiger_data_reduction_and_send_xy_to_pdfcurl(
    nexus_filepath: str | Path,
    output_xy_filepath: str | Path | None = None,
) -> Path:

    output_xy_filepath = do_eiger_data_reduction(
        nexus_filepath=nexus_filepath, output_xy_filepath=output_xy_filepath
    )

    eiger_data = EigerDataLoader(nexus_filepath)
    background_file_xy = _get_background_xy_filepath(eiger_data)

    try:
        logger.info("Sending xy to pdfcurl (pdfgetx3)")
        response_from_pdfcurl = send_xy_to_pdfcurl(
            xy_filepath=str(output_xy_filepath),
            composition=eiger_data.get_composition(),
            wavelength=eiger_data.get_wavelength(),
            background_file=background_file_xy,
        )
        logger.info(response_from_pdfcurl)

    except Exception as e:
        logger.error(e)

    return output_xy_filepath


def run_eiger_analysis(nexus_filepath: str | Path):

    wait_for_finished_file(nexus_filepath)

    logger.info(f"{nexus_filepath=}")

    eiger_data = EigerDataLoader(nexus_filepath)
    plan_name = eiger_data.get_plan_name()
    scan_type = eiger_data.get_scan_type()
    logger.info(f"{nexus_filepath=} {plan_name=} {scan_type=}")

    if scan_type == CollectionType.centring:
        logger.info(f"Nothing to do for {scan_type}. HeliotrAPI is doing it")
    elif scan_type == CollectionType.air:
        do_eiger_data_reduction(nexus_filepath=nexus_filepath)
    elif scan_type == CollectionType.empty:
        do_eiger_data_reduction(nexus_filepath=nexus_filepath)
    elif scan_type == CollectionType.calibrant:
        do_eiger_goniometer_calibration(nexus_filepath=nexus_filepath)
    elif scan_type == CollectionType.data_collection:
        # If it's actually a datacollections also send it to pdfcurl too
        do_eiger_data_reduction_and_send_xy_to_pdfcurl(nexus_filepath=nexus_filepath)
    else:
        error = f"No analysis for bluesky plan: {plan_name} & scan type: {scan_type}"
        logger.error(error)
        raise RuntimeError(error)


# if __name__ == "__main__":

#     def plot_final_data(output_xy: str | Path, title: str = ""):

#         import matplotlib.pyplot as plt
#         import numpy as np

#         x, y = np.genfromtxt(str(output_xy), unpack=True)

#         plt.title(title)
#         plt.plot(x, y)
#         plt.show()

#     # nexus_filepath = "/workspaces/outputs/i15-1/i15-1-98680.nxs"  # first si calib

#     # nexus_filepath = "/workspaces/outputs/i15-1/i15-1-98784.nxs"  # longer si calib

#     # nexus_filepath = "/workspaces/outputs/i15-1/i15-1-99380.nxs"

#     nexus_filepath = "/workspaces/outputs/i15-1/i15-1-98779.nxs"  # WB for calibration

#     # goniometer_cal_filepath, meatadata_filepath = do_eiger_goniometer_calibration(
#     #     cal_nexus_filepath,
#     #     calibrant_name="W",
#     #     plot_fits=True,
#     #     show_plots=False,
#     #     max_rings=[3, 5, 5, 5, 7, 7, 9, 11, 15, 17, 32, 64],
#     # )

#     #     nexus_filepath = "/workspaces/outputs/i15-1/i15-1-99340.nxs"  # test example

#     output_xy = do_eiger_data_reduction(
#         nexus_filepath,
#         edge_mask_width=(10, 10),
#         apply_absorption_correction=True,
#         apply_azimuthal_mask=True,
#         publish=False,
#     )

#     plot_final_data(output_xy, title="polarisation: 0.95")
