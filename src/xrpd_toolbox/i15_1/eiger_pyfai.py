"""Eiger500K goniometer calibration and integration."""

from __future__ import annotations

import contextlib
import datetime
import io
import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from pyFAI.calibrant import Calibrant
from pyFAI.detectors import Detector, detector_factory
from pyFAI.geometry import Geometry
from pyFAI.goniometer import (
    GeometryTransformation,
    Goniometer,
    GoniometerRefinement,
    MultiGeometry,
    SingleGeometry,
)

from xrpd_toolbox.i15_1.eiger_500k import ARM_ROTATION_SIGN
from xrpd_toolbox.i15_1.eiger_goniometer_models import GEOMETRY_TRANSFORMATION
from xrpd_toolbox.utils.utils import processed_directory_and_filename

logger = logging.getLogger(__name__)


# once the beam centre is off the detector a frame can't separate these from rot1,
# so they're held at what the model fitted to the earlier frames predicts
FIX_BETWEEN_FRAMES = ["wavelength", "dist", "poni1", "poni2"]

GONIOMETER_SAVE_NAME = "eiger_goniometer_calibration.json"
METADATA_SAVE_NAME = "calibration_metadata.json"
FITS_DIR_NAME = "calibration_fits"


def calibrate_single_geometry_from_rings(
    geometry: SingleGeometry,
    rings: list[int] = [3, 5, 5, 5, 7, 7, 9, 11, 15, 17],  # noqa
    fix: list | None = None,
):
    """Extract control points for geometry and refine it."""
    if fix is None:
        fix = []

    for max_rings in rings:
        geometry.extract_cp(max_rings=max_rings)
        # pyFAI leaves an empty 1D array, which fails obscurely inside refine2
        assert geometry.geometry_refinement.data is not None
        if geometry.geometry_refinement.data.ndim != 2:
            raise ValueError(
                f"No control points found for {geometry.label} "
                f"(max_rings={max_rings}) - check the image contains "
                "positive calibrant rings and the mask is correct"
            )
        npts = len(geometry.geometry_refinement.data)
        geometry.geometry_refinement.refine2(fix=fix)
        gr = geometry.geometry_refinement
        logger.info(
            "  %s max_rings=%d: %d points, chi2=%.3g, dist/poni1/poni2/rot1-3=%s",
            geometry.label,
            max_rings,
            npts,
            gr.chi2(),
            np.round(gr.param[:6], 5),
        )

    return geometry


def _load_goniometer(goniometer_filepath: Path) -> Goniometer:
    """Load a Goniometer which has previously been saved."""

    if not goniometer_filepath.exists():
        raise FileNotFoundError(
            f"{goniometer_filepath.name} not found in {goniometer_filepath.parent}"
        )

    gonio = Goniometer.sload(str(goniometer_filepath))
    logger.info("Loaded goniometer from %s", goniometer_filepath.parent)
    return gonio


def _resolve_detector(detector: Detector | str | None) -> Detector:
    """None -> the simulation Eiger500K; a name -> pyFAI's registry detector."""
    from xrpd_toolbox.i15_1.eiger_500k import Eiger500K

    if detector is None:
        return Eiger500K()
    if isinstance(detector, str):
        return detector_factory(detector)
    return detector


def _calibrate_single_frame(
    label: str,
    image: np.ndarray,
    two_theta_deg: float,
    calibrant: Calibrant,
    initial_dist_m: float,
    max_rings: int | None | Iterable[int],
    pts_per_deg: float,
    detector: Detector | str | None = None,
    initial_beam_centre_px: tuple[float, float] | None = None,
    seed_geometry: dict[str, float] | None = None,
    fix: list[str] | None = None,
) -> SingleGeometry:
    """Extract control points and refine the geometry of one frame.

    Starts from the beam centre (default: detector centre) with any values in
    `seed_geometry` taking priority. `detector` defaults to the sim Eiger500K.
    """
    # pass an instance: SingleGeometry looks up name strings in pyFAI's own
    # registry, and "eiger500k" there is a different shaped detector
    detector = _resolve_detector(detector)
    assert detector.max_shape is not None
    if initial_beam_centre_px is None:
        rows, cols = detector.max_shape
        initial_beam_centre_px = (rows / 2, cols / 2)
    centre_row, centre_col = initial_beam_centre_px

    initial_geometry = {
        "dist": initial_dist_m,
        # the PONI moves with the arm, so the 0° beam centre works at any angle
        "poni1": centre_row * detector.pixel1,
        "poni2": centre_col * detector.pixel2,
        "rot1": ARM_ROTATION_SIGN * np.deg2rad(two_theta_deg),
        "rot2": 0.0,
        "rot3": 0.0,
        "wavelength": calibrant.wavelength,
        "detector": detector,
    }
    if seed_geometry is not None:
        initial_geometry.update(seed_geometry)

    sg = SingleGeometry(
        label=label,
        image=image,
        metadata=two_theta_deg,
        # needed by GoniometerRefinement, which calls get_position()
        pos_function=lambda two_theta: (two_theta,),
        calibrant=calibrant,
        detector=detector,
        geometry=initial_geometry,
    )

    if isinstance(max_rings, Iterable):
        sg = calibrate_single_geometry_from_rings(
            geometry=sg, rings=list(max_rings), fix=fix
        )

    else:
        sg.extract_cp(max_rings=max_rings, pts_per_deg=pts_per_deg)
        sg.geometry_refinement.refine2(fix=fix)

    assert sg.geometry_refinement.data is not None
    return sg


def _start_goniometer(
    model: GeometryTransformation,
    first: SingleGeometry,
    detector: Detector,
    wavelength_m: float,
) -> GoniometerRefinement:
    """`model` with its parameters seeded from the first frame's fit."""
    gr = first.geometry_refinement
    assert first.metadata is not None
    seeds = {
        "dist": gr.dist,
        "poni1": gr.poni1,
        "poni2": gr.poni2,
        "rot1_scale": ARM_ROTATION_SIGN,
        "rot1_offset": gr.rot1 - ARM_ROTATION_SIGN * np.deg2rad(first.metadata),
        "rot2": gr.rot2,
        "rot3": gr.rot3,
        "pitch": gr.rot2,
        "roll": gr.rot3,
    }
    # anything else (quadratic terms, yaw, sample offsets) is a correction from 0
    params = {name: seeds.get(name, 0.0) for name in model.param_names}
    return GoniometerRefinement(
        params,
        pos_function=lambda two_theta: (two_theta,),
        trans_function=model,
        detector=detector,  # type: ignore[arg-type]
        wavelength=wavelength_m,
    )


def _predict_frame(gonioref: GoniometerRefinement, two_theta_deg: float) -> dict:
    """The model's geometry at two_theta, to start that frame's fit from."""
    ai = gonioref.get_ai((two_theta_deg,))
    names = ("dist", "poni1", "poni2", "rot1", "rot2", "rot3")
    return {name: float(getattr(ai, name)) for name in names}


def _plot_fit(
    sg: SingleGeometry,
    model: Geometry | None = None,
    save_path: Path | None = None,
    show: bool = False,
):
    """Rings over the image and Δ2θ residuals, for the frame fit and the model."""
    gr = sg.geometry_refinement
    assert gr.data is not None and sg.image is not None and sg.calibrant is not None
    d1, d2, rings = gr.data[:, 0], gr.data[:, 1], gr.data[:, 2].astype(int)
    chi = np.degrees(gr.chi(d1, d2))
    ring_tth = gr.calc_2th(rings)
    # only the rings with control points, the rest just clutter the image
    levels = np.unique(ring_tth)

    fig, axes = plt.subplot_mosaic(
        [["img", "chi"], ["img", "tth"]],
        figsize=(16, 6),
        width_ratios=[2.5, 1],
        layout="constrained",
    )
    ax_img = axes["img"]
    image = np.nan_to_num(sg.image)
    positive = image[image > 0]
    norm = LogNorm(*np.percentile(positive, [5, 99.9])) if positive.size else None
    ax_img.imshow(image, cmap="gray", norm=norm)
    ax_img.plot(d2, d1, ".", color="orange", ms=2, alpha=0.5)

    title = f"{sg.label}: {len(d1)} points on {len(np.unique(rings))} rings"
    for geometry, colour, name in [(gr, "cyan", "frame fit"), (model, "red", "model")]:
        if geometry is None:
            continue
        tth = geometry.center_array(unit="2th_rad", scale=False)
        # contour warns about levels outside the image
        in_image = levels[(levels > tth.min()) & (levels < tth.max())]
        if in_image.size:
            ax_img.contour(tth, levels=in_image, colors=colour, linewidths=0.8)
        residual = np.degrees(geometry.tth(d1, d2) - ring_tth) * 1e3
        axes["chi"].plot(chi, residual, ".", color=colour, ms=3, label=name)
        axes["tth"].plot(np.degrees(ring_tth), residual, ".", color=colour, ms=3)
        title += f", rms {name} {np.sqrt(np.mean(residual**2)):.1f} mdeg"

    for key, xlabel in [("chi", "χ (°)"), ("tth", "ring 2θ (°)")]:
        axes[key].axhline(0, color="k", lw=0.5)
        axes[key].set_xlabel(xlabel)
        axes[key].set_ylabel("Δ2θ (mdeg)")
    axes["chi"].legend()
    key = "cyan: frame fit, " + ("red: model, " if model is not None else "")
    fig.suptitle(f"{title}\n{key}orange: control points")

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
    if show:
        plt.show()
    plt.close(fig)


def build_and_save_goniometer(
    nexus_filepath: Path | str,
    images: np.ndarray,
    angles: np.ndarray,
    wavelength_in_angstrom: float,
    calibrant: Calibrant,
    initial_dist_m: float = 0.25,
    output_dir: Path | str | None = None,
    max_rings: list[int] | int | None = None,
    pts_per_deg: float = 1.0,
    unit: str = "2th_deg",
    radial_range: tuple[float, float] | None = None,
    npt: int = 2000,
    detector: Detector | str | None = None,
    plot_fits: bool = False,
    show_plots: bool = False,
    initial_beam_centre_px: tuple[float, float] | None = None,
    seed_from_previous: bool = True,
    model: GeometryTransformation = GEOMETRY_TRANSFORMATION,
) -> tuple[str, str]:
    """Fit each frame, then fit `model` (see eiger_goniometer_models) across them.

    Angles should be sorted from a frame with the beam on the detector. With
    `seed_from_previous`, each frame starts from `model` fitted to the ones before.
    `plot_fits` saves the fit for each frame, then with the model, and
    `show_plots` opens them.
    Returns the paths of the saved goniometer and metadata files.
    """
    detector = _resolve_detector(detector)

    nexus_path = Path(nexus_filepath)

    if output_dir is None:
        output_dir_str, _ = processed_directory_and_filename(
            nexus_path, nest_by_filename=False
        )
        output_dir = Path(output_dir_str)
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loaded %d calibration frames from %s", len(angles), nexus_path)
    wavelength_m = wavelength_in_angstrom / 1e10

    single_geometries: list[SingleGeometry] = []
    gonioref: GoniometerRefinement | None = None
    for i, (image, two_theta_deg) in enumerate(zip(images, angles, strict=True)):
        label = f"frame_{i:04d}_{two_theta_deg:.4f}deg"
        logger.info(
            "Calibrating frame %d / %d at %.4f°", i + 1, len(angles), two_theta_deg
        )
        seed_geometry, fix = None, None
        if seed_from_previous and gonioref is not None:
            seed_geometry = _predict_frame(gonioref, float(two_theta_deg))
            fix = list(FIX_BETWEEN_FRAMES)
            logger.info("  seeded from the model, fixing %s", ", ".join(fix))
        sg = _calibrate_single_frame(
            label,
            image,
            float(two_theta_deg),
            calibrant,
            initial_dist_m,
            max_rings,
            pts_per_deg,
            detector=detector,
            initial_beam_centre_px=initial_beam_centre_px,
            seed_geometry=seed_geometry,
            fix=fix,
        )
        single_geometries.append(sg)

        if gonioref is None:
            gonioref = _start_goniometer(model, sg, detector, wavelength_m)
        gonioref.single_geometries[sg.label] = sg
        if seed_from_previous and i < len(angles) - 1:
            # quietly: pyFAI prints on every refine2, the final one is below
            with contextlib.redirect_stdout(io.StringIO()):
                gonioref.refine2()
            logger.info("  model over %d frames: χ² = %.3g", i + 1, gonioref.chi2())

        if plot_fits or show_plots:
            fits_path = output_dir / FITS_DIR_NAME / f"{label}_frame.png"
            _plot_fit(sg, save_path=fits_path if plot_fits else None, show=show_plots)

    assert gonioref is not None
    logger.info(
        "Refining %s across %d frames …", ", ".join(model.param_names), len(angles)
    )

    # passed on to scipy, prints every iteration
    gonioref.refine2(iprint=2, disp=True)
    logger.info("Refinement done. χ² = %.6g", gonioref.chi2())

    for sg in single_geometries:
        gr = sg.geometry_refinement
        assert gr.data is not None
        model_ai = gonioref.get_ai(sg.get_position())
        model_param = [model_ai.dist, model_ai.poni1, model_ai.poni2]
        model_param += [model_ai.rot1, model_ai.rot2, model_ai.rot3]
        frame_rms, model_rms = np.degrees(
            np.sqrt([gr.chi2(), gr.chi2(model_param)]) / np.sqrt(len(gr.data))
        )
        logger.info(
            "  %s: rms Δ2θ %.4f° frame fit, %.4f° goniometer model",
            sg.label,
            frame_rms,
            model_rms,
        )

        if plot_fits or show_plots:
            fits_path = output_dir / FITS_DIR_NAME / f"{sg.label}_model.png"
            _plot_fit(
                sg,
                model_ai,
                save_path=fits_path if plot_fits else None,
                show=show_plots,
            )

    calibration_timestamp = datetime.datetime.now(datetime.UTC).strftime(
        "%Y-%m-%d_%H-%M-%S"
    )  # noqa - throws warning that is invalid

    calibration_save_filepath = str(
        output_dir / f"{nexus_path.stem}_{calibration_timestamp}_{GONIOMETER_SAVE_NAME}"
    )
    metadata_output_filepath = (
        output_dir / f"{nexus_path.stem}_{calibration_timestamp}_{METADATA_SAVE_NAME}"
    )

    gonioref.save(calibration_save_filepath)

    meta = {
        "filenumber": str(nexus_path.stem),
        "timestamp": str(calibration_timestamp),
        "unit": unit,
        "radial_range": list(radial_range) if radial_range is not None else None,
        "npt": npt,
        "wavelength": wavelength_m,
        "calibrant": calibrant.name,
        "calib_two_theta_deg": angles.tolist(),
    }
    metadata_output_filepath.write_text(json.dumps(meta, indent=2))

    metadata_output_filepath = str(metadata_output_filepath)

    logger.info("Saved goniometer to %s", calibration_save_filepath)
    logger.info("Saved goniometer calibration metadata to %s", metadata_output_filepath)

    return calibration_save_filepath, metadata_output_filepath


def integrate_with_goniometer(
    images: np.ndarray,
    positions: np.ndarray,
    goniometer_filepath: Path | str,
    output_xy_filepath: Path | str,
    npt: int = 2000,
    polarization_factor: float = 0.99,
    correct_solid_angle: bool = True,
    mask: np.ndarray | None = None,
    error_model: Literal["poisson", "azimuthal"] = "azimuthal",
    unit: str = "2th_deg",
    save_xye: bool = False,
    wavelength: float | None = None,
) -> Path:
    """Integrate images with a saved goniometer and write an .xy file."""
    output_xy_filepath = Path(output_xy_filepath)
    output_xy_filepath.parent.mkdir(parents=True, exist_ok=True)

    gonio = _load_goniometer(goniometer_filepath=Path(goniometer_filepath))

    frame_ais = [gonio.get_ai(float(two_theta)) for two_theta in positions]

    if wavelength is None:
        wavelength = gonio.wavelength

    mg = MultiGeometry(
        frame_ais,
        unit=unit,
        wavelength=wavelength,
    )

    n_frames = len(images)
    lst_mask = [mask.astype(bool)] * n_frames if mask is not None else None

    result = mg.integrate1d(
        list(images),
        npt=npt,
        correctSolidAngle=correct_solid_angle,
        polarization_factor=polarization_factor,
        lst_mask=lst_mask,
        error_model=error_model,
    )

    tth = np.array(result.radial)
    intensity = np.array(result.intensity)
    error = np.array(result.sigma)

    assert len(tth) == len(intensity) == len(error)

    np.savetxt(
        str(output_xy_filepath),
        np.column_stack([tth, intensity]),
        comments="",
        fmt="%.8g",
    )

    if save_xye:
        np.savetxt(
            str(output_xy_filepath).replace(".xy", ".xye"),
            np.column_stack([tth, intensity, error]),
            comments="",
            fmt="%.8g",
        )

    logger.info("Written %s", output_xy_filepath)
    return output_xy_filepath


# if __name__ == "__main__":
