"""Eiger500K goniometer calibration and integration."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from pyFAI.calibrant import Calibrant, get_calibrant
from pyFAI.detectors import Detector, detector_factory
from pyFAI.goniometer import (
    GeometryTransformation,
    Goniometer,
    GoniometerRefinement,
    MultiGeometry,
    SingleGeometry,
)
from pyFAI.gui import jupyter

from xrpd_toolbox.i15_1.eiger_500k import ARM_ROTATION_SIGN
from xrpd_toolbox.utils.utils import processed_directory_and_filename

logger = logging.getLogger(__name__)

# rigid arm swinging horizontally, so only rot1 changes with two_theta
GEOMETRY_TRANSFORMATION = GeometryTransformation(
    param_names=["dist", "poni1", "poni2", "rot1_scale", "rot1_offset", "rot2", "rot3"],
    pos_names=["two_theta"],
    dist_expr="dist",
    poni1_expr="poni1",
    poni2_expr="poni2",
    # numexpr has no deg2rad
    rot1_expr="rot1_scale * (two_theta * 0.017453292519943295) + rot1_offset",
    rot2_expr="rot2",
    rot3_expr="rot3",
)

# once the beam centre is off the detector a frame can't separate these from rot1
SEEDED_FRAME_FIX = ["wavelength", "dist", "poni1", "poni2"]

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


def _load_goniometer_dir(goniometer_filepath: Path) -> Goniometer:
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


def _seed_from(
    previous: SingleGeometry, two_theta_deg: float
) -> tuple[dict[str, float], list[str]]:
    """Previous frame's geometry with rot1 moved on by the arm step."""
    gr = previous.geometry_refinement
    names = ("dist", "poni1", "poni2", "rot1", "rot2", "rot3")
    seed = {name: float(getattr(gr, name)) for name in names}
    assert previous.metadata is not None
    step = np.deg2rad(two_theta_deg - float(previous.metadata))
    seed["rot1"] += ARM_ROTATION_SIGN * step
    return seed, list(SEEDED_FRAME_FIX)


def build_and_save_goniometer(
    nexus_filepath: Path | str,
    images: np.ndarray,
    angles: np.ndarray,
    wavelength_in_angstrom: float,
    calibrant_name: str = "Si",
    initial_dist_m: float = 0.25,
    output_dir: Path | str | None = None,
    max_rings: list[int] | int | None = None,
    pts_per_deg: float = 1.0,
    unit: str = "2th_deg",
    radial_range: tuple[float, float] | None = None,
    npt: int = 2000,
    detector: Detector | str | None = None,
    plot_fits: bool = False,
    initial_beam_centre_px: tuple[float, float] | None = None,
    seed_from_previous: bool = True,
) -> tuple[str, str]:
    """Fit each frame, then fit GEOMETRY_TRANSFORMATION across all of them.

    Angles should be ascending from a frame with the beam on the detector.
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

    calibrant = get_calibrant(calibrant_name=calibrant_name, wavelength=wavelength_m)

    single_geometries: list[SingleGeometry] = []
    for i, (image, two_theta_deg) in enumerate(zip(images, angles, strict=True)):
        label = f"frame_{i:04d}_{two_theta_deg:.4f}deg"
        logger.info(
            "Calibrating frame %d / %d at %.4f°", i + 1, len(angles), two_theta_deg
        )
        seed_geometry, fix = None, None
        if seed_from_previous and single_geometries:
            previous = single_geometries[-1]
            seed_geometry, fix = _seed_from(previous, float(two_theta_deg))
            logger.info("  seeded from %s, fixing %s", previous.label, ", ".join(fix))
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

    first = single_geometries[0].geometry_refinement
    initial_params = {
        "dist": first.dist,
        "poni1": first.poni1,
        "poni2": first.poni2,
        "rot1_scale": ARM_ROTATION_SIGN,
        "rot1_offset": first.rot1 - ARM_ROTATION_SIGN * np.deg2rad(angles[0]),
        "rot2": first.rot2,
        "rot3": first.rot3,
    }

    gonioref = GoniometerRefinement(
        initial_params,
        pos_function=lambda two_theta: (two_theta,),
        trans_function=GEOMETRY_TRANSFORMATION,
        detector=detector,  # type: ignore[arg-type]
        wavelength=wavelength_m,
    )

    for sg in single_geometries:
        gonioref.single_geometries[sg.label] = sg

    logger.info("Refining goniometer model across %d frames …", len(angles))

    # passed on to scipy, prints every iteration
    gonioref.refine2(iprint=2, disp=True)
    logger.info("Refinement done. χ² = %.6g", gonioref.chi2())

    for sg in single_geometries:
        gr = sg.geometry_refinement
        assert gr.data is not None
        model = gonioref.get_ai(sg.get_position())
        model_param = [model.dist, model.poni1, model.poni2]
        model_param += [model.rot1, model.rot2, model.rot3]
        frame_rms, model_rms = np.degrees(
            np.sqrt([gr.chi2(), gr.chi2(model_param)]) / np.sqrt(len(gr.data))
        )
        logger.info(
            "  %s: rms Δ2θ %.4f° frame fit, %.4f° goniometer model",
            sg.label,
            frame_rms,
            model_rms,
        )

        if plot_fits:
            fig, (ax_frame, ax_model) = plt.subplots(2, 1, figsize=(12, 10))
            jupyter.display(sg=sg, ax=ax_frame)
            jupyter.display(
                sg=sg, ai=model, ax=ax_model, label=f"{sg.label} goniometer model"
            )
            (output_dir / FITS_DIR_NAME).mkdir(exist_ok=True)
            fig.savefig(output_dir / FITS_DIR_NAME / f"{sg.label}.png")
            plt.close(fig)

    calibration_save_filepath = str(output_dir / GONIOMETER_SAVE_NAME)
    metadata_output_filepath = output_dir / METADATA_SAVE_NAME

    gonioref.save(calibration_save_filepath)

    meta = {
        "unit": unit,
        "radial_range": list(radial_range) if radial_range is not None else None,
        "npt": npt,
        "wavelength": wavelength_m,
        "calibrant": calibrant_name,
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

    gonio = _load_goniometer_dir(goniometer_filepath=Path(goniometer_filepath))

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
