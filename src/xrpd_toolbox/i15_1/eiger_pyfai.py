"""
Eiger500K goniometer calibration and integration.

Two public functions:

- ``build_and_save_goniometer``: calibrate a Goniometer model from images of
  a known calibrant and save to disk.
- ``integrate_with_goniometer``: load that model and integrate arbitrary
  detector images into a 1-D .xy pattern.

All data is read from NeXus/HDF5 files.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

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

from xrpd_toolbox.i15_1 import goniometer_diagnostics as diag
from xrpd_toolbox.i15_1.eiger_500k import ARM_ROTATION_SIGN
from xrpd_toolbox.utils.utils import processed_directory_and_filename

logger = logging.getLogger(__name__)

# Geometry model
# a detector arm swinging horizontally around tth (see ARM_ROTATION_SIGN):
# dist/poni1/poni2/rot2/rot3 should be constant; rot1 is linear in two_theta,
# with rot1_scale ~ ARM_ROTATION_SIGN.  Maybe extend expressions here for
# non-ideal arms?

GEOMETRY_TRANSFORMATION = GeometryTransformation(
    param_names=["dist", "poni1", "poni2", "rot1_scale", "rot1_offset", "rot2", "rot3"],
    pos_names=["two_theta"],
    dist_expr="dist",
    poni1_expr="poni1",
    poni2_expr="poni2",
    # numexpr can't call numpy functions (eg.. np.deg2rad,
    # and it has no built-in deg2rad,
    # so conversion factor pi/180 = 0.0174532...
    rot1_expr="rot1_scale * (two_theta * 0.017453292519943295) + rot1_offset",
    rot2_expr="rot2",
    rot3_expr="rot3",
)

# refined per frame only on the first frame when seeding from the previous one:
# once the beam centre is off the detector a frame can't separate these from rot1
SEEDED_FRAME_FIX = ["wavelength", "dist", "poni1", "poni2"]

GONIOMETER_SAVE_NAME = "eiger_goniometer_calibration.json"
METADATA_SAVE_NAME = "calibration_metadata.json"
DIAGNOSTICS_DIR_NAME = "goniometer_diagnostics"


def calibrate_single_geometry_from_rings(
    geometry: SingleGeometry,
    rings: list[int] = [3, 5, 5, 5, 7, 7, 9, 11, 15, 17],  # noqa
    fix: list | None = None,
):
    """Extract control points for geometry and refine it."""
    if fix is None:
        fix = []

    for step, max_rings in enumerate(rings, start=1):
        geometry.extract_cp(max_rings=max_rings)
        # pyFAI stores an empty 1D array when no points are found, which then
        # fails deep inside refine2 with an unhelpful unpacking error
        assert geometry.geometry_refinement.data is not None
        if geometry.geometry_refinement.data.ndim != 2:
            raise ValueError(
                f"No control points found for {geometry.label} "
                f"(max_rings={max_rings}) - check the image contains "
                "positive calibrant rings and the mask is correct"
            )
        rms_before = diag.rms_mdeg(geometry.geometry_refinement)
        geometry.geometry_refinement.refine2(fix=fix)
        diag.log_geometry_step(
            geometry.label,
            f"step {step}/{len(rings)} max_rings={max_rings}",
            geometry.geometry_refinement,
            rms_before,
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
    """Calibrate one frame independently and return the refined SingleGeometry.

    A SingleGeometry is initialised with an
    approximate sample-to-detector distance and beam centre (row, col) - the
    detector centre unless `initial_beam_centre_px` is given; any of dist,
    poni1, poni2, rot1, rot2, rot3 in `seed_geometry` override that guess.
    Control points are then extracted and the per-frame geometry is refined
    (keeping the parameters in `fix` constant) before being handed to the
    GoniometerRefinement

    `detector` defaults to the simulation Eiger500K; pass a pyFAI detector
    (or its registry name, e.g. "Eiger2CdTe_500k") for real data.

    Returns SingleGeometry with control points extracted and geometry refined.
    """
    # NOTE: must be an actual Eiger500K() instance, not the "Eiger500k" name
    # string - SingleGeometry.__init__ resolves a `detector=` string via
    # pyFAI's own detector registry, which (case-insensitively) maps
    # "eiger500k" to pyFAI's *own* built-in Eiger500k detector
    # (max_shape (514, 1030)), silently overriding whatever detector object
    # was set on `initial_geometry["detector"]" below - and that shape
    # doesn't match images produced by Eiger500K.simulate_data() (max_shape
    # DEFAULT_MAX_SHAPE = (1028, 512)), which crashes extract_cp(). Passing
    # an instance bypasses that string lookup entirely (detector_factory
    # returns a Detector instance unchanged).
    detector = _resolve_detector(detector)
    assert detector.max_shape is not None
    if initial_beam_centre_px is None:
        rows, cols = detector.max_shape
        initial_beam_centre_px = (rows / 2, cols / 2)
    centre_row, centre_col = initial_beam_centre_px

    # Approximate geometry: beam hits the detector centre, arm at two_theta.
    initial_geometry = {
        "dist": initial_dist_m,
        # the arm rotates about the sample, so the PONI stays fixed on the
        # detector: the 0° beam centre is the right guess at every angle
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
        # GoniometerRefinement.refine2()/chi2() call single.get_position(), which
        # calls pos_function(metadata) - without this, that's None and every
        # refinement crashes with "'NoneType' object is not callable".
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
        rms_before = diag.rms_mdeg(sg.geometry_refinement)
        sg.geometry_refinement.refine2(fix=fix)
        diag.log_geometry_step(
            label, f"max_rings={max_rings}", sg.geometry_refinement, rms_before
        )

    assert sg.geometry_refinement.data is not None
    return sg


def _seed_from(
    previous: SingleGeometry, two_theta_deg: float
) -> tuple[dict[str, float], list[str]]:
    """Starting geometry for a frame from the previously refined one: the same
    geometry with rot1 advanced by the arm step. Returns (seed, fix)."""
    gr = previous.geometry_refinement
    seed = {name: float(getattr(gr, name)) for name in diag.GEOMETRY_NAMES}
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
    show_plots: bool = False,
    initial_beam_centre_px: tuple[float, float] | None = None,
    seed_from_previous: bool = True,
) -> tuple[str, str]:
    """Calibrate a Goniometer from calibrant images and save it.

    Each calibration frame is first calibrated independently via SingleGeometry
    to extract rings and refine a per-frame geometry.  Those per-frame geometries
    create a .GoniometerRefinement  that fits GEOMETRY_TRANSFORMATION
    across all frames simultaneously, producing a model of how the detector g
    eometry varies with two_theta.

    `detector` defaults to the simulation Eiger500K; pass a pyFAI detector
    (or its registry name, e.g. "Eiger2CdTe_500k") for real data.

    Every step is logged (call ``diag.setup_calibration_logging()`` to see it
    on the console). With `plot_fits`, a figure of the fit at each angle is
    saved to ``output_dir/goniometer_diagnostics`` as each frame is
    calibrated, then again with the goniometer model overlaid, plus a summary
    figure. `show_plots` also opens the per-frame and summary figures
    interactively as they are made (blocking until each is closed).

    `initial_beam_centre_px` is the (row, col) of the direct beam with the arm
    at 0°, used as the starting PONI for every frame (default: detector
    centre). Each ring's control points are searched for within about a
    quarter of the spacing to the next ring, so a guess much further off than
    that assigns points to the wrong rings.

    With `seed_from_previous`, frames are calibrated in the order given (which
    should be ascending, starting with the beam on the detector): each starts
    from the previous frame's refined geometry with rot1 advanced by the arm
    step, and keeps dist/poni1/poni2 fixed (SEEDED_FRAME_FIX) - once the beam
    centre is off the detector a frame can't tell a shifted PONI from a change
    in rot1. The global goniometer refinement still refines everything.

    returns a tuple of strings to goniometer calibration, and calibration metadata
    """
    # see the NOTE in _calibrate_single_frame - resolve to an instance once so
    # every frame and the GoniometerRefinement share the same detector.
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

    plot_dir = output_dir / DIAGNOSTICS_DIR_NAME if plot_fits else None
    if plot_fits or show_plots:
        logger.info("Calibration diagnostic plots -> %s", plot_dir)

    # calibrate each frame independently
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

        if plot_fits or show_plots:
            diag.finish_figure(
                diag.plot_frame_fit(sg, float(two_theta_deg)),
                plot_dir / f"{label}_frame_fit.png" if plot_dir else None,
                show_plots,
            )

    # --- Step 2: seed GoniometerRefinement from the first refined frame ---
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

    # register each SingleGeometry with its control points
    for sg in single_geometries:
        gonioref.single_geometries[sg.label] = sg

    diag.log_frame_table(
        [
            diag.summarise_frame(sg, float(tth))
            for sg, tth in zip(single_geometries, angles, strict=True)
        ]
    )

    # --- Step 4: global refinement ---
    logger.info("Refining goniometer model across %d frames …", len(angles))

    trace = diag.refine_goniometer(gonioref)
    logger.info("Refinement done. χ² = %.6g", gonioref.chi2())

    summaries = [
        diag.summarise_frame(sg, float(tth), gonioref)
        for sg, tth in zip(single_geometries, angles, strict=True)
    ]
    diag.log_frame_table(summaries)

    if plot_fits or show_plots:
        for sg, summary in zip(single_geometries, summaries, strict=True):
            # only saved: showing these too would double the windows to close
            if plot_dir is not None:
                diag.finish_figure(
                    diag.plot_frame_fit(
                        sg, summary.two_theta_deg, summary.model_geometry
                    ),
                    plot_dir / f"{sg.label}_model_fit.png",
                    show=False,
                )
        diag.finish_figure(
            diag.plot_refinement_summary(summaries, trace),
            plot_dir / "refinement_summary.png" if plot_dir else None,
            show_plots,
        )

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
    """Integrate detector images using a saved Goniometer model.

    Return a Path to the written ``.xy`` file.
    """
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
