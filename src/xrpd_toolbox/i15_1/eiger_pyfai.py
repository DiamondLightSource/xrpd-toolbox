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
from collections.abc import Collection
from pathlib import Path

import h5py
import numpy as np
from pyFAI.calibrant import Calibrant, get_calibrant
from pyFAI.goniometer import (
    GeometryTransformation,
    Goniometer,
    GoniometerRefinement,
    MultiGeometry,
    SingleGeometry,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Geometry model
# ---------------------------------------------------------------------------
# Models a rotating detector arm: dist/poni1/poni2/rot1/rot3 are constant;
# rot2 is linear in two_theta.  Extend expressions here for non-ideal arms.

GEOMETRY_TRANSFORMATION = GeometryTransformation(
    param_names=["dist", "poni1", "poni2", "rot1", "rot2_scale", "rot2_offset", "rot3"],
    pos_names=["two_theta"],
    dist_expr="dist",
    poni1_expr="poni1",
    poni2_expr="poni2",
    rot1_expr="rot1",
    # numexpr can't call numpy functions (no dotted names like `np.deg2rad`,
    # hence "forbidden control characters"), and it has no built-in deg2rad,
    # so inline the conversion factor pi/180 as a literal.
    rot2_expr="rot2_scale * (two_theta * 0.017453292519943295) + rot2_offset",
    rot3_expr="rot3",
)

_DETECTOR = "Eiger500k"


def calibrate_single_geometry_from_rings(
    geometry: SingleGeometry,
    rings: Collection[int] = [5, 5, 5, 7, 7, 9, 11, 15, 17],
    fix: list | None = None,
):
    if fix is None:
        fix = []

    for n_rings in rings:
        geometry.extract_cp(max_rings=n_rings)
        geometry.geometry_refinement.refine2(fix=fix)

    return geometry


def _load_nexus(
    nexus_path: Path,
    images_dataset: str,
    angles_dataset: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(images, angles)`` from a NeXus file.

    Returns
    -------
    images : np.ndarray, shape (N, rows, cols), float32
    angles : np.ndarray, shape (N,), float64  [degrees]
    """
    with h5py.File(nexus_path, "r") as f:
        images = np.asarray(f[images_dataset], dtype=np.float32)
        angles = np.asarray(f[angles_dataset], dtype=np.float64)

    if images.ndim == 2:
        images = images[np.newaxis]
    if angles.ndim == 0:
        angles = angles[np.newaxis]
    if images.shape[0] != angles.shape[0]:
        raise ValueError(
            f"Image count ({images.shape[0]}) does not match "
            f"angle count ({angles.shape[0]})."
        )
    return images, angles


def _load_goniometer_dir(output_dir: Path) -> tuple[Goniometer, dict]:
    """Load a Goniometer and its metadata sidecar from *output_dir*."""
    gonio_path = output_dir / "goniometer.json"
    meta_path = output_dir / "meta.json"
    for p in (gonio_path, meta_path):
        if not p.exists():
            raise FileNotFoundError(f"{p.name} not found in {output_dir}")

    gonio = Goniometer.sload(str(gonio_path))
    meta: dict = json.loads(meta_path.read_text())
    logger.info("Loaded goniometer from %s", output_dir)
    return gonio, meta


def _calibrate_single_frame(
    label: str,
    image: np.ndarray,
    two_theta_deg: float,
    calibrant: Calibrant,
    initial_dist_m: float,
    max_rings: int | None,
    pts_per_deg: float,
) -> SingleGeometry:
    """Calibrate one frame independently and return the refined SingleGeometry.

    A SingleGeometry is initialised with an
    approximate sample-to-detector distance and beam centre at the detector
    centre.  Control points are then extracted and the per-frame geometry is
    refined before being handed to the GoniometerRefinement

    Returns SingleGeometry with control points extracted and geometry refined.
    """
    from pyFAI.detectors import Eiger500k

    detector = Eiger500k()
    rows, cols = image.shape

    # Approximate geometry: beam hits the detector centre, arm at two_theta.
    initial_geometry = {
        "dist": initial_dist_m,
        "poni1": rows / 2 * detector.pixel1,
        "poni2": cols / 2 * detector.pixel2,
        "rot1": 0.0,
        "rot2": np.deg2rad(two_theta_deg),
        "rot3": 0.0,
        "wavelength": calibrant.wavelength,
        "detector": detector,
    }

    sg = SingleGeometry(
        label=label,
        image=image,
        metadata=two_theta_deg,
        calibrant=calibrant,
        detector=_DETECTOR,
        geometry=initial_geometry,
    )
    assert sg.geometry_refinement.data is not None
    sg.extract_cp(max_rings=max_rings, pts_per_deg=pts_per_deg)
    sg.geometry_refinement.refine2()
    logger.debug(
        "  %s: dist=%.4f m  rot2=%.4f rad  npts=%d",
        label,
        sg.geometry_refinement.dist,
        sg.geometry_refinement.rot2,
        len(sg.geometry_refinement.data),
    )
    return sg


def build_and_save_goniometer(
    nexus_path: Path | str,
    *,
    images_dataset: str = "/entry/instrument/detector/data",
    angles_dataset: str = "/entry/instrument/goniometer/two_theta",
    wavelength: float,
    output_dir: Path | str = ".",
    calibrant_name: str = "LaB6",
    initial_dist_m: float = 0.2,
    max_rings: int | None = None,
    pts_per_deg: float = 1.0,
    unit: str = "2th_deg",
    radial_range: tuple[float, float] | None = None,
    npt: int = 2000,
) -> Path:
    """Calibrate a Goniometer from calibrant images and save it.

    Each calibration frame is first calibrated independently via SingleGeometry
    to extract rings and refine a per-frame geometry.  Those per-frame geometries
    create a .GoniometerRefinement  that fits GEOMETRY_TRANSFORMATION
    across all frames simultaneously, producing a model of how the detector g
    eometry varies with two_theta.

    returns a Path to goniometer refinement
    """
    nexus_path = Path(nexus_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images, angles = _load_nexus(nexus_path, images_dataset, angles_dataset)
    logger.info("Loaded %d calibration frames from %s", len(angles), nexus_path)

    calibrant = get_calibrant(calibrant_name)
    wavelength_m = wavelength / 1e10
    calibrant.wavelength = wavelength_m

    # --- Step 1: calibrate each frame independently ---
    single_geometries: list[SingleGeometry] = []
    for i, (image, two_theta_deg) in enumerate(zip(images, angles, strict=True)):
        label = f"frame_{i:04d}_{two_theta_deg:.4f}deg"
        logger.info(
            "Calibrating frame %d / %d at %.4f°", i + 1, len(angles), two_theta_deg
        )
        sg = _calibrate_single_frame(
            label,
            image,
            float(two_theta_deg),
            calibrant,
            initial_dist_m,
            max_rings,
            pts_per_deg,
        )
        single_geometries.append(sg)

    # --- Step 2: seed GoniometerRefinement from the first refined frame ---
    first = single_geometries[0].geometry_refinement
    initial_params = {
        "dist": first.dist,
        "poni1": first.poni1,
        "poni2": first.poni2,
        "rot1": first.rot1,
        "rot2_scale": 1.0,
        "rot2_offset": first.rot2 - np.deg2rad(angles[0]),
        "rot3": first.rot3,
    }

    gonioref = GoniometerRefinement(
        initial_params,
        pos_function=lambda two_theta: (two_theta,),
        trans_function=GEOMETRY_TRANSFORMATION,
        detector=_DETECTOR,
        wavelength=wavelength_m,
    )

    # register each SingleGeometry with its control points
    for sg in single_geometries:
        gonioref.single_geometries[sg.label] = sg

    # --- Step 4: global refinement ---
    logger.info("Refining goniometer model across %d frames …", len(angles))

    gonioref.refine2()
    logger.info("Refinement done. χ² = %.6g", gonioref.chi2())

    gonioref.save(str(output_dir / "goniometer.json"))

    meta = {
        "unit": unit,
        "radial_range": list(radial_range) if radial_range is not None else None,
        "npt": npt,
        "wavelength": wavelength_m,
        "calibrant": calibrant_name,
        "calib_two_theta_deg": angles.tolist(),
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    logger.info("Saved goniometer to %s", output_dir)
    return output_dir


def integrate_with_goniometer(
    images: np.ndarray,
    positions: np.ndarray,
    goniometer_dir: Path | str,
    output_xy_filepath: Path | str,
    npt: int | None = None,
    polarization_factor: float = 0.99,
    correct_solid_angle: bool = True,
    mask: np.ndarray | None = None,
) -> Path:
    """Integrate detector images using a saved Goniometer model.

    Return a Path to the written ``.xy`` file.
    """
    goniometer_dir = Path(goniometer_dir)
    output_xy_filepath = Path(output_xy_filepath)
    output_xy_filepath.parent.mkdir(parents=True, exist_ok=True)

    gonio, meta = _load_goniometer_dir(goniometer_dir)

    calib_angles = np.asarray(meta["calib_two_theta_deg"])
    out_of_range = (positions < calib_angles.min()) | (positions > calib_angles.max())
    if out_of_range.any():
        logger.warning(
            "%d frame(s) outside calibrated range [%.4f°, %.4f°] — extrapolating.",
            out_of_range.sum(),
            calib_angles.min(),
            calib_angles.max(),
        )

    effective_npt: int = npt if npt is not None else meta["npt"]
    radial_range: tuple[float, float] | None = (
        tuple(meta["radial_range"]) if meta.get("radial_range") else None  # type: ignore[assignment]
    )

    frame_ais = [gonio.get_ai(float(two_theta)) for two_theta in positions]

    mg = MultiGeometry(
        frame_ais,
        unit=meta["unit"],
        radial_range=radial_range,
        wavelength=meta["wavelength"],
    )

    n_frames = len(images)
    lst_mask = [mask.astype(bool)] * n_frames if mask is not None else None

    result = mg.integrate1d(
        list(images),
        effective_npt,
        correctSolidAngle=correct_solid_angle,
        polarization_factor=polarization_factor,
        lst_mask=lst_mask,
    )

    header = "\n".join(
        [
            f"# goniometer_dir: {goniometer_dir}",
            f"# frames: {n_frames}",
            f"# npt: {effective_npt}",
            f"# unit: {meta['unit']}",
            f"# wavelength_m: {meta['wavelength']}",
            f"# polarization_factor: {polarization_factor}",
            f"# correct_solid_angle: {correct_solid_angle}",
            f"# {meta['unit']}    Intensity",
        ]
    )
    np.savetxt(
        str(output_xy_filepath),
        np.column_stack([result.radial, result.intensity]),
        header=header,
        comments="",
        fmt="%.8g",
    )
    logger.info("Written %s", output_xy_filepath)
    return output_xy_filepath


# if __name__ == "__main__":
