"""
pyFAI MultiGeometry calibration and integration pipeline for an Eiger 500k
detector mounted on an arm with arbitrary, multi-axis motion.

Each calibration frame is refined independently into its own geometry and
saved as a standard pyFAI .poni file (one per arm position). Integration
loads those .poni files back into AzimuthalIntegrator objects, builds a
MultiGeometry from them, and stitches the corresponding data frames into a
single 1D pattern.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pyFAI
from numpy.typing import NDArray
from pyFAI.calibrant import get_calibrant
from pyFAI.goniometer import SingleGeometry
from pyFAI.integrator.azimuthal import AzimuthalIntegrator
from pyFAI.multi_geometry import MultiGeometry

from xrpd_toolbox.utils.utils import h5_to_array

DETECTOR_NAME = "eiger_500k"
PONI_INDEX_FILENAME = "poni_index.json"

Mask = NDArray[np.bool_]


def build_multigeometry_from_calibration(
    calib_nexus_path: Path,
    poni_dir: Path,
    calibrant_name: str,
    wavelength_m: float,
    detector_name: str = DETECTOR_NAME,
    data_path: str = "/entry/data/data",
    unit: str = "2th_deg",
    radial_range: tuple[float, float] | None = None,
    initial_distance_m: float = 0.1,
    max_rings: int = 10,
) -> Path:
    """
    Read calibrant frames from a NeXus file, refine the geometry of each
    frame independently, and write each as a .poni file under `poni_dir`
    (one per frame/position, in frame order). Writes a JSON index recording
    the .poni filenames and integration settings.

    Returns the path to the JSON index file.
    """

    detector = pyFAI.detector_factory(detector_name)
    calibrant = get_calibrant(calibrant_name)
    calibrant.wavelength = wavelength_m

    images = h5_to_array(calib_nexus_path, data_path)

    poni_dir.mkdir(parents=True, exist_ok=True)

    assert detector.shape is not None
    poni1_guess = detector.shape[0] / 2 * detector.pixel1
    poni2_guess = detector.shape[1] / 2 * detector.pixel2

    poni_filenames: list[str] = []
    for i in range(images.shape[0]):
        single_geom = SingleGeometry(
            f"frame_{i:04d}",
            images[i],
            calibrant=calibrant,
            detector=detector,
        )
        single_geom.geometry_refinement.set_param(
            [initial_distance_m, poni1_guess, poni2_guess, 0.0, 0.0, 0.0]
        )
        single_geom.extract_cp(max_rings=max_rings)
        single_geom.geometry_refinement.refine2(fix=["wavelength"])

        ai = single_geom.get_ai()
        poni_filename = f"frame_{i:04d}.poni"
        poni_filepath = poni_dir / poni_filename
        ai.save(str(poni_filepath))
        poni_filenames.append(poni_filename)

    index = {
        "calibrant_name": calibrant_name,
        "wavelength_m": wavelength_m,
        "detector_name": detector_name,
        "data_path": data_path,
        "unit": unit,
        "radial_range": list(radial_range) if radial_range is not None else None,
        "poni_files": poni_filenames,
    }

    index_path = poni_dir / PONI_INDEX_FILENAME
    index_path.write_text(json.dumps(index, indent=2))
    return index_path


def integrate_with_multigeometry(
    data_nexus_path: Path,
    poni_index_path: Path,
    npt: int = 2000,
    masks: Sequence[Mask] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Load the .poni files referenced by `poni_index_path`, build a
    MultiGeometry, load the data frames from `data_nexus_path` (same order
    as the .poni files), and stitch them into a single 1D pattern.

    Returns (radial, intensity).
    """

    index = json.loads(poni_index_path.read_text())
    poni_dir = poni_index_path.parent

    integrators: list[AzimuthalIntegrator] = []
    for poni_filename in index["poni_files"]:
        ai = AzimuthalIntegrator()
        ai.load(poni_dir / poni_filename)
        integrators.append(ai)

    radial_range = index["radial_range"]
    multigeometry = MultiGeometry(
        integrators,
        unit=index["unit"],
        radial_range=tuple(radial_range) if radial_range is not None else None,
        wavelength=index["wavelength_m"],
    )

    images = h5_to_array(data_nexus_path, index["data_path"])

    if images.shape[0] != len(integrators):
        raise ValueError(
            f"Number of frames ({images.shape[0]}) does not match "
            f"number of calibrated geometries ({len(integrators)})"
        )

    image_list = [images[i] for i in range(images.shape[0])]

    result = multigeometry.integrate1d(
        image_list,
        npt=npt,
        lst_mask=list(masks) if masks is not None else None,
    )

    radial = np.asarray(result.radial, dtype=np.float64)
    intensity = np.asarray(result.intensity, dtype=np.float64)
    return radial, intensity


__all__ = [
    "build_multigeometry_from_calibration",
    "integrate_with_multigeometry",
]
