"""Helpers for building minimal synthetic NeXus/HDF5 files that exercise
EigerDataLoader (xrpd_toolbox.i15_1.eiger_500k) without needing a real
Eiger500K data collection on disk.

Not a test module itself - nothing in here is named test_*.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

ENTRY = "entry"
EIGER_DATA_PATH = "/entry/instrument/fastcs_eiger"


def build_mask_file(path: str | Path, datapath: str, shape=(4, 5)) -> None:
    """Write a standalone HDF5 file holding a pixel mask dataset."""
    with h5py.File(path, "w") as f:
        f.create_dataset(datapath, data=np.zeros(shape, dtype=np.int32))


def build_eiger_nexus(
    path: str | Path,
    *,
    entry: str = ENTRY,
    eiger_data_path: str = EIGER_DATA_PATH,
    n_frames: int = 3,
    rows: int = 4,
    cols: int = 5,
    data: np.ndarray | None = None,
    data_is_group: bool = False,
    data_is_scalar: bool = False,
    include_data: bool = True,
    tth: np.ndarray | None = None,
    include_tth: bool = True,
    count_time: np.ndarray | None = None,
    include_count_time: bool = True,
    energy_kev: float = 12.4,
    include_energy_kev: bool = True,
    mask_ref: str | None = None,
    include_mask: bool = True,
    calibrant: str | None = "Si",
    include_calibrant: bool = True,
    background: int = 0,
    include_background: bool = True,
    plan_name: str | None = "data_collection",
    include_plan_name: bool = True,
    scan_type: str | None = "step_scan",
    include_scan_type: bool = True,
    i0: np.ndarray | None = None,
    include_i0: bool = True,
) -> Path:
    """Build a minimal NeXus-like HDF5 file with the datapaths that
    EigerDataLoader reads.

    Only the requested pieces are written, so callers can build files that
    are missing a given dataset in order to exercise the loader's error /
    fallback handling.
    """
    path = Path(path)

    if data is None:
        rng = np.random.default_rng(0)
        data = rng.integers(0, 100, size=(n_frames, rows, cols)).astype(np.uint32)

    if tth is None:
        tth = np.linspace(1.0, float(n_frames), n_frames)

    if count_time is None:
        count_time = np.full(n_frames, 0.1)

    if i0 is None:
        i0 = np.full(n_frames, 1.0)

    with h5py.File(path, "w") as f:
        entry_grp = f.create_group(entry)
        # absolute path, so this also creates /entry/instrument
        eiger_grp = f.create_group(eiger_data_path)

        if include_data:
            if data_is_group:
                eiger_grp.create_group("data")
            elif data_is_scalar:
                eiger_grp.create_dataset("data", data=1)
            else:
                eiger_grp.create_dataset("data", data=data)

        instrument_grp = entry_grp.require_group("instrument")

        if include_tth:
            tth_grp = instrument_grp.create_group("tth")
            tth_grp.create_dataset("data", data=np.asarray(tth))

        if include_mask:
            eiger_grp.create_dataset(
                "pixel_mask", data=mask_ref if mask_ref is not None else "//mask"
            )

        if include_energy_kev:
            xtal_grp = instrument_grp.create_group("xtal")
            xtal_grp.create_dataset("energy_kev", data=energy_kev)

        plan_metadata_grp = entry_grp.create_group("plan_metadata")

        if include_count_time:
            plan_metadata_grp.create_dataset(
                "exposure_time_per_frame", data=np.asarray(count_time)
            )

        if include_calibrant and calibrant is not None:
            plan_metadata_grp.create_dataset("calibrant", data=calibrant)

        if include_background:
            plan_metadata_grp.create_dataset("background", data=background)

        if include_plan_name and plan_name is not None:
            plan_metadata_grp.create_dataset("plan_name", data=plan_name)

        if include_scan_type and scan_type is not None:
            plan_metadata_grp.create_dataset("scan_type", data=scan_type)

        if include_i0:
            i0_grp = entry_grp.create_group("i0")
            i0_grp.create_dataset("data", data=np.asarray(i0))

    return path


# the real i15-1 Eiger frame shape - the saved detector corrections are this shape
I15_1_DETECTOR_SHAPE = (512, 1028)
I15_1_ENERGY_KEV = 76.69


def build_i15_1_scan(
    directory: str | Path,
    name: str = "scan",
    *,
    tth: tuple[float, ...] = (10.0, 14.0),
    counts: int = 100,
    composition: str | None = None,
    empty_capillary_filename: str | None = None,
) -> Path:
    """Write a small but full-size i15-1 Eiger scan (nexus + pixel mask file)
    that do_eiger_data_reduction can reduce for real.

    Every pixel reads `counts`, with one frame per two-theta position in `tth`.
    """
    directory = Path(directory)
    n_frames = len(tth)

    mask_filepath = directory / f"{name}_pixel_mask.h5"
    build_mask_file(mask_filepath, "mask", shape=I15_1_DETECTOR_SHAPE)

    nexus_filepath = build_eiger_nexus(
        directory / f"{name}.nxs",
        n_frames=n_frames,
        data=np.full((n_frames, *I15_1_DETECTOR_SHAPE), counts, dtype=np.uint32),
        tth=np.asarray(tth, dtype=float),
        energy_kev=I15_1_ENERGY_KEV,
        mask_ref=f"{mask_filepath}//mask",
    )

    with h5py.File(nexus_filepath, "a") as f:
        plan_metadata = f.require_group(f"/{ENTRY}/plan_metadata")
        if composition is not None:
            plan_metadata.create_dataset(
                "sample_info/data/composition", data=composition
            )
        if empty_capillary_filename is not None:
            plan_metadata.create_dataset(
                "auxiliary_scans/Empty Capillary/filename",
                data=empty_capillary_filename,
            )

    return nexus_filepath


def write_i15_1_goniometer_json(path: str | Path) -> Path:
    """Save the real i15-1 goniometer calibration (kept alongside the detector
    corrections in the package) as a pyFAI goniometer json file."""
    from xrpd_toolbox.i15_1.eiger_analysis import CORRECTION_h5
    from xrpd_toolbox.utils.utils import h5_to_string

    geometry_json = h5_to_string(CORRECTION_h5, "geometry_json")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(geometry_json)
    return path
