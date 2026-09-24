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
EIGER_DATA_PATH = "fastcs_eiger"


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
    plan_type: str | None = "step_scan",
    include_plan_type: bool = True,
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
        eiger_grp = entry_grp.create_group(eiger_data_path)

        if include_data:
            if data_is_group:
                eiger_grp.create_group("data")
            elif data_is_scalar:
                eiger_grp.create_dataset("data", data=1)
            else:
                eiger_grp.create_dataset("data", data=data)

        instrument_grp = entry_grp.create_group("instrument")
        instrument_eiger_grp = instrument_grp.create_group(eiger_data_path)

        if include_tth:
            tth_grp = instrument_grp.create_group("tth")
            tth_grp.create_dataset("data", data=np.asarray(tth))

        if include_mask:
            instrument_eiger_grp.create_dataset(
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

        if include_plan_type and plan_type is not None:
            plan_metadata_grp.create_dataset("plan_type", data=plan_type)

        if include_i0:
            i0_grp = entry_grp.create_group("i0")
            i0_grp.create_dataset("data", data=np.asarray(i0))

    return path
