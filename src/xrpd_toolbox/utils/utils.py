import json
import os
import re
from collections.abc import Iterable
from pathlib import Path

import h5py
import numpy as np
from h5py import Dataset, File


class NexusToDict:
    def __init__(self, nexus_filepath: str | Path):
        self.nexus_dict = {}

        with File(nexus_filepath, "r") as open_nexus_file:
            self.recursive_inspect(open_nexus_file)

    def recursive_inspect(self, dataset):
        for key in dataset.keys():
            if hasattr(dataset[key], "keys"):
                self.recursive_inspect(dataset[key])
            else:
                try:
                    self.nexus_dict[key] = dataset[key][()].decode("UTF-8")
                except Exception:
                    data = dataset[key][()]

                    if isinstance(data, np.ndarray) and (len(data) == 1):
                        data = data.flatten()[0]

                        if isinstance(data, bytes):
                            data = data.decode("UTF-8")

                    self.nexus_dict[key] = data

    def get_dict(self) -> dict:
        return self.nexus_dict


def h5_to_array(file_path: str | Path, data_path: str) -> np.ndarray:
    with File(file_path, "r") as file:
        data = file.get(data_path)
        if (data is not None) and isinstance(data, Dataset):
            return np.asarray(data)
        else:
            raise ValueError(f"Data is None at {data_path} in {file_path}")


def get_entry(nexus_filepath: str | Path) -> str:
    with File(nexus_filepath, "r") as file:
        return list(file.keys())[0]


def dict_to_json(dict_to_save: dict, filepath: str | Path):
    with open(filepath, "w") as file:
        json.dump(dict_to_save, file, indent=4)


def nexus_file_match(str_to_match, beamline: str = "i15-1"):
    return re.match(f"{beamline}" + r"-+[0-9]+\.nxs", str_to_match)


def get_nexus_files(
    instrument_session_folder: str | Path,
    beamline: str = "i15-1",
    exclude: str = "processed",
) -> list[str]:
    """Get all final data files ending with .nxs in some folder"""

    nexus_files = [
        os.path.join(str(instrument_session_folder), f)
        for f in os.listdir(instrument_session_folder)
        if nexus_file_match(f, beamline) and (exclude not in f)
    ]
    nexus_files.sort()

    return nexus_files


def get_filenumber_from_nxs(nexus_filepath: str | Path) -> int:
    basename = os.path.basename(str(nexus_filepath))
    filenumber_str = re.findall(r"\d+", basename)[-1]
    return int(filenumber_str)


def get_folder_paths(root_folder: str | Path) -> list[str]:
    """get all folder directories within another folder"""

    instrument_session_folders = [
        os.path.join(root_folder, f) for f in os.listdir(root_folder)
    ]
    instrument_session_folders.sort()

    return instrument_session_folders


def copy_datapath_to_nexus(
    source_file: str | Path,
    destination_file: str | Path,
    source_path: str,
    destination_path: str,
) -> None:
    """
    Copy a detector group from a source NeXus file to a destination NeXus file.
    Overwrites the destination path if it already exists.
    """

    with h5py.File(source_file, "r") as src, h5py.File(destination_file, "a") as dst:
        if source_path not in src:
            raise KeyError(f"Source path not found: {source_path}")

        # Remove destination path if it exists (overwrite behavior)
        if destination_path in dst:
            del dst[destination_path]

        # Ensure parent group exists
        parent_path = destination_path.rsplit("/", 1)[0]
        dst.require_group(parent_path)

        # Copy the group recursively
        src.copy(source_path, dst, destination_path)


def normalise_to(data: Iterable[float | int], minval: float | int = 0) -> np.ndarray:
    """
    normalises an array
    minval is  the minimum value that the
    processed array is scaled to.
    """

    data_array = np.array(data, dtype=float)

    return (data_array - minval) / (np.amax(data_array) - minval)


def normalise(data: np.ndarray | list) -> np.ndarray:
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def load_int_array_from_file(filepath: str | Path) -> np.ndarray:
    """
    File format is just a list of integers in a text file, one integer per line.

    If no file, will raise

    If empty file, will return no empty array.

    """

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{filepath} does not exist")
    elif os.path.getsize(filepath) == 0:
        return np.array([])
    else:
        return np.loadtxt(filepath, dtype=np.int64, comments="#", usecols=0, ndmin=1)


def create_bins(
    x: np.ndarray, rebin_step: float | int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create uniform bins for x, returning:
    - bin_edges: the edges of the bins
    - bin_centers: center of each bin
    - indices: which bin each x belongs to
    """
    bin_edges = np.arange(x.min(), x.max() + rebin_step, rebin_step)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    indices = np.digitize(x, bin_edges) - 1  # 0-based
    return bin_centers, bin_edges, indices


def bin_and_propagate_errors(
    x: np.ndarray,
    y: np.ndarray,
    e: np.ndarray,
    rebin_step: float | int,
    error_calc: str = "best",
    sum_counts: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rebin diffraction data with overlap-corrected averaging.

    sum_counts=False:
        return the average diffraction pattern

    sum_counts=True:
        return the pattern scaled by the effective number of detector steps
    """
    # --- Validation ---
    if not (x.ndim == y.ndim == e.ndim == 1):
        raise ValueError("x, y, and e must be 1D arrays")
    if not (x.shape == y.shape == e.shape):
        raise ValueError("x, y, and e must have the same length")

    # --- Bin definition ---
    bin_centres, bin_edges, _ = create_bins(x, rebin_step)

    # Prevent last point from falling exactly on the final bin edge
    if x[-1] == bin_edges[-1]:
        x = x.copy()
        x[-1] -= rebin_step / 10_000

    # --- Bin statistics ---
    y_sums, _ = np.histogram(x, bins=bin_edges, weights=y)
    y2_sums, _ = np.histogram(x, bins=bin_edges, weights=y**2)
    e2_sums, _ = np.histogram(x, bins=bin_edges, weights=e**2)
    bin_counts, _ = np.histogram(x, bins=bin_edges)

    # Effective scaling factor (heuristic)
    scale = np.max(bin_counts) - np.median(bin_counts)

    with np.errstate(divide="ignore", invalid="ignore"):
        mean = y_sums / bin_counts
        intensity = mean if not sum_counts else mean * scale

        # --- Errors ---
        # Internal (propagated) error
        prop_errors = np.sqrt(e2_sums) / bin_counts

        # External (sample) error with Bessel correction
        variance = (y2_sums - bin_counts * mean**2) / (bin_counts - 1)
        std_errors = np.sqrt(variance)

        # Invalidate bins with <2 points for external error
        std_errors[bin_counts < 2] = np.nan

        if error_calc == "internal":
            errors = prop_errors
        elif error_calc == "external":
            errors = std_errors
        elif error_calc == "best":
            errors = np.maximum(prop_errors, std_errors)
        else:
            raise ValueError(f"Invalid error_calc: {error_calc}")

        if sum_counts:
            errors *= scale

        ##remove Nan's
        nan_mask = np.isnan(intensity)
        nan_index = np.where(nan_mask)[0]

        bin_centres = np.delete(bin_centres, nan_index)
        intensity = np.delete(intensity, nan_index)
        errors = np.delete(errors, nan_index)

    return bin_centres, intensity, errors


def save_to_xye(xye_filepath_out, x: np.ndarray, y: np.ndarray, e: np.ndarray):
    xye_out_data = np.stack((x, y, e), axis=-1)

    np.savetxt(xye_filepath_out, xye_out_data, fmt="%.6f", delimiter=" ", newline="\n")
