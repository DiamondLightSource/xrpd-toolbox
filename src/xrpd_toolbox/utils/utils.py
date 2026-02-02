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
        os.path.join(instrument_session_folder, f)
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


def normalise(data) -> np.ndarray:
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def gaussian(x, amp, cen, fwhm, background) -> np.ndarray:
    # "1-d gaussian: gaussian(x, amp, cen, fwhm)"

    return (amp / (np.sqrt(2 * np.pi) * fwhm)) * np.exp(
        -((x - cen) ** 2) / (2 * fwhm**2)
    ) + background


def load_int_array_from_file(filepath: str | Path) -> np.ndarray:
    """
    File format is just a list of integers in a text file, one integer per line.

    If no file, will return no empty array.

    If empty file, will return no empty array.

    """

    if not os.path.exists(filepath):
        return np.array([])
    elif os.path.getsize(filepath) == 0:
        return np.array([])
    else:
        return np.loadtxt(filepath, dtype=np.int64, comments="#", usecols=0, ndmin=1)


def create_bins(
    tth_values: np.ndarray, rebin_step: float | int = 0.004
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return a suitable set of bin centres, and edges for histogramming this data.

    To match old GDA mythen2 behaviour, want start and stop to align with "multiples"ß
    of rebin step (as far as f.p. arithmetic allows this...).

    """
    rebin_step = float(rebin_step)
    mintth, maxtth = np.amin(tth_values), np.amax(tth_values)
    start = np.round((mintth / rebin_step), decimals=3) * rebin_step
    stop = np.round((maxtth / rebin_step), decimals=3) * rebin_step

    # start = mintth
    # stop = maxtth
    # self.logger.log("Min2th:",f'{start:.3f}'," | ","Max2th:",f"{stop:.3f}","\n")

    rebin_start = start - (rebin_step / 2)
    rebin_stop = stop + rebin_step + (rebin_step / 2)

    bin_edges = np.arange(
        rebin_start,
        rebin_stop,
        float(rebin_step),
        dtype=np.float64,
    )
    bin_centres = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    return bin_centres, bin_edges


def bin_and_propagate_errors(
    x: np.ndarray,
    y: np.ndarray,
    e: np.ndarray,
    rebin_step: float | int = 0.004,
    error_calc: str = "best",
    sum_counts: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """

    The bin centres and edges are calculated and used to bin the data.
    Binning of the data is done used searchsorted == np.digitize.

    Because we want to propagate the errors we will iterate though all the values of
    x, y and e that need to be binned together and propagate the errors

    Errors can be calculated using internal error = error propagation, external error
    std_dev of error or we can take the greatest of the two values.
    Which is probabaly the best idea.

    If you have a high spread of data (high noise), ie peaks with weak intensity surely
    the error can't be less than the spread.
    But equally if you have very large peaks with low spread the
    error should reflect that.

    """

    bin_centres, bin_edges = create_bins(x, rebin_step)

    if (
        x[-1] == bin_edges[-1]
    ):  # if the last value is exactly equal to the final bin edge it will be lost.
        x[-1] = x[-1] - (
            rebin_step / 10000
        )  # I think it would be better to move it inside bin edge, and include, rather
        # than remove all together or create a bin with a single value

    sums, bin_edges = np.histogram(x, bins=bin_edges, weights=y)
    occurances = np.histogram(x, bins=bin_edges)[0]

    # occurances = np.where(occurances != 0, occurances, 1)  # avoid division by zero

    if sum_counts:
        binned_counts = sums
    else:
        binned_counts = (
            sums / occurances
        )  # throws a warning if e missing counts in a bin as a result of missing module

    e_sums = np.histogram(x, bins=bin_edges, weights=e**2)[0]
    # https://faraday.physics.utoronto.ca/PVB/Harrison/ErrorAnalysis/Propagation.html
    prop_errors = np.sqrt(e_sums) / occurances

    repeated_mean = np.repeat(binned_counts, occurances)
    std_sums = np.histogram(x, bins=bin_edges, weights=(y - repeated_mean) ** 2)[0]
    std_errors = np.sqrt(std_sums / occurances)

    if error_calc == "internal":
        errors = prop_errors
    elif error_calc == "external":
        errors = std_errors
    else:
        errors = np.where(prop_errors > std_errors, prop_errors, std_errors)

    return bin_centres, binned_counts, errors
