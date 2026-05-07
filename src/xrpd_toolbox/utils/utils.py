import json
import os
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import datetime
from functools import wraps
from pathlib import Path
from time import time

import h5py
import numpy as np
from h5py import Dataset, File
from pyFAI.calibrant import get_calibrant
from scipy.interpolate import interp1d


def calculate_chi_squared(
    ycalc: np.ndarray, yobs: np.ndarray, y_err: np.ndarray | None
) -> float:
    """calculates chi_squared (the minimisation cost function) that is familiar to
    those who do retiveld refinements"""

    if y_err is not None:
        wi = 1 / y_err**2
    else:
        wi = 1 / yobs

    residual = (wi * (yobs - ycalc)) ** 2
    chi_squared = float(np.sum(residual))

    return chi_squared


# TODO: Decide whether we can just use the normal logging within python
class AnalysisLogger:
    """A class that can be use as a logger to log data to a
    filepath and print at the same time - possibly superfluous"""

    def __init__(self, log_filepath: str | Path, logging: bool = False):
        self.log_filepath = log_filepath
        self.logging = logging

        if not os.path.exists(self.log_filepath):
            os.makedirs(os.path.dirname(self.log_filepath), exist_ok=True)
        elif os.path.exists(self.log_filepath) and (
            os.path.getsize(self.log_filepath) > 1e7
        ):
            os.remove(self.log_filepath)
            with open(self.log_filepath, "a+") as f:
                f.write("Log File for I11 Data Reduction\n")

        with open(self.log_filepath, "a+") as f:
            f.write("================================\n")
            f.write(f"Datetime: {datetime.now()}\n")
            f.write("================================\n")

    def log(self, *args, print_to_console: bool = True):
        if print_to_console:
            print(*args)

        if self.logging:
            with open(self.log_filepath, "a") as f:
                [f.write(str(m)) for m in args]
                f.write("\n")


class NexusDatasetMapper:
    def __init__(self, filepath: str | Path):
        self.filepath = str(filepath)
        self._mapping = defaultdict(list)
        self._build_mapping()

    def _build_mapping(self):
        """Scan file once and build dataset name → path mapping."""
        with h5py.File(self.filepath, "r") as f:

            def visitor(name, obj):
                if isinstance(obj, h5py.Dataset):
                    dataset_name = name.split("/")[-1]
                    self._mapping[dataset_name].append(f"/{name}")

            f.visititems(visitor)

    def find(self, dataset_name):
        """
        Returns:
            - None if not found
            - Single string if exactly one match
            - List of paths if multiple matches
        """
        matches = self._mapping.get(dataset_name)
        if not matches:
            return None
        if len(matches) == 1:
            return matches[0]
        return matches

    def get(self, dataset_name, index=0):
        """
        Lazily open the file and return the h5py.Dataset object.

        NOTE:
        The returned dataset is only valid while the file is open.
        Use within a context manager.
        """
        paths = self._mapping.get(dataset_name)
        if not paths:
            raise KeyError(f"Dataset '{dataset_name}' not found.")

        path = paths[index]

        f = h5py.File(self.filepath, "r")
        return f[path]  # lazy dataset handle

    def keys(self):
        return list(self._mapping.keys())

    def items(self):
        return dict(self._mapping)


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


def normalise_to(
    data: Iterable[float | int],
    minval: float | int = 0,
) -> np.ndarray:
    """
    normalises an array
    minval is the minimum value that the
    processed array is scaled to.
    """

    data_array = np.array(data, dtype=float)

    return (data_array - minval) / (np.amax(data_array) - minval)


def normalise(data: np.ndarray | list, default_min: float = 1e-12) -> np.ndarray:
    """Normalises an array between 0 and 1, a specific version of normalise_to
    where minval is min(data)"""

    # Do not make this use normalise_to it keeps breaking for some reason

    array_min = float(np.amin(data))
    if array_min == 0.0:
        array_min = default_min

    return (data - np.min(data)) / (np.max(data) - array_min)


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
    error_calc: str = "max",
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

    # # Effective scaling factor (heuristic)
    # scale_factpr = scale or np.max(bin_counts) - np.median(bin_counts)

    with np.errstate(divide="ignore", invalid="ignore"):
        intensity = y_sums / bin_counts

        # --- Errors ---
        # Internal (propagated) error
        prop_errors = np.sqrt(e2_sums) / bin_counts

        # External (sample) error with Bessel correction
        variance = (y2_sums - bin_counts * intensity**2) / (bin_counts - 1)
        std_errors = np.sqrt(variance)

        if error_calc == "poisson":
            errors = prop_errors
        elif error_calc == "std_dev":
            # Invalidate bins with <2 points for external error
            std_errors[bin_counts < 2] = np.nan
            errors = std_errors
        elif error_calc == "max":
            std_errors[bin_counts < 2] = 0
            errors = np.maximum(prop_errors, std_errors)
        else:
            raise ValueError(f"Invalid error_calc: {error_calc}")

        ##remove NaN's - bins without counts
        nan_mask = np.isnan(intensity)
        nan_index = np.where(nan_mask)[0]

        bin_centres = np.delete(bin_centres, nan_index)
        intensity = np.delete(intensity, nan_index)
        errors = np.delete(errors, nan_index)

    return bin_centres, intensity, errors


def bin_and_propagate_errors_norm(
    x: np.ndarray,
    y: np.ndarray,
    e: np.ndarray,
    rebin_step: float | int,
    error_calc: str = "max",
    weighting: None | np.ndarray = None,
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
        x[-1] -= rebin_step / 1e3

        # --- Bin statistics ---

    y_sums, _ = np.histogram(x, bins=bin_edges, weights=y)
    y2_sums, _ = np.histogram(x, bins=bin_edges, weights=y**2)
    e2_sums, _ = np.histogram(x, bins=bin_edges, weights=e**2)
    bin_counts, _ = np.histogram(x, bins=bin_edges)

    # # Effective scaling factor (heuristic)
    # scale_factpr = scale or np.max(bin_counts) - np.median(bin_counts)

    with np.errstate(divide="ignore", invalid="ignore"):
        intensity = y_sums

        # --- Errors ---
        # Internal (propagated) error
        prop_errors = np.sqrt(e2_sums)

        # External (sample) error with Bessel correction
        variance = (y2_sums - bin_counts * intensity**2) / (bin_counts - 1)
        std_errors = np.sqrt(variance)

        if error_calc == "poisson":
            errors = prop_errors
        elif error_calc == "std_dev":
            # Invalidate bins with <2 points for external error
            std_errors[bin_counts < 2] = np.nan
            errors = std_errors
        elif error_calc == "max":
            std_errors[bin_counts < 2] = 0
            errors = np.maximum(prop_errors, std_errors)
        else:
            raise ValueError(f"Invalid error_calc: {error_calc}")

        ##remove NaN's - bins without counts
        nan_mask = np.isnan(intensity)
        nan_index = np.where(nan_mask)[0]

        bin_centres = np.delete(bin_centres, nan_index)
        intensity = np.delete(intensity, nan_index)
        errors = np.delete(errors, nan_index)

        if weighting is not None:
            w_sums, _ = np.histogram(x, bins=bin_edges, weights=weighting)
            scale = np.delete(w_sums, nan_index)

            intensity = intensity / scale
            errors = errors / scale

    return bin_centres, intensity, errors


def save_to_xye(xye_filepath_out, x: np.ndarray, y: np.ndarray, e: np.ndarray):
    """Takes in 3 equal sized arrays and writes them to a typical XRPD csv/xye file"""
    xye_out_data = np.stack((x, y, e), axis=-1)
    np.savetxt(xye_filepath_out, xye_out_data, fmt="%.6f", delimiter=" ", newline="\n")


def save_data_to_h5(filepath: str | Path, dataset_path: str, data: np.ndarray) -> None:
    group_path, name = dataset_path.rsplit("/", 1)

    with File(filepath, "a") as file:
        if dataset_path in file:
            del file[dataset_path]

        group = file.require_group(group_path)
        group.create_dataset(name, data=data, compression="gzip", compression_opts=4)


def get_calibrant_peaks(calibrant_name: str, wavelength_in_ang: float):
    calibrant = get_calibrant(calibrant_name)
    calibrant.wavelength = wavelength_in_ang / 1e10
    observed_reflections_in_tth = calibrant.get_peaks("2th_deg")

    return observed_reflections_in_tth


def rebin_together(x1, y1, x2, y2, num_points=None):
    # 1. Define overlapping x-range
    xmin = max(x1.min(), x2.min())
    xmax = min(x1.max(), x2.max())

    # 2. Decide number of points
    if num_points is None:
        num_points = min(len(x1), len(x2))

    # 3. Create common evenly spaced grid
    x_common = np.linspace(xmin, xmax, num_points)

    # 4. Interpolate both datasets
    f1 = interp1d(x1, y1, kind="linear", bounds_error=False, fill_value="extrapolate")  # type: ignore
    f2 = interp1d(x2, y2, kind="linear", bounds_error=False, fill_value="extrapolate")  # type: ignore

    y1_interp = f1(x_common)
    y2_interp = f2(x_common)

    return x_common, y1_interp, y2_interp


def timeit(f):
    """use this as a decorator to time a function

    eg:

    @timeit
    def my_func()
        time.sleep(1)

    myfunc()

    - my_func took 1 seconds

    """

    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        duration = te - ts
        print(f"{f.__name__} took {duration} seconds")
        return result

    return wrap


def parse_numbers_with_error(
    data: Sequence[str] | str | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Takes a str or list of str in the format 1.234(5)
    and returns two arrays: one with the numbers and one with the errors.
    ie 1.234(5) -> 1.234, 0.005"""

    arr = np.atleast_1d(np.array(data, dtype=str))
    arr = np.char.replace(arr, " ", "")

    # Split number and optional error
    num_str = np.char.partition(arr, "(")[:, 0]
    err_str = np.char.strip(np.char.partition(arr, "(")[:, 2], ")")
    err_str[err_str == ""] = "0"

    numbers = num_str.astype(float)

    # Compute decimal places safely as float
    decimals = np.where(
        np.char.find(num_str, ".") == -1,
        0,
        np.char.str_len(num_str) - np.char.find(num_str, ".") - 1,
    )
    errors = err_str.astype(float) * 10.0 ** (-decimals.astype(float))

    if isinstance(data, str):
        return numbers[0], errors[0]
    else:
        return numbers, errors


def cluster_points_auto(
    x: np.ndarray,
    y: np.ndarray,
    y_tol: float | None = None,
    x_gap: float | None = None,
):
    """
    Groups points based on:
    - Similar Y values (anchored to first point in group)
    - Local continuity in X

    Returns:
        labels: continuous group labels ordered left → right by X
        n_groups: number of groups
    """

    x = np.asarray(x)
    y = np.asarray(y)

    if y_tol is None:
        y_tol = (y.max() - y.min()) / 20

    if x_gap is None:
        x_gap = (x.max() - x.min()) / 5

    # IMPORTANT: sort by X first to enforce ordering consistency
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]

    n = len(x)
    labels_sorted = np.full(n, -1, dtype=int)

    n_groups = 0
    start = 0

    while start < n:
        end = start + 1

        while end < n:
            y_diff = abs(y_sorted[end] - y_sorted[start])
            x_diff = abs(x_sorted[end] - x_sorted[end - 1])

            if y_diff <= y_tol and x_diff <= x_gap:
                end += 1
            else:
                break

        labels_sorted[start:end] = n_groups
        n_groups += 1
        start = end

    # map back to original order
    labels = np.empty(n, dtype=int)
    labels[order] = labels_sorted

    return labels, n_groups
