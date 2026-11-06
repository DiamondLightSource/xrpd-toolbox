from collections.abc import Collection
from pathlib import Path
from typing import Self

import numpy as np
from h5py import Dataset, File


class BaseDataLoader:
    """
    Base class for detector data loaders.
    Reads the detector data in one nexus file, keeping the file open.
    """

    def __init__(self, filepath: str | Path, dataset_path: str):
        self.filepath = str(filepath)
        self._file: File | None = None

        self.entry = list(self.file.keys())[0]  # /entry

        self.dataset_path = dataset_path

    @property
    def file(self) -> File:
        if self._file is None:
            self._file = File(self.filepath, "r", libver="latest", swmr=True)
        return self._file

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def _read_array(self, data_path: str) -> np.ndarray:
        data = self.file.get(data_path)
        if (data is not None) and isinstance(data, Dataset):
            return np.asarray(data)
        else:
            raise ValueError(f"Data is None at {data_path} in {self.filepath}")

    def _read_string(self, data_path: str) -> str:
        data = self.file.get(data_path)
        if (data is not None) and isinstance(data, Dataset):
            value = data[()]

            if isinstance(value, bytes):
                value = value.decode()

            return value
        else:
            raise ValueError(f"Data is None at {data_path} in {self.filepath}")

    @property
    def dataset(self) -> Dataset:
        """The detector dataset, without loading any data"""

        if self.dataset_path not in self.file:
            raise ValueError(
                f"Dataset path {self.dataset_path} not found in HDF5 file: "
                f"{self.filepath}"
            )

        data = self.file.get(self.dataset_path)

        if not isinstance(data, Dataset):
            raise ValueError(f"Data is None at {self.dataset_path} in {self.filepath}")

        if data.ndim < 1:
            raise ValueError("Data has insufficient dimensions.")

        return data

    def get_entries(self) -> list[str]:
        paths = []
        self.file.visit(paths.append)
        return paths

    def get_data(self, frames: int | Collection[int] | slice) -> np.ndarray:
        return self.dataset[frames, ...]

    def load_all_data(self) -> np.ndarray:
        """Loads every frame into memory - prefer get_data with a slice."""
        return self.get_data(frames=slice(None))

    def get_data_dimensions(self) -> tuple[int, ...]:
        return self.dataset.shape

    def get_number_of_frames(self) -> int:
        """Number of detector frames, read from the shape without loading any data."""
        return self.dataset.shape[0]

    def _frame_indices(self) -> list[tuple[int, ...]]:
        """Every index of the dataset that isn't one of the 2 image axes"""

        data = self.dataset

        if data.ndim < 2:
            raise ValueError(f"Expected image data with ndim >= 2, got {data.ndim}")

        return list(np.ndindex(data.shape[:-2]))

    def sum_frames(self) -> np.ndarray:
        """Total counts in each frame."""

        data = self.dataset
        frame_indices = self._frame_indices()
        totals = np.zeros(len(frame_indices), dtype=np.float64)

        for n, index in enumerate(frame_indices):
            totals[n] = np.nansum(data[index], dtype=np.float64)

        return totals

    def max_value_per_frame(self) -> np.ndarray:
        """Maximum pixel value in each frame."""

        data = self.dataset
        frame_indices = self._frame_indices()
        max_values = np.zeros(len(frame_indices), dtype=np.float64)

        for n, index in enumerate(frame_indices):
            max_values[n] = np.nanmax(data[index])

        return max_values
