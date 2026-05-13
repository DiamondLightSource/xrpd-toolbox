from functools import cached_property
from pathlib import Path

import numpy as np
from h5py import Dataset, File

from xrpd_toolbox.utils.utils import get_entry, h5_to_array


def get_dataset(filepath, dataset_path: str) -> Dataset:
    with File(filepath, "r") as file:
        if dataset_path not in file:
            raise ValueError(f"Dataset path {dataset_path} not found in {filepath}")

        data = file.get(dataset_path)

        if not isinstance(data, Dataset):
            raise ValueError(f"{dataset_path} is not a dataset")

        return data


class BaseDataLoader:
    """
    Base class for detector data loaders.
    Handles Nexus/HDF5 access and metadata retrieval.
    """

    def __init__(self, filepath: str | Path, data_path: str):
        self.filepath = Path(filepath)
        self.data_path = data_path

        self.entry = get_entry(self.filepath)
        self.dataset_path = f"/{self.entry}/{self.data_path}/data"

    def get_entries(self):
        paths = []

        with File(self.filepath, "r") as file:
            file.visit(paths.append)

        return paths

    def get_data(self, dataset_path: str | None = None, selection=...) -> np.ndarray:
        dataset_path = dataset_path or self.dataset_path

        with File(self.filepath, "r") as file:
            if dataset_path not in file:
                raise ValueError(
                    f"Dataset path {dataset_path} not found in {self.filepath}"
                )

            data = file.get(dataset_path)

            if data is None or not isinstance(data, Dataset):
                raise ValueError(f"Data at {dataset_path} in {self.filepath} is None.")

            if data.ndim < 1:
                raise ValueError("Data has insufficient dimensions.")

            return np.asarray(data[selection])

    def sum_frames(self):
        data = get_dataset(filepath=self.filepath, dataset_path=self.dataset_path)

        n_frames = data.shape[1]

        summed_images = []

        for frame in range(n_frames):
            frame_image = data[:, frame, :, :]
            image_sum = np.sum(frame_image)

            summed_images.append(image_sum)

        summed_images = np.array(summed_images)

        return summed_images

    @property
    def data(self) -> np.ndarray:
        """Load the entire dataset."""
        return self.get_data()

    @cached_property
    def durations(self) -> np.ndarray:
        path = f"/{self.entry}/instrument/{self.data_path}/count_time"
        return h5_to_array(self.filepath, path)

    def read_array(self, path: str) -> np.ndarray:
        """Helper for reading arbitrary datasets."""
        return h5_to_array(self.filepath, path)
