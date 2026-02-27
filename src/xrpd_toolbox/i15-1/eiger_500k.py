from collections.abc import Collection
from functools import cached_property
from pathlib import Path
from typing import Literal

import numpy as np
from h5py import Dataset, File
from pyFAI.calibrant import get_calibrant
from pyFAI.detectors import Detector
from pyFAI.goniometer import SingleGeometry

from xrpd_toolbox.utils.settings import SettingsBase
from xrpd_toolbox.utils.utils import (
    get_entry,
    h5_to_array,
)

PIXEL_SIZE = 7.5e-5  # in m
INITIAL_DISTNACE = 700  # mm


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


class EigerSettings(SettingsBase):
    bad_channels_filepath: str | Path = "/dls_sw/i15-1/software/bad_channel_mask.hdf5"
    bad_channel_masking: bool = True
    flatfield_filepath: str | Path | None = None
    apply_flatfield: bool = False
    darkfield_filepath: str | Path | None = None
    send_to_ispyb: bool = False
    rebin_step: float = 0.004
    error_calc: Literal["poisson", "std_dev", "max"] = "poisson"
    poni_filepath: str | Path | None = None


class EigerDataLoader:
    def __init__(
        self,
        filepath: str | Path,
        eiger_data_path: str = "eiger",
        tth_path: str = "tth",
    ):
        self.filepath = filepath
        self.eiger_data_path = eiger_data_path
        self.tth_path = tth_path

        self.entry = get_entry(self.filepath)
        self.dataset_path = f"/{self.entry}/{self.eiger_data_path}/data"

    @cached_property
    def positions(self) -> np.ndarray:
        try:
            deltas = h5_to_array(self.filepath, self.tth_path)
            return deltas
        except ValueError as e:
            print(f"{e} - {self.tth_path} in data - returning 0")
            deltas = np.array([0])
            return deltas

    @cached_property
    def count_time_path(self) -> str:
        return f"/{self.entry}/instrument/{self.eiger_data_path}/count_time"

    @cached_property
    def durations(self) -> np.ndarray:
        return h5_to_array(self.filepath, self.count_time_path)

    @property
    def data(self):
        return self.get_data(self.dataset_path)

    def get_frame(self, frame: int | Collection[int] | slice):
        return self.get_data(self.dataset_path)

    def get_data(self, dataset_path) -> np.ndarray:
        with File(self.filepath, "r") as file:
            if self.dataset_path not in file:
                raise ValueError(f"Dataset path {dataset_path} not found in HDF5 file.")

            data = file.get(dataset_path)

            if (data is not None) and isinstance(data, Dataset):
                if data.ndim < 1:
                    raise ValueError("Data has insufficient dimensions.")
                module_frame_data = data[...]

                return np.asarray(module_frame_data)
            else:
                raise ValueError(f"Data at {dataset_path} in {self.filepath}is None.")


class Eiger500K(Detector):
    def __init__(self, filepath: str | Path, settings: EigerSettings):
        self.filepath = filepath
        self.settings = settings

        self.data_loader = EigerDataLoader(self.filepath)

        super().__init__(
            pixel1=PIXEL_SIZE, pixel2=PIXEL_SIZE, max_shape=self.data_loader.data.shape
        )

    def process_step_scan(self):
        for _position in self.data_loader.positions:
            # do geometry transformation

            pass

    def load_geometry(self, poni_files: str | list[str | Path]):
        # if isinstance(poni_files, list) and (len(poni_files) > 1):
        #     mg = MultiGeometry()

        # else:
        #     SingleGeometry()
        pass

    def simulate_data(self, calibrant_name: str, wavelength: float):
        pass

    def calibrate_single_geometry(
        self,
        calibrant_name: str,
        wavelength: float,
        poni_output_filepath: str | Path,
        wavelength_unit: Literal["Ang", "A", "Angstrom", "kev", "keV", "ev"],
    ):
        """Using pyfai and the current data file,
        this will attempt to calibrate the detector
        using a known calibrant and then output a poni file to disk"""

        if wavelength_unit.lower() in ["ang", "ansgtrom", "a"]:
            wavelength_in_ang = wavelength
        elif wavelength_unit.lower() in ["ang", "ansgtrom", "a"]:
            wavelength_in_ang = wavelength
        else:
            raise ValueError("wavelength_unit must be valid!")

        calibrant = get_calibrant(calibrant_name)
        calibrant.wavelength = wavelength_in_ang / 1e10

        single_geometry = SingleGeometry(
            self.name, self.data_loader.data, calibrant=calibrant, detector=self
        )

        single_geometry = calibrate_single_geometry_from_rings(geometry=single_geometry)
        single_geometry.geometry_refinement.save(str(poni_output_filepath))
