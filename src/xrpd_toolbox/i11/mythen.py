import tomllib
from collections.abc import Collection
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import yaml
from h5py import Dataset, File
from pydantic import BaseModel

from xrpd_toolbox.utils.utils import get_entry, load_int_array_from_file


class MythenReductionSettings(BaseModel):
    active_modules: list[int] = list(range(28))
    bad_modules: list[int] = []
    bad_channel_masking: bool = True
    flatfield_filepath: str | Path = ""
    apply_flatfield: bool = False
    modules_in_flatfield: list[int] = list(range(28))
    send_to_ispyb: bool = False
    rebin_step: float = 0.004
    default_counter: int = 0
    edge_bad_channels: int = 15
    error_calc: Literal["internal", "external", "best"] = "internal"
    data_reduction_mode: Literal[
        "step_scan", "time_resolved", "pump_probe", "flat_field", "bad_pixel"
    ] = "step_scan"
    bad_channels_filepath: str | Path = "/dls_sw/i11/software/mythen/badchannels.txt"
    angcal_filepath: str | Path = ""

    @classmethod
    def load_from_toml(cls, file_path: str | Path):
        settings_dict = tomllib.load(open(file_path, "rb"))
        return cls(**settings_dict)

    @classmethod
    def load_from_yaml(cls, file_path: str | Path):
        settings_dict = yaml.safe_load(open(file_path, "rb"))
        return cls(**settings_dict)

    def load_bad_channels(self):
        if not self.bad_channels_filepath:
            raise ValueError("Bad channels file path is not set.")
        self.bad_channels = load_int_array_from_file(self.bad_channels_filepath)
        return self.bad_channels

    def save_to_yaml(self, file_path: str | Path) -> None:
        print("Saving configuration to:", file_path)

        config_dict = self.model_dump()

        with open(file_path, "w") as outfile:
            yaml.dump(
                config_dict,
                outfile,
                default_flow_style=None,
                sort_keys=False,
                indent=2,
                explicit_start=True,
            )


class MythenDataLoader:
    def __init__(
        self,
        file_path: str | Path,
        active_modules: Collection[int] = tuple(range(28)),
        pixels_per_module: int = 1280,
        counter: int = 0,
    ):
        self.file_path = Path(file_path)
        self.active_modules = active_modules
        self.pixels_per_module = pixels_per_module
        self.counter = counter

        self.entry = get_entry(self.file_path)
        self.dataset_path = f"/{self.entry}/mythen_nx/data"

        self.n_modules_in_data, self.n_frames = self.read_nxs_metadata()

        if self.n_modules_in_data != len(self.active_modules):
            raise ValueError("Mismatch between active modules and data.")

        self.get_deltas()

        self.raw_data = self.load_data()
        self.module_data = np.array_split(
            self.raw_data, len(self.active_modules), axis=-1
        )
        self.n_modules = len(self.module_data)

    def get_deltas(self) -> list[float]:
        return [1.0]

    def read_nxs_metadata(self) -> tuple[int, int]:
        with h5py.File(self.file_path, "r") as file:
            data = file.get(self.dataset_path)
            if (data is not None) and isinstance(data, Dataset):
                first_frame = data[0, :, self.counter]
                first_frame_len = first_frame.shape[-1]
                n_modules_in_data = int(first_frame_len / self.pixels_per_module)
                n_frames = len(data)
                return n_modules_in_data, n_frames
            else:
                raise ValueError(f"Data is None at {self.dataset_path}")

    def load_data(self) -> np.ndarray:
        if not self.file_path.exists():
            raise FileNotFoundError(self.file_path)

        with File(self.file_path, "r") as file:
            if self.dataset_path not in file:
                raise ValueError(
                    f"Dataset path {self.dataset_path} not found in HDF5 file."
                )

            data = file.get(self.dataset_path)

            if (data is not None) and isinstance(data, Dataset):
                if data.ndim < 1:
                    raise ValueError("Data has insufficient dimensions.")
                self.n_frames = len(data)
                data = data[..., self.counter]

                return np.asarray(data)
            else:
                raise ValueError("Data is None.")

    def get_module_data(self, module_index: int) -> np.ndarray:
        if module_index not in self.active_modules:
            raise IndexError("Module index out of range.")

        return self.module_data[module_index]


class MythenModule:
    def __init__(self, data, pixels_per_modules: int = 1280):
        self.pixels_per_modules = pixels_per_modules

    def process(self):
        # Example processing: compute the mean
        return np.mean(self.pixels_per_modules)


class MythenDetector:
    def __init__(self, modules_per_detector: int = 28):
        self.modules_per_detector = modules_per_detector


if __name__ == "__main__":
    filepath = "/workspaces/XRPD-Toolbox/src/xrpd_toolbox/i11/mythen_calibration/mythen3_reduction_config.toml"  # noqa

    settings = MythenReductionSettings.load_from_toml(filepath)

    print("Loaded settings:", settings)

    MythenDataLoader("/dls/i11/data/2026/cm44155-1/1406733.nxs")

    # module = MythenModule(data)
    # result = module.process()
    # print("Processed result:", result)
