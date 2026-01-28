import tomllib
from collections.abc import Collection
from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from h5py import Dataset, File
from pydantic import BaseModel


class MythenReductionSettings(BaseModel):
    active_modules: list[int]
    bad_modules: list[int]
    bad_channel_masking: bool
    flatfield_filepath: str | Path
    apply_flatfield: bool
    modules_in_flatfield: list[int]
    send_to_ispyb: bool
    rebin_step: float
    default_counter: int
    edge_bad_channels: int
    error_calc: Literal["internal", "external", "best"]
    data_reduction_mode: int
    bad_channels_filepath: str | Path
    angcal_filepath: str | Path

    @classmethod
    def load_from_toml(cls, file_path: str | Path):
        settings_dict = tomllib.load(open(file_path, "rb"))
        return cls(**settings_dict)

    @classmethod
    def load_from_yaml(cls, file_path: str | Path):
        settings_dict = yaml.safe_load(open(file_path, "rb"))
        return cls(**settings_dict)

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
        counter: int = 0,
    ):
        self.file_path = Path(file_path)
        self.active_modules = active_modules
        self.counter = counter
        self.dataset_path = "/entry/mythen_nx/data"
        self.raw_data = self.load_data()
        self.module_data = np.array_split(
            self.raw_data, len(self.active_modules), axis=-1
        )
        self.n_modules = len(self.module_data)

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
    settings = MythenReductionSettings.load_from_toml(
        "/Users/akz63626/projects/XRPD-Toolbox/src/xrpd_toolbox/i11/mythen_calibration/mythen3_reduction_config.toml"
    )

    print("Loaded settings:", settings)

    MythenDataLoader("/Users/akz63626/cm44155-1/1407178.nxs")

    # module = MythenModule(data)
    # result = module.process()
    # print("Processed result:", result)
