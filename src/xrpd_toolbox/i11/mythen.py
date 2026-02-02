import json
import math
import os
import re
import tomllib
from collections import OrderedDict
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import toml
import yaml
from h5py import Dataset, File
from pydantic import BaseModel

from xrpd_toolbox.utils.utils import (
    get_entry,
    h5_to_array,
    load_int_array_from_file,
)

SUPPORTED_FILE_TYPES = ["json", "toml", "yaml"]


class SettingsBase(BaseModel):
    @classmethod
    def load_from_toml(cls, file_path: str | Path):
        with open(file_path, "rb") as file:
            settings_dict = tomllib.load(file)

        return cls(**settings_dict)

    @classmethod
    def load_from_yaml(cls, file_path: str | Path):
        with open(file_path, "rb") as file:
            settings_dict = yaml.safe_load(file)
        return cls(**settings_dict)

    @classmethod
    def load_from_json(cls, file_path: str | Path):
        with open(file_path, "rb") as file:
            settings_dict = json.load(file)
        return cls(**settings_dict)

    @classmethod
    def load(cls, file_path: str | Path):
        filename, file_extension = os.path.splitext(str(file_path))

        if file_extension == "json":
            return cls.load_from_json(file_path)
        elif file_extension == "yaml":
            return cls.load_from_yaml(file_path)
        elif file_extension == "toml":
            return cls.load_from_toml(file_path)
        else:
            raise ValueError(f"Filetype must be: {SUPPORTED_FILE_TYPES}")

    def save_to_toml(self, file_path: str | Path) -> None:
        if not str(file_path).endswith(".toml"):
            raise ValueError("file_path name must end with .toml")

        print("Saving configuration to:", file_path)

        config_dict = self.model_dump()

        with open(file_path, "w") as outfile:
            toml.dump(config_dict, outfile)

    def save_to_json(self, file_path: str | Path) -> None:
        if not str(file_path).endswith(".json"):
            raise ValueError("file_path name must end with .json")

        print("Saving configuration to:", file_path)

        config_dict = self.model_dump()

        with open(file_path, "w") as outfile:
            json.dump(config_dict, outfile, indent=2, sort_keys=False)

    def save_to_yaml(self, file_path: str | Path) -> None:
        if not str(file_path).endswith(".yaml"):
            raise ValueError("file_path name must end with .yaml")

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


class ModuleConversion(BaseModel):
    conv: float
    offset: float

    @property
    def module_sign(self) -> int:  # returns -1 or 1 depending on sign of conv
        return int(math.copysign(1, self.conv))


class AngularCalibration(SettingsBase):
    beamline_offset: float
    centre: float
    module_0: ModuleConversion
    module_1: ModuleConversion
    module_2: ModuleConversion
    module_3: ModuleConversion
    module_4: ModuleConversion
    module_5: ModuleConversion
    module_6: ModuleConversion
    module_7: ModuleConversion
    module_8: ModuleConversion
    module_9: ModuleConversion
    module_10: ModuleConversion
    module_11: ModuleConversion
    module_12: ModuleConversion
    module_13: ModuleConversion
    module_14: ModuleConversion
    module_15: ModuleConversion
    module_16: ModuleConversion
    module_17: ModuleConversion
    module_18: ModuleConversion
    module_19: ModuleConversion
    module_20: ModuleConversion
    module_21: ModuleConversion
    module_22: ModuleConversion
    module_23: ModuleConversion
    module_24: ModuleConversion
    module_25: ModuleConversion
    module_26: ModuleConversion
    module_27: ModuleConversion


class MythenReductionSettings(SettingsBase):
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

    def load_bad_channels(self):
        if not self.bad_channels_filepath:
            raise ValueError("Bad channels file path is not set.")
        self.bad_channels = load_int_array_from_file(self.bad_channels_filepath)
        return self.bad_channels


class MythenDataLoader:
    def __init__(
        self,
        file_path: str | Path,
        pixels_per_module: int = 1280,
        counter: int = 0,
        mythen_data_path="mythen_nx",
    ):
        self.file_path = Path(file_path)
        self.pixels_per_module = pixels_per_module
        self.counter = counter
        self.mythen_data_path = mythen_data_path

        self.entry = get_entry(self.file_path)
        self.dataset_path = f"/{self.entry}/{self.mythen_data_path}/data"

        self.n_modules_in_data, self.n_frames = self.read_nxs_metadata()

        self.delta_path = self.get_delta_path()
        self.deltas_or_ds = self.get_deltas()

        self.raw_data = self.load_data(self.counter)
        self.module_data = np.array_split(
            self.raw_data, self.n_modules_in_data, axis=-1
        )

    def get_delta_path(self) -> str:
        delta_subpaths = ("delta", "deltas", "ds")

        with h5py.File(self.file_path, "r") as file:
            base = f"/{self.entry}/{self.mythen_data_path}"

            for name in delta_subpaths:
                path = f"{base}/{name}"
                if path in file and isinstance(file[path], h5py.Dataset):
                    return path

            raise KeyError(
                f"No delta dataset found. Tried: "
                f"{', '.join(f'{base}/{n}' for n in delta_subpaths)}"
            )

    def get_deltas(self) -> np.ndarray:
        deltas_or_ds = h5_to_array(self.file_path, self.delta_path)

        return deltas_or_ds

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

    def load_data(self, counter: int) -> np.ndarray:
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
                data = data[..., counter]

                return np.asarray(data)
            else:
                raise ValueError("Data is None.")

    def get_module_data(self, module_index):
        return self.module_data[module_index]


class MythenModule:
    def __init__(self, data: np.ndarray, pixels_per_modules: int = 1280):
        self.pixels_per_modules = pixels_per_modules
        self.data = data

    def process(self):
        # Example processing: compute the mean
        return np.mean(self.pixels_per_modules)


class MythenDetector:
    def __init__(
        self,
        file_path: str | Path,
        settings: MythenReductionSettings | None = None,
        angular_calibration: AngularCalibration | None = None,
    ):
        self.file_path = file_path
        self.settings = settings or MythenReductionSettings()
        self.angular_calibration = angular_calibration

        # mythen data loader, just loads the data,
        # it has no information about which modules are which
        self.mythen_data = MythenDataLoader(
            file_path=file_path,
        )

        self.modules = OrderedDict()

        for n_module, module in enumerate(self.settings.active_modules):
            self.modules[module] = MythenModule(
                data=self.mythen_data.get_module_data(n_module)
            )


def convert_angcal_to_pydantic_json(
    ang_cal_json_path: str | Path, new_path: str | Path
):
    pydantic_dict = {}

    with open(ang_cal_json_path, "rb") as file:
        legacy_dict = json.load(file)

    pydantic_dict["beamline_offset"] = legacy_dict["beamline_offset"]
    pydantic_dict["centre"] = legacy_dict["centre"]

    for entry in legacy_dict.keys():
        numbers = re.findall(r"-?\d*\.?\d+", str(entry))

        if len(numbers) > 0:
            module = numbers[0]
            pydantic_dict[f"module_{module}"] = {
                "conv": legacy_dict[f"conv_{module}"],
                "offset": legacy_dict[f"offset_{module}"],
            }

    pydantic_model = AngularCalibration(**pydantic_dict)
    pydantic_model.save_to_json(new_path)


if __name__ == "__main__":
    PARENT_PATH = Path(__file__).parent.parent

    print(PARENT_PATH)

    CONFIG_FILE = (
        PARENT_PATH / "i11" / "mythen_calibration" / "mythen3_reduction_config.toml"
    )

    DATA_FILE = "/workspaces/XRPD-Toolbox/examples/i11/step_scan/1406731.nxs"

    ANG_CAL = PARENT_PATH / "i11" / "mythen_calibration" / "ang_cal_171125_new.json"
    settings = MythenReductionSettings.load_from_toml(CONFIG_FILE)
    print("Loaded settings:", settings)

    print(DATA_FILE)

    MythenDataLoader(DATA_FILE)

    angular_calibration = AngularCalibration.load_from_json(ANG_CAL)

    # MythenDetector(
    #     file_path=DATA_FILE,
    #     settings=settings,
    #     angular_calibration=angular_calibration
    # )
