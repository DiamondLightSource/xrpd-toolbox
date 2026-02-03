import json
import math
import os
import re
from collections import OrderedDict
from functools import cached_property
from pathlib import Path
from shutil import copy2
from typing import Literal

import h5py
import numpy as np
from h5py import Dataset, File
from pydantic import BaseModel

from xrpd_toolbox.utils.daq_messenger import DaqMessenger
from xrpd_toolbox.utils.mythen_utils import channel_to_angle
from xrpd_toolbox.utils.settings import SettingsBase
from xrpd_toolbox.utils.utils import (
    bin_and_propagate_errors,
    get_entry,
    h5_to_array,
    load_int_array_from_file,
)

# np.set_printoptions(threshold=sys.maxsize)

MODULES_IN_DETECTOR = 28
PIXELS_PER_MODULE = 1280
PSD_RADIUS = 762  # mm
MYTHEN_PIXEL_SIZE = 0.05  # mm


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

    def __getitem__(self, name):
        if name in AngularCalibration.model_fields:
            value = getattr(self, name)
            return value
        else:
            raise ValueError(f"{name} not in {self}")


class MythenReductionSettings(SettingsBase):
    active_modules: list[int] = list(range(MODULES_IN_DETECTOR))
    bad_modules: list[int] = []
    bad_channel_masking: bool = True
    flatfield_filepath: str | Path = ""
    apply_flatfield: bool = False
    modules_in_flatfield: list[int] = list(range(MODULES_IN_DETECTOR))
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


class BadChannels:
    def __init__(self, filepath: str | Path):
        self.filepath = filepath
        self.array = self.load_bad_channels()
        self.masks = self.bad_channels_to_mask()

    def load_bad_channels(self):
        if not self.filepath:
            raise ValueError("Bad channels file path is not set.")
        self.bad_channels = load_int_array_from_file(self.filepath)
        return self.bad_channels

    def _split_bad_channels_into_modules(self, bad_channels: np.ndarray):
        """Takes a long array eg. 0, 1, 67, 7655, 32000
        turns it into 28 arrays, [0, 1, 67], ... [67], ... [7655], ...[32000]
        """

        bins = np.arange(
            0, MODULES_IN_DETECTOR * PIXELS_PER_MODULE + 1, PIXELS_PER_MODULE
        )
        indices = (
            np.digitize(bad_channels, bins) - 1
        )  # subtract 1 because digitize returns 1-based

        detector_bad_channels_per_module = [
            bad_channels[indices == i] for i in range(MODULES_IN_DETECTOR)
        ]

        return detector_bad_channels_per_module

    def bad_channels_to_mask(self) -> OrderedDict:
        """PyFAI considers masks with values equal to zero 0 as valid pixels (mnemonic:
        non zero pixels are masked out) - Therefore we keep the convention here
        https://pyfai.readthedocs.io/en/latest/conventions.html

        Takes the long araay of numbers and convert, used by the SLSDet Package
        and converts it to a module by module mask
        """

        bad_channel_mask = OrderedDict()

        bad_channels_per_module = self._split_bad_channels_into_modules(
            self.bad_channels
        )

        module_bad_channels = [
            bad_channels_per_module[f] - (PIXELS_PER_MODULE * f)
            for f in range(MODULES_IN_DETECTOR)
        ]

        for module in range(MODULES_IN_DETECTOR):
            mask = np.zeros(PIXELS_PER_MODULE, dtype=bool)
            mask[module_bad_channels[module]] = True
            bad_channel_mask[module] = mask

        return bad_channel_mask  # bad channels are 1


class MythenDataLoader:
    def __init__(
        self,
        file_path: str | Path,
        pixels_per_module: int = PIXELS_PER_MODULE,
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

    @cached_property
    def deltas(self) -> np.ndarray:
        try:
            self.delta_path = self.get_delta_path()
            deltas = h5_to_array(self.file_path, self.delta_path)
            return deltas
        except ValueError as e:
            print(f"{e} - {self.delta_path} in data - returning 0")
            deltas = np.array([0])
            return deltas

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
                raise ValueError(
                    f"Data at {self.dataset_path} in {self.file_path}is None."
                )


class MythenModule:
    def __init__(
        self,
        data: np.ndarray,
        conversion: ModuleConversion,
        centre: float | int,
        beamline_offset: float,
        deltas: np.ndarray,
        module_id: int,
        bad_channel_mask: np.ndarray | None = None,
        pixels_per_modules: int = PIXELS_PER_MODULE,
    ):
        self.data = data
        self.conversion = conversion
        self.centre = centre
        self.beamline_offset = beamline_offset
        self.deltas = deltas
        self.module_id = module_id
        self.pixels_per_modules = pixels_per_modules

        if bad_channel_mask is None:
            self.bad_channel_mask = np.zeros(self.data.shape[-1], dtype=bool)
        else:
            self.bad_channel_mask = bad_channel_mask

    @cached_property
    def pixel_number(self) -> np.ndarray:
        """returns a 1d array of integers between 0 and 1280"""
        return np.arange(self.pixels_per_modules, dtype=np.int64)

    @cached_property
    def raw_tth(self) -> np.ndarray:
        """this calculated the module raw tth, ie the tth of the detector
        without taking the delta angle into account"""

        raw_tth = channel_to_angle(
            pixel_number=self.pixel_number,
            centre=self.centre,
            conv=self.conversion.conv,
            offset=self.conversion.offset,
            beamline_offset=self.beamline_offset,
        )
        return raw_tth

    @cached_property
    def tth_shaped(self) -> np.ndarray:
        """Creates an array with the same shape as the data,
        with the tth values at the corresponding indexes"""

        raw_tth_shaped = np.broadcast_to(self.raw_tth, self.data.shape)
        tth_shaped = raw_tth_shaped + self.deltas[:, None]
        return tth_shaped

    @cached_property
    def unmasked_counts(self) -> np.ndarray:
        return self.data.flatten()

    @cached_property
    def unmasked_tth(self) -> np.ndarray:
        return self.tth_shaped.flatten()

    @cached_property
    def unmasked_error(self) -> np.ndarray:
        return np.sqrt(self.unmasked_counts)

    @cached_property
    def counts(self) -> np.ndarray:
        masked_counts = self.data[:, ~self.bad_channel_mask]
        return masked_counts.flatten()

    @cached_property
    def tth(self) -> np.ndarray:
        masked_tth = self.tth_shaped[:, ~self.bad_channel_mask]
        return masked_tth.flatten()

    @cached_property
    def error(self) -> np.ndarray:
        return np.sqrt(self.counts)


class MythenDetector:
    def __init__(
        self,
        filepath: str | Path,
        angular_calibration: AngularCalibration | None = None,
        settings: MythenReductionSettings | None = None,
        xye_filepath_out: str | Path | None = None,
        output_directory: str | Path | None = None,
        filename_suffix: str = "",
    ):
        self.filepath = filepath
        self.filename_suffix = filename_suffix

        if not str(self.filepath).lower().endswith(".nxs"):
            raise ValueError(f"{self.filepath} should be a Nexus File!!")

        self.output_directory = output_directory or os.path.join(
            str(self.filepath), "processed"
        )

        self.file_dir = os.path.dirname(str(self.filepath))
        self.filename = os.path.basename(str(self.filepath))

        self.xye_filepath_out = xye_filepath_out or os.path.join(
            self.file_dir,
            f"{self.filename}_summed_mythen3{self.filename_suffix}.xye",
        )

        self.settings = settings or MythenReductionSettings()
        self.calibration = angular_calibration or AngularCalibration.load(
            self.settings.angcal_filepath
        )

        self.active_modules: list[int] = self.settings.active_modules
        self.bad_modules: list[int] = self.settings.bad_modules
        self.bad_channels = BadChannels(self.settings.bad_channels_filepath)

        # mythen data loader, just loads the data,
        # it has no information about which modules are which
        self.mythen_data = MythenDataLoader(
            file_path=filepath,
        )

        self.contruct_modules()

    def contruct_modules(self):
        self.modules = OrderedDict()

        for n_module, nth_active_module in enumerate(self.settings.active_modules):
            module = MythenModule(
                data=self.mythen_data.module_data[n_module],
                conversion=self.calibration[f"module_{nth_active_module}"],
                centre=self.calibration.centre,
                beamline_offset=self.calibration.beamline_offset,
                deltas=self.mythen_data.deltas,
                bad_channel_mask=self.bad_channels.masks[nth_active_module],
                module_id=nth_active_module,
            )

            self.modules[nth_active_module] = module

    @cached_property
    def good_modules(self) -> list[MythenModule]:
        return [
            self.modules[f]
            for f in list(self.modules.keys())
            if f not in self.bad_modules
        ]  # noqa

    def generate_xye(
        self, masked: bool = True
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if masked:
            module_tth_list = [f.tth for f in self.good_modules]
            module_counts_list = [f.counts for f in self.good_modules]
            module_errors_list = [f.error for f in self.good_modules]
        else:
            module_tth_list = [f.unmasked_tth for f in self.good_modules]
            module_counts_list = [f.unmasked_counts for f in self.good_modules]
            module_errors_list = [f.unmasked_error for f in self.good_modules]

        tth = np.concatenate(module_tth_list)
        counts = np.concatenate(module_counts_list)
        error = np.concatenate(module_errors_list)

        return tth, counts, error

    def generate_summed_xye(
        self,
        masked: bool = True,
        rebin_step: float = 0.004,
        error_calc: str = "internal",
        sum_counts: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        tth, counts, error = self.generate_xye(masked=masked)

        summed_tth, summed_counts, summed_error = bin_and_propagate_errors(
            tth,
            counts,
            error,
            rebin_step=rebin_step,
            error_calc=error_calc,
            sum_counts=sum_counts,
        )

        return summed_tth, summed_counts, summed_error

    def communicate_with_control(self, send_to_ispyb: bool = False):
        """
        Attempts to connect to i11-control and send a message indicating
        that a file has been processed. This will cause gda to plot the latest file

        Also may send xye to ispyb so that users can lookup data

        """

        daq = DaqMessenger("i11-control")
        daq.connect()
        daq.send_file(str(self.xye_filepath_out))  # sends message to GDA

        if send_to_ispyb:
            p = Path(self.filepath)
            magic_path = p.parent / ".ispyb" / (p.stem + "_mythen_nx/data.dat")
            copy2(self.xye_filepath_out, magic_path)  # copies to ispyb

    def process_step_scan(self):
        summed_tth, summed_counts, summed_error = self.generate_summed_xye(
            masked=self.settings.bad_channel_masking,
            rebin_step=self.settings.rebin_step,
            error_calc=self.settings.error_calc,
            sum_counts=True,
        )

        # plt.plot(normalise(summed_tth), normalise(summed_counts))
        # plt.show()

        # save_to_xye(self.xye_filepath_out, summed_tth, summed_counta, summed_error)
        self.save_processed_nexus()
        # self.communicate_with_control(send_to_ispyb=self.settings.send_to_ispyb)

    def save_processed_nexus(self):
        pass

    def process_pump_probe(self):
        pass

    def process_time_resolved(self):
        pass


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

    DATA_FILE = "/workspaces/XRPD-Toolbox/examples/i11/step_scan/1410286.nxs"

    ANG_CAL = PARENT_PATH / "i11" / "mythen_calibration" / "ang_cal_171125_new.json"
    settings = MythenReductionSettings.load_from_toml(CONFIG_FILE)
    print("Loaded settings:", settings)

    # print(DATA_FILE)

    # MythenDataLoader(DATA_FILE)

    BAD_CHAN_FILE = "/workspaces/XRPD-Toolbox/examples/i11/bad_channels.txt"

    angular_calibration = AngularCalibration.load_from_json(ANG_CAL)

    settings.bad_channels_filepath = BAD_CHAN_FILE

    mythen3 = MythenDetector(
        filepath=DATA_FILE, settings=settings, angular_calibration=angular_calibration
    )

    mythen3.process_step_scan()
