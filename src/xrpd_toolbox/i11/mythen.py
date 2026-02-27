import json
import math
import os
import re
from collections import OrderedDict
from collections.abc import Collection, Iterable
from functools import cached_property
from pathlib import Path
from shutil import copy, copy2
from typing import Literal

import h5py
import matplotlib.pyplot as plt
import numpy as np
from h5py import Dataset, File
from pydantic import BaseModel, Field

from xrpd_toolbox.utils.messenger import Messenger
from xrpd_toolbox.utils.mythen_utils import channel_to_angle, modules_to_pixels
from xrpd_toolbox.utils.peaks import fit_peaks
from xrpd_toolbox.utils.settings import SettingsBase
from xrpd_toolbox.utils.utils import (
    bin_and_propagate_errors,
    get_calibrant_peaks,
    get_entry,
    h5_to_array,
    load_int_array_from_file,
    save_data_to_h5,
    save_to_xye,
)

# np.set_printoptions(threshold=sys.maxsize)

MODULES_IN_DETECTOR = 28
PIXELS_PER_MODULE = 1280
PSD_RADIUS = 762  # mm
MYTHEN_PIXEL_SIZE = 0.05  # mm

PIXEL_NUMBER = np.arange(PIXELS_PER_MODULE, dtype=np.int64)


class ModuleConversion(BaseModel):
    conv: float
    offset: float
    centre: float

    @property
    def module_sign(self) -> int:  # returns -1 or 1 depending on sign of conv
        return int(math.copysign(1, self.conv))

    @property
    def distance(self) -> float:
        """returns the theoretical distance of the module based on the conv"""
        return abs(MYTHEN_PIXEL_SIZE / self.conv)

    def return_raw_tth(self, zero_offset: float) -> np.ndarray:
        """this calculated the module raw tth, ie the tth of the detector
        without taking the delta angle into account"""

        raw_tth = channel_to_angle(
            pixel_number=PIXEL_NUMBER,
            centre=self.centre,
            conv=self.conv,
            module_offset=self.offset,
            zero_offset=zero_offset,
        )
        return raw_tth


class AngularCalibration(SettingsBase):
    beamline_offset: float | int
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


class MythenSettings(SettingsBase):
    active_modules: list[int] = list(range(MODULES_IN_DETECTOR))
    bad_modules: list[int] = []
    bad_channel_masking: bool = True
    flatfield_filepath: str | Path = ""
    apply_flatfield: bool = False
    modules_in_flatfield: list[int] = list(range(MODULES_IN_DETECTOR))
    send_to_ispyb: bool = False
    rebin_step: float = 0.004
    default_counter: int = Field(default=0, ge=0, le=3)
    edge_bad_channels: int = 15
    error_calc: Literal["poisson", "std_dev", "max"] = "poisson"
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
        """Takes a long array eg. 0, 1, 67, 1285, 7655, 32000
        turns it into 28 arrays, [0, 1, 67], ... [1285], ... [7655], ...[32000]
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

        return bad_channel_mask  # bad channels are denoted by 1


class MythenDataLoader:
    def __init__(
        self,
        filepath: str | Path,
        counter: int = 0,
        mythen_data_path="mythen_nx",
        frames: int | slice | Collection[int] = slice(None),
        modules: int | slice | Collection[int] = slice(None),
    ):
        self.filepath = Path(filepath)
        self.frames = frames
        self.modules = modules
        self.counter = counter
        self.mythen_data_path = mythen_data_path

        self.entry = get_entry(self.filepath)
        self.dataset_path = f"/{self.entry}/{self.mythen_data_path}/data"
        self.n_modules_in_data, self.n_frames = self.read_nxs_metadata()

        if self.frames == slice(None) and self.modules == slice(None):
            self.data, self.module_data = self.load()
        else:
            self.data = self.get_data(modules=self.modules, frame=self.frames)

    def load(self):
        self.data = self.load_all_data(self.counter)
        self.module_data = np.array_split(self.data, self.n_modules_in_data, axis=-1)

        return self.data, self.module_data

    @property
    def pixels_per_module(self):
        return PIXELS_PER_MODULE

    def get_data(
        self,
        modules: int | Collection[int] | slice,
        frame: int | Collection[int] | slice,
        counter: int = 0,
    ):
        if isinstance(modules, int) or isinstance(modules, Collection):
            pixels = modules_to_pixels(modules=modules)
        else:
            pixels = modules

        with File(self.filepath, "r") as file:
            if self.dataset_path not in file:
                raise ValueError(
                    f"Dataset path {self.dataset_path} not found in HDF5 file."
                )

            data = file.get(self.dataset_path)

            if (data is not None) and isinstance(data, Dataset):
                if data.ndim < 1:
                    raise ValueError("Data has insufficient dimensions.")
                module_frame_data = data[frame, pixels, counter]

                return np.asarray(module_frame_data)
            else:
                raise ValueError(
                    f"Data at {self.dataset_path} in {self.filepath}is None."
                )

    @cached_property
    def delta_path(self) -> str:
        delta_subpaths = ("delta", "deltas", "ds")

        with h5py.File(self.filepath, "r") as file:
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
    def count_time_path(self) -> str:
        return f"/{self.entry}/instrument/{self.mythen_data_path}/count_time"

    @cached_property
    def positions(self) -> np.ndarray:
        try:
            deltas = h5_to_array(self.filepath, self.delta_path)
            return deltas
        except ValueError as e:
            print(f"{e} - {self.delta_path} in data - returning 0")
            deltas = np.array([0])
            return deltas

    @cached_property
    def durations(self) -> np.ndarray:
        return h5_to_array(self.filepath, self.count_time_path)

    def read_nxs_metadata(self) -> tuple[int, int]:
        with h5py.File(self.filepath, "r") as file:
            data = file.get(self.dataset_path)
            if (data is not None) and isinstance(data, Dataset):
                first_frame = data[0, :, self.counter]
                first_frame_len = first_frame.shape[-1]
                n_modules_in_data = int(first_frame_len / self.pixels_per_module)
                n_frames = len(data)
                return n_modules_in_data, n_frames
            else:
                raise ValueError(f"Data is None at {self.dataset_path}")

    def load_all_data(self, counter: int) -> np.ndarray:
        return self.get_data(slice(None), slice(None), counter)


class MythenModule:
    def __init__(
        self,
        data: np.ndarray,
        conversion: ModuleConversion,
        beamline_offset: float,
        module_id: int,
        positions: np.ndarray,
        durations: np.ndarray | None = None,
        bad_channel_mask: np.ndarray | None = None,
    ):
        self.data = data
        self.conversion = conversion
        self.beamline_offset = beamline_offset
        self.positions = positions
        self.module_id = module_id

        if bad_channel_mask is None:
            self.bad_channel_mask = np.zeros(self.data.shape[-1], dtype=bool)
        else:
            self.bad_channel_mask = bad_channel_mask

        if durations is None:
            self.durations = np.ones(self.data.shape[-1], dtype=bool)
        else:
            self.durations = durations

    # @cached_property
    # def pixel_number(self) -> np.ndarray:
    #     """returns a 1d array of integers between 0 and 1280"""
    #     return PIXEL_NUMBER

    @cached_property
    def raw_tth(self) -> np.ndarray:
        """this calculated the module raw tth, ie the tth of the detector
        without taking the delta angle into account"""

        raw_tth = self.conversion.return_raw_tth(self.beamline_offset)

        return raw_tth

    @cached_property
    def positions_2d(self):
        print(len(self.positions))
        positions_2d = np.broadcast_to(self.positions, self.data.shape)
        return positions_2d

    @cached_property
    def duration_2d(self):
        duration_2d = np.broadcast_to(self.durations[:, np.newaxis], self.data.shape)
        return duration_2d

    @cached_property
    def tth_2d(self) -> np.ndarray:
        """Creates an array with the same shape as the data,
        with the tth values at the corresponding indexes"""

        tth_2d = np.broadcast_to(self.raw_tth, self.data.shape)
        tth_2d = tth_2d + self.positions[:, None]
        return tth_2d

    @cached_property
    def mask_2d(self) -> np.ndarray:
        """Creates an array with the same shape as the data,
        with the tth values at the corresponding indexes"""

        mask_2d = np.broadcast_to(self.bad_channel_mask, self.data.shape)
        return mask_2d

    @cached_property
    def unmasked_counts(self) -> np.ndarray:
        return self.data.flatten()

    @cached_property
    def unmasked_duration(self) -> np.ndarray:
        return self.duration_2d.flatten()

    @cached_property
    def unmasked_tth(self) -> np.ndarray:
        return self.tth_2d.flatten()

    @cached_property
    def unmasked_error(self) -> np.ndarray:
        return np.sqrt(self.unmasked_counts)

    @cached_property
    def counts(self) -> np.ndarray:
        masked_counts = self.data[:, ~self.bad_channel_mask]
        return masked_counts.flatten()

    @cached_property
    def duration(self) -> np.ndarray:
        duration = self.duration_2d[:, ~self.bad_channel_mask]
        return duration.flatten()

    @cached_property
    def tth(self) -> np.ndarray:
        masked_tth = self.tth_2d[:, ~self.bad_channel_mask]
        return masked_tth.flatten()

    @cached_property
    def error(self) -> np.ndarray:
        return np.sqrt(self.counts)


class MythenDetector:
    def __init__(
        self,
        filepath: str | Path,
        angular_calibration: AngularCalibration | None = None,
        settings: MythenSettings | None = None,
        xye_filepath_out: str | Path | None = None,
        output_directory: str | Path | None = None,
        filename_suffix: str = "",
    ):
        self.filepath = filepath
        self.filename_suffix = filename_suffix

        if str(self.filepath) == "dev":
            pass
        elif not str(self.filepath).lower().endswith(".nxs"):
            raise ValueError(f"{self.filepath} should be a Nexus File!!")
        elif not os.path.exists(self.filepath):
            raise ValueError(f"{self.filepath} does not exist!!")

        self.file_dir = os.path.dirname(str(self.filepath))
        self.filename = os.path.basename(str(self.filepath))

        self.output_directory = output_directory or os.path.join(
            str(self.file_dir), "processed"
        )

        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)

        self.processed_nexus_filepath = os.path.join(
            str(self.output_directory),
            f"{Path(self.filename).stem}_reduced_mythen3{self.filename_suffix}.nxs",
        )

        self.xye_filepath_out = xye_filepath_out or os.path.join(
            self.file_dir,
            f"{self.filename}_summed_mythen3{self.filename_suffix}.xye",
        )

        self.settings = settings or MythenSettings()
        self.calibration = angular_calibration or AngularCalibration.load(
            self.settings.angcal_filepath
        )

        self.active_modules: list[int] = self.settings.active_modules
        self.bad_modules: list[int] = self.settings.bad_modules
        self.good_modules: list[int] = list(
            set(self.active_modules) ^ set(self.bad_modules)
        )
        self.ring1_modules: list[int] = list(range(0, 14))
        self.ring2_modules: list[int] = list(range(14, 28))

        self.active_ring1_modules: list[int] = list(
            set(self.active_modules) ^ set(self.ring2_modules)
        )
        self.active_ring2_modules: list[int] = list(
            set(self.active_modules) ^ set(self.ring1_modules)
        )

        self.bad_channels = BadChannels(self.settings.bad_channels_filepath)

        # mythen data loader, just loads the data,
        # it has no information about which modules are which

        self.mythen_data = MythenDataLoader(
            filepath=filepath,
        )

        self.contruct_modules()

    def _make_mythen_module_kwargs(self, n_module, nth_active_module):
        """MythenModule requires quite a lot of info,
        so it's easier to make a contructor of it's kwargs"""

        return {
            "data": self.mythen_data.module_data[n_module],
            "conversion": self.calibration[f"module_{nth_active_module}"],
            "beamline_offset": self.calibration.beamline_offset,
            "module_id": nth_active_module,
            "positions": self.mythen_data.positions,
            "durations": self.mythen_data.durations,
            "bad_channel_mask": self.bad_channels.masks[nth_active_module],
        }

    def get_module(self, mod: int) -> MythenModule:
        return self.modules[mod]

    def contruct_modules(self):
        self.modules = OrderedDict()

        for n_module, nth_active_module in enumerate(self.settings.active_modules):
            module = MythenModule(
                **self._make_mythen_module_kwargs(n_module, nth_active_module)
            )

            self.modules[nth_active_module] = module

    def generate_xye(
        self, modules: Iterable[int], masked: bool = True
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if masked:
            module_tth_list = [self.get_module(f).tth for f in modules]
            module_counts_list = [self.get_module(f).counts for f in modules]
            module_errors_list = [self.get_module(f).error for f in modules]
        else:
            module_tth_list = [self.get_module(f).unmasked_tth for f in modules]
            module_counts_list = [self.get_module(f).unmasked_counts for f in modules]
            module_errors_list = [self.get_module(f).unmasked_error for f in modules]

        unsorted_tth = np.concatenate(module_tth_list)
        unsorted_counts = np.concatenate(module_counts_list)
        unsorted_error = np.concatenate(module_errors_list)

        sort_indexes = np.argsort(unsorted_tth)

        tth = unsorted_tth[sort_indexes]
        counts = unsorted_counts[sort_indexes]
        error = unsorted_error[sort_indexes]

        return tth, counts, error

    def generate_binned_xye(
        self,
        masked: bool = True,
        rebin_step: float = 0.004,
        error_calc: str = "poisson",
        normalise: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns the final dataset that you would use to generate a xye file,
        binned and all
        """

        tth, counts, error = self.generate_xye(modules=self.good_modules, masked=masked)

        # TODO: Should count times really be in MythenModule and then recreated this way
        # - seems like extra wasteful computation?

        if normalise:
            module_duration_list = [
                self.get_module(f).duration for f in self.good_modules
            ]
            durations = np.concatenate(module_duration_list)
            counts = counts / durations

        binned_tth, binned_counts, binned_error = bin_and_propagate_errors(
            tth,
            counts,
            error,
            rebin_step=rebin_step,
            error_calc=error_calc,
        )

        return binned_tth, binned_counts, binned_error

    def communicate_with_control(self, send_to_ispyb: bool = False):
        """
        Attempts to connect to i11-control and send a message indicating
        that a file has been processed. This will cause gda to plot the latest file

        Also may send xye to ispyb so that users can lookup data

        """

        daq = Messenger(host="i11-control")
        daq.connect()
        daq.send_file(str(self.xye_filepath_out))  # sends message to GDA

        if send_to_ispyb:
            p = Path(self.filepath)
            magic_path = p.parent / ".ispyb" / (p.stem + "_mythen_nx/data.dat")
            copy2(self.xye_filepath_out, magic_path)  # copies to ispyb

    def process_step_scan(self):
        """Analyses the data using the settings provided by the MythenSettings class
        Takes all of the data from each module and each step of the multistep scan
        and calculates the tth for each delta position, orders them, concatenates
        and puts them into a single array. Bins it, and saves it to xye/nexus file"""

        self.binned_tth, self.binned_counts, self.binned_error = (
            self.generate_binned_xye(
                masked=self.settings.bad_channel_masking,
                rebin_step=self.settings.rebin_step,
                error_calc=self.settings.error_calc,
            )
        )

        save_to_xye(
            self.xye_filepath_out,
            self.binned_tth,
            self.binned_counts,
            self.binned_error,
        )

        xye_names_and_data = {
            "tth": self.binned_tth,
            "counts": self.binned_counts,
            "error": self.binned_error,
        }

        self.save_data_to_nexus(subpath="/xye", names_and_data=xye_names_and_data)

        ring1_data, ring2_data = self.get_ring1_ring2_data()
        self.save_data_to_nexus(subpath="/ring1", names_and_data={"data": ring1_data})
        self.save_data_to_nexus(subpath="/ring1", names_and_data={"data": ring2_data})

        print(f"Data saved to: {self.processed_nexus_filepath}")

        try:
            self.communicate_with_control(send_to_ispyb=self.settings.send_to_ispyb)
        except Exception as e:
            print(f"Could not connect with control - {e}")
            pass

    def get_ring1_ring2_data(self):
        first_ring_pixels = len(self.active_ring1_modules) * PIXELS_PER_MODULE
        ring1_data = self.mythen_data.get_data(slice(0, first_ring_pixels), slice(None))
        ring2_data = self.mythen_data.get_data(
            slice(first_ring_pixels, None), slice(None)
        )

        return ring1_data, ring2_data

    def save_data_to_nexus(
        self, subpath: str, names_and_data: dict[str, np.ndarray], **kwargs
    ):
        copy(self.filepath, self.processed_nexus_filepath)

        for name, data in names_and_data.items():
            save_data_to_h5(
                self.filepath,
                f"{self.mythen_data.entry}{subpath}/{name}",
                data,
                **kwargs,
            )

    def process_pump_probe(self):
        pass

    def process_time_resolved(self):
        pass

    def plot_diffraction(self, filepath: str | Path | None = None):
        plt.figure(figsize=(10, 7))

        tth, counts, error = self.generate_binned_xye(
            masked=self.settings.bad_channel_masking,
            rebin_step=self.settings.rebin_step,
            error_calc=self.settings.error_calc,
        )

        si_tth = get_calibrant_peaks("Si", 0.828783)
        plt.vlines(si_tth, 0, np.amax(counts), color="red")

        plt.errorbar(tth, counts, error, label=self.settings.error_calc)
        plt.legend()
        plt.xlabel("tth")
        plt.ylabel("Intensity (arb. units)")

        if filepath:
            plt.savefig(filepath)

        plt.show()
        plt.close()

        amps, fit_xpos, fwhms = fit_peaks(tth, counts, si_tth)

        plt.plot(si_tth, fit_xpos - si_tth)
        plt.show()

    def plot_diffraction_by_mod(self, filepath: str | Path | None = None):
        plt.figure(figsize=(10, 7))

        for module in self.good_modules:
            sort_index = np.argsort(self.modules[module].tth)
            tth = (self.modules[module].tth)[sort_index]
            counts = (self.modules[module].counts)[sort_index]

            plt.plot(
                tth,
                counts,
                label=str(self.modules[module].module_id),
            )
            plt.text(
                np.mean(tth),
                np.amin(counts),
                str(self.modules[module].module_id),
            )  # type: ignore

        plt.xlabel("tth")
        plt.ylabel("Intensity (arb. units)")
        # plt.legend()

        if filepath:
            plt.savefig(filepath)

        plt.show()
        plt.close()


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


def convert_angcal_to_new_pydantic_json(
    ang_cal_json_path: str | Path, new_path: str | Path
):
    pydantic_dict = {}
    module_conv_list = []

    with open(ang_cal_json_path, "rb") as file:
        legacy_dict = json.load(file)

    for entry in legacy_dict.keys():
        numbers = re.findall(r"-?\d*\.?\d+", str(entry))

        if len(numbers) > 0:
            module = numbers[0]
            module_conv_list.append(
                {
                    "name": f"module_{module}",
                    "conv": legacy_dict[f"conv_{module}"],
                    "beamline_offset": legacy_dict[f"conv_{module}"],
                    "centre": legacy_dict["beamline_offset"],
                    "offset": legacy_dict["centre"],
                }
            )

    pydantic_dict["modules"] = module_conv_list
    pydantic_model = AngularCalibration(**pydantic_dict)
    pydantic_model.save_to_json(new_path)


if __name__ == "__main__":
    PARENT_PATH = Path(__file__).parent.parent

    print(PARENT_PATH)

    CONFIG_FILE = (
        PARENT_PATH / "i11" / "mythen_calibration" / "mythen3_reduction_config.toml"
    )

    DATA_FILE = "/workspaces/XRPD-Toolbox/examples/i11/step_scan/1410289.nxs"

    ANG_CAL = "/workspaces/XRPD-Toolbox/src/xrpd_toolbox/i11/mythen_calibration/processed/ang_cal_020426_cen_639.5_leastsq_[11, 17, 27]_new.json"  # noqa

    settings = MythenSettings.load_from_toml(CONFIG_FILE)
    print("Loaded settings:", settings)

    # print(DATA_FILE)

    # MythenDataLoader(DATA_FILE)

    BAD_CHAN_FILE = "/workspaces/XRPD-Toolbox/examples/i11/bad_channels.txt"

    angular_calibration = AngularCalibration.load_from_json(ANG_CAL)
    # angular_calibration.beamline_offset = -0.4979739

    print(angular_calibration)

    settings.bad_channels_filepath = BAD_CHAN_FILE

    # DATA_FILE = "/workspaces/XRPD-Toolbox/examples/i11/step_scan/1414223.nxs"
    DATA_FILE = "/workspaces/XRPD-Toolbox/examples/i11/angular_calibration/1410289.nxs"

    mythen3 = MythenDetector(
        filepath=DATA_FILE, settings=settings, angular_calibration=angular_calibration
    )

    mythen3.plot_diffraction()
    mythen3.plot_diffraction_by_mod()
    # print(mythen3.counts_times)
