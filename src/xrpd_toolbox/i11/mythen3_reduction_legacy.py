# type: ignore
import argparse
import os
import sys
from itertools import product
from pathlib import Path
from shutil import copy, copy2
from tomllib import load

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from h5py import File as h5pyFile
from matplotlib.gridspec import GridSpec

from xrpd_toolbox.utils.energy import tth_to_q
from xrpd_toolbox.utils.messenger import Messenger
from xrpd_toolbox.utils.mythen_utils import paired_modules
from xrpd_toolbox.utils.utils import AnalysisLogger, load_int_array_from_file

matplotlib.use("Qt5Agg")  # or TkAgg


np.seterr(
    divide="ignore", invalid="ignore"
)  # dividing by zero throws a warning, this is expected due to some pixels being dead


pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 1000)
np.set_printoptions(threshold=sys.maxsize)


class I11Reduction:
    __slots__ = (
        "filepath",
        "reduced_nxs_filepath_out",
        "xye_filepath_out",
        "xye_filepath_out_q",
        "file_dir",
        "out_directory",
        "file_name",
        "file_extension",
        "filename_suffix",
        "config_filepath",
        "STRIPS_PER_MODULE",
        "MODULES_PER_DETECTOR",
        "config",
        "mythen3_config_dir",
        "flatfield_filepath",
        "apply_flatfield",
        "active_modules",
        "bad_modules",
        "bad_frames",
        "bad_channel_masking",
        "default_counter",
        "n_bad_edge_channels",
        "rebin_step",
        "error_calc",
        "beam_energy",
        "beamline_offset",
        "save_nxs_out",
        "verbose_nxs",
        "debug_mode",
        "save_in_q_space",
        "data_reduction_mode",
        "wavelength",
        "raw_flatfield_counts",
        "flatfield_modules",
        "bad_channels",
        "module_angular_cal",
        "module_raw_tth",
        "deltas",
        "duration",
        "n_frames",
        "wholedetector_raw_frames",
        "raw_frame_counts",
        "frames_range",
        "all_module_data",
        "angular_corrected_data",
        "module_raw_data",
        "xyedata",
        "frame_data",
        "angular_corrected_data_unmasked",
        "out_raw_data",
        "n_modules_in_data",
        "good_modules",
        "Ie",
        "Ic4",
        "whole_data_raw_tth",
        "beam_intensity",
        "ffcorr",
        "bad_channels_filepath",
        "modules_in_flatfield",
        "angcal_filepath",
        "live",
        "send_to_ispyb",
        "execute_reduction",
        "logger",
        "logging",
    )

    @staticmethod
    def read_singular_angcal_files(angcal_filepath: str) -> dict:
        """

        Reads a single of ang.off files and returns a dict with the
        each modules anngular calibrations contains within a dict

        each module dict contains "offset", "conv" and "centre"

        eg. self.module_angular_cal[module]["offset"]

        """

        module_angular_cal = {}
        beamline_offset = None

        with open(angcal_filepath) as f:
            for line in f:
                if "beamline_offset" in line:
                    elements = line.split()
                    beamline_offset = float(elements[1])

                elif line := line.strip():
                    elements = line.split()
                    module_cal = {}

                    (
                        module_in_file,
                        module_cal["offset"],
                        module_cal["conv"],
                        module_cal["centre"],
                    ) = (
                        int(elements[1]),
                        float(elements[3]),
                        float(elements[5]),
                        float(elements[7]),
                    )

                    module_angular_cal[module_in_file] = module_cal

        return module_angular_cal, beamline_offset

    @staticmethod
    def read_angular_calibration_and_create_cal_dict(
        config: dict, active_modules: list[int]
    ) -> dict:
        """

        Reads a load of ang_d .off files and returns a dict with
        the each modules anngular calibrations contains within a dict

        each module dict contains "offset", "conv" and "centre"

        eg. self.module_angular_cal[module]["offset"]

        """

        print(config)

        module_angular_cal = {}

        for active_mod in active_modules:
            single_cal_file = config["".join(["module_", str(active_mod)])][
                "angular_calibration_path"
            ]

            with open(single_cal_file) as f:
                module_cal = {}
                for line in f:
                    if "beamline_offset" in line:
                        elements = line.split()
                        beamline_offset = float(elements[1])

                    elif line := line.strip():
                        elements = line.split()
                        (
                            module_in_file,  # noqa
                            module_cal["offset"],
                            module_cal["conv"],
                            module_cal["centre"],
                        ) = (
                            int(elements[1]),
                            float(elements[3]),
                            float(elements[5]),
                            float(elements[7]),
                        )

                        module_angular_cal[active_mod] = module_cal

        return module_angular_cal, beamline_offset

    @staticmethod
    def channel_to_angle(module_pixel_number, centre, conv, offset, beamline_offset):
        module_conversions = module_pixel_number - centre
        module_conversions = module_conversions * conv
        module_conversions = np.arctan(module_conversions)
        raw_tth = offset + np.rad2deg(module_conversions) + beamline_offset

        return raw_tth

    @staticmethod
    def channel_to_angle_in_real_units(
        module_pixel_number,
        centre,
        offset,
        beamline_offset,
        radius,
        p=0.05,
        direction=1,
    ):
        """
        module_pixel_number: channel number, 0-1280
        centre: centre (in pixel number - ie 1280/2)
        offset: module offset, degrees
        radius: radius, mm - approx 760
        direction: 1 or -1 depending if module is flipped or not
        p: pixel size, mm = 0.05
        """

        raw_tth = I11Reduction.channel_to_angle(
            module_pixel_number, centre, (p / radius), offset, beamline_offset
        )

        return raw_tth

    def calculate_modules_tth(self):
        """
        Given a set of calibration parameters, return a numpy array
        describing the angle, in degrees, of each pixel in that module

        Ref:
        section 1.1 of "Angular conversion 1-D" by A. Cervellino (ANGCONV_2024.pdf).
        beamline_offset
        file:///home/akz63626/Downloads/ANGCONV_2024.pdf

        """

        module_raw_tth = {}
        module_pixel_number = np.arange(self.STRIPS_PER_MODULE, dtype=np.int64)
        module_raw_tth["mod_channel"] = module_pixel_number

        for n_mod in self.active_modules:
            # cm = self.module_angular_cal[n_mod]["centre"]
            # km = self.module_angular_cal[n_mod]["conv"]
            # om = self.module_angular_cal[n_mod]["offset"]
            # sm  = +1/-1 # this is already included in the km in the ang.off files

            raw_tth = I11Reduction.channel_to_angle(
                module_pixel_number,
                self.module_angular_cal[n_mod]["centre"],
                self.module_angular_cal[n_mod]["conv"],
                self.module_angular_cal[n_mod]["offset"],
                self.beamline_offset,
            )

            module_raw_tth[n_mod] = (
                raw_tth  # this is raw because it doesn't take into account the  delta
            )

        whole_data_raw_tth = np.array(
            [module_raw_tth[f] for f in self.active_modules]
        ).flatten()

        return module_raw_tth, whole_data_raw_tth

    def generate_badchannel_dict(self) -> dict:
        """

        loads the bad channels specified with the bad_channels_path
        which are specified in the .toml config file.

        Creates a dictionary of bad channels so that they can be
        accessed self.bad_channels[n] where n is 0-27

        """

        bad_channels = {}

        self.logger.log("Using the following bad channels files:")
        for active_mod in self.active_modules:
            badchan_file = self.config["".join(["module_", str(active_mod)])][
                "bad_channels_path"
            ]
            bad_channels[active_mod] = load_int_array_from_file(badchan_file)
            self.logger.log(
                f"Module: {active_mod} badchan_file | Bad Chans: {len(bad_channels[active_mod])}"  # noqa
            )

        self.logger.log("\n")
        return bad_channels

    @staticmethod
    def read_config(mythen3_config_dir):
        """
        reads the config file and works out what modules are currently active

        """

        enabled_modules_hostnames = None

        with open(mythen3_config_dir) as file:
            lines = [line.rstrip() for line in file]

        for _, line in enumerate(lines):
            if line.startswith("hostname"):
                enabled_modules_hostnames = line.split()[1::]

        enabled_modules = [
            int(n_mod.rstrip()[-3::]) - 100 for n_mod in enabled_modules_hostnames
        ]

        return enabled_modules

    def apply_flatfield_correction(
        self, no_ff_corr: np.ndarray, flatfield_counts: np.ndarray
    ) -> np.ndarray:
        """
        Divide raw counts by flatfield counts to get scaled counts.
        Where the flatfield counts are zero, return zero.  Then rescale
        by the mean value of flatfield counts to get back to a unit of
        counts.

        This MUST be scaled on the whole detector not on a module/module
        basis otherwise counts will be thrown off massively by bad
        modules/channels and the module will scale incorrectly!!!
        """

        scaled_counts = np.divide(
            no_ff_corr,
            flatfield_counts,
            where=flatfield_counts != 0,
            out=np.zeros(no_ff_corr.shape),
        )

        # self.logger.log(np.mean(flatfield_counts))

        return scaled_counts

    def read_nxs_metadata(self, filepath: str):
        with h5pyFile(filepath, "r") as file:
            try:
                entry = file["entry"]
            except Exception:
                entry = file["entry1"]

            if "ds" in list(entry["mythen_nx"].keys()):
                dummy_array = entry["mythen_nx"]["ds"][()]
                self.deltas = [dummy_array[0]] * len(dummy_array)

            elif "delta" in list(entry["mythen_nx"].keys()):
                self.deltas = entry["mythen_nx"]["delta"][()]

            elif "deltas" in list(entry["mythen_nx"].keys()):
                self.deltas = entry["mythen_nx"]["deltas"][()]

            self.n_frames = len(entry["mythen_nx"]["data"])
            first_frame_len = len(
                entry["mythen_nx"]["data"][0, :, self.default_counter][()]
            )
            self.n_modules_in_data = int(first_frame_len / self.STRIPS_PER_MODULE)

            self.Ie = np.full(self.n_frames, 1)
            try:
                self.Ie = entry["Ie"]["data"][()]
            except Exception:
                try:
                    self.Ie = entry["mythen_nx"]["Ie"][()]
                except Exception:
                    pass

            self.Ic4 = np.full(self.n_frames, 1)
            try:
                self.Ic4 = entry["Ic4"]["data"][()]
            except Exception:
                try:
                    self.Ic4 = entry["mythen_nx"]["Ic4"][()]
                except Exception:
                    pass

            self.beam_intensity = (self.Ic4 + self.Ie) / 2

            return self.n_frames, self.deltas, self.n_modules_in_data

    def read_nxs_data(
        self,
        filepath: str,
        frame: int | None = None,
        sum_frames: bool = False,
        bad_frames=(),
    ) -> dict:
        """
        Note: [()] causes data to be copied to a numpy array
        rather than just referencing
        a h5py dataset (which goes out of scope after the context manager exits)
        Axes at this level are:
                x_dim: always 1
                y_dim: channelS_PER_MODULE * NUM_MODULES
                counters: (always 3 counters).

        Reading nxs files is pretty fast - not really a bottleneck

        """

        with h5pyFile(filepath, "r") as file:
            try:
                entry = file["entry"]
            except Exception:
                entry = file["entry1"]

            normalised_beam_intensity = self.beam_intensity / np.median(
                self.beam_intensity
            )

            if (frame is not None) and (sum_frames is False):
                # for when we want to only read a specific frame - ie time resolved

                self.wholedetector_raw_frames = entry["mythen_nx"]["data"][
                    frame, :, self.default_counter
                ][()]  # [n, 17920, 3] where n is the number of frames, 3 = 3 counters
                self.wholedetector_raw_frames = (
                    self.wholedetector_raw_frames / normalised_beam_intensity[frame]
                )

                self.raw_frame_counts = {}
                self.raw_frame_counts[frame] = self.wholedetector_raw_frames
                self.logger.log(
                    f"Frame: {frame + 1}/{self.n_frames} | Delta: {self.deltas[frame]}"
                )

            elif (frame is None) and (sum_frames is True):
                # we want to read all frames and then sum them, ie pump-probe mode
                self.wholedetector_raw_frames = np.zeros_like(
                    entry["mythen_nx"]["data"][0, :, self.default_counter][()]
                )

                for frame in range(self.n_frames):
                    if frame in bad_frames:
                        self.logger.log(f"{frame} is a bad frame")
                        continue

                    wholedetector_n_frame = entry["mythen_nx"]["data"][
                        frame, :, self.default_counter
                    ][()]
                    wholedetector_n_frame_normalised = (
                        wholedetector_n_frame / normalised_beam_intensity[frame]
                    )
                    self.wholedetector_raw_frames = (
                        self.wholedetector_raw_frames + wholedetector_n_frame_normalised
                    )

                self.raw_frame_counts = {}
                self.raw_frame_counts[0] = self.wholedetector_raw_frames
                self.deltas = [np.mean(self.deltas)]
                self.logger.log(f"Summing {self.n_frames} frames")

            elif (frame is None) and not (sum_frames):
                # standard data reduction mode
                self.wholedetector_raw_frames = entry["mythen_nx"]["data"][
                    :, :, self.default_counter
                ][()]  # [n, 17920, 3] where n is the number of frames, 3 = 3 counters
                self.raw_frame_counts = {
                    frame: self.wholedetector_raw_frames[frame, :]
                    for frame in range(len(self.wholedetector_raw_frames))
                }  # coverts it from list into dict
                self.logger.log(f"Deltas position(s): {self.deltas}")
                self.logger.log(
                    f"Frames: {len(self.raw_frame_counts)}/{self.n_frames} | Specified frame: {frame}"  # noqa
                )

            self.frames_range = list(self.raw_frame_counts.keys())

            return (
                self.raw_frame_counts,
                self.frames_range,
                self.wholedetector_raw_frames,
            )

    def load_flatfield(self, flatfield_filepath: str) -> np.ndarray:
        """

        loads the flatfield file. The flatfield file is (usually) an
        h5 file containing a calibrated beam on the detector. This

        allows to calibrate the response of the detector to the beam
        ie. work out the efficiency of the detector

        the flatfield can then be used to adjust the raw counts to correct the raw data.
        If flatfield correction is set to False then

        the entire flatfield is set to 1, such that it effectively makes no correction

        """

        if self.apply_flatfield:
            if not os.path.exists(flatfield_filepath):
                self.logger.log("Flatfield file does not exist")
                flatfield = np.full(
                    (self.STRIPS_PER_MODULE * self.MODULES_PER_DETECTOR), 1
                )

            try:
                with h5pyFile(flatfield_filepath, "r") as file:
                    if "flatfield" in file:
                        flatfield = file["flatfield"][()]
                    elif "flat_total_rescaled" in file:
                        flatfield = file["flat_total_rescaled"][()]
            except Exception:
                with h5pyFile(flatfield_filepath, "r") as file:
                    flatfield_tuple = file["data"][()]
                    flatfield = np.full(
                        (self.STRIPS_PER_MODULE * self.MODULES_PER_DETECTOR), 0
                    )

                    for flatfield_frame in flatfield_tuple:
                        flatfield += flatfield_frame

            # flatfield = I11Reduction.NormalizeTo(flatfield,minval=0)

            return flatfield

        else:
            flatfield = np.full((self.STRIPS_PER_MODULE * self.MODULES_PER_DETECTOR), 1)
            return flatfield

    def split_flatfield(self) -> dict:
        """

        splits the raw_flatfield_counts into array of arrays,
        where each array corresponds to the module counts using a white beam

        """
        if len(self.raw_flatfield_counts) == (
            self.STRIPS_PER_MODULE * self.MODULES_PER_DETECTOR
        ):
            flatfield_module_array = np.split(
                self.raw_flatfield_counts, self.MODULES_PER_DETECTOR
            )
            iter_modules = range(self.MODULES_PER_DETECTOR)

        elif (
            (len(self.modules_in_flatfield) != 0)
            and (
                len(self.raw_flatfield_counts)
                != (self.STRIPS_PER_MODULE * self.MODULES_PER_DETECTOR)
            )
            and (
                (len(self.raw_flatfield_counts) / self.STRIPS_PER_MODULE)
                == self.n_modules_in_data
            )
        ):
            flatfield_module_array = np.split(
                self.raw_flatfield_counts, len(self.modules_in_flatfield)
            )
            iter_modules = self.modules_in_flatfield

        flatfield_modules = {}

        for n, mod in enumerate(iter_modules):
            flatfield_modules[mod] = flatfield_module_array[n]

        return flatfield_modules

    def save_nxs_outfile(
        self,
        reduced_nxs_filepath_out: str,
        xyedata: pd.DataFrame,
        module_raw_data: pd.DataFrame,
        frame_data: pd.DataFrame,
        angular_corrected_data: pd.DataFrame,
        debug=False,
    ) -> None:
        """

        Saves a hdf5 file suitable for analysis in dawn or calibration of the detector

        """

        if not self.verbose_nxs:
            columns_to_export = ["tth", "det_channel", "no_ff_corr", "counts"]
        else:
            columns_to_export = angular_corrected_data.columns

        self.logger.log(reduced_nxs_filepath_out)

        #####################################
        if os.path.exists(reduced_nxs_filepath_out):
            os.remove(reduced_nxs_filepath_out)
            print("remove")

        copy(self.filepath, reduced_nxs_filepath_out)

        with h5pyFile(self.filepath, "r", libver="latest", swmr=True) as file:
            left_ring = file["entry"]["mythen_nx"]["data"][:, 0:17920, :][
                ()
            ]  # [n, 17920, 3] where n is the number of frames, 3 = 3 counters
            right_ring = file["entry"]["mythen_nx"]["data"][:, 17920::, :][
                ()
            ]  # [n, 17920, 3] where n is the number of frames, 3 = 3 counters

        with h5pyFile(reduced_nxs_filepath_out, "a", libver="latest") as out_file:
            ##################################
            # save rings exactly as they were from the hdf5 file but split ring 1 and 2
            ##################################
            print(out_file["entry"].keys())

            out_file["entry"]["mythen_nx"]["ring1"] = left_ring
            out_file["entry"]["mythen_nx"]["ring2"] = right_ring

            tth_sorted_data = angular_corrected_data.sort_values(by="tth")

            ##################################
            # save fully reduced xye data
            ##################################

            xye_group = out_file["entry"].create_group("xye")
            out_file["entry"]["xye"].create_dataset(
                "error", data=xyedata["error"].values, dtype="f"
            )
            out_file["entry"]["xye"].create_dataset(
                "counts", data=xyedata["counts"].values, dtype="f"
            )
            out_file["entry"]["xye"].create_dataset(
                "tth", data=xyedata["tth"].values, dtype="f"
            )

            xye_group.attrs["NX_class"] = "NXdata"
            xye_group.attrs["signal"] = "counts"
            xye_group.attrs["axes"] = ["tth"]
            xye_group.attrs["tth_indices"] = [0]

            nxentry = out_file["entry"]
            nxentry.attrs["default"] = "/entry/xye/counts"

            # out_file.attrs["default"] = "/entry/xye/counts"

            ##################################
            # save all reduced modules seperately
            ##################################

            out_file["entry"].create_group("modules")
            out_file["entry"]["modules"]["active_modules"] = np.array(
                self.active_modules
            )

            for column in columns_to_export:
                column_array = np.zeros(
                    (len(self.active_modules), self.STRIPS_PER_MODULE * self.n_frames)
                )

                for n, module in enumerate(self.active_modules):
                    module_data = tth_sorted_data[tth_sorted_data["n_mod"] == module]
                    column_array[n, 0 : len(module_data[column].values)] = module_data[
                        column
                    ].values

                out_file["entry"]["modules"][column] = column_array

            ##################################
            # save all reduced frames seperately
            ##################################

            out_file["entry"].create_group("frames")
            out_file["entry"]["frames"]["frame_ids"] = np.arange(self.n_frames)

            for col in columns_to_export:
                column_array = []
                for frame in frame_data.values():
                    column_array.append(frame[col])
                column_array = np.array(column_array)
                out_file["entry"]["frames"][col] = column_array

            ##################################
            # save reduced rings seperately
            ##################################
            nxs_modules_group_ring1 = out_file["entry"].create_group("ring1")

            if np.amin(self.active_modules) < 14:
                ring1 = angular_corrected_data[
                    angular_corrected_data["n_mod"].isin(np.arange(0, 14, 1, dtype=int))
                ]
                for col in columns_to_export:
                    nxs_modules_group_ring1[col] = ring1[col]

                nxs_modules_group_ring1.attrs["NX_class"] = "NXdata"
                nxs_modules_group_ring1.attrs["signal"] = "counts"
                nxs_modules_group_ring1.attrs["axes"] = ["tth"]
                nxs_modules_group_ring1.attrs["tth_indices"] = [0]

            nxs_modules_group_ring2 = out_file["entry"].create_group("ring2")

            if np.amax(self.active_modules) > 14:
                ring2 = angular_corrected_data[
                    angular_corrected_data["n_mod"].isin(
                        np.arange(14, 28, 1, dtype=int)
                    )
                ]

                for col in columns_to_export:
                    nxs_modules_group_ring2[col] = ring2[col]

                nxs_modules_group_ring2.attrs["NX_class"] = "NXdata"
                nxs_modules_group_ring2.attrs["signal"] = "counts"
                nxs_modules_group_ring2.attrs["axes"] = ["tth"]
                nxs_modules_group_ring2.attrs["tth_indices"] = [0]

            #############################
            # run debug processing
            ##############################

            if debug:
                guide = self.make_module_boundary_guide(len(self.active_modules))
                out_file["entry"]["guide"] = guide

                detchannel_data = angular_corrected_data.sort_values(by="det_channel")

                ring1_ffcorr = []
                ring2_ffcorr = []

                ring1_ffcorr_flipped = []
                ring2_ffcorr_flipped = []

                for frame in range(self.n_frames):
                    det_channel_frame = detchannel_data[
                        detchannel_data["frame"] == frame
                    ]
                    det_channel_frame_ring1 = (det_channel_frame["counts"].values)[
                        0:17920
                    ]
                    det_channel_frame_ring2 = (det_channel_frame["counts"].values)[
                        17920::
                    ]

                    flipped_frame = tth_sorted_data[tth_sorted_data["frame"] == frame]
                    flipped_frame_ring1 = (flipped_frame["counts"].values)[0:17920]
                    flipped_frame_ring2 = (flipped_frame["counts"].values)[17920::]

                    ring1_ffcorr.append(det_channel_frame_ring1)
                    ring2_ffcorr.append(det_channel_frame_ring2)

                    ring1_ffcorr_flipped.append(flipped_frame_ring1)
                    ring2_ffcorr_flipped.append(flipped_frame_ring2)

                ring1_ffcorr = np.array(ring1_ffcorr)
                ring2_ffcorr = np.array(ring2_ffcorr)
                ring1_ffcorr_flipped = np.array(ring1_ffcorr_flipped)
                ring2_ffcorr_flipped = np.array(ring2_ffcorr_flipped)

                out_file["entry"]["frames"].create_dataset(
                    "ring1_ffcorr", data=ring1_ffcorr, dtype="f"
                )
                out_file["entry"]["frames"].create_dataset(
                    "ring2_ffcorr", data=ring2_ffcorr, dtype="f"
                )

                out_file["entry"]["frames"].create_dataset(
                    "ring1_ffcorr_flipped", data=ring1_ffcorr_flipped, dtype="f"
                )
                out_file["entry"]["frames"].create_dataset(
                    "ring2_ffcorr_flipped", data=ring2_ffcorr_flipped, dtype="f"
                )

        self.logger.log(f"Saving NXS file to: {reduced_nxs_filepath_out}")

    def align_modules_dict(self, deltas: np.ndarray) -> dict:
        """

        Creates a self.all_module_data which ia a dicitionary containing delta position
        and module datae broken up into an accessable way

        self.all_module_data[n_frame][n_mod]

        n_frame may relate to a delta position or potentially the same delta
        if running a time resolved experiment

        If flatfield correction is True in the config file
        it will apply the flatfield correction

        """

        self.all_module_data = {}  # is a dictionary containing a load of dataframes.
        # Each dict is self.all_module_data[n_frame][n_mod] = module_data_at_delta

        det_channels = np.arange(
            self.STRIPS_PER_MODULE * len(self.active_modules), dtype=np.int64
        )
        module_det_channels = np.array_split(det_channels, len(self.active_modules))

        start_bad_chans = np.arange(0, self.n_bad_edge_channels, 1)
        end_bad_chans = np.arange(
            self.STRIPS_PER_MODULE - self.n_bad_edge_channels, self.STRIPS_PER_MODULE, 1
        )

        for n_frame, delta_pos in zip(self.frames_range, deltas, strict=True):
            self.all_module_data[n_frame] = {}
            counts_in_each_frame_and_module = np.array_split(
                self.raw_frame_counts[n_frame], len(self.active_modules)
            )

            for n, (n_mod) in enumerate(self.active_modules):
                # self.logger.log(n_frame, n_mod)

                # mod_start_chan = n_mod*self.STRIPS_PER_MODULE
                # self.logger.log(n_mod, module_det_channels[n])

                module_data_at_delta = pd.DataFrame()
                module_data_at_delta["raw_tth"] = self.module_raw_tth[n_mod]
                module_data_at_delta["tth"] = (
                    module_data_at_delta["raw_tth"].values + delta_pos
                )
                module_data_at_delta["mod_channel"] = self.module_raw_tth[
                    "mod_channel"
                ]  # always 0-1280 for each module
                module_data_at_delta["det_channel"] = module_det_channels[n]
                module_data_at_delta["frame"] = n_frame
                module_data_at_delta["delta_pos"] = delta_pos
                module_data_at_delta["n_mod"] = n_mod
                module_data_at_delta["no_ff_corr"] = counts_in_each_frame_and_module[n]

                # if flatfield is not enabled,
                # the entire array is set to 1, so it makes no difference

                # self.logger.log(n_mod, self.flatfield_modules[n_mod])
                module_data_at_delta["counts"] = self.apply_flatfield_correction(
                    module_data_at_delta["no_ff_corr"].values,
                    self.flatfield_modules[n_mod],
                )

                module_data_at_delta["error"] = np.sqrt(
                    module_data_at_delta["counts"]
                )  # poisson errors

                module_data_at_delta["badchannel"] = (
                    (module_data_at_delta["det_channel"].isin(self.bad_channels))
                    | (module_data_at_delta["mod_channel"].isin(start_bad_chans))
                    | (module_data_at_delta["mod_channel"].isin(end_bad_chans))
                )  # new way

                self.all_module_data[n_frame][n_mod] = module_data_at_delta

            # 	plt.plot(module_data_at_delta["tth"],module_data_at_delta["counts"])
            # 	plt.plot(module_data_at_delta["tth"],module_data_at_delta["no_ff_corr"])
            # plt.show()

        return self.all_module_data

    def remove_bad_channels_modules_frames(self) -> pd.DataFrame:
        if self.bad_channel_masking:
            angular_corrected_data = self.angular_corrected_data_unmasked[
                (~self.angular_corrected_data_unmasked["n_mod"].isin(self.bad_modules))
                & (self.angular_corrected_data_unmasked["badchannel"] == False)  # noqa
                & (~self.angular_corrected_data_unmasked["frame"].isin(self.bad_frames))
            ]  # keep non-bad modules #remove bad channels

        else:
            angular_corrected_data = self.angular_corrected_data_unmasked[
                (~self.angular_corrected_data_unmasked["n_mod"].isin(self.bad_modules))
                & (~self.angular_corrected_data_unmasked["frame"].isin(self.bad_frames))
            ]  # keep non-bad modules only

        return angular_corrected_data

    def concatenate_frames_and_modules(self) -> pd.DataFrame:
        """

        it contatenates the data in self.all_module_data and then appends
        it into a dataframe containing lots of info. Then remove the bad channels

        I suspect when fast_shutter_mode is available we would probably want to iterate
        frames first, modules second and build a
        self.frame_outputdata dict instead of modules

        This way we could iterate the frames and save an xye for each frame. Which means
        we wouldn't concatenate everything into a dataframe, we would concatenate all

        the modules together and be able to look frame by frame

        """

        angular_corrected_data_unmasked = pd.DataFrame()

        for n_mod, n_frame in product(self.active_modules, self.frames_range):
            angular_corrected_data_unmasked = pd.concat(
                [angular_corrected_data_unmasked, self.all_module_data[n_frame][n_mod]],
                axis=0,
            )

        angular_corrected_data_unmasked = angular_corrected_data_unmasked.sort_values(
            by="tth", ascending=True
        )
        angular_corrected_data_unmasked = angular_corrected_data_unmasked.reset_index()

        del self.all_module_data

        return angular_corrected_data_unmasked

    def check_active_modules(self) -> bool:
        """

        checks to see if the modules specified in the config are
        correct for the data shape. If not it will try to infer the correct shape.

        """

        self.logger.log(f"\nModules in data: {self.n_modules_in_data}")

        if (len(self.active_modules) != self.n_modules_in_data) and (
            os.path.exists(self.mythen3_config_dir)
        ):
            self.logger.log(
                f"\nNumber of active modules in {self.config_filepath} does not reflect data!!!!"  # noqa
            )
            self.logger.log("Modules will be determined from mythen3 config")
            self.active_modules = I11Reduction.read_config(self.mythen3_config_dir)

        if (len(self.active_modules) != self.n_modules_in_data) and (
            self.n_modules_in_data == 14
        ):
            self.logger.log("Modules will be determined by using a range()")
            self.active_modules = np.arange(self.n_modules_in_data, dtype=int)

        elif (len(self.active_modules) != self.n_modules_in_data) and (
            self.n_modules_in_data == 28
        ):
            self.logger.log("Modules will be determined by using a range()")
            self.active_modules = np.arange(self.n_modules_in_data, dtype=int)

        self.logger.log(f"Using = {self.active_modules}", "\n")

        if len(self.active_modules) != self.n_modules_in_data:
            self.logger.log(
                f"The modules must be specified correctly in active_modules in: {self.config_filepath}"  # noqa
            )
            quit()

        return True

    @staticmethod
    def calculate_wavelength(beam_energy: float) -> float:
        """

        Calculates wavelength (Angstrom) from beam energy in kev.

        To allow convertion of tth to Q space, using the energy of the beam.
        beam energy is converted to wavlength because it's better

        """

        beam_energy_ev = beam_energy * 1000
        ev_to_j, h_planck, c_speed_of_light = (
            1.602176634e-19,
            6.62607015e-34,
            299792458.0,
        )  # electron volt-joule relationship (in J), plancks constant and speeed of l
        beam_energy_j = beam_energy_ev * ev_to_j
        wavelength_m = (h_planck * c_speed_of_light) / (beam_energy_j)
        wavelength = wavelength_m * 1e10

        return wavelength

    def create_bins(self, tth_values: np.ndarray, rebin_step) -> np.ndarray:
        """
        Return a suitable set of bin centres, and edges for histogramming this data.

        To match old GDA mythen2 behaviour, want start and
        stop to align with "multiples" of rebin step
        (as far as f.p. arithmetic allows this...).

        """
        mintth, maxtth = np.amin(tth_values), np.amax(tth_values)
        start = np.round((mintth / rebin_step), decimals=3) * rebin_step
        stop = np.round((maxtth / rebin_step), decimals=3) * rebin_step

        # start = mintth
        # stop = maxtth
        # self.logger.log("Min2th:",f'{start:.3f}'," | ","Max2th:",f"{stop:.3f}","\n")

        bin_edges = np.arange(
            start=start - (rebin_step / 2),
            stop=stop + rebin_step + (rebin_step / 2),
            step=rebin_step,
            dtype=np.float64,
        )
        bin_centres = 0.5 * (bin_edges[1:] + bin_edges[:-1])

        return bin_centres, bin_edges

    def bin_and_propagate_errors(
        self, x: np.ndarray, y: np.ndarray, e: np.ndarray, error_calc: str = "poisson"
    ) -> np.ndarray:
        """

        The bin centres and edges are calculated and used to bin the data.
        Binning of the data is done used searchsorted == np.digitize.

        Because we want to propagate the errors we will iterate though all the values
        of x, y and e that need to be binned together and propagate the errors

        Errors can be calculated using internal error = error propagation, external
        error std_dev of error or we can take the greatest of the two values.
        Which is probabaly the best idea.

        If you have a high spread of data (high noise), ie peaks with weak intensity
        surely the error can't be less than the spread.
        But equally if you have very large peaks with low spread the

        error should reflect that.

        """

        bin_centres, bin_edges = self.create_bins(x, self.rebin_step)

        if (
            x[-1] == bin_edges[-1]
        ):  # if the last value is exactly equal to the final bin edge it will be lost.
            x[-1] = x[-1] - (
                self.rebin_step / 10000
            )  # I think it would be better to move it inside bin edge,
            # and include, rather than remove all together or create bin with 1 value

        sums, bin_edges = np.histogram(x, bins=bin_edges, weights=y)
        counts = np.histogram(x, bins=bin_edges)[0]
        mean_counts = (
            sums / counts
        )  # this will throw a warning if there are missing counts

        e_sums = np.histogram(
            x, bins=bin_edges, weights=e**2
        )[
            0
        ]  # https://faraday.physics.utoronto.ca/PVB/Harrison/ErrorAnalysis/Propagation.html
        prop_errors = np.sqrt(e_sums) / counts

        repeated_mean = np.repeat(mean_counts, counts)
        std_sums = np.histogram(x, bins=bin_edges, weights=(y - repeated_mean) ** 2)[0]
        std_errors = np.sqrt(std_sums / counts)

        if error_calc == "poisson":
            errors = prop_errors
        elif error_calc == "std_dev":
            errors = std_errors
        elif error_calc == "max":
            errors = np.where(prop_errors > std_errors, prop_errors, std_errors)
        else:
            raise ValueError(
                f"Invalid error_calc value: {error_calc}. Must be 'poisson', 'std_dev', or 'max'."  # noqa
            )

        return bin_centres, bin_edges, mean_counts, errors

    def bin_data(
        self, angular_corrected_data: pd.DataFrame, error_calc: str
    ) -> pd.DataFrame:
        """

        creates the bins, fins the bins centres (which gives the 2th angle)
        Then sticks this all in a dataframe called xyedata with tth,
        counts, error and if wavelength is provided Q

        """

        rebinned_tth, tth_bin_edges, rebinned_counts, errors = (
            self.bin_and_propagate_errors(
                angular_corrected_data["tth"].values,
                angular_corrected_data["counts"].values,
                angular_corrected_data["error"].values,
                error_calc,
            )
        )

        xyedata = pd.DataFrame()

        xyedata["tth"] = rebinned_tth
        xyedata["counts"] = rebinned_counts
        xyedata["error"] = errors

        xyedata = xyedata[
            (xyedata["counts"] != 0) & (~xyedata["counts"].isnull())
        ]  # remove no counts #remove null counts

        if (self.beam_energy) and (self.save_in_q_space):
            q_space = tth_to_q(xyedata["tth"].to_numpy(), self.wavelength)
            xyedata["Q"] = q_space

        return xyedata

    def save_xye(
        self, xye_filepath_out: str, xyedata: pd.DataFrame, x: str
    ) -> np.ndarray:
        """

        given a filepath out, and a dataframe containing some
        x axis (Q or tth) and an x axis label

        it concatenated the data into a ascii format and then saves it

        """

        xye_out_data = np.stack(
            (xyedata[x].values, xyedata["counts"].values, xyedata["error"].values),
            axis=-1,
        )

        np.savetxt(
            xye_filepath_out, xye_out_data, fmt="%.6f", delimiter=" ", newline="\n"
        )

        self.logger.log(f"Saving xye to: {xye_filepath_out}")

        return xye_out_data

    def split_data(self, unmasked: bool = False):
        """

        splits data into frames and modules, depending on how you want to
        look at the data, creates self.frame_data and self.module_raw_data

        """

        if unmasked:
            data_to_split = self.angular_corrected_data_unmasked
        else:
            data_to_split = self.angular_corrected_data

        frame_data = {
            frame: data_to_split[data_to_split["frame"] == frame]
            for frame in data_to_split.frame.unique()
        }
        module_raw_data = {
            n_mod: data_to_split[data_to_split["n_mod"] == n_mod]
            for n_mod in data_to_split.n_mod.unique()
        }

        return frame_data, module_raw_data

    def load_toml_config(self) -> bool:
        """

        Loads the config from a toml file for the data reduction.
        Returns true when success.

        """

        if not os.path.exists(self.config_filepath):
            self.logger.log("Config file does not exist")
            quit()

        with open(self.config_filepath, "rb") as file:
            self.logger.log(f"Using config: {self.config_filepath}")

            self.config = load(file)

        self.logger.log("\n")
        self.mythen3_config_dir = self.config["mythen3_detector_config"]

        if self.flatfield_filepath is None:
            self.flatfield_filepath = self.config["flatfield_filepath"]

        if self.apply_flatfield is None:
            self.apply_flatfield = self.config["apply_flatfield"]

        if self.active_modules is None:
            self.active_modules = self.config["active_modules"]
            self.active_modules.sort()

        if self.bad_modules is None:
            self.bad_modules = self.config["bad_modules"]

        self.bad_channel_masking = self.config["bad_channel_masking"]
        self.default_counter = self.config["default_counter"]
        self.n_bad_edge_channels = self.config["edge_bad_channels"]
        self.rebin_step = self.config["rebin_step"]
        self.beam_energy = self.config["beam_energy"]

        if self.beamline_offset is None:
            self.beamline_offset = self.config.get("beamline_offset")

        self.save_nxs_out = self.config["save_nxs_out"]
        self.out_raw_data = self.config["out_raw_data"]
        self.save_in_q_space = self.config["save_in_Q_space"]
        self.debug_mode = self.config["debug_mode"]
        self.modules_in_flatfield = self.config["modules_in_flatfield"]
        self.send_to_ispyb = self.config["send_to_ispyb"]

        if self.angcal_filepath is None:
            self.angcal_filepath = self.config["angcal_filepath"]

        if self.data_reduction_mode is None:
            self.data_reduction_mode = int(self.config["data_reduction_mode"])

        self.error_calc = self.config["error_calc"]
        self.verbose_nxs = self.config["verbose_nxs"]
        self.bad_channels_filepath = self.config["bad_channels_filepath"]

        self.logger.log("Active modules:", self.active_modules)
        self.logger.log("Bad modules:", self.bad_modules)
        self.logger.log("Bad frames:", self.bad_frames)
        self.logger.log("Beam energy (keV):", self.beam_energy)
        self.logger.log("Beamline offset:", self.beamline_offset)
        self.logger.log("Flatfield filepath:", self.flatfield_filepath)
        self.logger.log("Saving in Q space:", self.save_in_q_space)
        self.logger.log("Saving in NXS:", self.save_nxs_out)
        self.logger.log("Apply Flatfield:", self.apply_flatfield)
        self.logger.log("Using counter:", self.default_counter)
        self.logger.log(
            "Number of bad channels at module edge:", self.n_bad_edge_channels
        )
        self.logger.log("Rebin step:", self.rebin_step)

        data_reduction_mode_dict = {
            0: "standard",
            1: "time-resolved",
            2: "pump-probe",
            3: "flatfield",
        }

        self.logger.log(
            "Data reduction mode:", data_reduction_mode_dict[self.data_reduction_mode]
        )

        return True

    def set_save_filepaths(self) -> bool:
        if not self.xye_filepath_out:
            self.xye_filepath_out = os.path.join(
                self.file_dir,
                f"{self.file_name}_summed_mythen3{self.filename_suffix}.xye",
            )

        if not os.path.exists(os.path.join(self.file_dir, "processed")):
            os.makedirs(os.path.join(self.file_dir, "processed"))

        if not self.reduced_nxs_filepath_out:
            self.reduced_nxs_filepath_out = os.path.join(
                self.file_dir,
                "processed",
                f"{self.file_name}_reduced_mythen3{self.filename_suffix}.nxs",
            )

        if not self.xye_filepath_out_q:
            self.xye_filepath_out_q = os.path.join(
                self.file_dir,
                f"{self.file_name}_summed_mythen3_Q{self.filename_suffix}.xye",
            )

        return True

    def _refine_rebin(self):
        best_so_far = [[1e-8, 1e19]]

        for n, new_rebin in enumerate(
            np.linspace(
                self.rebin_step / 10, self.rebin_step + (self.rebin_step / 10), 10000
            )
        ):
            bin_centres, bin_edges = self.create_bins(
                self.angular_corrected_data["tth"].values, new_rebin
            )

            sums, bin_edges = np.histogram(
                self.angular_corrected_data["tth"].values,
                bins=bin_edges,
                weights=self.angular_corrected_data["counts"].values,
            )
            counts = np.histogram(
                self.angular_corrected_data["tth"].values, bins=bin_edges
            )[0]

            if np.amin(counts) == 0:
                continue

            # mean_counts = (
            #     sums / counts
            # )  # this will throw a warning if there are
            # missing counts in a bin as a result of missing module

            bin_centres_repeated = np.repeat(bin_centres, counts)

            chi = np.sum(
                np.abs(bin_centres_repeated - self.angular_corrected_data["tth"].values)
            ) * len(bin_centres)

            if n % 100 == 0:
                self.logger.log(new_rebin)

            if chi < best_so_far[-1][1]:
                best_so_far.append([new_rebin, chi])
                self.logger.log(new_rebin, chi)

        best_rebin = best_so_far[-1][0]

        return best_rebin

    def plot_modules(self, block=True):
        for n_mod in self.active_modules:
            n_mod_theta = self.module_raw_tth[n_mod]
            plt.plot([n_mod] * 1280, n_mod_theta)

        plt.xlabel("module number")
        plt.ylabel("angle tth")
        plt.show(block=block)
        if block is True:
            plt.close()

    def plot_modules_by_ring(self, mask=(), block=True):
        plt.figure(figsize=(10, 4))

        modules_min_max = {}

        for n_mod in self.active_modules:
            if n_mod in mask:
                continue

            n_mod_theta = self.module_raw_tth[n_mod]
            if n_mod <= 13:
                plt.plot(n_mod_theta, [0] * 1280, label=str(n_mod))
                plt.text(n_mod_theta[640], 0.1, str(n_mod), fontsize=8)
            else:
                plt.plot(n_mod_theta, [1] * 1280, label=str(n_mod))
                plt.text(n_mod_theta[640], 0.9, str(n_mod), fontsize=8)

            modules_min_max[n_mod] = [float(min(n_mod_theta)), float(max(n_mod_theta))]

        plt.ylabel("Ring number")
        plt.yticks([0, 1])
        plt.xlabel("Angle (tth)")
        plt.savefig("./outputs/module_arrangment.png")
        plt.show(block=block)
        if block is True:
            plt.close()

        total_degrees = np.amax(self.module_raw_tth[0]) - np.amin(
            self.module_raw_tth[13]
        )

        radius = 762
        circum = 2 * 3.14159 * radius

        length_of_arc = circum * (total_degrees / 360)

        mm_per_degree = length_of_arc / total_degrees
        print("Total Angular Coverage", total_degrees)
        print("Radius of detector (mm):", 762)
        print("mm/tth", mm_per_degree)

        module_tth_spans = []
        module_sizes = []
        print(len(self.active_modules))
        print(len(self.good_modules))

        spec_module_size = 1280 * 0.05

        for n_mod in list(self.good_modules):
            module_1 = modules_min_max[n_mod]
            module_tth_span = np.amax(module_1) - np.amin(module_1)
            module_tth_spans.append(module_tth_span)
            module_size = module_tth_span * mm_per_degree
            module_sizes.append(module_size)

            mm_per_deg_size = spec_module_size / module_tth_span

            print(n_mod, module_tth_span, module_size, mm_per_deg_size)

        fig, ax1 = plt.subplots(figsize=(10, 7))
        ax1.scatter(list(self.good_modules), module_tth_spans)
        ax1.set_ylabel("Size (tth)")
        ax2 = ax1.twinx()

        ax2.scatter(list(self.good_modules), module_sizes)
        ax2.set_ylabel("Size (mm)")
        ax1.set_xlabel("Module")
        plt.savefig("./outputs/sizes.png")
        plt.close()

        for n_mod in self.active_modules[0:-1]:
            if n_mod == 13:
                continue

            n_mod1 = n_mod
            n_mod2 = n_mod + 1

            module_1 = modules_min_max[n_mod1]
            module_2 = modules_min_max[n_mod2]

            sorted_modules = np.sort(np.concatenate((module_1, module_2)))

            diff = mm_per_degree * (sorted_modules[2] - sorted_modules[1])

            print("Modules:", n_mod1, n_mod2)
            print("Sorted Module Edges (tth):", sorted_modules)
            print("Gap (mm)", diff, "\n")

        mod1spans = []
        mod2spans = []

        for mod1, mod2 in paired_modules():
            n_mod1_theta = self.module_raw_tth[mod1]
            mod1span = np.amax(n_mod1_theta) - np.amin(n_mod1_theta)
            n_mod2_theta = self.module_raw_tth[mod2]
            mod2span = np.amax(n_mod2_theta) - np.amin(n_mod2_theta)

            if mod1 in [11, 17, 27]:
                mod1spans.append(np.nan)
            else:
                mod1spans.append(mod1span)

            if mod2 in [11, 17, 27]:
                mod2spans.append(np.nan)
            else:
                mod2spans.append(mod2span)

        plt.plot(np.array(mod1spans) * mm_per_degree)
        plt.plot(np.array(mod2spans) * mm_per_degree)
        plt.ylabel("size (mm)")
        plt.xlabel("Index from module")
        plt.savefig("./outputs/module_compare.png")

        plt.show()

    def plot_by(self, parameters=(), x="tth", at_a_time=False):
        if isinstance(parameters, str):
            parameters = [parameters]

        for parameter in parameters:
            for val in np.unique(self.angular_corrected_data[parameter]):
                parameter_data = self.angular_corrected_data[
                    self.angular_corrected_data[parameter] == val
                ]
                p = plt.scatter(
                    parameter_data[x], parameter_data["counts"], label=str(val), s=2
                )
                col = p.get_facecolors()[-1].tolist()
                plt.text(
                    np.median(parameter_data[x]),
                    -np.amax(self.angular_corrected_data["counts"]) / 30,
                    str(val),
                    color=col,
                )

                if at_a_time:
                    plt.legend()
                    plt.xlabel("Angle (tth)")
                    plt.ylabel("Intensity (arb. units)")
                    plt.show()

        if not at_a_time:
            plt.legend()
            plt.xlabel("Angle (tth)")
            plt.ylabel("Intensity (arb. units)")
            plt.show()

    def plot_module_offsets(self, mask=()):
        plt.figure(figsize=(10, 4))

        ring_1_angles = {}
        ring_2_angles = {}

        for n_mod in self.active_modules:
            if n_mod in mask:
                continue

            n_mod_theta = self.module_raw_tth[n_mod]
            if n_mod <= 13:
                ring_1_angles[n_mod] = n_mod_theta

            else:
                ring_2_angles[n_mod] = n_mod_theta

        ring_1_array = np.array(list(ring_1_angles.values())).flatten()
        ring_1_diff = np.gradient(ring_1_array)

        ring_2_array = np.array(list(ring_2_angles.values())).flatten()
        ring_2_diff = np.gradient(ring_2_array)

        ring1_steps = ring_1_diff[np.abs(ring_1_diff) > 1]
        ring2_steps = ring_2_diff[np.abs(ring_2_diff) > 1]

        for x, y in zip(ring_1_angles.keys(), ring1_steps, strict=True):
            print(x, y)

        for x, y in zip(ring_2_angles.keys(), ring2_steps, strict=True):
            print(x, y)

        plt.plot(ring_1_diff, label="ring1")
        plt.plot(ring_2_diff, label="ring2")
        plt.legend()
        plt.show()

    def plot_module_counts(self, block=True):
        plt.figure()

        for n_mod in self.active_modules:
            if n_mod in self.angular_corrected_data["n_mod"].values:
                plt.bar(n_mod, np.sum(self.module_raw_data[n_mod]))

        plt.xlabel("module number")
        plt.ylabel("module raw counts")
        plt.show(block=block)
        if block is True:
            plt.close()

    def plot_diffraction(self, out_only=True, filepath=None):
        plt.figure(figsize=(15, 10))

        if not out_only:
            for n_frame in self.frame_data.keys():
                framedata = self.frame_data[n_frame]
                plt.plot(framedata["tth"], framedata["counts"], label=n_frame)

        plt.errorbar(
            self.xyedata["tth"],
            self.xyedata["counts"],
            self.xyedata["error"],
            label=self.error_calc,
        )
        plt.legend()
        plt.xlabel("tth")
        plt.ylabel("Intensity (arb. units)")

        if filepath:
            plt.savefig(filepath)

        plt.show()
        plt.close()

    def plot_raw_vs_xye(self):
        fig, axes = plt.subplots(2)

        axes[0].step(
            self.angular_corrected_data["tth"],
            self.angular_corrected_data["counts"],
            label="ff corrected counts",
            color="k",
        )
        axes[0].plot(self.xyedata["tth"], self.xyedata["counts"], label="xye")
        axes[0].legend()

        axes[1].step(
            self.angular_corrected_data["tth"],
            self.angular_corrected_data["no_ff_corr"],
            label="ff uncorrected counts",
            color="k",
        )
        axes[1].plot(self.xyedata["tth"], self.xyedata["counts"], label="xye")

        axes[1].legend()
        plt.show()

    def plot_by_region_of_interest(self, peaks, tol=0.03, filepath=None):
        fig = plt.figure(figsize=(15, 10))

        max_rows = 4
        max_cols = 4

        gs = GridSpec(max_rows, max_cols, figure=fig)  # upper bound on plots

        i = 0

        for peak in np.sort(peaks):
            if i > 15:
                break

            region_of_interest = self.angular_corrected_data[
                (self.angular_corrected_data["tth"] > peak - tol)
                & (self.angular_corrected_data["tth"] < peak + tol)
            ]

            present_modules = np.unique(region_of_interest["n_mod"])

            if len(region_of_interest) == 0:  # or (len(present_modules) < 2):
                continue

            row = i // max_rows
            col = i % max_cols

            ax = fig.add_subplot(gs[row, col])

            for n_mod in present_modules:
                mod_data = region_of_interest[region_of_interest["n_mod"] == n_mod]
                ax.scatter(mod_data["tth"], mod_data["counts"], label=str(n_mod))
                ax.vlines(peak, np.min(mod_data["counts"]), np.max(mod_data["counts"]))
                ax.legend()

            i = i + 1

        plt.xlabel("tth")
        plt.ylabel("Intensity (arb. units)")
        if filepath:
            plt.savefig(filepath)
        plt.show()
        plt.close()

    def plot_diffraction_by_mod(self, filepath=None, block=True):
        plt.figure(figsize=(15, 10))

        for n_mod in np.unique(self.angular_corrected_data["n_mod"]):
            mod_data = self.angular_corrected_data[
                self.angular_corrected_data["n_mod"] == n_mod
            ]
            plt.scatter(mod_data["tth"], mod_data["counts"], label=str(n_mod))
            plt.text(
                np.median(mod_data["tth"]), np.amin(mod_data["counts"]), str(n_mod)
            )

        plt.xlabel("tth")
        plt.ylabel("Intensity (arb. units)")
        plt.legend()

        if filepath:
            plt.savefig(filepath)

        plt.show(block=block)
        if block is True:
            plt.close()

    def plot_rings(self):
        ringmod0_13 = self.angular_corrected_data[
            self.angular_corrected_data["n_mod"].isin(np.arange(0, 14, 1, dtype=int))
        ]
        ringmod14_27 = self.angular_corrected_data[
            self.angular_corrected_data["n_mod"].isin(np.arange(14, 28, 1, dtype=int))
        ]
        plt.errorbar(
            ringmod0_13["tth"],
            ringmod0_13["counts"],
            ringmod0_13["error"],
            label="ring1",
        )
        plt.errorbar(
            ringmod14_27["tth"],
            ringmod14_27["counts"],
            ringmod14_27["error"],
            label="ring2",
        )
        plt.show()
        plt.close()

    def make_module_boundary_guide(self, nmodules):
        guide = np.zeros([nmodules * 1280])

        for mod in range(nmodules):
            guide[(mod - 1) * 1280 : ((mod - 1) * 1280) + 10] = 100000000000

        return guide

    def return_outliers(
        self,
        factor: int | float = 3,
        low_bound: float | None = None,
        plot: bool = False,
    ) -> None:
        """
        This is used to return bad pixels.
        This should only be run on a scan conducted on water at high angles (
        or something else that scatters very flat) -
        otherwise the results are invalid"""

        all_bad_channels = np.array([], dtype=int)

        for n_mod in np.unique(self.angular_corrected_data["n_mod"]):
            hist_model = "fd"

            mod_data = self.angular_corrected_data[
                self.angular_corrected_data["n_mod"] == n_mod
            ]

            hist, bin_edges = np.histogram(mod_data["counts"].values, bins=hist_model)
            mean_hist = np.mean(bin_edges)
            std_hist = np.std(bin_edges)
            stdfact = 1.5 * std_hist

            print("hist", mean_hist, std_hist, mean_hist - stdfact, mean_hist + stdfact)

            if plot:
                plt.hist(
                    mod_data["counts"].values, bins=hist_model
                )  # arguments are passed to np.histogram
                plt.title(f"Histogram with {hist_model} bins")
                plt.show()

            median_count = np.median(mod_data["counts"])
            stddev = np.std(mod_data["counts"])

            low_data_points = mod_data[(mod_data["counts"] < median_count / factor)]
            high_data_points = mod_data[(mod_data["counts"] > median_count * factor)]

            low_channels = low_data_points["det_channel"].values
            high_channels = high_data_points["det_channel"].values

            bad_channels = np.sort(np.unique(np.append(low_channels, high_channels)))
            all_bad_channels = np.append(all_bad_channels, bad_channels)

            print(n_mod, bad_channels, median_count, stddev)

            if low_bound:
                low_bound_points = mod_data[(mod_data["counts"] < low_bound)]
                low_bound_channels = low_bound_points["det_channel"].values

                if len(low_bound_channels) > 0:
                    print("\n", "low bound", low_bound_channels, "\n")

            if plot:
                plt.scatter(mod_data["det_channel"], mod_data["counts"], label=n_mod)
                plt.scatter(
                    bad_channels, [median_count] * len(bad_channels), color="red"
                )
                plt.legend()
                plt.show()

        for bc in all_bad_channels:
            print(bc)

    def data_reduction_mode_standard(self):
        # standard data reduction

        self.raw_frame_counts, self.frames_range, self.wholedetector_raw_frames = (
            self.read_nxs_data(self.filepath)
        )
        # self.ffcorr = self.ffcorr_calc(self.wholedetector_raw_frames)

        self.all_module_data = self.align_modules_dict(
            self.deltas
        )  # its a dict of dataframes

        self.angular_corrected_data_unmasked = self.concatenate_frames_and_modules()
        self.angular_corrected_data = self.remove_bad_channels_modules_frames()

        self.frame_data, self.module_raw_data = self.split_data(
            unmasked=self.out_raw_data
        )  # if this is done after remove bad channels and modules
        self.xyedata = self.bin_data(
            self.angular_corrected_data, error_calc=self.error_calc
        )

        #####save data
        self.save_xye(self.xye_filepath_out, self.xyedata, "tth")

        if (self.save_in_q_space) and (self.beam_energy):
            self.save_xye(self.xye_filepath_out_q, self.xyedata, "Q")

        if self.save_nxs_out:
            try:
                self.save_nxs_outfile(
                    self.reduced_nxs_filepath_out,
                    self.xyedata,
                    self.module_raw_data,
                    self.frame_data,
                    self.angular_corrected_data_unmasked,
                    debug=self.debug_mode,
                )
            except Exception as e:
                self.logger.log(e, self.filepath, "is open?")

    def data_reduction_mode_time_resolved(self):
        ###where every frame is a unique dataset and you want lots of final xye's
        # This iterates through every frame and load them
        # seperately because for large datasets it will eat up memory

        for n_frame in range(self.n_frames):
            self.logger.log(f"Analysing frame: {n_frame}")

            self.raw_frame_counts, self.frames_range, self.wholedetector_raw_frames = (
                self.read_nxs_data(self.filepath, frame=n_frame)
            )
            self.all_module_data = self.align_modules_dict(
                [self.deltas[n_frame]]
            )  # align all modules for the specfic frame of data
            self.angular_corrected_data_unmasked = self.concatenate_frames_and_modules()
            self.angular_corrected_data = self.remove_bad_channels_modules_frames()
            self.frame_data, self.module_raw_data = self.split_data(
                unmasked=self.out_raw_data
            )  # if this is done after remove bad channels and modules
            self.xyedata = self.bin_data(
                self.angular_corrected_data, error_calc=self.error_calc
            )

            self.save_xye(
                self.xye_filepath_out.replace(
                    ".xye", f"_frame_{n_frame + 1}{self.filename_suffix}.xye"
                ),
                self.xyedata,
                "tth",
            )
            if (self.save_in_q_space) and (self.beam_energy):
                self.save_xye(
                    self.xye_filepath_out_q.replace(
                        ".xye", f"_frame_{n_frame + 1}{self.filename_suffix}.xye"
                    ),
                    self.xyedata,
                    "Q",
                )

    def data_reduction_mode_pump_probe(self):
        # pump probe

        self.raw_frame_counts, self.frames_range, self.wholedetector_raw_frames = (
            self.read_nxs_data(self.filepath, sum_frames=True)
        )
        self.all_module_data = self.align_modules_dict(
            self.deltas
        )  # its a dict of dataframes
        self.angular_corrected_data_unmasked = self.concatenate_frames_and_modules()
        self.angular_corrected_data = self.remove_bad_channels_modules_frames()
        self.frame_data, self.module_raw_data = self.split_data(
            unmasked=self.out_raw_data
        )  # if this is done after remove bad channels and modules
        self.xyedata = self.bin_data(
            self.angular_corrected_data, error_calc=self.error_calc
        )

        #####save data

        self.save_xye(self.xye_filepath_out, self.xyedata, "tth")

        if (self.save_in_q_space) and (self.beam_energy):
            self.save_xye(self.xye_filepath_out_q, self.xyedata, "Q")

        if self.save_nxs_out:
            self.save_nxs_outfile(
                self.reduced_nxs_filepath_out,
                self.xyedata,
                self.module_raw_data,
                self.frame_data,
                self.angular_corrected_data,
                debug=self.debug_mode,
            )

    def data_reduction_mode_flatfield(self, cleanup=True, peak_centre=80.67, tol=3.05):
        # flatfield
        """
        This assumes that the angular calibration is correct,
        if it isn't then the peak centre will be wrong and the flatfield will be wrong.
        Do the angular calibration first, then run this.
        """

        from datetime import datetime

        ###analyse the middle frame (where we hope the peak is present)
        # throw away bad channels, and find the peak maxima.
        # If this is a flatfield, that peak will always be at that tth
        #

        self.raw_frame_counts, self.frames_range, self.wholedetector_raw_frames = (
            self.read_nxs_data(self.filepath, frame=int(self.n_frames / 2))
        )
        self.all_module_data = self.align_modules_dict(
            [self.deltas[int(self.n_frames / 2)]]
        )  # its a dict of dataframes
        self.angular_corrected_data_unmasked = self.concatenate_frames_and_modules()
        self.angular_corrected_data = self.remove_bad_channels_modules_frames()

        max_index = np.argmax(self.angular_corrected_data["counts"])
        peak_centre = (self.angular_corrected_data["tth"].values)[max_index]

        new_flatfield = np.full((len(self.active_modules) * 1280), 0)
        normalised_beam_intensity = self.beam_intensity / np.median(self.beam_intensity)

        new_flatfield = np.full((len(self.active_modules) * 1280), 0)

        for n_frame in range(self.n_frames):
            self.logger.log(
                f"Analysing frame: {n_frame}, Beam: {normalised_beam_intensity[n_frame]}"  # noqa
            )
            self.raw_frame_counts, self.frames_range, self.wholedetector_raw_frames = (
                self.read_nxs_data(self.filepath, frame=n_frame)
            )

            beam_normalised_flatfield = (
                self.wholedetector_raw_frames / normalised_beam_intensity[n_frame]
            )

            if cleanup:
                tth = self.deltas[n_frame] + self.whole_data_raw_tth
                tth_throw_index = np.where(
                    ~((tth < (peak_centre + tol)) & (tth > (peak_centre - tol)))
                )[0]
                beam_normalised_flatfield[tth_throw_index] = 0

                peak_counts = beam_normalised_flatfield[beam_normalised_flatfield != 0]

                if len(peak_counts) == 0:
                    maxx = 10
                else:
                    maxx = np.median(peak_counts) * 1000

                self.logger.log(maxx)

                count_throw_index = np.where(beam_normalised_flatfield > maxx)[0]
                beam_normalised_flatfield[count_throw_index] = 0

            new_flatfield = new_flatfield + beam_normalised_flatfield

        new_flatfield = new_flatfield / np.median(new_flatfield)

        count_throw_index = np.where(new_flatfield > 10)[0]
        new_flatfield[count_throw_index] = 0

        datetimestr = datetime.now().strftime("%Y-%m-%d")  # _%H:%M")

        if not self.out_directory:
            save_dir = "/dls_sw/i11/software/mythen3/diamond/flatfield"
        else:
            save_dir = self.out_directory

        flatfield_dir_flatfield_save_path = os.path.join(
            save_dir,
            f"{self.file_name}_flatfield_{datetimestr}{self.filename_suffix}.h5",
        )

        with h5pyFile(flatfield_dir_flatfield_save_path, "w") as out_file:
            out_file.create_dataset("flatfield", data=new_flatfield)

        self.logger.log(
            f"New flatfield has been saved to {flatfield_dir_flatfield_save_path}"
        )

    def data_reduction_mode_0_fast(self):
        # standard fast - testing

        for n_frame, delta in range(self.n_frames), self.deltas:
            self.raw_frame_counts, self.frames_range, self.wholedetector_raw_frames = (
                self.read_nxs_data(self.filepath, frame=n_frame)
            )
            self.all_module_data = self.align_modules_dict(
                [delta]
            )  # its a dict of dataframes

        self.angular_corrected_data_unmasked = self.concatenate_frames_and_modules()
        self.angular_corrected_data = self.remove_bad_channels_modules_frames()
        self.frame_data, self.module_raw_data = self.split_data(
            unmasked=self.out_raw_data
        )  # if this is done after remove bad channels and modules
        self.xyedata = self.bin_data(
            self.angular_corrected_data, error_calc=self.error_calc
        )

        #####save data

        self.save_xye(self.xye_filepath_out, self.xyedata, "tth")

        if (self.save_in_q_space) and (self.beam_energy):
            self.save_xye(self.xye_filepath_out_q, self.xyedata, "Q")

        if self.save_nxs_out:
            self.save_nxs_outfile(
                self.reduced_nxs_filepath_out,
                self.xyedata,
                self.module_raw_data,
                self.frame_data,
                self.angular_corrected_data,
                debug=self.debug_mode,
            )

        if self.debug_mode:
            self.debug_reduction()

    def communicate_with_control(self, send_to_ispyb: bool = False):
        """
        Attempts to connect to i11-control and send a message indicating
        that a file has been processed. This will cause gda to plot the latest file

        Also may send xye to ispyb so that users can lookup data

        """

        try:
            daq = Messenger("i11-control")
            daq.connect()
            daq.send_file(str(self.xye_filepath_out))  # sends message to GDA

            if send_to_ispyb:
                p = Path(self.filepath)
                magic_path = p.parent / ".ispyb" / (p.stem + "_mythen_nx/data.dat")
                copy2(self.xye_filepath_out, magic_path)  # copies to ispyb

        except Exception as e:
            self.logger.log(f"{e}: No messenger")

    def __init__(
        self,
        filepath: str | None = None,
        reduced_nxs_filepath_out: str | None = None,
        xye_filepath_out: str | None = None,
        xye_filepath_out_q: str | None = None,
        out_directory: str | None = None,
        config_filepath: str | None = None,
        beam_energy: float | None = None,
        data_reduction_mode: int | None = None,
        bad_frames: list[int] | None = None,
        bad_modules: list[int] | None = None,
        beamline_offset: float | None = None,
        active_modules: list[int] | None = None,
        flatfield_filepath: str | None = None,
        apply_flatfield: bool | None = None,
        angcal_filepath: str | None = None,
        filename_suffix: str = "",
        live: bool | None = False,
        execute_reduction: bool | None = True,
        logging: bool = True,
    ):
        self.reduced_nxs_filepath_out = reduced_nxs_filepath_out
        self.xye_filepath_out = xye_filepath_out
        self.xye_filepath_out_q = xye_filepath_out_q
        self.config_filepath = config_filepath
        self.beam_energy = beam_energy
        self.flatfield_filepath = flatfield_filepath
        self.active_modules = active_modules
        self.angcal_filepath = angcal_filepath
        self.filename_suffix = filename_suffix
        self.out_directory = out_directory
        self.bad_frames = bad_frames
        self.bad_modules = bad_modules
        self.beamline_offset = beamline_offset
        self.data_reduction_mode = data_reduction_mode
        self.apply_flatfield = apply_flatfield
        self.live = live
        self.execute_reduction = execute_reduction
        self.logging = logging
        self.filepath = filepath

        if self.bad_frames is None:
            self.bad_frames = []  # frames that should be removed, because they are bad

        if self.out_directory is not None:
            self.file_dir = (
                self.out_directory
            )  # this will replace the directory with the one specified by the user
        else:
            self.file_dir = os.path.dirname(
                self.filepath
            )  # this will be something like /dls/i11/data/2025/cm40643-1

        self.logger = AnalysisLogger(
            os.path.join(self.file_dir, "processed", "mythen3_reduction.log")
        )

        if not os.path.exists(self.filepath):
            self.logger.log("NXS file does not exist")
            quit()

        self.logger.log("######################################\n")
        self.logger.log(f"Data reduction being performed on: {self.filepath}")

        self.file_name = os.path.splitext(os.path.basename(self.filepath))[
            0
        ]  # this will be something like 1290222
        self.file_extension = os.path.splitext(self.filepath)[-1]  # this will be .nxs

        # Each module is divided into this many pixels.
        self.STRIPS_PER_MODULE = 1280
        self.MODULES_PER_DETECTOR = 28

        if not self.config_filepath:
            self.config_filepath = (
                "/dls_sw/i11/software/mythen3/diamond/mythen3_reduction_config.toml"
            )

        self.load_toml_config()
        self.set_save_filepaths()

        self.good_modules = set(self.active_modules).difference(set(self.bad_modules))

        self.logger.log("Modules in output data:", self.good_modules)

        if self.beam_energy:
            self.wavelength = I11Reduction.calculate_wavelength(self.beam_energy)
            self.logger.log(
                f"Beam Energy: {self.beam_energy} (keV) | Wavelength = {self.wavelength:.3f} (Angstrom)\n"  # noqa
            )

        # self.bad_channels = self.generate_badchannel_dict()
        self.bad_channels = load_int_array_from_file(self.bad_channels_filepath)

        if self.angcal_filepath:
            self.logger.log(
                f"Using the following angular calibrations file: {angcal_filepath}"
            )

            self.module_angular_cal, self.beamline_offset = (
                I11Reduction.read_singular_angcal_files(self.angcal_filepath)
            )
        else:
            # self.logger.log("Using the following angular calibrations files")
            # self.logger.log(f"{list(self.config["angular_calibrations"].values())}")
            self.module_angular_cal, self.beamline_offset = (
                I11Reduction.read_angular_calibration_and_create_cal_dict(
                    self.config, self.active_modules
                )
            )

        #####################################################################################################################################
        # everything before here is just setting up calibrations,
        # and hasn't read the actual dataset

        if self.filepath.lower().endswith(".nxs"):
            self.n_frames, self.deltas, self.n_modules_in_data = self.read_nxs_metadata(
                self.filepath
            )
        else:
            self.logger.log("\n\nAborting: Must be a nexus!!\n\n")
            quit()

        self.raw_flatfield_counts = self.load_flatfield(self.flatfield_filepath)
        self.flatfield_modules = self.split_flatfield()

        self.check_active_modules()
        # dict containing angular calibrations for each module
        self.module_raw_tth, self.whole_data_raw_tth = self.calculate_modules_tth()

        ############################################################################################################

        # data reduction happens here differently for each data reduction mode

        if self.execute_reduction:
            if (
                (self.data_reduction_mode == 0)
                and (self.n_frames > 50)
                and ((np.amax(self.deltas) - np.amin(self.deltas)) > 60)
            ):
                self.logger.log(
                    "This looks like a flatfield scan...treating it as such"
                )
                self.data_reduction_mode = 3

            if self.data_reduction_mode == 0:
                self.data_reduction_mode_standard()
                # self.data_reduction_mode_0_fast()
            elif self.data_reduction_mode == 1:
                self.data_reduction_mode_time_resolved()
            elif self.data_reduction_mode == 2:
                self.data_reduction_mode_pump_probe()
            elif self.data_reduction_mode == 3:
                self.data_reduction_mode_flatfield()
            else:
                self.logger.log(
                    "Data reduction mode must be one of the specified values"
                )

            if self.live:
                self.communicate_with_control(self.send_to_ispyb)

            zeros = (
                self.angular_corrected_data[self.angular_corrected_data["counts"] == 0]
            ).sort_values(by="det_channel", ascending=True)
            # print("bad channel", zeros)
            self.logger.log(f"Possible bad channels: {zeros['det_channel'].unique()}")
            ##########################################################################
            self.logger.log("###############END###############\n")


if __name__ == "__main__":
    ##################################################
    ##################################################

    parser = argparse.ArgumentParser(
        description="Post-processor for mythen3 data; "
        "converts an uncalibrated .nxs files "
        "written by the detector into a calibrated and corrected .xye ASCII file.",
        add_help=True,
    )

    parser.add_argument(
        "-d", "--data", help="Path to the nxs data file to reduce", required=True
    )

    parser.add_argument(
        "-c",
        "--config",
        help="Path to .toml config file, containing the bad channels, "
        "angular cal files, binning size, etc",
        required=False,
    )

    parser.add_argument(
        "-o",
        "--out-xye-file",
        help="Path to write output .xye file (2th, counts, error)",
        required=False,
    )

    parser.add_argument(
        "-q",
        "--out-q-space-file",
        help="Path to write output Q .xye file (Q, counts, error)",
        required=False,
    )

    parser.add_argument(
        "-nxs",
        "--out-nxs-file",
        help="Path to write output processed .nxs file",
        required=False,
    )

    parser.add_argument(
        "-ang",
        "--ang-cal-file",
        help="Path to angular calibration file",
        required=False,
    )

    parser.add_argument(
        "-drm",
        "--data-reduction-mode",
        help="How the data should be reduced. 0, 1 or 2 \
        0 = standard (all data will be reduced into 1 file, possibly multiple angles) \
        1 = time-resolved mode (many frames per .nxs saved into "
        "seperate .xye files for each frame)\
        2 = pump-probe (all frames will be read and summed together - "
        "data has been taken at static angle, optimised for lots of frames)",
        required=False,
    )

    parser.add_argument(
        "-bf",
        "--bad_frames",
        help="A list of 'bad frames' that will be removed from the final dataset."
        "Specified comma seperated eg. -bf 0,1,2,3 ",
        required=False,
        type=lambda s: [int(item) for item in s.split(",")],
    )

    parser.add_argument(
        "-l",
        "--live",
        help="Is this a live experiment, and therefore shuold we send messenges to gda?",  # noqa
        required=False,
    )

    args = parser.parse_args()

    import time

    start = time.time()
    Analysis = I11Reduction(
        filepath=args.data,
        config_filepath=args.config,
        xye_filepath_out_q=args.out_q_space_file,
        reduced_nxs_filepath_out=args.out_nxs_file,
        angcal_filepath=args.ang_cal_file,
        live=args.live,
    )
    end = time.time()

    Analysis.logger.log(f"Time: {end - start} sec")
