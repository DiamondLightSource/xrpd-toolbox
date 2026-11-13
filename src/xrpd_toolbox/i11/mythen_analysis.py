import logging
from collections.abc import Collection
from functools import cached_property
from pathlib import Path

import numpy as np

from xrpd_toolbox.data_loader import BaseDataLoader
from xrpd_toolbox.i11.mythen import PIXELS_PER_MODULE, MythenDetector, MythenSettings
from xrpd_toolbox.plotting import DataPlot
from xrpd_toolbox.utils.files import get_filenumber_from_filepath
from xrpd_toolbox.utils.utils import (
    get_entry,
    load_int_array_from_file,
    wait_for_finished_file,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)


class MythenBlueskyDataLoader(BaseDataLoader):
    """Reads the Mythen data in one nexus file, keeping the file open.
    The whole detector is in one dataset, so it is split into modules."""

    def __init__(
        self,
        filepath: str | Path,
        modules: Collection[int] | slice,
        frames: int | slice | Collection[int] = slice(None),
    ):
        entry = get_entry(filepath)
        super().__init__(
            filepath, dataset_path=f"/{entry}/fastcs_mythen3/fastcs_mythen3"
        )

        self.frames = frames
        self.modules = modules

    def get_data(
        self,
        frames: int | Collection[int] | slice,
        pixels: slice = slice(None),
    ) -> np.ndarray:
        # the mythen data has a size 1 axis between the frames and the channels
        return np.atleast_2d(self.dataset[frames, 0, pixels])

    @cached_property
    def data(self) -> np.ndarray:
        """The selected frames - only read from file when first used"""
        return self.get_data(self.frames)

    @cached_property
    def pixels_per_module(self):
        return 1280

    @cached_property
    def n_modules_in_data(self):
        # dataset is (frames, 1, pixels) - use the shape so no data is read
        return int(self.dataset.shape[-1] // self.pixels_per_module)

    @cached_property
    def module_data(self) -> list[np.ndarray]:
        """self.data split into modules - only read from file when first used"""
        n_modules = (
            self.n_modules_in_data
            if isinstance(self.modules, slice)
            else len(self.modules)
        )
        return np.array_split(self.data, n_modules, axis=-1)

    def get_module_data(self, n_module: int) -> np.ndarray:
        """Reads only the nth module from file, for the selected frames.
        Only the modules that are present are in the data, so the nth module in
        the data is the nth module in self.modules"""

        pixels = slice(n_module * PIXELS_PER_MODULE, (n_module + 1) * PIXELS_PER_MODULE)

        return self.get_data(self.frames, pixels)

    @property
    def n_frames(self) -> int:
        """Number of selected frames, without loading any data"""

        if isinstance(self.frames, int):
            return 1
        elif isinstance(self.frames, slice):
            return len(range(self.get_number_of_frames())[self.frames])
        else:
            return len(self.frames)

    def get_positions(self):

        position_path = f"/{self.entry}/instrument/diff_stage/delta"
        deltas = self._read_array(position_path)

        return deltas

    @cached_property
    def positions(self) -> np.ndarray:

        positions = self.get_positions()

        if len(positions) != self.n_frames:
            mean_delta = np.mean(positions)
            positions = np.full(self.n_frames, mean_delta)

        return positions

    @cached_property
    def durations(self) -> np.ndarray:

        return np.ones(self.n_frames)


def _publish_xye(
    output_xy_filepath: str | Path, data_type: str, filenumber: int | None = None
):
    try:
        data_plot = DataPlot.from_csv(output_xy_filepath)
        data_plot.filenumber = filenumber
        data_plot.x_label = "2θ (deg)"
        data_plot.data_type = data_type
        data_plot.publish(beamline="i11")

    except Exception as e:
        logger.error(e)


def run_mythen_one_diffraction_per_trigger(
    filepath: str,
    settings: MythenSettings,
    frames_to_plot: int = 10,
):

    detector = MythenDetector(
        filepath=filepath, settings=settings, data_loader=MythenBlueskyDataLoader
    )

    xye_filepaths = detector.process_time_resolved()

    filenumber = get_filenumber_from_filepath(filepath)

    n_files = len(xye_filepaths)

    plot_index = max(1, n_files // frames_to_plot)

    for n, xye_filepath in enumerate(xye_filepaths):
        if (n == 0) or (n == n_files - 1) or (n % plot_index == 0):
            logger.info(f"Publishing frame: {n} filepath: {xye_filepath}")

            _publish_xye(xye_filepath, data_type="pxrd", filenumber=filenumber)

    return xye_filepaths


def run_mythen_sum_all_positions(filepath: str, settings: MythenSettings):

    filenumber = get_filenumber_from_filepath(filepath)

    detector = MythenDetector(
        filepath=filepath, settings=settings, data_loader=MythenBlueskyDataLoader
    )

    xye_filepath = detector.process_step_scan()

    logger.info(f"Completed dat reduciton for{xye_filepath}")
    _publish_xye(xye_filepath, data_type="pxrd", filenumber=filenumber)


def bad_channels_to_physical(
    bad_channels: np.ndarray, active_modules: Collection[int]
) -> np.ndarray:
    """The bluesky data only holds the active modules, packed together, so the
    bad channels file counts channels by position in the data
    (max len(active_modules) * PIXELS_PER_MODULE). MythenDetector picks masks by
    physical module, so convert each channel to its physical channel number:
    channel in the nth module of the data -> same pixel in active_modules[n]"""

    bad_channels = np.asarray(bad_channels, dtype=int)
    active_modules = np.asarray(list(active_modules), dtype=int)

    n_module, pixel = np.divmod(bad_channels, PIXELS_PER_MODULE)

    in_data = (n_module >= 0) & (n_module < len(active_modules))
    if not np.all(in_data):
        logger.warning(
            f"Ignoring bad channels outside the {len(active_modules)} active "
            f"modules: {bad_channels[~in_data].tolist()}"
        )

    physical = active_modules[n_module[in_data]] * PIXELS_PER_MODULE + pixel[in_data]

    return np.unique(physical)


def write_physical_bad_channels_file(
    bad_channels_filepath: str | Path,
    active_modules: Collection[int],
    output_filepath: str | Path,
) -> Path:
    """Writes bad_channels_to_physical of the given file to output_filepath"""

    output_filepath = Path(output_filepath)
    output_filepath.parent.mkdir(parents=True, exist_ok=True)

    bad_channels = load_int_array_from_file(bad_channels_filepath)
    physical = bad_channels_to_physical(bad_channels, active_modules)
    np.savetxt(output_filepath, physical, fmt="%d")

    logger.info(
        f"Bad channels from {bad_channels_filepath} converted to physical "
        f"channels in {output_filepath}"
    )

    return output_filepath


def run_mythen_analysis(
    nexus_filepath: str,
    absent_modules: Collection[int] = (4, 11, 17),
):
    wait_for_finished_file(nexus_filepath)

    folder = Path(nexus_filepath).parent

    bad_channel_filepath = folder / "badchannels.txt"

    active_modules = [m for m in range(28) if m not in absent_modules]
    logger.info(f"Active modules {active_modules}")

    # the bad channels file counts channels by position in the data, so convert
    # it to physical channels (resolving the default file if there isn't one)
    bad_channel_filepath = write_physical_bad_channels_file(
        bad_channels_filepath=MythenSettings(
            bad_channels_filepath=bad_channel_filepath
        ).bad_channels_filepath,
        active_modules=active_modules,
        output_filepath=folder / "processed" / "badchannels_physical.txt",
    )

    mythen_data = MythenBlueskyDataLoader(
        filepath=nexus_filepath, modules=active_modules
    )

    plan_name = mythen_data.get_plan_name()
    scan_type = mythen_data.get_scan_type()

    logger.info(f"{plan_name=} {scan_type=}")

    if scan_type == "time_resolved":
        settings = MythenSettings(
            active_modules=active_modules,
            data_reduction_mode="time_resolved",
            bad_channels_filepath=bad_channel_filepath,
        )

        run_mythen_one_diffraction_per_trigger(
            filepath=nexus_filepath,
            settings=settings,
        )
    elif scan_type == "step_scan":
        settings = MythenSettings(
            active_modules=active_modules,
            data_reduction_mode="step_scan",
            bad_channels_filepath=bad_channel_filepath,
        )
        run_mythen_sum_all_positions(filepath=nexus_filepath, settings=settings)
    else:
        settings = MythenSettings(
            active_modules=active_modules,
            data_reduction_mode="step_scan",
            bad_channels_filepath=bad_channel_filepath,
        )
        run_mythen_sum_all_positions(filepath=nexus_filepath, settings=settings)

        raise ValueError(
            f"There is no process for {scan_type=}, treated it as a step scan anyway"
        )


if __name__ == "__main__":
    filepath = "/workspaces/outputs/i11/fast_trigger/i11-1487561.nxs"

    absent = (4, 11, 17)

    run_mythen_analysis(filepath, absent_modules=absent)

    filepath = "/workspaces/outputs/i11/fast_trigger/i11-1487562.nxs"

    run_mythen_analysis(filepath, absent_modules=absent)
