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
        modules: Collection[int],
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
    def module_data(self) -> list[np.ndarray]:
        """self.data split into modules - only read from file when first used"""
        return np.array_split(self.data, len(self.modules), axis=-1)

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

    @cached_property
    def positions(self) -> np.ndarray:

        position_path = f"/{self.entry}/instrument/diff_stage/delta"
        deltas = self._read_array(position_path)

        mean_delta = np.mean(deltas)

        interpolated_delta = np.full(self.n_frames, mean_delta)

        return interpolated_delta

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


def run_mythen_time_resolved_scan(
    filepath: str, absent_modules: Collection[int] = (4, 11, 17)
):

    wait_for_finished_file(filepath)

    active_modules = [m for m in range(28) if m not in absent_modules]

    logger.info(f"Active modules {active_modules}")

    settings = MythenSettings(
        active_modules=active_modules, data_reduction_mode="time_resolved"
    )

    detector = MythenDetector(
        filepath=filepath, settings=settings, data_loader=MythenBlueskyDataLoader
    )

    xye_filepaths = detector.process_time_resolved()

    filenumber = get_filenumber_from_filepath(filepath)

    for xye_filepath in xye_filepaths:
        logger.info(f"Filepath: {filepath}")
        _publish_xye(xye_filepath, data_type="pxrd", filenumber=filenumber)

    return xye_filepaths


if __name__ == "__main__":
    filepath = "/workspaces/outputs/i11/fast_trigger/i11-1487459.nxs"

    absent = (4, 11, 17)
    active_modules = [m for m in range(28) if m not in absent]

    mythen_data = MythenBlueskyDataLoader(filepath, active_modules)

    data = mythen_data.get_data(slice(None))

    print(data.shape)

    print(data)

    run_mythen_time_resolved_scan(filepath)
