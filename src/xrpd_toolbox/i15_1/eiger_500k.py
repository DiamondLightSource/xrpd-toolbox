import logging
from collections.abc import Collection
from copy import deepcopy
from functools import cached_property
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyFAI
from h5py import Dataset, File
from pyFAI import units
from pyFAI.calibrant import get_calibrant
from pyFAI.detectors import Detector
from pyFAI.goniometer import MultiGeometry
from pyFAI.gui import jupyter
from pyFAI.integrator.azimuthal import AzimuthalIntegrator
from pyFAI.method_registry import IntegrationMethod

from xrpd_toolbox.utils.unit_conversion import beam_energy_to_wavelength
from xrpd_toolbox.utils.utils import h5_to_array

PIXEL_SIZE = 7.5e-5  # in m
INITIAL_DISTNACE = 250  # mm

DEFAULT_MAX_SHAPE = (512, 1028)

# the arm swings horizontally so it drives pyFAI's rot1, and on i15-1 the beam
# centre moves to higher columns as two-theta increases, hence the sign
ARM_ROTATION_SIGN = -1.0
logger = logging.getLogger(__name__)


# the tth readback jitters by ~6e-5 deg
TTH_GROUP_TOLERANCE_DEG = 1e-3
SUM_CHUNK_FRAMES = 100


def group_positions(
    positions: Collection[float], tolerance: float = TTH_GROUP_TOLERANCE_DEG
) -> tuple[np.ndarray, np.ndarray]:
    """Group readbacks within `tolerance` of each other, as the readback jitters.

    Returns the group of each frame and the mean position of each group.
    """
    positions = np.asarray(positions, dtype=np.float64)
    if positions.size == 0:
        return np.empty(0, dtype=int), np.empty(0)

    order = np.argsort(positions, kind="stable")
    sorted_labels = np.concatenate(
        [[0], np.cumsum(np.diff(positions[order]) > tolerance)]
    )
    labels = np.empty(positions.size, dtype=int)
    labels[order] = sorted_labels

    n_groups = sorted_labels[-1] + 1
    counts = np.bincount(labels, minlength=n_groups)
    means = np.bincount(labels, weights=positions, minlength=n_groups) / counts
    return labels, means


def _contiguous_runs(labels: np.ndarray) -> list[slice]:
    """Slices over which `labels` doesn't change."""
    starts = np.flatnonzero(np.diff(labels)) + 1
    bounds = np.concatenate([[0], starts, [labels.size]])
    return [slice(a, b) for a, b in zip(bounds[:-1], bounds[1:], strict=True)]


def apply_mask(image_frames: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Zero the masked (nonzero) pixels in every frame."""

    bad_pixels = np.asarray(mask).astype(bool)

    masked_image_frames = np.where(bad_pixels, 0, np.asarray(image_frames))

    return masked_image_frames


class EigerDataLoader:
    """Reads the Eiger data in one nexus file, keeping the file open."""

    def __init__(
        self,
        filepath: str | Path,
        eiger_data_path: str = "fastcs_eiger",
    ):
        self.filepath = str(filepath)
        self.eiger_data_path = eiger_data_path
        self._file: File | None = None

        self.entry = list(self.file.keys())[0]  # /entry
        self.dataset_path = f"/{self.entry}/{self.eiger_data_path}/data"

    @property
    def file(self) -> File:
        if self._file is None:
            self._file = File(self.filepath, "r", libver="latest", swmr=True)
        return self._file

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> "EigerDataLoader":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def get_data_dimensions(self):
        data = self.file.get(self.dataset_path)
        if (data is not None) and isinstance(data, Dataset):
            return np.shape(data[()])
        else:
            raise ValueError(f"Data is None at {self.dataset_path} in {self.filepath}")

    def _read_array(self, data_path: str) -> np.ndarray:
        data = self.file.get(data_path)
        if (data is not None) and isinstance(data, Dataset):
            return np.asarray(data)
        else:
            raise ValueError(f"Data is None at {data_path} in {self.filepath}")

    def _read_string(self, data_path: str) -> str:
        data = self.file.get(data_path)
        if (data is not None) and isinstance(data, Dataset):
            value = data[()]

            if isinstance(value, bytes):
                value = value.decode()

            return value
        else:
            raise ValueError(f"Data is None at {data_path} in {self.filepath}")

    @cached_property
    def positions(self) -> np.ndarray:

        position_path = f"/{self.entry}/instrument/tth/data"

        deltas = self._read_array(position_path)
        return deltas

    @cached_property
    def tth_groups(self) -> tuple[np.ndarray, np.ndarray]:
        labels, group_tth = group_positions(self.positions)
        counts = np.bincount(labels, minlength=len(group_tth))
        logger.info(
            "Grouped %d frames into %d two-theta positions", labels.size, counts.size
        )
        for tth, count in zip(group_tth, counts, strict=True):
            logger.debug("  2θ=%.5f°: %d frames", tth, count)
        return labels, group_tth

    def get_unique_tth_positions(self) -> np.ndarray:
        """Ascending, in the same order as the summed frames."""
        return self.tth_groups[1]

    @cached_property
    def durations(self) -> np.ndarray:

        count_time_path = f"/{self.entry}/plan_metadata/exposure_time_per_frame"

        return self._read_array(count_time_path)

    @cached_property
    def energy_kev(self) -> float:

        energy_kev_data_path = f"/{self.entry}/instrument/xtal/energy_kev"

        energy_kev_arr = self._read_array(energy_kev_data_path)

        energy_kev = float(np.mean(energy_kev_arr))

        return energy_kev

    @cached_property
    def wavelength(self) -> float:
        """Returns the wavelength in angstrom"""

        wavelength = beam_energy_to_wavelength(beam_energy=self.energy_kev, unit="kev")

        return wavelength

    def get_wavelength(self) -> float:
        """Returns the wavelength in angstrom"""
        return self.wavelength

    def load_all_data(self) -> np.ndarray:
        """Loads every frame into memory - prefer get_data with a slice."""
        return self.get_data(frames=slice(None))

    def get_data(
        self,
        frames: int | Collection[int] | slice,
    ):

        if self.dataset_path not in self.file:
            raise ValueError(
                f"Dataset path {self.dataset_path} not found in HDF5 file."
            )

        data = self.file.get(self.dataset_path)

        if (data is not None) and isinstance(data, Dataset):
            if data.ndim < 1:
                raise ValueError("Data has insufficient dimensions.")
            module_frame_data = data[frames, ...]

            return module_frame_data
        else:
            raise ValueError(f"Data at {self.dataset_path} in {self.filepath}is None.")

    @cached_property
    def mask_filepath(self):

        pixel_mask_path = f"{self.entry}/instrument/{self.eiger_data_path}/pixel_mask"

        mask_filepath = self._read_string(pixel_mask_path)

        return mask_filepath

    def get_pixel_mask_filepath_and_datapath(self) -> tuple[str, str]:

        mask_filepath, mask_datapath = str(self.mask_filepath).split("//")

        if not Path(mask_filepath).exists():
            mask_filepath = Path(self.filepath).parent / Path(mask_filepath).stem
            mask_filepath = (
                str(mask_filepath) + ".h5"
            )  # when odin/ophyd async fixes this remove the .h5

        return mask_filepath, mask_datapath

    def get_calibrant(self) -> str | None:

        calibrant_path = (
            f"/{self.entry}/plan_metadata/auxiliary_scans/Standard Sample/pin/contents"
        )
        return self._read_string(calibrant_path)

    def get_air_scan_filepath(self):

        air_scan_filename_dataset_path = (
            f"/{self.entry}/plan_metadata/auxiliary_scans/Air/filename"  # noqa
        )

        air_scan_filename = self._read_string(air_scan_filename_dataset_path)

        air_scan_filepath = Path(self.filepath).parent / air_scan_filename

        return air_scan_filepath

    def get_sample_environment_scan_filepath(self) -> str:

        sample_environment_filename_dataset_path = (
            f"/{self.entry}/plan_metadata/auxiliary_scans/Empty Capillary/filename"  # noqa
        )

        air_scan_filename = self._read_string(sample_environment_filename_dataset_path)

        sample_environment_filepath = Path(self.filepath).parent / air_scan_filename

        return str(sample_environment_filepath)

    def get_mask(self, as_nan: bool = False):

        mask_filepath, mask_datapath = self.get_pixel_mask_filepath_and_datapath()

        mask = h5_to_array(filepath=mask_filepath, data_path=mask_datapath)
        if as_nan:
            nan_mask = np.where(
                mask != 0, np.nan, 1.0
            )  # Eiger convention: nonzero = bad
            return nan_mask
        else:
            return mask.astype(bool)

    def get_plan_type(self) -> str:

        plan_type_path = f"/{self.entry}/plan_metadata/plan_type"

        return self._read_string(plan_type_path)

    @cached_property
    def plan_name(self) -> str:
        """returns the plan name as a string, eg static_collection or data_collection"""

        return self.get_plan_name()

    def get_plan_name(self) -> str:
        """returns the plan name as a string, eg static_collection or data_collection"""

        plan_name_path = f"{self.entry}/plan_metadata/plan_name"

        return self._read_string(plan_name_path)

    def get_data_shape(self) -> dict:

        data_shape_path = f"/{self.entry}/plan_metadata/data_shape"

        data_shape_array = self._read_array(data_shape_path)

        data_shape_dict = dict(data_shape_array)

        return data_shape_dict

    def get_composition(self) -> str:
        """returns the composition of the sample eg. SiO2 or Tb(HCO2)3. etc"""

        composition_path = f"{self.entry}/plan_metadata/sample_info/data/composition"

        return self._read_string(composition_path)

    def get_summed_and_normalised_frames(self) -> np.ndarray:
        summed_and_normalised_frames = (
            self.sum_unique_two_theta_positions_and_normalise()
        )

        return summed_and_normalised_frames

    def get_summed_and_masked_frames(self) -> np.ndarray:
        """Summed and masked but not normalised, for calibration."""

        summed_frames = self.sum_unique_two_theta_positions_and_normalise(
            normalise=False
        )

        return apply_mask(image_frames=summed_frames, mask=self.get_mask())

    def get_summed_normalised_and_masked_frames(self) -> np.ndarray:

        summed_and_normalised_frames = self.get_summed_and_normalised_frames()
        mask = self.get_mask()

        summed_normalised_and_masked_frames = apply_mask(
            image_frames=summed_and_normalised_frames, mask=mask
        )

        return summed_normalised_and_masked_frames

    @property
    def i0(self) -> np.ndarray:
        i0_data_path = f"/{self.entry}/i0/data"

        # areaDetector writes (n_frames, 1)
        return self._read_array(i0_data_path).flatten()

    def get_i0(self, abs: bool = True) -> np.ndarray:
        if abs:
            return np.abs(self.i0)
        else:
            return self.i0

    def sum_frames(self) -> np.ndarray:
        """Total counts in each frame."""

        data = self.file.get(self.dataset_path)

        if not isinstance(data, Dataset):
            raise ValueError(f"Data is None at {self.dataset_path} in {self.filepath}")

        if data.ndim < 2:
            raise ValueError(f"Expected image data with ndim >= 2, got {data.ndim}")

        frame_indices = list(np.ndindex(data.shape[:-2]))
        totals = np.zeros(len(frame_indices), dtype=np.float64)

        for n, index in enumerate(frame_indices):
            totals[n] = np.sum(data[index], dtype=np.float64)

        return totals

    def sum_unique_two_theta_positions_and_normalise(
        self, normalise: bool = True
    ) -> np.ndarray:
        """One summed image per two-theta position, divided by its total i0.

        Without normalise that's the mean frame at each position.
        """

        labels, group_tth = self.tth_groups
        i0 = self.get_i0(abs=True) if normalise else np.ones(labels.size)

        summed_frames: np.ndarray | None = None
        summed_i0 = np.zeros(len(group_tth))

        # chunked so a position with ~1000 frames doesn't all load at once
        for run in _contiguous_runs(labels):
            group = labels[run.start]
            for start in range(run.start, run.stop, SUM_CHUNK_FRAMES):
                chunk = slice(start, min(start + SUM_CHUNK_FRAMES, run.stop))
                frames = np.asarray(self.get_data(chunk))
                if summed_frames is None:
                    summed_frames = np.zeros(
                        (len(group_tth), *frames.shape[1:]), dtype=np.float64
                    )
                summed_frames[group] += frames.sum(axis=0, dtype=np.float64)
                summed_i0[group] += i0[chunk].sum()

        if summed_frames is None:
            raise ValueError(f"No frames to sum in {self.filepath}")

        return summed_frames / summed_i0[:, np.newaxis, np.newaxis]


class Eiger500K(Detector):
    IS_FLAT = False  # this detector is flat
    IS_CONTIGUOUS = True

    """This is simple a test platform for data simution - not used for real data"""

    def __init__(
        self,
        filepath: str | Path | None = None,
        poni: str | Path | dict | None = None,
        wavelength: float | None = None,  # in Angstrom
    ):
        self.filepath = filepath
        self.poni = poni
        self.calibrant = None
        self.wavelength = wavelength

        self.max_shape = DEFAULT_MAX_SHAPE  # Default shape if no data

        if self.filepath is not None:
            self.data_loader = EigerDataLoader(self.filepath)

        super().__init__(pixel1=PIXEL_SIZE, pixel2=PIXEL_SIZE, max_shape=self.max_shape)

        if isinstance(self.poni, (str, Path)):
            self.ai = pyFAI.load(str(self.poni))
        elif isinstance(self.poni, dict):
            self.ai = AzimuthalIntegrator(detector=self, **self.poni)
        else:
            self.ai = None

        if self.ai is not None and (
            self.ai.pixel1 != PIXEL_SIZE or self.ai.pixel2 != PIXEL_SIZE
        ):
            raise ValueError(
                f"Pixel size in poni file ({self.ai.pixel1}, {self.ai.pixel2}) does not match expected pixel size ({PIXEL_SIZE})."  # noqa
            )

    def set_calibrant(self, calibrant_name: str, wavelength_in_ang: float):
        self.calibrant = get_calibrant(calibrant_name)
        self.calibrant.wavelength = wavelength_in_ang / 1e10

        return self.calibrant

    def test(self):
        poni1 = 0.06144
        poni2 = 0.06144
        wavelength = 1e-10

        lab6 = get_calibrant("LaB6")
        lab6.wavelength = wavelength

        ai = pyFAI.load(
            {
                "dist": 0.1,
                "poni1": poni1,
                "poni2": poni2,
                "detector": self,
                "wavelength": wavelength,
            }
        )

        method = IntegrationMethod.parse("full", dim=1)
        img = lab6.fake_calibration_image(ai)

        plt.imshow(img)
        plt.show()

        step = 15 * np.pi / 180
        ais = []
        imgs = []
        fig, ax = plt.subplots(1, 5, figsize=(20, 4))
        for i in range(5):
            my_ai = deepcopy(ai)
            my_ai.rot1 += ARM_ROTATION_SIGN * i * step
            my_img = lab6.fake_calibration_image(my_ai)
            jupyter.display(
                my_img,
                label=f"Angle rot1: {np.degrees(my_ai.rot1)}",
                ax=ax[i],
            )
            ais.append(my_ai)
            imgs.append(my_img)
            print(my_ai)

        mg = MultiGeometry(ais, unit="2th_deg", radial_range=(0, 90))
        print(mg)
        fig, ax = plt.subplots(2, 1, figsize=(12, 16))
        jupyter.plot1d(mg.integrate1d(imgs, 10000, method=method), ax=ax[0])
        plt.show()

    def integrate_images(
        self, images: Collection[np.ndarray], ais: Collection[AzimuthalIntegrator]
    ):
        method = IntegrationMethod.parse("full", dim=1)
        mg = MultiGeometry(ais, unit=units.TTH_DEG)
        x_data, y_data = mg.integrate1d(images, npt=10000, method=method)

        return x_data, y_data

    def simulate_data(
        self,
        positions_in_tth: Collection[float],
        calibrant_name: str,
        wavelength_in_ang: float,
        resolution: float = 0.03,
    ) -> tuple[list[np.ndarray], list[AzimuthalIntegrator]]:
        simulated_data = []
        simulated_ais = []

        if self.calibrant is None:
            self.calibrant = self.set_calibrant(calibrant_name, wavelength_in_ang)
        if self.ai is None:
            raise AttributeError("No Azimuthal Integrator set")

        positions_rad = np.deg2rad(np.asarray(positions_in_tth, dtype=float))

        for position in positions_rad:
            ai_copy = deepcopy(self.ai)
            ai_copy.rot1 = ARM_ROTATION_SIGN * position

            simulated_image = self.calibrant.fake_calibration_image(
                ai_copy, shape=self.max_shape, resolution=resolution
            )

            simulated_ais.append(ai_copy)
            simulated_data.append(simulated_image)

        return simulated_data, simulated_ais

    def simulate_1d_pattern(
        self,
        positions_in_tth: Collection[float],
        calibrant_name: str,
        wavelength_in_ang: float,
    ):
        simulated_step_scan, simulated_ais = self.simulate_data(
            positions_in_tth=positions_in_tth,
            calibrant_name=calibrant_name,
            wavelength_in_ang=wavelength_in_ang,
        )

        simulated_x_data, simulated_y_data = self.integrate_images(
            simulated_step_scan, simulated_ais
        )

        return simulated_x_data, simulated_y_data


# if __name__ == "__main__":  # pragma: no cover - manual/interactive smoke test
#     from matplotlib.colors import LogNorm

#     SETTINGS = EigerSettings()
#     FILEPATH = "/workspaces/XRPD-Toolbox/examples/i15-1/eiger_500k/1414223.nxs"
#     eiger = Eiger500K(filepath=FILEPATH, settings=SETTINGS)

#     calibrant = get_calibrant(calibrant_name="Si")
#     calibrant.wavelength = 0.161699 / 1e10

#     pixel1 = PIXEL_SIZE
#     pixel2 = PIXEL_SIZE

#     shape = DEFAULT_MAX_SHAPE

#     poni1 = pixel1 * shape[0] / 2
#     poni2 = pixel2 * shape[1] / 2

#     import pyFAI.detectors

#     ai = AzimuthalIntegrator(
#         detector=pyFAI.detectors.Detector(
#             pixel1=pixel1, pixel2=pixel2, max_shape=shape
#         ),
#         wavelength=calibrant.wavelength,
#         dist=0.25,
#         poni1=poni1,
#         poni2=poni2,
#         rot1=0,
#         rot2=0.1,
#         rot3=0,
#     )

#     calibration_image = calibrant.fake_calibration_image(
#         ai, shape=shape, resolution=0.01
#     )

#     plt.imshow(calibration_image, norm=LogNorm())
#     plt.colorbar()
#     plt.show()
