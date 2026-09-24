from collections.abc import Collection
from copy import deepcopy
from functools import cached_property
from pathlib import Path
from typing import Literal

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

from xrpd_toolbox.core import XRPDBaseModel
from xrpd_toolbox.utils.unit_conversion import beam_energy_to_wavelength
from xrpd_toolbox.utils.utils import h5_to_array

PIXEL_SIZE = 7.5e-5  # in m
INITIAL_DISTNACE = 250  # mm
DEFAULT_MAX_SHAPE = (1028, 512)


def unique_slices(arr: np.ndarray):
    """Retturns slices at which the all the values for the input array are the same
    assumes that the input array is sorted and only increases/decreases

    if it's not will have to use np.argwhere - but that isn't this functons

    """
    arr = np.asarray(arr)
    _, start_idx = np.unique(arr, return_index=True)
    start_idx = np.sort(start_idx)
    end_idx = np.append(start_idx[1:], len(arr))
    return [slice(s, e) for s, e in zip(start_idx, end_idx, strict=True)]


def sum_unique_two_theta_positions_and_normalise(eiger_data: "EigerDataLoader"):

    slices_of_data = unique_slices(eiger_data.positions)

    assert len(slices_of_data) == len(np.unique(eiger_data.positions))

    summed_and_normalised_frames = []

    for slice in slices_of_data:
        frames_with_position = eiger_data.get_data(slice)
        i0_for_frames = eiger_data.get_i0()[slice]

        summed_frames_at_tth_position = np.sum(frames_with_position, axis=0)

        assert summed_frames_at_tth_position.ndim > 1

        summed_and_normalised_frames_at_tth_position = (
            summed_frames_at_tth_position / np.sum(i0_for_frames)
        )

        summed_and_normalised_frames.append(
            summed_and_normalised_frames_at_tth_position
        )

    summed_and_normalised_frames = np.array(summed_and_normalised_frames)

    return summed_and_normalised_frames


def apply_mask(image_frames: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """applys a mask to all frames in the image"""

    masked_image_frames = [image * mask for image in image_frames]

    masked_image_frames = np.array(masked_image_frames)

    return masked_image_frames


class EigerSettings(XRPDBaseModel):
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
    """Reads a single nexus file's worth of Eiger data.

    Keeps one h5py.File handle open for the lifetime of the instance instead
    of reopening the file on every read - most of these methods are called
    once per unique two-theta position when reducing a scan (see
    sum_unique_two_theta_positions_and_normalise), and re-opening the file
    from scratch every time was the dominant cost there. Call close() (or
    use as a context manager) to release the handle deterministically;
    otherwise it is released when the instance is garbage collected.
    """

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

    def get_unique_tth_positions(self):
        """returns uniuqe positions as defined by tth"""

        return np.unique(self.positions)

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
        """Dangerous as it might contains a lot of data,
        which will then be loaded into memory - you have been warned

        ideally use get_data with specific frames as slice

        """
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

            return np.asarray(module_frame_data)
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
        summed_and_normalised_frames = sum_unique_two_theta_positions_and_normalise(
            eiger_data=self
        )

        return summed_and_normalised_frames

    def get_summed_normalised_and_masked_frames(self) -> np.ndarray:

        summed_and_normalised_frames = self.get_summed_and_normalised_frames()
        mask = self.get_mask()

        summed_normalised_and_masked_frames = apply_mask(
            image_frames=summed_and_normalised_frames, mask=mask
        )

        return summed_normalised_and_masked_frames

    @property
    def i0(self) -> np.ndarray:
        i0_data_path = f"{self.entry}/i0/data"

        return self._read_array(i0_data_path)

    def get_i0(self) -> np.ndarray:
        return self.i0

    def sum_frames(self) -> np.ndarray:
        """Returns a 1D array containing the total counts of each frame.

        Any leading (scan) dimensions are flattened, so the output has one
        entry per frame. Frames are read one at a time to limit memory use.
        """

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


class Eiger500K(Detector):
    IS_FLAT = False  # this detector is flat
    IS_CONTIGUOUS = True

    """This is simple a test platform for data simution - not used for real data"""

    def __init__(
        self,
        filepath: str | Path | None = None,
        settings: EigerSettings | None = None,
        poni: str | Path | dict | None = None,
        wavelength: float | None = None,  # in Angstrom
    ):
        self.filepath = filepath
        self.settings = settings
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
        elif self.settings is not None:
            self.ai = pyFAI.load(str(self.settings.poni_filepath))
        else:
            self.ai = None

        if self.ai is not None and (
            self.ai.pixel1 != PIXEL_SIZE or self.ai.pixel2 != PIXEL_SIZE
        ):
            raise ValueError(
                f"Pixel size in poni file ({self.ai.pixel1}, {self.ai.pixel2}) does not match expected pixel size ({PIXEL_SIZE})."  # noqa
            )

    def process_step_scan(self):
        for _position in self.data_loader.positions:
            # do geometry transformation

            pass

    def load_geometry(self, poni_files: str | list[str | Path]):
        # if isinstance(poni_files, list) and (len(poni_files) > 1):
        #     mg = MultiGeometry()

        # else:
        #     self.ai = pyFAI.load(str(poni_files))
        pass

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
            my_ai.rot2 -= i * step
            my_img = lab6.fake_calibration_image(my_ai)
            jupyter.display(
                my_img,
                label=f"Angle rot2: {np.degrees(my_ai.rot2)}",
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
            ai_copy.rot2 = position

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
