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
from xrpd_toolbox.utils.utils import (
    get_entry,
    h5_to_array,
    h5_to_float,
    h5_to_string,
)

PIXEL_SIZE = 7.5e-5  # in m
INITIAL_DISTNACE = 700  # mm
DEFAULT_MAX_SHAPE = (1028, 512)


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
    def __init__(
        self,
        filepath: str | Path,
        eiger_data_path: str = "fastcs_eiger",
    ):
        self.filepath = str(filepath)
        self.eiger_data_path = eiger_data_path

        self.entry = get_entry(self.filepath)  # /entry
        self.dataset_path = f"/{self.entry}/{self.eiger_data_path}/data"

    @cached_property
    def positions(self) -> np.ndarray:

        position_path = f"{self.entry}/{self.eiger_data_path}/tth"

        try:
            deltas = h5_to_array(self.filepath, position_path)
            return deltas
        except ValueError as e:
            print(f"{e} - {position_path} in data - returning 0")
            deltas = np.array([0])
            return deltas

    @cached_property
    def durations(self) -> np.ndarray:

        count_time_path = f"/{self.entry}/instrument/{self.eiger_data_path}/count_time"

        return h5_to_array(self.filepath, count_time_path)

    @cached_property
    def energy_kev(self) -> float:

        energy_kev_data_path = f"/{self.entry}/instrument/xtal/energy_kev"

        energy_kev = h5_to_float(self.filepath, energy_kev_data_path)

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
        return self.get_data(frames=slice(None))

    def get_data(
        self,
        frames: int | Collection[int] | slice,
    ):

        with File(self.filepath, "r") as file:
            if self.dataset_path not in file:
                raise ValueError(
                    f"Dataset path {self.dataset_path} not found in HDF5 file."
                )

            data = file.get(self.dataset_path)

            if (data is not None) and isinstance(data, Dataset):
                if data.ndim < 1:
                    raise ValueError("Data has insufficient dimensions.")
                module_frame_data = data[frames, ...]

                return np.asarray(module_frame_data)
            else:
                raise ValueError(
                    f"Data at {self.dataset_path} in {self.filepath}is None."
                )

    def get_pixel_mask_filepath_and_datapath(self) -> tuple[str, str]:

        pixel_mask_path = f"{self.entry}/instrument/{self.eiger_data_path}/pixel_mask"

        mask_filepath = h5_to_string(self.filepath, pixel_mask_path)

        mask_filepath, mask_datapath = str(mask_filepath).split("//")

        if not Path(mask_filepath).exists():
            mask_filepath = Path(self.filepath).parent / Path(mask_filepath).stem
            mask_filepath = (
                str(mask_filepath) + ".h5"
            )  # when odin/ophyd async fixes this remove the .h5

        return mask_filepath, mask_datapath

    def get_calibrant(self) -> str | None:

        calibrant_path = f"{self.entry}/plan_metadata/calibrant"
        try:
            h5_to_string(self.filepath, calibrant_path)
        except Exception as e:
            print(e)
            return None

    def get_mask(self):

        mask_filepath, mask_datapath = self.get_pixel_mask_filepath_and_datapath()

        mask = h5_to_array(filepath=mask_filepath, data_path=mask_datapath)

        return mask.astype(bool)

    def is_background(self) -> bool:

        background = f"{self.entry}/plan_metadata/background"

        return bool(h5_to_array(self.filepath, background))

    @cached_property
    def plan_name(self) -> str:
        """returns the plan name as a string, eg static_collection or data_collection"""

        return self.get_plan_name()

    def get_plan_name(self) -> str:
        """returns the plan name as a string, eg static_collection or data_collection"""

        plan_name_path = f"{self.entry}/plan_metadata/plan_name"

        return h5_to_string(self.filepath, plan_name_path)

    def get_composition(self) -> str:
        """returns the composition of the sample eg. SiO2 or Tb(HCO2)3. etc"""

        composition_path = f"{self.entry}/plan_metadata/sample_info/data/composition"

        return h5_to_string(self.filepath, composition_path)


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


if __name__ == "__main__":  # pragma: no cover - manual/interactive smoke test
    from matplotlib.colors import LogNorm

    SETTINGS = EigerSettings()
    FILEPATH = "/workspaces/XRPD-Toolbox/examples/i15-1/eiger_500k/1414223.nxs"
    eiger = Eiger500K(filepath=FILEPATH, settings=SETTINGS)

    calibrant = get_calibrant(calibrant_name="Si")
    calibrant.wavelength = 0.161699 / 1e10

    pixel1 = PIXEL_SIZE
    pixel2 = PIXEL_SIZE

    shape = DEFAULT_MAX_SHAPE

    poni1 = pixel1 * shape[0] / 2
    poni2 = pixel2 * shape[1] / 2

    import pyFAI.detectors

    ai = AzimuthalIntegrator(
        detector=pyFAI.detectors.Detector(
            pixel1=pixel1, pixel2=pixel2, max_shape=shape
        ),
        wavelength=calibrant.wavelength,
        dist=0.25,
        poni1=poni1,
        poni2=poni2,
        rot1=0,
        rot2=0.1,
        rot3=0,
    )

    calibration_image = calibrant.fake_calibration_image(
        ai, shape=shape, resolution=0.01
    )

    plt.imshow(calibration_image, norm=LogNorm())
    plt.colorbar()
    plt.show()
