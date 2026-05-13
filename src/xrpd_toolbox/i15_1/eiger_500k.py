from collections.abc import Collection
from copy import deepcopy
from functools import cached_property
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pyFAI
from h5py import Dataset, File
from matplotlib.colors import LogNorm
from pyFAI import units
from pyFAI.calibrant import get_calibrant
from pyFAI.detectors import Detector
from pyFAI.goniometer import MultiGeometry, SingleGeometry
from pyFAI.gui import jupyter
from pyFAI.integrator.azimuthal import AzimuthalIntegrator
from pyFAI.method_registry import IntegrationMethod

from xrpd_toolbox.utils.settings import XRPDBaseModel
from xrpd_toolbox.utils.utils import (
    get_entry,
    h5_to_array,
)

PIXEL_SIZE = 7.5e-5  # in m
INITIAL_DISTNACE = 700  # mm
DEFAULT_MAX_SHAPE = (1024, 512)


def calibrate_single_geometry_from_rings(
    geometry: SingleGeometry,
    rings: Collection[int] = [5, 5, 5, 7, 7, 9, 11, 15, 17],
    fix: list | None = None,
):
    if fix is None:
        fix = []

    for n_rings in rings:
        geometry.extract_cp(max_rings=n_rings)
        geometry.geometry_refinement.refine2(fix=fix)

    return geometry


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
        eiger_data_path: str = "data",
        tth_path: str = "tth",
    ):
        self.filepath = filepath
        self.eiger_data_path = eiger_data_path
        self.tth_path = tth_path

        self.entry = get_entry(self.filepath)
        self.dataset_path = f"/{self.entry}/{self.eiger_data_path}/data"

    @cached_property
    def positions(self) -> np.ndarray:
        try:
            deltas = h5_to_array(self.filepath, self.tth_path)
            return deltas
        except ValueError as e:
            print(f"{e} - {self.tth_path} in data - returning 0")
            deltas = np.array([0])
            return deltas

    @cached_property
    def count_time_path(self) -> str:
        return f"/{self.entry}/instrument/{self.eiger_data_path}/count_time"

    @cached_property
    def durations(self) -> np.ndarray:
        return h5_to_array(self.filepath, self.count_time_path)

    @property
    def data(self):
        return self.get_data(self.dataset_path)

    def get_frame(self, frame: int | Collection[int] | slice):
        return self.get_data(self.dataset_path)

    def get_data(self, dataset_path) -> np.ndarray:
        with File(self.filepath, "r") as file:
            if self.dataset_path not in file:
                raise ValueError(f"Dataset path {dataset_path} not found in HDF5 file.")

            data = file.get(dataset_path)

            if (data is not None) and isinstance(data, Dataset):
                if data.ndim < 1:
                    raise ValueError("Data has insufficient dimensions.")
                module_frame_data = data[...]

                return np.asarray(module_frame_data)
            else:
                raise ValueError(f"Data at {dataset_path} in {self.filepath}is None.")


class Eiger500K(Detector):
    IS_FLAT = False  # this detector is flat
    IS_CONTIGUOUS = True

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

        if self.filepath is not None:
            self.data_loader = EigerDataLoader(self.filepath)
            self.max_shape = self.data_loader.data.shape
        else:
            self.max_shape = DEFAULT_MAX_SHAPE  # Default shape if no data

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

            simulated_image = calibrant.fake_calibration_image(
                ai, shape=self.max_shape, resolution=resolution
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

    def calibrate_single_geometry(
        self,
        calibrant_name: str,
        wavelength: float,
        poni_output_filepath: str | Path,
        wavelength_unit: Literal["Ang", "A", "Angstrom", "kev", "keV", "ev"],
    ):
        """Using pyfai and the current data file,
        this will attempt to calibrate the detector
        using a known calibrant and then output a poni file to disk"""

        if wavelength_unit.lower() in ["ang", "ansgtrom", "a"]:
            wavelength_in_ang = wavelength
        elif wavelength_unit.lower() in ["ang", "ansgtrom", "a"]:
            wavelength_in_ang = wavelength
        else:
            raise ValueError("wavelength_unit must be valid!")

        calibrant = get_calibrant(calibrant_name)
        calibrant.wavelength = wavelength_in_ang / 1e10

        single_geometry = SingleGeometry(
            self.name, self.data_loader.data, calibrant=calibrant, detector=self
        )

        single_geometry = calibrate_single_geometry_from_rings(geometry=single_geometry)
        single_geometry.geometry_refinement.save(str(poni_output_filepath))


if __name__ == "__main__":
    #     SETTINGS = EigerSettings()
    #     FILEPATH = "/workspaces/XRPD-Toolbox/examples/i15-1/eiger_500k/1414223.nxs"
    #     eiger = Eiger500K(filepath=FILEPATH, settings=SETTINGS)
    calibrant = get_calibrant(calibrant_name="Si")
    calibrant.wavelength = 0.161699 / 1e10

    pixel1 = PIXEL_SIZE
    pixel2 = PIXEL_SIZE

    shape = (1024, 512)

    poni1 = pixel1 * shape[0] / 2
    poni2 = pixel2 * shape[1] / 2

    import pyFAI.detectors

    ai = AzimuthalIntegrator(
        detector=pyFAI.detectors.Detector(
            pixel1=pixel1, pixel2=pixel2, max_shape=shape
        ),
        wavelength=calibrant.wavelength,
        dist=0.7,
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
