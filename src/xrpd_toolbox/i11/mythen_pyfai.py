from pathlib import Path
from typing import Literal

# import ipywidgets as widgets
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

# from pyFAI import goniometer, units
# from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
from pyFAI.calibrant import get_calibrant

# from pyFAI.containers import Integrate1dResult
# from pyFAI.control_points import ControlPoints
from pyFAI.detectors import Detector
from pyFAI.geometry import Geometry

# from pyFAI.geometryRefinement import GeometryRefinement
from pyFAI.goniometer import (
    ExtendedTransformation,
    SingleGeometry,
)

# from pyFAI.gui import jupyter
# from pyFAI.units import hc
# from scipy.interpolate import interp1d
# from scipy.optimize import bisect, minimize
from scipy.signal import find_peaks_cwt

# from scipy.spatial import distance_matrix
# from silx.resources import ExternalResources
from xrpd_toolbox.i11.mythen import MythenDataLoader


class Mythen3(Detector):
    "Verical Mythen dtrip detector from Dectris"

    aliases = ["Mythen3 1280"]
    force_pixel = True
    MAX_SHAPE = (1280, 1)

    def __init__(self, pixel1=50e-6, pixel2=8e-3):
        super().__init__(pixel1=pixel1, pixel2=pixel2)


class AngularCalibrationPyFAI:
    def __init__(
        self,
        filepath: str | Path,
        wavelength_in_m: float,
        calibrant_name: Literal["Si", "LaB6", "CeO2"],
    ):
        self.filepath = filepath
        self.data_loader = MythenDataLoader(filepath=filepath)
        self.data = self.data_loader.module_data
        self.calibrant_name = calibrant_name
        self.calibrant = get_calibrant(self.calibrant_name)
        self.calibrant.wavelength = wavelength_in_m

        self.modules = {}
        for name, module_dataset in enumerate(self.data):
            detector_module = Mythen3()
            mask = module_dataset[0] < 0
            print(name, module_dataset.shape)
            # discard the first 20 and last 20 pixels
            # as their intensities are less reliable
            mask[:20] = True
            mask[-20:] = True
            detector_module.mask = mask.reshape(-1, 1)
            self.modules[name] = detector_module

        ExtendedTransformation(
            dist_expr="dist",
            poni1_expr="poni1",
            poni2_expr="poni2",
            rot1_expr="rot1",
            rot2_expr="pi*(offset+scale*angle)/180.",
            rot3_expr="0.0",
            wavelength_expr="wavelength",
            param_names=["dist", "poni1", "poni2", "rot1", "offset", "scale", "nrj"],
            pos_names=["angle"],
            constants={"wavelength": wavelength_in_m},
        )

        peaks = self.peak_picking(0, 0)
        print(peaks)

        step_idx = 1
        # Approximate offset for the module #0 at 0°
        print(
            f"Approximated offset for the first module at step {step_idx}: ",
            self.get_position(step_idx),
        )

    def single_geometry_calibration(self, wavelength_in_m: float):
        for name, module_dataset in enumerate(self.data):
            detector_module = self.modules[name]
            initial = Geometry(detector=detector_module, wavelength=wavelength_in_m)

            plt.title(f"Module {name}")
            plt.imshow(module_dataset, cmap=cm.inferno, norm=LogNorm(), origin="lower")  # type: ignore
            plt.show()

            # peak_indices = get_peaks_from_images(data_img, n_peaks=n_peaks)

            # print(peak_indices)
            beam_centre_x = -10  # x-coordinate of the beam-center in pixels
            beam_centre_y = 1  # y-coordinate of the beam-center in pixels
            distance = 762  # This is the distance in mm (unit used by Fit2d)
            rot3 = -0.78539816339

            initial.setFit2D(distance, beam_centre_x, beam_centre_y)

            single_geometry = SingleGeometry(
                "pe2AD",
                module_dataset,
                calibrant=self.calibrant,
                detector=detector_module,
                geometry=initial,
            )
            single_geometry.geometry_refinement.rot3 = (
                rot3  # https://confluence.diamond.ac.uk/display/I151/Making+a+poni+file
            )

            for rings in [5, 5, 5, 5, 7, 7, 9, 11, 15, 15, 17, 21, 23, 25, 31, 41, 51]:
                single_geometry.extract_cp(max_rings=rings)
                # Refine the geometry ... here in SAXS geometry,
                # the rotation is fixed in orthogonal setup
                single_geometry.geometry_refinement.refine2(fix=["wavelength"])

    def get_data(self, module_id, frame_id: int):
        return self.data[module_id][frame_id]

    def get_position(self, idx: int):
        "Returns the postion of the goniometer for the given frame_id"
        return self.data_loader.positions[idx]

    # Define a peak-picking function based on the dataset-name and the frame_id:
    def peak_picking(self, module_name, frame_id, threshold=500):
        """Peak-picking base on find_peaks_cwt from scipy plus
        second-order tailor exapention refinement for sub-pixel resolution.

        The half-pixel offset is accounted here, i.e pixel #0 has its center at 0.5

        """
        module = self.modules[module_name]
        msk = module.mask.ravel()

        spectrum = self.data[module_name][frame_id]
        guess = find_peaks_cwt(spectrum, [20])

        valid = np.logical_and(np.logical_not(msk[guess]), spectrum[guess] > threshold)
        guess = guess[valid]

        # Based on maximum is f'(x) = 0 ~ f'(x0) + (x-x0)*(f''(x0))
        df = np.gradient(spectrum)
        d2f = np.gradient(df)
        bad = d2f == 0
        d2f[bad] = 1e-10  # prevent devision by zero. Discared later on
        cor = df / d2f
        cor[abs(cor) > 1] = 0
        cor[bad] = 0
        ref = guess - cor[guess] + 0.5  # half a pixel offset
        x = np.zeros_like(ref) + 0.5  # half a pixel offset
        return np.vstack((ref, x)).T

    # def update(self, module_id, frame_id):
    #     spectrum = self.data[module_id][frame_id]
    #     self.line.set_data(np.arange(spectrum.size), spectrum)
    #     self.ax.set_title(f"Module {module_id}, Frame {frame_id}")
    #     self.fig.canvas.draw()


if __name__ == "__main__":
    filepath = "/host-home/projects/outputs/angular_calibration/1410290.nxs"
    wavelength = 0.828783 * 1e-10  # 0.828773
    calibrant_name = "Si"

    AngularCalibrationPyFAI(filepath, wavelength, calibrant_name)
