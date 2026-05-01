import math
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

from xrpd_toolbox.core import Model, Parameter
from xrpd_toolbox.fit_engine.background import ConstantBackground
from xrpd_toolbox.fit_engine.peaks import (
    PeakType,
    TopHatPeak,
    calculate_profile,
)
from xrpd_toolbox.fit_engine.profile_calculation import XYEData
from xrpd_toolbox.fit_engine.refiner import refine_model
from xrpd_toolbox.utils.utils import cluster_points_auto


class SampleAligner(Model):
    data: XYEData
    background: ConstantBackground
    sample_and_capillary: list[PeakType] = []

    def get_peak_info(self, labels, peak_position, peak_intensity):
        grouped_peaks = []

        group_numbers = np.unique(labels)

        for group in group_numbers:
            index = np.where(labels == group)[0]

            group_positions = peak_position[index]

            average_peak_position = np.median(group_positions)
            average_peak_intensity = np.median(peak_intensity[index])

            average_peak_spread = np.ptp(group_positions)

            if average_peak_spread == 0:
                average_peak_spread = np.ptp(self.data.x) / 10

            peak = {
                "group": int(group),
                "centre": float(average_peak_position),
                "amplitude": float(average_peak_intensity),
                "fwhm": float(average_peak_spread),
            }

            grouped_peaks.append(peak)

        return grouped_peaks

    # def use_simple_peak_model(self):
    #     capillary_edge = GaussianPeak(
    #         amplitude=np.ptp(self.data.y),
    #         centre=self.data.x.max() / 3,
    #         fwhm=np.ptp(self.data.x) / 20,
    #         normalised=False,
    #     )
    #     capillary_edge.parameterise_all(refine=True)
    #     self.sample_and_capillary.append(capillary_edge)

    #     sample1 = TopHatPeak(
    #         amplitude=np.ptp(self.data.y),
    #         centre=self.data.x.max() / 2,
    #         fwhm=np.ptp(self.data.x) / 5,
    #         normalised=False,
    #     )
    #     sample1.parameterise_all(refine=True)
    #     sample1.amplitude.bounds = [0, np.amax(self.data.y)]
    #     sample1.fwhm.bounds = [0, np.inf]
    #     sample1.centre.bounds = [0, np.amax(self.data.x)]

    #     sample1.epsilon.refine = False

    #     sample2 = sample1.__deepcopy__()
    #     sample3 = sample2.__deepcopy__()

    #     sample1.centre.value = sample1.centre.value - 10
    #     sample2.centre.value = sample2.centre.value + 20

    #     # print(self.background)
    #     # print(sample1)
    #     # print(sample2)
    #     # quit()

    #     self.sample_and_capillary.extend([sample1, sample2, sample3])

    def calculate_profile(self):
        return calculate_profile(
            self.data.x, self.sample_and_capillary
        ) + self.background.calculate(self.data.x)

    def get_initial_peaks(self, smoothing: int = 5):
        peak_indexes = find_peaks(self.data.y)[0]

        peak_position = self.data.x[peak_indexes]
        peak_intensity = self.data.y[peak_indexes]

        labels, n_groups = cluster_points_auto(peak_position, peak_intensity)

        peaks = self.get_peak_info(labels, peak_position, peak_intensity)

        for peak in peaks:
            if math.isclose(peak["amplitude"], self.data.y.min(), rel_tol=0.1) or (
                peak["amplitude"] < 0
            ):
                continue
            else:
                sample_peak = TopHatPeak.model_validate(peak)
                sample_peak.parameterise_all(refine=True)
                sample_peak.normalised = False
                assert isinstance(sample_peak.centre, Parameter)
                sample_peak.centre.bounds = [np.amin(self.data.x), np.amax(self.data.x)]

                self.sample_and_capillary.append(sample_peak)

        # print(self.sample_and_capillary)
        # plt.scatter(peak_position, peak_intensity)
        # plt.plot(self.data.x, self.data.y)
        # plt.show()

    def plot_data(self):
        if self.data.source is not None:
            plt.title(os.path.basename(self.data.source))

        # profile = calculate_profile(self.data.x, self.sample_and_capillary)

        # profile = profile + self.background.calculate(self.data.x)

        # plt.plot(self.data.x, profile)
        plt.plot(self.data.x, self.data.y)
        plt.plot(self.data.x, self.background.calculate(data.x))
        plt.show()


if __name__ == "__main__":
    folder = "/workspaces/XRPD-Toolbox/src/xrpd_toolbox/i15_1/sample_alignment_data"

    sample_alignment_files = [os.path.join(folder, f) for f in os.listdir(folder)]

    for filepath in sample_alignment_files:
        data = XYEData.from_csv(filepath)
        background = ConstantBackground.estimate(data.x, data.y)
        background.refine_none()

        print(background.value)

        sample_alignment = SampleAligner(data=data, background=background)

        # sample_alignment.plot_data()
        sample_alignment.get_initial_peaks()

        updated, new_model, result = refine_model(
            sample_alignment, plot=True, step_time=0.1, max_nfev=10
        )

        # sample_alignment.plot_data()
