import matplotlib.pyplot as plt
import numpy as np

from xrpd_toolbox.data_loader import BaseDataLoader
from xrpd_toolbox.fit_engine.peaks import fit_peaks, smooth_tophat


class PE2AD(BaseDataLoader):
    def __init__(self, filepath, dataset_path: str = "/entry/pe2AD/data"):
        super().__init__(filepath, dataset_path)

        self.entries = self.get_entries()

    def find_centre1(self, x: np.ndarray):
        positions = np.arange(len(x))
        center = np.sum(positions * x) / np.sum(x)

        return center

    def find_center2(self, x):
        threshold = np.percentile(x, 99)
        sample = x[x < threshold]
        densest_idx = np.argmax(sample)

        return int(densest_idx)

    def find_sample_centre(self):
        summed_images = self.sum_frames()
        centre_index = self.find_center2(summed_images)

        # x = np.arange(len(summed_images))

        indices = np.arange(
            int(centre_index) - 30, int(centre_index) + 31, 1, dtype=int
        )

        print(indices)

        trimmed_sample_density = summed_images[indices]

        s = 3

        trimmed_sample_density = np.convolve(trimmed_sample_density, np.ones(s) / s)

        trimmed_sample_density = trimmed_sample_density[s - 1 : :]

        peaks = fit_peaks(
            x=indices, y=trimmed_sample_density, initial_x_pos=[centre_index]
        )

        # p0 = [x_guess, amp_guess, width_guess]
        # popt, _ = curve_fit(gaussian, x_fit, y_fit, p0=p0, maxfev=10000)

        for peak in peaks:
            print(peak)
            plt.plot(indices, peak.calculate(indices))

            tophat = smooth_tophat(
                indices,
                float(peak.amplitude),
                float(peak.amplitude),
                float(peak.amplitude),
                0.1,
            )

            plt.plot(indices, tophat)

        plt.plot(indices, trimmed_sample_density)
        plt.show()


if __name__ == "__main__":
    import os

    folder = "/dls/i15-1/data/2026/cm44163-1/"
    file = "i15-1-95016.nxs"

    prefix = "i15-1-"

    sample_alignment_scans = {
        "carbon_black": 94519,
        "water": 94520,
        "GaIn": 94521,
        "NIST_Si": 95016,
        "NaCl": 95017,
        "HKUST1": 95018,
    }

    for sample, number in sample_alignment_scans.items():
        filepath = folder + prefix + str(number) + ".nxs"

        output_file = f"/workspaces/outputs/i15-1/{sample}-{number}.csv"

        print(filepath)

        if not os.path.exists(filepath):
            print("it doesn't exist")

        pe2ad = PE2AD(filepath)
        summed_images = pe2ad.sum_frames()

        index = np.linspace(0, len(summed_images), len(summed_images))

        csv_data = np.stack((index, summed_images), axis=-1)

        np.savetxt(output_file, csv_data)

        x, y = np.genfromtxt(output_file, unpack=True)

        plt.plot(x, y)
        plt.show()
