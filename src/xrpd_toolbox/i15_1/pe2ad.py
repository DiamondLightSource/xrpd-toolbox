import matplotlib.pyplot as plt
import numpy as np

from xrpd_toolbox.data_loader import BaseDataLoader
from xrpd_toolbox.utils.peaks import fit_peaks, smooth_tophat


class PE2AD(BaseDataLoader):
    def __init__(self, filepath, data_path: str = "pe2AD"):
        super().__init__(filepath, data_path)

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

    def sum_frames(self):
        data = self._get_dataset(dataset_path=self.dataset_path)

        n_frames = data.shape[1]

        summed_images = []

        for frame in range(n_frames):
            frame_image = data[:, frame, :, :]
            image_sum = np.sum(frame_image)

            summed_images.append(image_sum)

        summed_images = np.array(summed_images)

        return summed_images

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

            tophat = smooth_tophat(indices, peak.amplitude, peak.centre, peak.fwhm, 0.1)

            plt.plot(indices, tophat)

        plt.plot(indices, trimmed_sample_density)
        plt.show()


if __name__ == "__main__":
    filepath = "/host-home/projects/data/i15-1-95016.nxs"

    pe2ad = PE2AD(filepath)
    pe2ad.find_sample_centre()
