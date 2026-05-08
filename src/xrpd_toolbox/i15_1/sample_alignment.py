"""Sample alignment utilities for the XRPD Toolbox.

This module provides a SampleAligner model for I15-1 XRPD data,
including peak detection, peak profile construction, and plotting.
"""

import math
import os

import numpy as np
from scipy.signal import find_peaks

from xrpd_toolbox.core import Parameter, XYEData
from xrpd_toolbox.fit_engine.background import (
    BackgroundType,
    ConstantBackground,
)
from xrpd_toolbox.fit_engine.fitting_core import Model, RefinementBaseModel
from xrpd_toolbox.fit_engine.peaks import (
    Peak,
    PeakType,
    calculate_profile,
    peak_factory,
)
from xrpd_toolbox.plotting import PlotData
from xrpd_toolbox.utils.messenger import Messenger
from xrpd_toolbox.utils.utils import cluster_points_auto


class SampleCenteringResult(RefinementBaseModel):
    """This is what contains the results and can
    be serisalised and sent back to the bluesky plan"""

    centre: float
    peaks: list[Peak]
    scores: list[float]


class SampleAligner(Model[XYEData]):
    """Model for aligning XRPD sample patterns using initial peak detection.

    The SampleAligner detects peaks from observed XYEData, groups them,
    constructs initial peak models, and evaluates the resulting profile
    together with an estimated background.
    """

    background: BackgroundType
    sample_and_capillary: list[PeakType] = []
    centre: float | None = None

    def get_peak_info(self, labels, peak_position, peak_intensity):
        """Summarize grouped peak detections into peak initialization data.

        Parameters
        ----------
        labels:
            Cluster labels for each detected peak.
        peak_position:
            X positions of the detected peaks.
        peak_intensity:
            Intensities of the detected peaks.

        Returns
        -------
        list[dict]
            A list of peak summaries containing group id, centre,
            amplitude, and FWHM estimates.
        """
        grouped_peaks = []

        group_numbers = np.unique(labels)

        for group in group_numbers:
            index = np.where(labels == group)[0]

            group_positions = peak_position[index]

            average_peak_position = np.median(group_positions)
            average_peak_intensity = np.median(peak_intensity[index])

            average_peak_spread = np.ptp(group_positions)

            if average_peak_spread == 0:
                average_peak_spread = np.ptp(self.data.x) / 20

            peak = {
                "group": int(group),
                "centre": float(average_peak_position),
                "amplitude": float(average_peak_intensity),
                "fwhm": float(average_peak_spread),
            }

            grouped_peaks.append(peak)

        return grouped_peaks

    def calculate_profile(self):
        """Calculate the combined peak profile plus background for the model.

        Returns
        -------
        numpy.ndarray
            The combined intensity profile evaluated at the current x positions.
        """
        return calculate_profile(
            self.data.x, self.sample_and_capillary
        ) + self.background.calculate(self.data.x)

    def get_initial_peaks(self, peak_type: str = "tophat", smoothing: int = 5):
        """Detect observed peaks and build initial peak models for fitting.

        Parameters
        ----------
        peak_type:
            The type of peak model to create for each detected peak.
        smoothing:
            Smoothing parameter for peak detection. Currently accepted for
            API compatibility but not applied in the current implementation.
        """
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
                peak_cls = peak_factory(peak_type)

                sample_peak = peak_cls.model_validate(peak)
                sample_peak.parameterise_all(refine=True)
                sample_peak.normalised = False
                assert isinstance(sample_peak.centre, Parameter)
                sample_peak.centre.bounds = [
                    self.data.x.min(),
                    self.data.x.max(),
                ]
                # assert isinstance(sample_peak.fwhm, Parameter)
                # sample_peak.fwhm.bounds = [
                #     0,
                #     float(np.ptp(self.data.x)),
                # ]
                self.sample_and_capillary.append(sample_peak)

            # plt.plot(self.data.x, self.calculate_profile())
            # plt.plot(self.data.x, self.data.y)
            # plt.scatter(peak_position, peak_intensity)
            # plt.show()

        # print(self.sample_and_capillary)
        # plt.scatter(peak_position, peak_intensity)
        # plt.plot(self.data.x, self.data.y)
        # plt.show()

    def plot_data(self):
        """Plot the observed data, calculated profile, background, and residual.

        The plot shows the observed intensities, the current fitted profile,
        the background model, and the residual (observed minus calculated).
        """
        if self.data.source is not None:
            title = os.path.basename(self.data.source)
        else:
            title = None

        profile = calculate_profile(self.data.x, self.sample_and_capillary)
        profile = profile + self.background.calculate(self.data.x)

        plot_data = PlotData(
            data=self.data,
            calc=profile,
            diff=self.data.y - profile,
            background=self.background.calculate(self.data.x),
            title=title,
            markers=np.array([self.centre]),
        )
        plot_data.plot()

        return plot_data

    def peaks_to_arrays(
        self, peaks: list[PeakType]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert a list of peaks into numpy arrays of centre, amplitude, and FWHM.

        Parameters
        ----------
        peaks:
            A list of peak model objects.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
            Arrays containing peak centres, amplitudes, and FWHMs.
        """
        centres = [
            p.centre.value if isinstance(p.centre, Parameter) else p.centre
            for p in peaks
        ]
        amplitudes = [
            p.amplitude.value if isinstance(p.amplitude, Parameter) else p.amplitude
            for p in peaks
        ]
        fwhms = [
            p.fwhm.value if isinstance(p.fwhm, Parameter) else p.fwhm for p in peaks
        ]
        return np.array(centres), np.array(amplitudes), np.array(fwhms)

    def middle_elements(self, lst: list):
        """Return the central element(s) from a list.

        If the list length is odd, returns the middle three elements. If even,
        returns the two elements at the center of the list.
        """
        n = len(lst)
        mid = n // 2

        if n % 2 == 0:
            return lst[mid - 1 : mid + 1]  # 2 elements
        else:
            return lst[mid - 1 : mid + 2]  # 3 elements

    def get_middle_peaks(self):
        """Return the central peaks from the current sample and capillary list.

        This is useful for selecting the most representative peaks from the
        middle of the detector pattern.
        """
        middle_peaks = self.middle_elements(self.sample_and_capillary)

        return middle_peaks

    def get_sample_centre(self) -> SampleCenteringResult:
        middle_peaks = self.get_middle_peaks()
        centres, amplitudes, fwhms = self.peaks_to_arrays(middle_peaks)

        scores = fwhms * amplitudes
        max_index = np.argmax(scores)

        sample_centre = centres[max_index]

        self.centre = sample_centre

        return SampleCenteringResult(
            centre=sample_centre, peaks=middle_peaks, scores=scores.tolist()
        )

    # def get_best_peaks(self):
    #     middle_peaks = self.get_middle_peaks()
    #     centres, amplitudes, fwhms = self.peaks_to_arrays(self.sample_and_capillary)

    #     scores = fwhms * amplitudes
    #     sort_index = np.argsort(scores)

    #     centre_distance = np.abs(centres - np.mean(centres))
    #     centre_distance[centre_distance == 0] = 1e-12

    #     centre_scores = normalise(np.log10(centre_distance))

    #     for cen, amp, fw, cs, score in zip(
    #         centres, amplitudes, fwhms, centre_scores, scores, strict=True
    #     ):
    #         print(cen, amp, fw, cs, score)


def sample_alignment_builder(
    data: XYEData | str, peak_type: str = "gaussian"
) -> SampleAligner:
    """Construct a SampleAligner from XYE data or a CSV file path, and a peak type
    A sample aligner is what is used to align an abtract data set (usually a sample)

    Parameters
    ----------
    data:
        XYEData instance or a path to a CSV file containing x, y, and e data.
    peak_type:
        The peak model type to use when generating initial peaks.

    Returns
    -------
    SampleAligner
        A model with estimated background and initial peak definitions.
    """
    if isinstance(data, str):
        data = XYEData.from_csv(data)

    background = ConstantBackground.estimate(data.x, data.y)
    sample_alignment_model = SampleAligner(data=data, background=background)
    sample_alignment_model.get_initial_peaks(peak_type=peak_type)

    return sample_alignment_model


def run_sample_alignment(data: XYEData | str) -> SampleAligner:
    gauss_model = sample_alignment_builder(data, "gaussian")
    _, gauss_model, gauss_result = gauss_model.refine()

    tophat_model = sample_alignment_builder(data, "tophat")
    _, tophat_model, tophat_result = tophat_model.refine()

    if gauss_result.cost < tophat_result.cost:
        print("gaussian wins")
        best_model = gauss_model
    else:
        print("top hat wins")
        best_model = tophat_model

    # sample_centre = best_model.get_best_peaks()
    best_model.get_sample_centre()

    # SampleCenteringResult(centre=sample_centre, peaks=)

    return best_model


if __name__ == "__main__":
    BEAMLINE = "i15-1"

    folder = "/workspaces/XRPD-Toolbox/src/xrpd_toolbox/i15_1/sample_alignment_data"

    sample_alignment_files = [os.path.join(folder, f) for f in os.listdir(folder)]

    listener = Messenger("i15-1", destinations=["/topic/public.data.plot"])

    for filepath in sample_alignment_files:
        best_model = run_sample_alignment(data=filepath)

        sample_centre_result = best_model.get_sample_centre()
        # _ = best_model.get_best_peaks()

        print(sample_centre_result.model_dump_json())

        sample_centre_result.deparameterise_all()

        print(sample_centre_result.model_dump_json())

        plot_data = best_model.plot_data()
        # plot_data.publish(BEAMLINE)

        # messenger.send_plot_data(plot_data)
        # listener.listen(max_iter=5)
        # listener.stop()

        print("-----")
