"""
Angular calibration of a multi-module Mythen strip-detector goniometer,
following pyFAI's tutorial:
https://www.silx.org/doc/pyFAI/latest/usage/tutorial/Soleil/Cristal_Mythen.html

Workflow per module
--------------------
1. Bootstrap: over a small frame window where exactly one calibrant peak is
   visible, assign it to ring 0 and get an approximate goniometer geometry.
2. Scan: over all remaining frames, use the approximate geometry to predict
   how many rings should be visible and assign detected peaks to rings.
3. Refine: tighten/loosen bounds and re-run the least-squares refinement.
4. Complete: re-scan any still-unassigned frames (this recovers peaks on the
   "far side" of the beam center that weren't indexable before refinement).
5. Prune: optionally drop frames whose peaks were mis-assigned (large chi2).
6. Merge: integrate every module's data through its own goniometer geometry
   and sum them into a single 1D pattern.
"""

import json
from pathlib import Path
from typing import Literal

import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from pyFAI.calibrant import get_calibrant
from pyFAI.containers import Integrate1dResult
from pyFAI.control_points import ControlPoints
from pyFAI.detectors import Detector
from pyFAI.goniometer import ExtendedTransformation, Goniometer, GoniometerRefinement
from pyFAI.gui import jupyter
from scipy.interpolate import interp1d
from scipy.optimize import bisect
from scipy.signal import find_peaks_cwt
from scipy.spatial import distance_matrix

from xrpd_toolbox.i11.mythen import MythenDataLoader


class Mythen3(Detector):
    "Vertical Mythen strip detector from Dectris"

    aliases = ["Mythen3 1280"]
    force_pixel = True
    MAX_SHAPE = (1280, 1)

    def __init__(self, pixel1=50e-6, pixel2=8e-3):
        super().__init__(pixel1=pixel1, pixel2=pixel2)


def calc_fwhm(integrate_result, calibrant, tth_min=None, tth_max=None):
    "Compute the (2theta, FWHM) of every calibrant ring visible in a 1D pattern."
    delta = integrate_result.intensity[1:] - integrate_result.intensity[:-1]
    maxima = np.where(np.logical_and(delta[:-1] > 0, delta[1:] < 0))[0] + 1
    minima = np.where(np.logical_and(delta[:-1] < 0, delta[1:] > 0))[0] + 1

    if tth_min is None:
        tth_min = integrate_result.radial[0]
    if tth_max is None:
        tth_max = integrate_result.radial[-1]

    tth, fwhm = [], []
    for tth_rad in calibrant.get_2th():
        tth_deg = tth_rad * integrate_result.unit.scale
        if tth_deg <= tth_min or tth_deg >= tth_max:
            continue
        idx_theo = abs(integrate_result.radial - tth_deg).argmin()
        id0_max = abs(maxima - idx_theo).argmin()
        id0_min = abs(minima - idx_theo).argmin()
        i_max = integrate_result.intensity[maxima[id0_max]]
        i_min = integrate_result.intensity[minima[id0_min]]
        tth_maxi = integrate_result.radial[maxima[id0_max]]
        i_thres = (i_max + i_min) / 2.0

        if minima[id0_min] > maxima[id0_max]:
            min_lo = (
                integrate_result.radial[0]
                if id0_min == 0
                else integrate_result.radial[minima[id0_min - 1]]
            )
            min_hi = integrate_result.radial[minima[id0_min]]
        else:
            min_hi = (
                integrate_result.radial[-1]
                if id0_min == len(minima) - 1
                else integrate_result.radial[minima[id0_min + 1]]
            )
            min_lo = integrate_result.radial[minima[id0_min]]

        f = interp1d(integrate_result.radial, integrate_result.intensity - i_thres)
        try:
            tth_lo = bisect(f, min_lo, tth_maxi)
            tth_hi = bisect(f, tth_maxi, min_hi)
        except ValueError:
            continue
        fwhm.append(tth_hi - tth_lo)
        tth.append(tth_deg)
    return tth, fwhm


def calc_peak_error(integrate_result, calibrant, tth_min=10, tth_max=95):
    "Compute the (2theta, observed - expected) error of every calibrant ring."
    peaks = find_peaks_cwt(integrate_result.intensity, [10])
    df = np.gradient(integrate_result.intensity)
    d2f = np.gradient(df)
    bad = d2f == 0
    d2f[bad] = 1e-10
    cor = df / d2f
    cor[abs(cor) > 1] = 0
    cor[bad] = 0
    got = np.interp(
        peaks - cor[peaks],
        np.arange(len(integrate_result.radial)),
        integrate_result.radial,
    )
    mask = np.logical_and(got >= tth_min, got <= tth_max)
    got = got[mask]

    target = np.array(calibrant.get_2th()) * integrate_result.unit.scale
    mask = np.logical_and(target >= tth_min, target <= tth_max)
    target = target[mask]

    d2 = distance_matrix(target.reshape(-1, 1), got.reshape(-1, 1), p=1)
    return target, target - got[d2.argmin(axis=-1)]


class AngularCalibrationPyFAI:
    def __init__(
        self,
        filepath: str | Path,
        wavelength_in_m: float,
        calibrant_name: Literal["Si", "LaB6", "CeO2"],
        design_geometry_path: str | Path | None = None,
    ):
        self.filepath = filepath
        self.wavelength = wavelength_in_m
        self.data_loader = MythenDataLoader(filepath=filepath)
        self.data = self.data_loader.module_data
        self.calibrant_name = calibrant_name
        self.calibrant = get_calibrant(self.calibrant_name)
        self.calibrant.wavelength = wavelength_in_m

        self.modules = {}
        for name, module_dataset in enumerate(self.data):
            detector_module = Mythen3()
            mask = module_dataset[0] < 0
            # discard the first 20 and last 20 pixels
            # as their intensities are less reliable
            mask[:20] = True
            mask[-20:] = True
            detector_module.mask = mask.reshape(-1, 1)
            self.modules[name] = detector_module

        # Goniometer coordinate transform: the scanned motor (delta, i.e.
        # rot2) drives the module's angular position, via an (offset, scale)
        # calibration; dist/poni1/poni2/rot1 stay fixed per module and the
        # wavelength is fixed for the whole calibration.
        self.trans = ExtendedTransformation(
            dist_expr="dist",
            poni1_expr="poni1",
            poni2_expr="poni2",
            rot1_expr="rot1",
            rot2_expr="pi*(offset+scale*delta)/180.",
            rot3_expr="0.0",
            wavelength_expr="wavelength",
            param_names=["dist", "poni1", "poni2", "rot1", "offset", "scale"],
            pos_names=["delta"],
            constants={"wavelength": wavelength_in_m},
        )

        self.goniometers: dict[int, GoniometerRefinement] = {}
        self.results: dict[int, Integrate1dResult] = {}

        self.design_params: dict[int, dict] = {}
        self.beamline_offset = 0.0
        self.rotation_centre = (0.0, 0.0)
        if design_geometry_path is not None:
            self.load_design_geometry(design_geometry_path)

    def get_data(self, module_id, frame_id: int):
        return self.data[module_id][frame_id]

    def get_position(self, idx: int):
        "Returns delta (the scanned goniometer motor position) for the given frame_id"
        return self.data_loader.positions[idx]

    def peak_picking(self, module_name, frame_id, threshold=500):
        """Peak-picking based on find_peaks_cwt from scipy plus
        second-order Taylor-expansion refinement for sub-pixel resolution.

        The half-pixel offset is accounted here, i.e. pixel #0 has its center at 0.5

        Returns an (N, 2) array of [pixel_row, 0.5] control-point coordinates.
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
        d2f[bad] = 1e-10  # prevent division by zero. Discarded later on
        cor = df / d2f
        cor[abs(cor) > 1] = 0
        cor[bad] = 0
        ref = guess - cor[guess] + 0.5  # half a pixel offset
        x = np.zeros_like(ref) + 0.5  # half a pixel offset
        return np.vstack((ref, x)).T

    def plot_module_frame(self, module_id, frame_id):
        """Plot a single spectrum, to help you visually pick zero_pos /
        frame_start / frame_stop for `add_module` (see the tutorial's
        Figure 1: find the frame where the beam-stop is centred, and the
        frames where the first calibrant ring enters/leaves the strip)."""
        spectrum = self.data[module_id][frame_id]
        fig, ax = plt.subplots()
        ax.plot(spectrum)
        ax.axvline(640, color="red", linestyle="--")
        ax.set_title(f"Module {module_id}, frame {frame_id}")
        fig.show()
        return fig, ax

    def show_module_image(self, module_id):
        "Plot the full module dataset (all frames) as a 2D image, log-scaled."
        plt.figure()
        plt.title(f"Module {module_id}")
        plt.imshow(
            self.data[module_id], cmap=cm.inferno, norm=LogNorm(), origin="lower"
        )
        plt.xlabel("pixel")
        plt.ylabel("frame")
        plt.show()

    def load_design_geometry(self, design_geometry_path):
        """Load a mechanical-design geometry file and convert it into pyFAI
        initial-guess parameters (dist, poni1, poni2, rot1, offset, scale)
        for every module.

        Expected file format (units: radius & pixel_size in mm, angles in
        degrees):
            {
              "beamline_offset": ...,
              "rotation_centre_x": ..., "rotation_centre_y": ...,
              "module_0": {"radius": ..., "module_angle": ..., "tilt": ...,
                           "pixel_direction": 1, "centre": 639.5,
                           "pixel_size": 0.05},
              ...
            }

        `dist` comes from `radius`, `poni1` from `centre * pixel_size`,
        `rot1` from `tilt`, `offset` from `module_angle + beamline_offset`,
        and `scale` from `pixel_direction` (the module's mounting direction,
        +1 or -1). `poni2` isn't in this file, so a generic default is used.
        `rotation_centre_x/y` is stored but not currently folded into the
        per-module geometry -- treat it as informational until/unless the
        refinement shows it's needed.
        """
        design_geometry_path = Path(design_geometry_path)
        with open(design_geometry_path) as f:
            design = json.load(f)

        self.beamline_offset = design.get("beamline_offset", 0.0)
        self.rotation_centre = (
            design.get("rotation_centre_x", 0.0),
            design.get("rotation_centre_y", 0.0),
        )

        self.design_params = {}
        for key, entry in design.items():
            if not key.startswith("module_"):
                continue
            module_id = int(key.split("_")[1])
            pixel_size_m = entry["pixel_size"] * 1e-3  # mm -> m
            self.design_params[module_id] = {
                "dist": entry["radius"] * 1e-3,  # mm -> m
                "poni1": entry["centre"] * pixel_size_m,
                "poni2": 4e-3,  # not in the design file; generic default
                "rot1": np.radians(entry["tilt"]),
                "offset": entry["module_angle"] + self.beamline_offset,
                "scale": float(entry["pixel_direction"]),
            }

        print(
            f"Loaded design geometry for {len(self.design_params)} modules "
            f"from {design_geometry_path} "
            f"(beamline_offset={self.beamline_offset}, "
            f"rotation_centre={self.rotation_centre})"
        )
        return self.design_params

    def _assign_remaining_frames(self, module_id, frame_range):
        """Scan the given frames, predict the expected number of visible
        rings from the current geometry, and assign peaks to rings when the
        count matches. Returns the number of newly-assigned frames."""
        gonioref = self.goniometers[module_id]
        ds = self.data[module_id]
        tths = self.calibrant.get_2th()
        n_assigned = 0

        for i in frame_range:
            frame_name = f"{module_id}_{i:04d}"
            if frame_name in gonioref.single_geometries:
                continue

            peak = self.peak_picking(module_id, i)
            ai = gonioref.get_ai(self.get_position(i))
            tth = ai.array_from_unit(unit="2th_rad", scale=False)
            tth_low, tth_hi = tth[20], tth[-20]
            ttmin, ttmax = min(tth_low, tth_hi), max(tth_low, tth_hi)
            valid_rings = np.logical_and(ttmin <= tths, tths < ttmax)
            n_expected = valid_rings.sum()

            if len(peak) == n_expected and n_expected > 0:
                if tth_hi < tth_low:
                    peak = peak[::-1]
                cp = ControlPoints(calibrant=self.calibrant, wavelength=self.wavelength)
                for p, r in zip(peak, np.where(valid_rings)[0]):
                    cp.append([p], ring=r)
                img = ds[i].reshape((-1, 1))
                sg = gonioref.new_geometry(
                    frame_name,
                    image=img,
                    metadata=i,
                    control_points=cp,
                    calibrant=self.calibrant,
                )
                sg.geometry_refinement.data = np.array(cp.getList())
                n_assigned += 1

        return n_assigned

    def add_module(
        self,
        module_id,
        zero_pos,
        frame_start,
        frame_stop,
        dist_guess=0.72,
        poni1_guess=None,
        poni2_guess=4e-3,
        dist_bounds=(0.70, 0.80),
    ):
        """Bootstrap and refine the goniometer geometry for one module.

        :param zero_pos: frame index where the beam-stop is centred on this module
        :param frame_start: frame index where the first calibrant ring enters the strip
        :param frame_stop: frame index where a second ring appears (end of the
            single-ring bootstrap window)
        """
        ds = self.data[module_id]
        detector = self.modules[module_id]
        if poni1_guess is None:
            poni1_guess = 640 * detector.pixel1

        param = {
            "dist": dist_guess,
            "poni1": poni1_guess,
            "poni2": poni2_guess,
            "rot1": 0.0,
            "offset": -self.get_position(zero_pos),
            "scale": 1.0,
        }
        bounds = {
            "dist": dist_bounds,
            "poni2": (poni2_guess, poni2_guess),
            "rot1": (0.0, 0.0),
            "scale": (1.0, 1.0),
        }

        gonioref = GoniometerRefinement(
            param,
            self.get_position,
            self.trans,
            detector=detector,
            wavelength=self.wavelength,
            bounds=bounds,
        )
        self.goniometers[module_id] = gonioref

        # Bootstrap on the window where only ring 0 is visible.
        for i in range(frame_start, frame_stop):
            peak = self.peak_picking(module_id, i)
            if len(peak) != 1:
                continue
            cp = ControlPoints(calibrant=self.calibrant, wavelength=self.wavelength)
            cp.append([peak[0]], ring=0)
            img = ds[i].reshape((-1, 1))
            sg = gonioref.new_geometry(
                f"{module_id}_{i:04d}",
                image=img,
                metadata=i,
                control_points=cp,
                calibrant=self.calibrant,
            )
            sg.geometry_refinement.data = np.array(cp.getList())

        if not gonioref.single_geometries:
            raise RuntimeError(
                f"No single-peak frames found for module {module_id} in "
                f"[{frame_start}, {frame_stop}). Check zero_pos/frame_start/frame_stop."
            )

        print(f"Module {module_id}: bootstrap chi2 = {gonioref.chi2()}")
        gonioref.refine2()

        # Now scan the rest of the dataset with the (rough) refined geometry.
        n_new = self._assign_remaining_frames(module_id, range(frame_stop, ds.shape[0]))
        print(f"Module {module_id}: assigned {n_new} additional frames")

        gonioref.refine2()
        gonioref.set_bounds("poni1", -1, 1)
        gonioref.set_bounds("poni2", -1, 1)
        gonioref.set_bounds("rot1", -1, 1)
        gonioref.set_bounds("scale", 0.9, 1.1)
        gonioref.refine2()

        return gonioref

    def add_module_from_design(
        self,
        module_id,
        offset_tolerance_deg=2.0,
        dist_tolerance_m=0.01,
        rot1_tolerance_rad=0.05,
        poni1_tolerance_m=0.01,
        scale_tolerance=0.1,
    ):
        """Bootstrap a module's geometry directly from `load_design_geometry`,
        skipping the frame-window bootstrap entirely: since dist/poni1/rot1/
        offset/scale are already known to good precision, control points can
        be assigned across the *whole* scan in one pass before refining.

        The `*_tolerance*` arguments set how far the refinement is allowed
        to move each parameter away from its design value.
        """
        if module_id not in self.design_params:
            raise KeyError(
                f"No design geometry loaded for module {module_id}. "
                f"Call load_design_geometry() first, or pass its path to __init__."
            )

        design = self.design_params[module_id]
        detector = self.modules[module_id]
        ds = self.data[module_id]

        param = dict(design)  # dist, poni1, poni2, rot1, offset, scale
        bounds = {
            "dist": (
                param["dist"] - dist_tolerance_m,
                param["dist"] + dist_tolerance_m,
            ),
            "poni1": (
                param["poni1"] - poni1_tolerance_m,
                param["poni1"] + poni1_tolerance_m,
            ),
            "poni2": (
                param["poni2"] - poni1_tolerance_m,
                param["poni2"] + poni1_tolerance_m,
            ),
            "rot1": (
                param["rot1"] - rot1_tolerance_rad,
                param["rot1"] + rot1_tolerance_rad,
            ),
            "offset": (
                param["offset"] - offset_tolerance_deg,
                param["offset"] + offset_tolerance_deg,
            ),
            "scale": (
                min(param["scale"] - scale_tolerance, param["scale"] + scale_tolerance),
                max(param["scale"] - scale_tolerance, param["scale"] + scale_tolerance),
            ),
        }

        gonioref = GoniometerRefinement(
            param,
            self.get_position,
            self.trans,
            detector=detector,
            wavelength=self.wavelength,
            bounds=bounds,
        )
        self.goniometers[module_id] = gonioref

        n_assigned = self._assign_remaining_frames(module_id, range(ds.shape[0]))
        print(
            f"Module {module_id}: assigned {n_assigned}/{ds.shape[0]} frames from design geometry"
        )

        if not gonioref.single_geometries:
            raise RuntimeError(
                f"Module {module_id}: design geometry didn't match any frames. "
                f"Check the sign/units of offset ({param['offset']:.3f} deg) and "
                f"scale ({param['scale']}), or widen offset_tolerance_deg."
            )

        gonioref.refine2()
        return gonioref

    def complete_gonio(self, module_id):
        """Re-scan every frame of a module for peaks that weren't indexable
        before refinement (typically on the far/negative side of the beam)."""
        gonioref = self.goniometers[module_id]
        ds = self.data[module_id]
        before = sum(
            len(sg.geometry_refinement.data)
            for sg in gonioref.single_geometries.values()
        )
        self._assign_remaining_frames(module_id, range(ds.shape[0]))
        after = sum(
            len(sg.geometry_refinement.data)
            for sg in gonioref.single_geometries.values()
        )
        print(f"Module {module_id}: peaks {before} -> {after}")
        return gonioref

    def search_outliers(self, module_id, threshold=1.2):
        "Return the labels of frames whose peaks look mis-assigned (large chi2)."
        gonioref = self.goniometers[module_id]
        labels, errors = [], []
        for lbl, sg in gonioref.single_geometries.items():
            labels.append(lbl)
            errors.append(sg.geometry_refinement.chi2())

        order = np.argsort(errors)
        last = errors[order[-1]]
        to_remove = []
        for i in order[::-1]:
            current = errors[i]
            if threshold * current < last:
                break
            last = current
            to_remove.append(labels[i])
        return to_remove

    def _peak_counts(self, module_id, threshold=500):
        "Peak-pick every frame of a module once; return counts and the peaks themselves."
        ds = self.data[module_id]
        counts = np.zeros(ds.shape[0], dtype=int)
        peaks_per_frame = []
        for i in range(ds.shape[0]):
            peak = self.peak_picking(module_id, i, threshold=threshold)
            peaks_per_frame.append(peak)
            counts[i] = len(peak)
        return counts, peaks_per_frame

    def auto_detect_module_params(
        self, module_id, thresholds=(500, 300, 150, 80), min_run=5
    ):
        """Automatically find (zero_pos, frame_start, frame_stop) for a module,
        replacing the manual/visual step from the tutorial.

        Logic: as delta steps, ring 0 sweeps across the strip; the longest
        run of consecutive frames with exactly one detected peak is used as
        the single-ring bootstrap window (frame_start, frame_stop). Within
        that window, the frame whose peak sits closest to the strip's centre
        pixel is used as the zero-reference frame (zero_pos), giving the
        initial `offset` guess. Several peak-picking thresholds are tried in
        case the calibrant signal is weaker/stronger than the default.
        """
        last_exc = None
        for threshold in thresholds:
            counts, peaks_per_frame = self._peak_counts(module_id, threshold=threshold)
            is_single = counts == 1

            runs = []
            start = None
            for i, val in enumerate(is_single):
                if val and start is None:
                    start = i
                elif not val and start is not None:
                    runs.append((start, i))
                    start = None
            if start is not None:
                runs.append((start, len(is_single)))

            long_runs = [r for r in runs if r[1] - r[0] >= min_run]
            if not long_runs:
                last_exc = RuntimeError(
                    f"Module {module_id}: no run of >= {min_run} consecutive "
                    f"single-peak frames found at threshold={threshold}."
                )
                continue

            frame_start, frame_stop = max(long_runs, key=lambda r: r[1] - r[0])

            centre_pixel = self.modules[module_id].shape[0] / 2.0
            best_i, best_d = frame_start, None
            for i in range(frame_start, frame_stop):
                peak = peaks_per_frame[i]
                if len(peak) != 1:
                    continue
                d = abs(peak[0][0] - centre_pixel)
                if best_d is None or d < best_d:
                    best_d = d
                    best_i = i

            print(
                f"Module {module_id}: auto-detected zero_pos={best_i}, "
                f"frame_start={frame_start}, frame_stop={frame_stop} "
                f"(threshold={threshold})"
            )
            return best_i, frame_start, frame_stop

        raise last_exc

    def calibrate_module(
        self,
        module_id,
        zero_pos,
        frame_start,
        frame_stop,
        prune=True,
        outlier_threshold=1.2,
    ):
        "Full per-module pipeline: bootstrap, refine, complete, and optionally prune outliers."
        self.add_module(module_id, zero_pos, frame_start, frame_stop)
        self.complete_gonio(module_id)
        self.goniometers[module_id].refine2()

        if prune:
            removed = self.search_outliers(module_id, outlier_threshold)
            for lbl in removed:
                self.goniometers[module_id].single_geometries.pop(lbl)
            if removed:
                print(
                    f"Module {module_id}: dropped {len(removed)} outlier frames: {removed}"
                )
                self.complete_gonio(module_id)
                self.goniometers[module_id].refine2()

        return self.goniometers[module_id]

    def calibrate(self, module_params: dict, prune=True, outlier_threshold=1.2):
        """Run the full calibration for several modules.

        :param module_params: {module_id: (zero_pos, frame_start, frame_stop)}

        Modules that fail (e.g. detection window too noisy to refine) are
        skipped with a warning rather than aborting the whole run.
        """
        for module_id, (zero_pos, frame_start, frame_stop) in module_params.items():
            print(f"\n=== Calibrating module {module_id} ===")
            try:
                self.calibrate_module(
                    module_id,
                    zero_pos,
                    frame_start,
                    frame_stop,
                    prune=prune,
                    outlier_threshold=outlier_threshold,
                )
            except Exception as exc:  # noqa: BLE001 - keep the pipeline going
                print(f"Module {module_id}: calibration FAILED ({exc}) -- skipping")
                self.goniometers.pop(module_id, None)
        return self.goniometers

    def auto_calibrate(
        self,
        module_ids=None,
        thresholds=(500, 300, 150, 80),
        min_run=5,
        prune=True,
        outlier_threshold=1.2,
    ):
        """End-to-end calibration with no manual input: auto-detect each
        module's (zero_pos, frame_start, frame_stop), then run `calibrate`.
        This is the "just run it" entry point.
        """
        if module_ids is None:
            module_ids = sorted(self.modules.keys())

        module_params = {}
        for module_id in module_ids:
            try:
                module_params[module_id] = self.auto_detect_module_params(
                    module_id,
                    thresholds=thresholds,
                    min_run=min_run,
                )
            except RuntimeError as exc:
                print(f"Module {module_id}: auto-detection FAILED ({exc}) -- skipping")

        if not module_params:
            raise RuntimeError(
                "Auto-detection failed for every module. Inspect the data with "
                "show_module_image()/plot_module_frame() and call calibrate() "
                "with manually chosen (zero_pos, frame_start, frame_stop) instead."
            )

        return self.calibrate(
            module_params, prune=prune, outlier_threshold=outlier_threshold
        )

    def calibrate_module_from_design(
        self, module_id, prune=True, outlier_threshold=1.2, **design_tolerances
    ):
        "Full per-module pipeline seeded from the design geometry, instead of an auto-detected window."
        self.add_module_from_design(module_id, **design_tolerances)
        self.complete_gonio(module_id)
        self.goniometers[module_id].refine2()

        if prune:
            removed = self.search_outliers(module_id, outlier_threshold)
            for lbl in removed:
                self.goniometers[module_id].single_geometries.pop(lbl)
            if removed:
                print(
                    f"Module {module_id}: dropped {len(removed)} outlier frames: {removed}"
                )
                self.complete_gonio(module_id)
                self.goniometers[module_id].refine2()

        return self.goniometers[module_id]

    def calibrate_from_design(
        self, module_ids=None, prune=True, outlier_threshold=1.2, **design_tolerances
    ):
        """End-to-end calibration seeded from the mechanical design geometry
        (`load_design_geometry`). This is the preferred "just run it" entry
        point when a design file is available -- no frame-window detection
        needed, and it's more robust than `auto_calibrate` since the initial
        geometry is already close to correct.

        Modules that fail are skipped with a warning rather than aborting
        the whole run.
        """
        if module_ids is None:
            module_ids = sorted(self.design_params.keys())

        for module_id in module_ids:
            print(f"\n=== Calibrating module {module_id} (design prior) ===")
            try:
                self.calibrate_module_from_design(
                    module_id,
                    prune=prune,
                    outlier_threshold=outlier_threshold,
                    **design_tolerances,
                )
            except Exception as exc:  # noqa: BLE001 - keep the pipeline going
                print(f"Module {module_id}: calibration FAILED ({exc}) -- skipping")
                self.goniometers.pop(module_id, None)

        return self.goniometers
        return self.goniometers

    # ------------------------------------------------------------------
    # Reporting & persistence
    # ------------------------------------------------------------------

    @staticmethod
    def _param_dict(gonioref):
        """Map a GoniometerRefinement's flat `.param` list back to named
        values. The parameter names live on the namedtuple factory
        `gonioref.nt_param` (built from the transformation's `param_names`),
        not on the goniometer object itself.
        """
        return dict(zip(gonioref.nt_param._fields, gonioref.param))

    def print_calibration(self):
        "Print the refined geometry parameters and fit quality for every module."
        for module_id, gonioref in sorted(self.goniometers.items()):
            params = self._param_dict(gonioref)
            n_frames = len(gonioref.single_geometries)
            n_peaks = sum(
                len(sg.geometry_refinement.data)
                for sg in gonioref.single_geometries.values()
            )
            print(
                f"Module {module_id}: chi2={gonioref.chi2():.6g}  "
                f"frames={n_frames}  peaks={n_peaks}"
            )
            for name, value in params.items():
                print(f"    {name:8s} = {value:.8g}")

    def save_calibration(self, output_dir):
        """Save each module's refined geometry to its own .json calibration
        file (pyFAI's native `Goniometer.save` format), plus a summary.json
        with the numeric parameters for quick inspection.

        Reload later with `load_calibration(output_dir)`.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        summary = {}
        for module_id, gonioref in sorted(self.goniometers.items()):
            path = output_dir / f"module_{module_id}.json"
            gonioref.save(str(path))

            params = self._param_dict(gonioref)
            entry = {
                "file": str(path),
                "params": params,
                "chi2": gonioref.chi2(),
                "n_frames": len(gonioref.single_geometries),
                "wavelength": self.wavelength,
                "calibrant": self.calibrant_name,
            }
            summary[str(module_id)] = entry
            print(f"Module {module_id}: chi2={entry['chi2']:.6g}  -> {path}")

        summary_path = output_dir / "calibration_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary written to {summary_path}")

        return summary

    def load_calibration(self, input_dir):
        """Load per-module calibration files previously written by
        `save_calibration`. The loaded objects are plain `Goniometer`
        instances (not `GoniometerRefinement`) -- enough to build an
        AzimuthalIntegrator/MultiGeometry from a goniometer position, but
        not to re-refine. Use `add_module`/`calibrate` again for that.
        """
        input_dir = Path(input_dir)
        self.goniometers = {}
        for path in sorted(input_dir.glob("module_*.json")):
            module_id = int(path.stem.split("_")[1])
            self.goniometers[module_id] = Goniometer.sload(str(path))
            print(f"Loaded module {module_id} from {path}")
        return self.goniometers

    # ------------------------------------------------------------------
    # Applying the calibration: raw frames -> diffraction pattern
    # ------------------------------------------------------------------

    def get_ai(self, module_id, delta):
        "Build an AzimuthalIntegrator for one module at a given delta (motor) position."
        return self.goniometers[module_id].get_ai(delta)

    def integrate_frame(
        self, module_id, image, delta, npt=10000, unit="2th_deg", radial_range=None
    ):
        """Convert a single new detector frame into a 1D diffraction pattern,
        using the stored calibration for `module_id` at the given `delta`
        (goniometer motor position). `image` is the raw 1280-pixel spectrum
        from that module."""
        ai = self.get_ai(module_id, delta)
        img = np.asarray(image).reshape((-1, 1))
        return ai.integrate1d(img, npt, unit=unit, radial_range=radial_range)

    def integrate_module(
        self, module_id, npt=50000, radial_range=(0, 95), data=None, positions=None
    ):
        """Integrate a whole scan (multiple frames at multiple goniometer
        positions) for one module through its calibrated geometry.

        By default integrates the calibration dataset itself; pass `data`
        (an (n_frames, n_pixels) array) and `positions` (n_frames angles) to
        apply the calibration to a new scan instead.
        """
        gonioref = self.goniometers[module_id]
        ds = self.data[module_id] if data is None else data
        pos = self.data_loader.positions if positions is None else positions
        mg = gonioref.get_mg(pos)
        mg.radial_range = radial_range
        images = [frame.reshape(-1, 1) for frame in ds]
        res = mg.integrate1d(images, npt)
        if data is None:
            self.results[module_id] = res
        return res

    def integrate_all(
        self, npt=50000, radial_range=(0, 95), plot=True, data=None, positions=None
    ):
        """Integrate every calibrated module and sum them into one merged
        pattern. By default uses the calibration dataset; pass `data` (a
        {module_id: array} dict) and `positions` to apply the calibration to
        a new scan instead.
        """
        summed = counted = radial = None
        fig, ax = plt.subplots() if plot else (None, None)

        for module_id in self.goniometers:
            module_data = None if data is None else data[module_id]
            res = self.integrate_module(
                module_id,
                npt=npt,
                radial_range=radial_range,
                data=module_data,
                positions=positions,
            )
            if summed is None:
                summed, counted = res.sum, res.count
            else:
                summed = summed + res.sum
                counted = counted + res.count
            radial = res.radial
            if plot:
                jupyter.plot1d(
                    res, label=f"module {module_id}", calibrant=self.calibrant, ax=ax
                )

        merged = Integrate1dResult(radial, summed / np.maximum(counted, 1e-10))
        merged._set_unit(res.unit)
        merged._set_count(counted)
        merged._set_sum(summed)

        if plot:
            ax.plot(radial, summed / np.maximum(counted, 1e-10), label="Merged")
            ax.legend()
            fig.show()

        return merged

    def plot_calibration_quality(self, integrate_result, tth_min=10, tth_max=95):
        "Plot FWHM and peak-position error vs. angle for a merged/single pattern."
        fig, ax = plt.subplots()
        ax.plot(
            *calc_fwhm(integrate_result, self.calibrant, tth_min, tth_max),
            "o",
            label="FWHM",
        )
        ax.plot(
            *calc_peak_error(integrate_result, self.calibrant, tth_min, tth_max),
            "o",
            label="error",
        )
        ax.set_title("Peak shape & error as function of the angle")
        ax.set_xlabel(integrate_result.unit.label)
        ax.legend()
        fig.show()
        return fig, ax


if __name__ == "__main__":
    filepath = "/host-home/projects/outputs/angular_calibration/1410290.nxs"
    design_geometry_path = "/workspaces/outputs/mythen_calibration/processed/ang_cal_82026_cen_639.5_leastsq_[17, 27].json"

    wavelength = 0.828783e-10
    calibrant_name = "Si"
    output_dir = "/host-home/projects/outputs/angular_calibration/calib"

    cal = AngularCalibrationPyFAI(
        filepath,
        wavelength,
        calibrant_name,
        design_geometry_path=output_dir,
    )

    # Seeds every module's geometry from the mechanical design file (no
    # frame-window detection needed), calibrates every module, prints and
    # saves the result, then produces the merged diffraction pattern.
    # cal.calibrate_from_design()
    # cal.print_calibration()
    # cal.save_calibration(output_dir)

    merged = cal.integrate_all()
    cal.plot_calibration_quality(merged)

    # --- Later / in another script: reuse the saved calibration ---
    # cal2 = AngularCalibrationPyFAI(filepath, wavelength, calibrant_name)
    # cal2.load_calibration(output_dir)
    #
    # # Single new frame from module 0, taken at delta = 12.3 deg:
    # pattern = cal2.integrate_frame(0, new_spectrum, delta=12.3, unit="2th_deg")
    #
    # # Or a full new multi-module scan (raw data + matching delta array):
    # new_data = {0: module0_frames, 1: module1_frames, ...}  # each (n_frames, 1280)
    # new_positions = new_scan_deltas                          # (n_frames,)
    # merged_pattern = cal2.integrate_all(data=new_data, positions=new_positions)

    # If a module fails to calibrate from the design geometry -- e.g. the
    # sign convention of offset/scale doesn't match this dataset -- inspect
    # it directly and widen the tolerance, or fall back to auto-detection:
    #   cal.show_module_image(3)
    #   cal.calibrate_module_from_design(3, offset_tolerance_deg=10.0)
    #   # or: cal.calibrate({3: (zero_pos, frame_start, frame_stop)})

    # No design file? Fall back to the auto-detected frame-window approach:
    #   cal = AngularCalibrationPyFAI(filepath, wavelength, calibrant_name)
    #   cal.auto_calibrate()
