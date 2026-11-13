"""Tests for the bluesky mythen loader, the step scan saving and the
run_mythen_* entry points. Everything uses small synthetic h5 files or a fake
loader - none of the real nexus test data is used."""

from pathlib import Path

import h5py
import numpy as np
import pytest

from xrpd_toolbox.data_loader import BaseDataLoader
from xrpd_toolbox.i11 import mythen_analysis
from xrpd_toolbox.i11.mythen import PIXELS_PER_MODULE, MythenDetector, MythenSettings
from xrpd_toolbox.i11.mythen_analysis import (
    MythenBlueskyDataLoader,
    bad_channels_to_physical,
    run_mythen_analysis,
    run_mythen_one_diffraction_per_trigger,
    run_mythen_sum_all_positions,
)
from xrpd_toolbox.utils.utils import h5_to_array

N_FRAMES = 4
MODULES = [0, 1]


def write_bluesky_h5(
    filepath: Path,
    deltas,
    n_frames: int = N_FRAMES,
    plan_name: str | bytes | None = "scan_delta_with_mythen",
    scan_type: str | None = "step_scan",
) -> Path:
    n_channels = len(MODULES) * PIXELS_PER_MODULE
    data = np.ones((n_frames, 1, n_channels), dtype=np.uint32)

    with h5py.File(filepath, "w") as file:
        file.create_dataset("entry/fastcs_mythen3/fastcs_mythen3", data=data)
        file.create_dataset("entry/instrument/diff_stage/delta", data=deltas)
        if plan_name is not None:
            file.create_dataset("entry/plan_metadata/plan_name", data=plan_name)
        if scan_type is not None:
            file.create_dataset("entry/plan_metadata/scan_type", data=scan_type)

    return filepath


# BaseDataLoader


def test_base_loader_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="doesn't exist"):
        BaseDataLoader(tmp_path / "missing.nxs", dataset_path="/entry/data")


@pytest.mark.parametrize("plan_name", ["fast_trigger_mythen", b"fast_trigger_mythen"])
def test_base_loader_plan_name(tmp_path: Path, plan_name):
    filepath = write_bluesky_h5(tmp_path / "scan.nxs", [1.0], plan_name=plan_name)

    with BaseDataLoader(filepath, dataset_path="/entry/data") as loader:
        assert loader.get_plan_name() == "fast_trigger_mythen"
        assert loader.plan_name == "fast_trigger_mythen"


def test_base_loader_plan_name_is_cached(tmp_path: Path):
    filepath = write_bluesky_h5(tmp_path / "scan.nxs", [1.0])

    with BaseDataLoader(filepath, dataset_path="/entry/data") as loader:
        assert "plan_name" not in loader.__dict__
        _ = loader.plan_name
        assert loader.__dict__["plan_name"] == "scan_delta_with_mythen"


def test_base_loader_scan_type(tmp_path: Path):
    filepath = write_bluesky_h5(tmp_path / "scan.nxs", [1.0], scan_type="trigger")

    with BaseDataLoader(filepath, dataset_path="/entry/data") as loader:
        assert loader.get_scan_type() == "trigger"


def test_base_loader_missing_plan_metadata_raises(tmp_path: Path):
    filepath = write_bluesky_h5(
        tmp_path / "scan.nxs", [1.0], plan_name=None, scan_type=None
    )

    with BaseDataLoader(filepath, dataset_path="/entry/data") as loader:
        with pytest.raises(ValueError):
            loader.get_plan_name()
        with pytest.raises(ValueError):
            loader.get_scan_type()


# MythenBlueskyDataLoader positions


def test_bluesky_loader_get_positions_returns_raw_deltas(tmp_path: Path):
    filepath = write_bluesky_h5(tmp_path / "scan.nxs", [10.0, 12.0])

    with MythenBlueskyDataLoader(filepath, modules=MODULES) as loader:
        np.testing.assert_array_equal(loader.get_positions(), [10.0, 12.0])


def test_bluesky_loader_step_scan_keeps_each_position(tmp_path: Path):
    # one delta per frame - a step scan, so the positions are used as they are
    deltas = [10.0, 11.0, 12.0, 13.0]
    filepath = write_bluesky_h5(tmp_path / "scan.nxs", deltas)

    with MythenBlueskyDataLoader(filepath, modules=MODULES) as loader:
        np.testing.assert_array_equal(loader.positions, deltas)


def test_bluesky_loader_trigger_scan_uses_mean_position(tmp_path: Path):
    # fewer deltas than frames - a trigger scan at (roughly) one position
    filepath = write_bluesky_h5(tmp_path / "scan.nxs", [10.0, 12.0])

    with MythenBlueskyDataLoader(filepath, modules=MODULES) as loader:
        np.testing.assert_allclose(loader.positions, np.full(N_FRAMES, 11.0))


def test_bluesky_loader_positions_follow_frame_selection(tmp_path: Path):
    # selecting frames means the deltas no longer match, so the mean is used
    filepath = write_bluesky_h5(tmp_path / "scan.nxs", [10.0, 11.0, 12.0, 13.0])

    with MythenBlueskyDataLoader(
        filepath, modules=MODULES, frames=slice(0, 2)
    ) as loader:
        np.testing.assert_allclose(loader.positions, [11.5, 11.5])


# MythenDetector.process_step_scan / save_step_scan_nexus


class FakeLoader:
    """Stands in for the mythen data loader, no nexus needed.
    Counts in module n are all n + 1"""

    def __init__(self, filepath, modules):
        self.filepath = filepath
        self.modules = list(modules)
        self.entry = "entry"
        self.module_data = [
            np.full((N_FRAMES, PIXELS_PER_MODULE), m + 1, dtype=float)
            for m in self.modules
        ]
        self.data = np.concatenate(self.module_data, axis=-1)
        self.get_module_data = self.module_data.__getitem__
        self.positions = np.linspace(0, 1, N_FRAMES)
        self.durations = np.ones(N_FRAMES)


@pytest.fixture
def detector(tmp_path: Path) -> MythenDetector:
    bad_channels_file = tmp_path / "badchannels.txt"
    bad_channels_file.write_text("")

    # the processed nexus starts as a copy of the raw file, so it must be h5
    filepath = tmp_path / "fake.nxs"
    with h5py.File(filepath, "w") as file:
        file.create_group("entry")

    settings = MythenSettings(
        active_modules=MODULES,
        bad_modules=[],
        bad_channels_filepath=bad_channels_file,
        edge_bad_channels=0,
    )

    return MythenDetector(
        filepath=filepath,
        settings=settings,
        filename_suffix="_test",
        data_loader=FakeLoader,
    )


def test_process_step_scan_returns_xye_filepath(detector: MythenDetector):
    xye_filepath = detector.process_step_scan()

    assert xye_filepath == detector.xye_filepath_out
    assert Path(xye_filepath).exists()


def test_process_step_scan_does_not_contact_control(
    detector: MythenDetector, monkeypatch
):
    def fail(*args, **kwargs):
        raise AssertionError("control should not be contacted")

    monkeypatch.setattr(detector, "communicate_with_control", fail)

    detector.process_step_scan()


def test_save_step_scan_nexus_writes_xye_and_rings(detector: MythenDetector):
    detector.process_step_scan()

    processed = detector.processed_nexus_filepath

    np.testing.assert_allclose(
        h5_to_array(processed, "entry/xye/tth"), detector.binned_tth
    )
    np.testing.assert_allclose(
        h5_to_array(processed, "entry/xye/counts"), detector.binned_counts
    )
    np.testing.assert_allclose(
        h5_to_array(processed, "entry/xye/error"), detector.binned_error
    )

    ring1, ring2 = detector.get_ring1_ring2_data()
    np.testing.assert_array_equal(h5_to_array(processed, "entry/ring1/data"), ring1)
    np.testing.assert_array_equal(h5_to_array(processed, "entry/ring2/data"), ring2)


def test_save_step_scan_nexus_leaves_raw_file_alone(detector: MythenDetector):
    detector.process_step_scan()

    with h5py.File(detector.filepath, "r") as file:
        assert "entry/xye" not in file


def test_ring_data_with_bluesky_loader(tmp_path: Path, detector: MythenDetector):
    # regression: the bluesky loader's get_data has no modules argument
    filepath = write_bluesky_h5(tmp_path / "scan.nxs", [1.0])
    detector.mythen_data = MythenBlueskyDataLoader(filepath, modules=MODULES)

    ring1, ring2 = detector.get_ring1_ring2_data()

    first_ring_pixels = len(detector.active_ring1_modules) * PIXELS_PER_MODULE
    assert ring1.shape == (N_FRAMES, first_ring_pixels)
    assert ring2.shape == (
        N_FRAMES,
        len(MODULES) * PIXELS_PER_MODULE - first_ring_pixels,
    )


def test_save_data_to_nexus_uses_given_filepath(
    detector: MythenDetector, tmp_path: Path
):
    other = tmp_path / "other.nxs"

    detector.save_data_to_nexus(
        other, subpath="/extra", names_and_data={"x": np.array([1, 2])}
    )

    np.testing.assert_array_equal(h5_to_array(other, "entry/extra/x"), [1, 2])
    assert not Path(detector.processed_nexus_filepath).exists()


# run_mythen_* entry points


class RecordingDetector:
    """Replaces MythenDetector so the run functions can be tested alone"""

    instances: list["RecordingDetector"] = []

    def __init__(self, filepath, settings, data_loader):
        self.filepath = filepath
        self.settings = settings
        self.data_loader = data_loader
        RecordingDetector.instances.append(self)

    def process_time_resolved(self):
        return [f"frame_{n}.xye" for n in range(25)]

    def process_step_scan(self):
        return "summed.xye"


@pytest.fixture
def fake_run(monkeypatch):
    """Patches out the detector, the file wait and the publishing"""

    RecordingDetector.instances = []
    published = []

    monkeypatch.setattr(mythen_analysis, "MythenDetector", RecordingDetector)
    monkeypatch.setattr(mythen_analysis, "wait_for_finished_file", lambda f: None)
    monkeypatch.setattr(
        mythen_analysis,
        "_publish_xye",
        lambda path, data_type, filenumber=None: published.append(
            (path, data_type, filenumber)
        ),
    )

    return published


def test_run_one_diffraction_per_trigger(fake_run):
    settings = MythenSettings(
        active_modules=[m for m in range(28) if m not in (4, 11, 17)],
        data_reduction_mode="time_resolved",
    )
    xye_filepaths = run_mythen_one_diffraction_per_trigger(
        "/data/i11-123456.nxs", settings=settings, frames_to_plot=10
    )

    (detector,) = RecordingDetector.instances
    assert detector.settings.active_modules == [
        m for m in range(28) if m not in (4, 11, 17)
    ]
    assert detector.settings.data_reduction_mode == "time_resolved"
    assert detector.data_loader is MythenBlueskyDataLoader

    assert len(xye_filepaths) == 25

    published_paths = [path for path, _, _ in fake_run]
    # first, last, and every 25 // 10 = 2nd frame
    assert published_paths[0] == "frame_0.xye"
    assert published_paths[-1] == "frame_24.xye"
    assert published_paths == [f"frame_{n}.xye" for n in range(0, 25, 2)]
    assert all(filenumber == 123456 for _, _, filenumber in fake_run)


def test_run_sum_all_positions(fake_run):
    settings = MythenSettings(
        active_modules=list(range(2, 28)), data_reduction_mode="step_scan"
    )
    run_mythen_sum_all_positions("/data/i11-123456.nxs", settings=settings)

    (detector,) = RecordingDetector.instances
    assert detector.settings is settings
    assert detector.settings.active_modules == list(range(2, 28))
    assert detector.data_loader is MythenBlueskyDataLoader

    assert fake_run == [("summed.xye", "pxrd", 123456)]


@pytest.fixture
def dispatch(monkeypatch):
    """Records which processing run_mythen_analysis chooses, while still
    running the real processing so the files it makes can be checked"""

    calls = []
    real_trigger = mythen_analysis.run_mythen_one_diffraction_per_trigger
    real_step = mythen_analysis.run_mythen_sum_all_positions

    def trigger(filepath, settings):
        calls.append(("time_resolved", filepath, settings))
        return real_trigger(filepath, settings)

    def step(filepath, settings):
        calls.append(("step_scan", filepath, settings))
        return real_step(filepath, settings)

    monkeypatch.setattr(
        mythen_analysis, "run_mythen_one_diffraction_per_trigger", trigger
    )
    monkeypatch.setattr(mythen_analysis, "run_mythen_sum_all_positions", step)
    monkeypatch.setattr(mythen_analysis, "wait_for_finished_file", lambda f: None)
    monkeypatch.setattr(mythen_analysis, "_publish_xye", lambda *args, **kwargs: None)

    return calls


# the synthetic data only has MODULES in it
ABSENT_MODULES = [m for m in range(28) if m not in MODULES]


def processed_files(folder: Path, pattern: str) -> list[Path]:
    return sorted((folder / "processed").rglob(pattern))


def test_run_mythen_analysis_time_resolved(tmp_path: Path, dispatch):
    filepath = write_bluesky_h5(
        tmp_path / "i11-1.nxs",
        [1.0],
        plan_name="fast_trigger_mythen",
        scan_type="time_resolved",
    )

    run_mythen_analysis(str(filepath), absent_modules=ABSENT_MODULES)

    ((kind, called_filepath, settings),) = dispatch
    assert (kind, called_filepath) == ("time_resolved", str(filepath))
    assert settings.active_modules == MODULES
    assert settings.data_reduction_mode == "time_resolved"

    # one xye per frame, in processed/<filename>/, and no summed xye or nexus
    xye_files = processed_files(tmp_path, "*.xye")
    assert xye_files == [
        tmp_path / "processed" / "i11-1" / f"i11-1_summed_mythen3_{n}.xye"
        for n in range(N_FRAMES)
    ]
    assert processed_files(tmp_path, "*.nxs") == []


def test_run_mythen_analysis_step_scan(tmp_path: Path, dispatch):
    filepath = write_bluesky_h5(
        tmp_path / "i11-12345.nxs",
        [10.0, 11.0, 12.0, 13.0],
        plan_name="scan_delta_with_mythen",
        scan_type="step_scan",
    )

    run_mythen_analysis(str(filepath), absent_modules=ABSENT_MODULES)

    ((kind, called_filepath, settings),) = dispatch
    assert (kind, called_filepath) == ("step_scan", str(filepath))
    assert settings.active_modules == MODULES
    assert settings.data_reduction_mode == "step_scan"

    # every position summed into a single xye and a single processed nexus
    assert processed_files(tmp_path, "*.xye") == [
        tmp_path / "processed" / "i11-12345_summed_mythen3.xye"
    ]
    assert processed_files(tmp_path, "*.nxs") == [
        tmp_path / "processed" / "i11-12345_reduced_mythen3.nxs"
    ]


def test_run_mythen_analysis_unknown_scan_type_runs_step_scan_then_raises(
    tmp_path: Path, dispatch
):
    filepath = write_bluesky_h5(
        tmp_path / "i11-12345.nxs",
        [10.0, 11.0, 12.0, 13.0],
        plan_name="mystery_plan",
        scan_type="mystery",
    )

    with pytest.raises(ValueError, match="mystery"):
        run_mythen_analysis(str(filepath), absent_modules=ABSENT_MODULES)

    ((kind, called_filepath, settings),) = dispatch
    assert (kind, called_filepath) == ("step_scan", str(filepath))
    assert settings.active_modules == MODULES
    assert settings.data_reduction_mode == "step_scan"

    assert processed_files(tmp_path, "*.xye") == [
        tmp_path / "processed" / "i11-12345_summed_mythen3.xye"
    ]
    assert processed_files(tmp_path, "*.nxs") == [
        tmp_path / "processed" / "i11-12345_reduced_mythen3.nxs"
    ]


def test_bad_channels_to_physical_skips_absent_modules():
    active = [m for m in range(28) if m not in (4, 11, 17)]
    ppm = PIXELS_PER_MODULE
    data_position = np.array([5, 4 * ppm + 7, 10 * ppm + 1, 24 * ppm + 3, 25 * ppm])

    physical = bad_channels_to_physical(data_position, active)

    # slot 4 -> module 5, slot 10 -> module 12, slot 24 -> module 27,
    # 25 * ppm is past the active modules so is dropped
    np.testing.assert_array_equal(
        physical, [5, 5 * ppm + 7, 12 * ppm + 1, 27 * ppm + 3]
    )
