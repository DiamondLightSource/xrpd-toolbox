import json
import shutil
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pytest
from pydantic import ValidationError

from xrpd_toolbox import BASE_PATH
from xrpd_toolbox.core import XYEData
from xrpd_toolbox.i11 import mythen
from xrpd_toolbox.i11.mythen import (
    DEFAULT_ANG_CAL,
    DEFAULT_BAD_CHANS,
    MODULES_IN_DETECTOR,
    PIXELS_PER_MODULE,
    AngularCalibration,
    BadChannels,
    ModuleConversion,
    MythenDataLoader,
    MythenDetector,
    MythenModule,
    MythenSettings,
    convert_angcal_to_pydantic_json,
)
from xrpd_toolbox.utils.utils import h5_to_array, normalise

matplotlib.use("Agg")

CONFIG_DIR = BASE_PATH.parent.parent / "config" / "i11"
CONFIG_FILE = CONFIG_DIR / "mythen3_reduction_config.toml"
LEGACY_CONFIG = CONFIG_DIR / "legacy_config.toml"

TEST_DATA_DIR = BASE_PATH.parent.parent / "tests" / "data" / "i11"
SI_DATA_FILE = TEST_DATA_DIR / "1410696.nxs"
SI_MYTHEN_FILE = TEST_DATA_DIR / "mythen3_1410696.hdf5"
PREPROCESSED_SI_XYE_15KEV = TEST_DATA_DIR / "1410696_summed_mythen3.xye"

N_FRAMES = 3


@pytest.fixture
def si_nexus(tmp_path: Path) -> Path:
    # reduction writes into the nexus file, so work on a copy - the nexus
    # links to its mythen data file by name, so that is copied alongside it
    nexus_filepath = tmp_path / SI_DATA_FILE.name
    shutil.copy(SI_DATA_FILE, nexus_filepath)
    shutil.copy(SI_MYTHEN_FILE, tmp_path)
    return nexus_filepath


@pytest.fixture
def mythen_settings() -> MythenSettings:
    return MythenSettings(
        active_modules=[1, 2, 3],
        bad_modules=[4, 5],
        bad_channel_masking=True,
        flatfield_filepath="flatfield.h5",
        apply_flatfield=False,
        modules_in_flatfield=[1, 2],
        send_to_ispyb=False,
        rebin_step=0.004,
        default_counter=0,
        edge_bad_channels=10,
        error_calc="poisson",
        data_reduction_mode="step_scan",
        bad_channels_filepath="bad_channels.txt",
        angcal_filepath="angcal.txt",
    )


@pytest.fixture
def conversion() -> ModuleConversion:
    return ModuleConversion(conv=6.5e-05, offset=10.0, centre=640)


@pytest.fixture
def bad_channels_file(tmp_path: Path) -> Path:
    # channel 5 of module 0, channel 5 of module 1, last channel of module 27
    filepath = tmp_path / "badchannels.txt"
    filepath.write_text("5\n1285\n35839\n")
    return filepath


@pytest.fixture
def no_show(monkeypatch):
    monkeypatch.setattr(mythen.plt, "show", lambda: None)


def make_fake_loader(n_frames: int = N_FRAMES, durations: np.ndarray | None = None):
    """Builds a data loader for MythenDetector that doesn't need a nexus file.
    Counts in module n are all n + 1, so modules can be told apart"""

    class FakeMythenDataLoader:
        def __init__(self, filepath, modules):
            self.filepath = filepath
            self.modules = list(modules)
            self.entry = "entry"

            self.module_data = [
                np.full((n_frames, PIXELS_PER_MODULE), m + 1, dtype=float)
                for m in self.modules
            ]
            self.data = np.concatenate(self.module_data, axis=-1)
            self.positions = np.linspace(0, 1, n_frames)
            self.durations = (
                np.ones(n_frames) if durations is None else np.asarray(durations)
            )

    return FakeMythenDataLoader


def make_detector(
    tmp_path: Path,
    bad_channels_file: Path,
    active_modules: list[int],
    bad_modules: list[int],
    data_loader: type | None = None,
    **settings_kwargs,
) -> MythenDetector:
    filepath = tmp_path / "fake.nxs"
    filepath.touch()

    settings = MythenSettings(
        active_modules=active_modules,
        bad_modules=bad_modules,
        bad_channels_filepath=bad_channels_file,
        edge_bad_channels=0,
        **settings_kwargs,
    )

    return MythenDetector(
        filepath=filepath,
        settings=settings,
        filename_suffix="_test",
        data_loader=data_loader or make_fake_loader(),
    )


# ModuleConversion


@pytest.mark.parametrize("conv, sign", [(6.5e-05, 1), (-6.5e-05, -1)])
def test_module_conversion_sign(conv: float, sign: int):
    assert ModuleConversion(conv=conv, offset=0, centre=640).module_sign == sign


def test_module_conversion_distance_matches_psd_radius(conversion: ModuleConversion):
    assert conversion.distance == pytest.approx(mythen.MYTHEN_PIXEL_SIZE / 6.5e-05)
    assert conversion.distance == pytest.approx(mythen.PSD_RADIUS, rel=0.01)


def test_module_conversion_raw_tth_at_centre_is_offset(conversion: ModuleConversion):
    raw_tth = conversion.return_raw_tth(zero_offset=-0.5)

    assert raw_tth.shape == (PIXELS_PER_MODULE,)
    assert raw_tth[640] == pytest.approx(conversion.offset - 0.5)


@pytest.mark.parametrize("conv", [6.5e-05, -6.5e-05])
def test_module_conversion_raw_tth_direction_follows_sign(conv: float):
    conversion = ModuleConversion(conv=conv, offset=10, centre=640)
    raw_tth = conversion.return_raw_tth(zero_offset=0)

    assert np.all(np.sign(np.diff(raw_tth)) == conversion.module_sign)


def test_default_angular_calibration_loads():
    calibration = AngularCalibration.load(DEFAULT_ANG_CAL)

    for module in range(MODULES_IN_DETECTOR):
        assert isinstance(calibration[f"module_{module}"], ModuleConversion)


# MythenSettings


def test_mythen_settings(mythen_settings: MythenSettings):
    assert mythen_settings.data_reduction_mode == "step_scan"
    assert mythen_settings.rebin_step == 0.004


def test_mythen_settings_load_from_toml():
    settings = MythenSettings.load_from_toml(CONFIG_FILE)

    assert isinstance(settings, MythenSettings)


def test_mythen_settings_load_legacy_toml():
    settings = MythenSettings.load(LEGACY_CONFIG)

    assert isinstance(settings, MythenSettings)


@pytest.mark.parametrize("extension", ["toml", "yaml"])
def test_mythen_settings_save_load(
    tmp_path: Path, mythen_settings: MythenSettings, extension: str
):
    filepath = tmp_path / f"settings.{extension}"

    mythen_settings.save(filepath)
    loaded_mythen_settings = MythenSettings.load(filepath)

    assert mythen_settings == loaded_mythen_settings


def test_mythen_settings_save_fails_with_incorrect_file_extension(
    tmp_path: Path, mythen_settings: MythenSettings
):
    filepath = tmp_path / "settings.txt"

    with pytest.raises(ValueError):
        mythen_settings.save_to_yaml(filepath)

    with pytest.raises(ValueError):
        mythen_settings.save_to_toml(filepath)


def test_mythen_settings_defaults():
    settings = MythenSettings()

    assert settings.active_modules == list(range(MODULES_IN_DETECTOR))
    assert settings.data_reduction_mode == "step_scan"
    assert settings.error_calc == "poisson"


def test_mythen_settings_missing_paths_fall_back_to_defaults():
    settings = MythenSettings(
        bad_channels_filepath="does_not_exist.txt",
        angcal_filepath="does_not_exist.json",
    )

    assert settings.bad_channels_filepath == str(DEFAULT_BAD_CHANS)
    assert settings.angcal_filepath == str(DEFAULT_ANG_CAL)


def test_mythen_settings_existing_paths_are_kept(tmp_path: Path):
    bad_chans = tmp_path / "bad.txt"
    bad_chans.touch()
    angcal = tmp_path / "angcal.json"
    angcal.touch()

    settings = MythenSettings(bad_channels_filepath=bad_chans, angcal_filepath=angcal)

    assert settings.bad_channels_filepath == str(bad_chans)
    assert settings.angcal_filepath == str(angcal)


def test_mythen_settings_off_angcal_is_swapped_for_json(tmp_path: Path):
    angcal = tmp_path / "angcal.json"
    angcal.touch()

    settings = MythenSettings(angcal_filepath=tmp_path / "angcal.off")

    assert settings.angcal_filepath == str(angcal)


@pytest.mark.parametrize(
    "mode, expected", [(0, "step_scan"), (1, "time_resolved"), (2, "pump_probe")]
)
def test_mythen_settings_int_data_reduction_mode(mode: int, expected: str):
    assert MythenSettings(data_reduction_mode=mode).data_reduction_mode == expected  # type: ignore


@pytest.mark.parametrize(
    "kwargs",
    [
        {"default_counter": 4},
        {"default_counter": -1},
        {"error_calc": "not_a_method"},
        {"data_reduction_mode": "not_a_mode"},
    ],
)
def test_mythen_settings_invalid_values_raise(kwargs: dict):
    with pytest.raises(ValidationError):
        MythenSettings(**kwargs)


# BadChannels


def test_bad_channels_masks_one_per_module(bad_channels_file: Path):
    bad_channels = BadChannels(bad_channels_file, n_edge_bad_channels=0)

    assert list(bad_channels.masks.keys()) == list(range(MODULES_IN_DETECTOR))
    for mask in bad_channels.masks.values():
        assert mask.shape == (PIXELS_PER_MODULE,)
        assert mask.dtype == bool


def test_bad_channels_are_put_in_the_right_module(bad_channels_file: Path):
    masks = BadChannels(bad_channels_file, n_edge_bad_channels=0).masks

    assert np.flatnonzero(masks[0]).tolist() == [5]
    assert np.flatnonzero(masks[1]).tolist() == [5]
    assert np.flatnonzero(masks[27]).tolist() == [PIXELS_PER_MODULE - 1]
    assert sum(mask.sum() for mask in masks.values()) == 3


def test_bad_channels_edge_channels(bad_channels_file: Path):
    n_edge = 3
    bad_channels = BadChannels(bad_channels_file, n_edge_bad_channels=n_edge)

    assert len(bad_channels.edge_bad_channels) == 2 * n_edge * MODULES_IN_DETECTOR

    for mask in bad_channels.masks.values():
        assert np.all(mask[:n_edge])
        assert np.all(mask[-n_edge:])


def test_bad_channels_default_file_loads():
    bad_channels = BadChannels(DEFAULT_BAD_CHANS)
    assert len(bad_channels.masks) == MODULES_IN_DETECTOR


def test_bad_channels_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        BadChannels(tmp_path / "does_not_exist.txt")


def test_bad_channels_unset_filepath_raises():
    with pytest.raises(ValueError):
        BadChannels("")


@pytest.mark.xfail(
    strict=True,
    reason="an empty file loads as a float array, which can't be used as a mask index",
)
def test_bad_channels_empty_file(tmp_path: Path):
    filepath = tmp_path / "empty.txt"
    filepath.touch()

    masks = BadChannels(filepath, n_edge_bad_channels=2).masks

    assert masks[0].sum() == 4


@pytest.mark.xfail(
    strict=True,
    reason="edge channels are applied to modules 0..n_modules-1, not the active ids",
)
def test_detector_edge_channels_masked_on_active_modules(
    tmp_path: Path, bad_channels_file: Path
):
    detector = make_detector(
        tmp_path,
        bad_channels_file,
        active_modules=[20],
        bad_modules=[],
        edge_bad_channels=5,
    )

    assert np.all(detector.get_module(20).bad_channel_mask[:5])


# MythenModule


@pytest.fixture
def module_data() -> np.ndarray:
    return np.arange(N_FRAMES * PIXELS_PER_MODULE, dtype=float).reshape(
        N_FRAMES, PIXELS_PER_MODULE
    )


@pytest.fixture
def mythen_module(module_data: np.ndarray, conversion: ModuleConversion):
    mask = np.zeros(PIXELS_PER_MODULE, dtype=bool)
    mask[[0, 1, 100]] = True

    return MythenModule(
        data=module_data,
        conversion=conversion,
        beamline_offset=-0.5,
        module_id=3,
        positions=np.array([0.0, 1.0, 2.0]),
        durations=np.array([1.0, 2.0, 3.0]),
        bad_channel_mask=mask,
    )


def test_mythen_module_raw_tth(mythen_module: MythenModule):
    expected = mythen_module.conversion.return_raw_tth(-0.5)
    np.testing.assert_allclose(mythen_module.raw_tth, expected)


def test_mythen_module_tth_2d_adds_positions(mythen_module: MythenModule):
    tth_2d = mythen_module.tth_2d

    assert tth_2d.shape == (N_FRAMES, PIXELS_PER_MODULE)
    np.testing.assert_allclose(tth_2d[0], mythen_module.raw_tth)
    np.testing.assert_allclose(tth_2d[2] - tth_2d[0], 2.0)


def test_mythen_module_duration_2d(mythen_module: MythenModule):
    duration_2d = mythen_module.duration_2d

    assert duration_2d.shape == (N_FRAMES, PIXELS_PER_MODULE)
    np.testing.assert_allclose(duration_2d[:, 0], [1.0, 2.0, 3.0])


def test_mythen_module_mask_2d(mythen_module: MythenModule):
    mask_2d = mythen_module.mask_2d

    assert mask_2d.shape == (N_FRAMES, PIXELS_PER_MODULE)
    assert mask_2d[:, [0, 1, 100]].all()
    assert mask_2d.sum() == 3 * N_FRAMES


def test_mythen_module_unmasked_outputs(
    mythen_module: MythenModule, module_data: np.ndarray
):
    n_points = N_FRAMES * PIXELS_PER_MODULE

    assert mythen_module.unmasked_counts.shape == (n_points,)
    assert mythen_module.unmasked_tth.shape == (n_points,)
    assert mythen_module.unmasked_duration.shape == (n_points,)
    np.testing.assert_allclose(mythen_module.unmasked_counts, module_data.flatten())
    np.testing.assert_allclose(
        mythen_module.unmasked_error, np.sqrt(module_data.flatten())
    )


def test_mythen_module_masked_outputs(
    mythen_module: MythenModule, module_data: np.ndarray
):
    n_points = N_FRAMES * (PIXELS_PER_MODULE - 3)

    assert mythen_module.counts.shape == (n_points,)
    assert mythen_module.tth.shape == (n_points,)
    assert mythen_module.duration.shape == (n_points,)
    np.testing.assert_allclose(mythen_module.error, np.sqrt(mythen_module.counts))

    # masked channels are gone from every frame
    for masked_channel in [0, 1, 100]:
        assert not np.isin(module_data[:, masked_channel], mythen_module.counts).any()


def test_mythen_module_masked_tth_matches_counts(mythen_module: MythenModule):
    keep = ~mythen_module.bad_channel_mask
    expected_tth = mythen_module.tth_2d[:, keep].flatten()
    expected_counts = mythen_module.data[:, keep].flatten()

    np.testing.assert_allclose(mythen_module.tth, expected_tth)
    np.testing.assert_allclose(mythen_module.counts, expected_counts)


def test_mythen_module_default_mask_keeps_everything(
    module_data: np.ndarray, conversion: ModuleConversion
):
    module = MythenModule(
        data=module_data,
        conversion=conversion,
        beamline_offset=0,
        module_id=0,
        positions=np.zeros(N_FRAMES),
        durations=np.ones(N_FRAMES),
    )

    assert not module.bad_channel_mask.any()
    np.testing.assert_allclose(module.counts, module.unmasked_counts)


@pytest.mark.xfail(
    strict=True,
    reason="default durations have one entry per channel (as bool), not per frame",
)
def test_mythen_module_default_durations(
    module_data: np.ndarray, conversion: ModuleConversion
):
    module = MythenModule(
        data=module_data,
        conversion=conversion,
        beamline_offset=0,
        module_id=0,
        positions=np.zeros(N_FRAMES),
    )

    np.testing.assert_allclose(module.duration_2d, 1.0)


@pytest.mark.xfail(
    strict=True, reason="positions_2d broadcasts per-frame positions along channels"
)
def test_mythen_module_positions_2d(mythen_module: MythenModule):
    positions_2d = mythen_module.positions_2d

    assert positions_2d.shape == (N_FRAMES, PIXELS_PER_MODULE)
    np.testing.assert_allclose(positions_2d[:, 0], [0.0, 1.0, 2.0])


# MythenDetector (fake data)


def test_detector_nonexistent_file_raises(tmp_path: Path):
    with pytest.raises(ValueError):
        MythenDetector(filepath=tmp_path / "does_not_exist.nxs")


def test_detector_output_paths(tmp_path: Path, bad_channels_file: Path):
    detector = make_detector(tmp_path, bad_channels_file, [0, 1], [])

    assert Path(detector.output_directory) == tmp_path / "processed"
    assert Path(detector.output_directory).is_dir()
    assert Path(detector.xye_filepath_out).name == "fake_summed_mythen3_test.xye"
    assert (
        Path(detector.processed_nexus_filepath).name == "fake_reduced_mythen3_test.nxs"
    )


def test_detector_custom_output_paths(tmp_path: Path, bad_channels_file: Path):
    filepath = tmp_path / "fake.nxs"
    filepath.touch()
    output_directory = tmp_path / "somewhere" / "else"
    xye_filepath = tmp_path / "custom.xye"

    detector = MythenDetector(
        filepath=filepath,
        settings=MythenSettings(bad_channels_filepath=bad_channels_file),
        output_directory=output_directory,
        xye_filepath_out=xye_filepath,
        data_loader=make_fake_loader(),
    )

    assert output_directory.is_dir()
    assert detector.xye_filepath_out == xye_filepath


def test_detector_good_modules_excludes_bad_and_inactive(
    tmp_path: Path, bad_channels_file: Path
):
    detector = make_detector(
        tmp_path,
        bad_channels_file,
        active_modules=[0, 1, 2, 3, 5, 20],
        bad_modules=[3, 4, 27],
    )

    assert sorted(detector.good_modules) == [0, 1, 2, 5, 20]


@pytest.mark.xfail(
    strict=True, reason="active ring modules use symmetric difference (^)"
)
def test_detector_active_ring_modules(tmp_path: Path, bad_channels_file: Path):
    detector = make_detector(
        tmp_path, bad_channels_file, active_modules=[0, 1, 2, 15, 20], bad_modules=[]
    )

    assert sorted(detector.active_ring1_modules) == [0, 1, 2]
    assert sorted(detector.active_ring2_modules) == [15, 20]


def test_detector_modules_keyed_by_module_id(tmp_path: Path, bad_channels_file: Path):
    active_modules = [0, 5, 20]
    detector = make_detector(tmp_path, bad_channels_file, active_modules, [])

    assert list(detector.modules.keys()) == active_modules

    for module_id in active_modules:
        module = detector.get_module(module_id)
        assert module.module_id == module_id
        # the fake loader fills module n with n + 1
        assert np.all(module.data == module_id + 1)
        assert module.conversion == detector.calibration[f"module_{module_id}"]


def test_detector_modules_use_their_own_bad_channel_mask(
    tmp_path: Path, bad_channels_file: Path
):
    detector = make_detector(tmp_path, bad_channels_file, [0, 1, 27], [])

    for module_id in [0, 1, 27]:
        np.testing.assert_array_equal(
            detector.get_module(module_id).bad_channel_mask,
            detector.bad_channels.masks[module_id],
        )


def test_detector_get_inactive_module_raises(tmp_path: Path, bad_channels_file: Path):
    detector = make_detector(tmp_path, bad_channels_file, [0, 1], [])

    with pytest.raises(KeyError):
        detector.get_module(2)


@pytest.mark.parametrize("masked", [True, False])
def test_detector_generate_xye_is_sorted(
    tmp_path: Path, bad_channels_file: Path, masked: bool
):
    detector = make_detector(tmp_path, bad_channels_file, [0, 1, 27], [])

    tth, counts, error = detector.generate_xye(modules=[0, 1, 27], masked=masked)

    assert tth.shape == counts.shape == error.shape
    assert np.all(np.diff(tth) >= 0)
    np.testing.assert_allclose(error, np.sqrt(counts))


def test_detector_generate_xye_masking_removes_points(
    tmp_path: Path, bad_channels_file: Path
):
    detector = make_detector(tmp_path, bad_channels_file, [0, 1, 27], [])

    masked_tth, _, _ = detector.generate_xye(modules=[0, 1, 27], masked=True)
    unmasked_tth, _, _ = detector.generate_xye(modules=[0, 1, 27], masked=False)

    assert len(unmasked_tth) == 3 * N_FRAMES * PIXELS_PER_MODULE
    # one bad channel in each of modules 0, 1 and 27
    assert len(unmasked_tth) - len(masked_tth) == 3 * N_FRAMES


def test_detector_generate_xye_keeps_counts_with_their_tth(
    tmp_path: Path, bad_channels_file: Path
):
    detector = make_detector(tmp_path, bad_channels_file, [0, 27], [])

    tth, counts, _ = detector.generate_xye(modules=[0, 27], masked=True)

    for module_id in [0, 27]:
        module = detector.get_module(module_id)
        in_module = np.isin(tth, module.tth)
        assert np.all(counts[in_module] == module_id + 1)


def test_detector_generate_binned_xye(tmp_path: Path, bad_channels_file: Path):
    detector = make_detector(tmp_path, bad_channels_file, [0, 1, 2], [2])
    rebin_step = 0.01

    tth, counts, error = detector.generate_binned_xye(rebin_step=rebin_step)

    assert tth.shape == counts.shape == error.shape
    np.testing.assert_allclose(np.diff(tth), rebin_step)

    # only modules 0 and 1 (counts of 1 and 2) are good, so module 2 (counts of 3)
    # shouldn't appear - overlapping bins average to somewhere in between
    finite_counts = counts[np.isfinite(counts) & (counts > 0)]
    assert finite_counts.min() >= 1.0
    assert finite_counts.max() <= 2.0


@pytest.mark.xfail(
    strict=True,
    reason="durations are concatenated in module order but counts are sorted by tth",
)
def test_detector_generate_binned_xye_normalise(
    tmp_path: Path, bad_channels_file: Path
):
    # counts in module 0 are 1, so normalising by a duration of 2 should give 0.5
    detector = make_detector(
        tmp_path,
        bad_channels_file,
        active_modules=[0, 1],
        bad_modules=[],
        data_loader=make_fake_loader(durations=np.array([2.0, 2.0, 1.0])),
    )
    module_0 = detector.get_module(0)
    first_frame_tth = module_0.tth_2d[0, ~module_0.bad_channel_mask]

    tth, counts, _ = detector.generate_binned_xye(normalise=True, rebin_step=0.004)

    in_first_frame = (tth > first_frame_tth.min()) & (tth < first_frame_tth.max())
    assert np.all(counts[in_first_frame & (counts > 0)] == pytest.approx(0.5))


def test_detector_plot_diffraction(
    tmp_path: Path, bad_channels_file: Path, no_show: None
):
    detector = make_detector(tmp_path, bad_channels_file, [0, 1], [])
    plot_filepath = tmp_path / "plot.png"

    tth, counts, error = detector.plot_diffraction(filepath=plot_filepath)

    assert plot_filepath.exists()
    expected_tth, expected_counts, _ = detector.generate_binned_xye(
        rebin_step=detector.settings.rebin_step
    )
    np.testing.assert_allclose(tth, expected_tth)
    np.testing.assert_allclose(counts, expected_counts)


def test_detector_plot_diffraction_by_mod(
    tmp_path: Path, bad_channels_file: Path, no_show: None
):
    detector = make_detector(tmp_path, bad_channels_file, [0, 1], [])
    plot_filepath = tmp_path / "plot_by_mod.png"

    detector.plot_diffraction_by_mod(filepath=plot_filepath)

    assert plot_filepath.exists()


# MythenDataLoader (real data)


def test_data_loader_rejects_non_nexus(tmp_path: Path):
    with pytest.raises(ValueError):
        MythenDataLoader(tmp_path / "data.h5")


def test_data_loader_reads_metadata(si_nexus: Path):
    loader = MythenDataLoader(si_nexus)

    assert loader.n_modules_in_data == MODULES_IN_DETECTOR
    assert loader.n_frames == 4
    assert loader.data.shape == (4, MODULES_IN_DETECTOR * PIXELS_PER_MODULE)


def test_data_loader_splits_into_modules(si_nexus: Path):
    loader = MythenDataLoader(si_nexus)

    assert len(loader.module_data) == MODULES_IN_DETECTOR
    for module_data in loader.module_data:
        assert module_data.shape == (4, PIXELS_PER_MODULE)

    np.testing.assert_array_equal(
        loader.module_data[1], loader.data[:, PIXELS_PER_MODULE : 2 * PIXELS_PER_MODULE]
    )


def test_data_loader_positions_and_durations(si_nexus: Path):
    loader = MythenDataLoader(si_nexus)

    assert loader.delta_path.endswith("/delta")
    assert loader.positions.shape == (4,)
    assert loader.durations.shape == (4,)

    np.testing.assert_allclose(
        loader.positions, h5_to_array(si_nexus, "entry/mythen_nx/delta")
    )


def test_data_loader_get_data_by_module(si_nexus: Path):
    loader = MythenDataLoader(si_nexus)

    module_2 = loader.get_data(modules=2, frame=slice(None))

    np.testing.assert_array_equal(module_2, loader.module_data[2])


@pytest.mark.parametrize("counter", [0, 1, 2])
def test_data_loader_load_all_data_by_counter(si_nexus: Path, counter: int):
    loader = MythenDataLoader(si_nexus)

    data = loader.load_all_data(counter)

    all_counters = h5_to_array(si_nexus, "entry/mythen_nx/data")
    np.testing.assert_array_equal(data, all_counters[:, :, counter])


@pytest.mark.xfail(
    strict=True,
    reason="module_data is split by modules in the file, not the modules loaded",
)
def test_data_loader_subset_of_modules(si_nexus: Path):
    loader = MythenDataLoader(si_nexus, modules=[0, 1, 2])

    assert len(loader.module_data) == 3
    for module_data in loader.module_data:
        assert module_data.shape == (4, PIXELS_PER_MODULE)


# MythenDetector (real data)


def test_detector_real_data_modules(si_nexus: Path):
    detector = MythenDetector(filepath=si_nexus)

    assert list(detector.modules.keys()) == detector.settings.active_modules
    assert not set(detector.good_modules) & set(detector.settings.bad_modules)


def test_detector_real_data_process_step_scan(si_nexus: Path):
    detector = MythenDetector(filepath=si_nexus, filename_suffix="_test")

    detector.process_step_scan(control=False)

    processed_dir = si_nexus.parent / "processed"
    assert Path(detector.output_directory) == processed_dir
    assert Path(detector.xye_filepath_out).parent == processed_dir
    assert Path(detector.xye_filepath_out).exists()
    assert Path(detector.processed_nexus_filepath).exists()
    assert np.all(detector.binned_tth > 0)
    assert len(detector.binned_tth) == len(detector.binned_counts)


def test_detector_real_data_matches_preprocessed(si_nexus: Path):
    detector = MythenDetector(filepath=si_nexus, filename_suffix="_test")

    detector.process_step_scan(control=False)

    processed_now = XYEData.from_csv(detector.xye_filepath_out)
    preprocessed = XYEData.from_csv(PREPROCESSED_SI_XYE_15KEV)

    np.testing.assert_allclose(processed_now.x, preprocessed.x)
    normalised_diff = normalise(processed_now.y) - normalise(preprocessed.y)
    assert np.amax(np.abs(normalised_diff)) < 0.01


@pytest.mark.xfail(
    strict=True, reason="save_data_to_nexus writes into the raw file, not the copy"
)
def test_detector_process_step_scan_leaves_raw_nexus_alone(si_nexus: Path):
    # the test data already has reduced data in it from earlier runs
    with h5py.File(si_nexus, "a") as file:
        for group in ["entry/xye", "entry/ring1", "entry/ring2"]:
            if group in file:
                del file[group]

    detector = MythenDetector(filepath=si_nexus, filename_suffix="_test")

    detector.process_step_scan(control=False)

    with h5py.File(si_nexus, "r") as file:
        assert "entry/xye" not in file


@pytest.mark.xfail(strict=True, reason="ring2 data is saved over /ring1")
def test_detector_process_step_scan_saves_both_rings(si_nexus: Path):
    detector = MythenDetector(filepath=si_nexus, filename_suffix="_test")

    detector.process_step_scan(control=False)

    with h5py.File(si_nexus, "r") as file:
        assert "entry/ring2/data" in file


# angcal conversion


@pytest.mark.xfail(
    strict=True,
    raises=ValidationError,
    reason="centre is read as a top level field, not per module",
)
def test_convert_legacy_angcal(tmp_path: Path):

    legacy = {"beamline_offset": -0.5, "centre": 639.5}
    for module in range(MODULES_IN_DETECTOR):
        legacy[f"conv_{module}"] = 6.5e-05
        legacy[f"offset_{module}"] = 5.0 * module

    legacy_path = tmp_path / "legacy.json"
    legacy_path.write_text(json.dumps(legacy))
    new_path = tmp_path / "new.json"

    convert_angcal_to_pydantic_json(legacy_path, new_path)

    calibration = AngularCalibration.load(new_path)
    assert calibration.module_3.offset == 15.0
