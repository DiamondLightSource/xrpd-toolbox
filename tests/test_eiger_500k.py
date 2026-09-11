"""Tests for xrpd_toolbox.i15_1.eiger_500k.

Uses small synthetic NeXus/HDF5 files (see eiger_fixtures.py) instead of a
real Eiger500K data collection, and small synthetic .poni files for the
pyFAI geometry-loading branches of Eiger500K.__init__.
"""

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError
from pyFAI.detectors import Detector
from pyFAI.integrator.azimuthal import AzimuthalIntegrator

from eiger_fixtures import build_eiger_nexus, build_mask_file
from xrpd_toolbox.i15_1.eiger_500k import (
    DEFAULT_MAX_SHAPE,
    PIXEL_SIZE,
    Eiger500K,
    EigerDataLoader,
    EigerSettings,
)

WAVELENGTH_ANGSTROM = 0.161699

PONI_DICT = {
    "dist": 0.7,
    "poni1": 0.0,
    "poni2": 0.1,
    "rot1": 0.0,
    "rot2": 0.0,
    "rot3": 0.0,
    "pixel1": PIXEL_SIZE,
    "pixel2": PIXEL_SIZE,
    "wavelength": WAVELENGTH_ANGSTROM / 1e10,
}


def make_poni_file(path: Path, pixel_size: float = PIXEL_SIZE) -> Path:
    detector = Detector(
        pixel1=pixel_size, pixel2=pixel_size, max_shape=DEFAULT_MAX_SHAPE
    )
    ai = AzimuthalIntegrator(
        detector=detector,
        dist=0.25,
        poni1=0.01,
        poni2=0.02,
        wavelength=WAVELENGTH_ANGSTROM / 1e10,
    )
    ai.save(str(path))
    return path


# ---------------------------------------------------------------------------
# EigerSettings
# ---------------------------------------------------------------------------


def test_eiger_settings_defaults():
    settings = EigerSettings()

    assert settings.bad_channel_masking is True
    assert settings.apply_flatfield is False
    assert settings.error_calc == "poisson"
    assert settings.poni_filepath is None


def test_eiger_settings_overrides():
    settings = EigerSettings(
        bad_channels_filepath="mask.h5",
        bad_channel_masking=False,
        flatfield_filepath="flat.h5",
        apply_flatfield=True,
        darkfield_filepath="dark.h5",
        send_to_ispyb=True,
        rebin_step=0.01,
        error_calc="std_dev",
        poni_filepath="calib.poni",
    )

    assert settings.apply_flatfield is True
    assert settings.error_calc == "std_dev"
    assert settings.poni_filepath == "calib.poni"


def test_eiger_settings_rejects_invalid_error_calc():
    with pytest.raises(ValidationError):
        EigerSettings(error_calc="not_a_valid_choice")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# EigerDataLoader
# ---------------------------------------------------------------------------


@pytest.fixture
def nexus_file(tmp_path) -> Path:
    mask_file = tmp_path / "mask.h5"
    build_mask_file(mask_file, "entry/mask", shape=(4, 5))
    return build_eiger_nexus(
        tmp_path / "scan.nxs",
        n_frames=3,
        rows=4,
        cols=5,
        tth=np.array([1.0, 2.0, 3.0]),
        mask_ref=f"{mask_file}//entry/mask",
    )


def test_data_loader_entry_and_dataset_path(nexus_file):
    loader = EigerDataLoader(nexus_file)

    assert loader.entry == "entry"
    assert loader.dataset_path == "/entry/fastcs_eiger/data"


def test_data_loader_positions(nexus_file):
    loader = EigerDataLoader(nexus_file)

    assert np.array_equal(loader.positions, [1.0, 2.0, 3.0])
    # cached_property: second access returns the same cached array
    assert loader.positions is loader.positions


def test_data_loader_positions_missing_falls_back_to_zero(tmp_path, capsys):
    nxs = build_eiger_nexus(tmp_path / "scan.nxs", include_tth=False)
    loader = EigerDataLoader(nxs)

    assert np.array_equal(loader.positions, [0])
    assert "tth" in capsys.readouterr().out


def test_data_loader_durations(nexus_file):
    loader = EigerDataLoader(nexus_file)

    assert np.array_equal(loader.durations, [0.1, 0.1, 0.1])


def test_data_loader_beam_energy_and_wavelength(nexus_file):
    loader = EigerDataLoader(nexus_file)

    assert loader.beam_energy == pytest.approx(12.4)
    assert loader.wavelength == pytest.approx(0.99987, abs=1e-4)


def test_load_all_data(nexus_file):
    loader = EigerDataLoader(nexus_file)

    data = loader.load_all_data()

    assert data.shape == (3, 4, 5)


def test_get_data_with_int_and_list(nexus_file):
    loader = EigerDataLoader(nexus_file)

    single = loader.get_data(0)
    assert single.shape == (4, 5)

    subset = loader.get_data([0, 2])
    assert subset.shape == (2, 4, 5)


def test_get_data_missing_dataset_raises(tmp_path):
    nxs = build_eiger_nexus(tmp_path / "scan.nxs", include_data=False)
    loader = EigerDataLoader(nxs)

    with pytest.raises(ValueError, match="not found in HDF5 file"):
        loader.get_data(slice(None))


def test_get_data_scalar_dataset_raises(tmp_path):
    nxs = build_eiger_nexus(tmp_path / "scan.nxs", data_is_scalar=True)
    loader = EigerDataLoader(nxs)

    with pytest.raises(ValueError, match="insufficient dimensions"):
        loader.get_data(slice(None))


def test_get_data_non_dataset_raises(tmp_path):
    nxs = build_eiger_nexus(tmp_path / "scan.nxs", data_is_group=True)
    loader = EigerDataLoader(nxs)

    with pytest.raises(ValueError, match="is None"):
        loader.get_data(slice(None))


def test_get_pixel_mask_filepath_and_datapath_direct(nexus_file):
    loader = EigerDataLoader(nexus_file)

    mask_filepath, mask_datapath = loader.get_pixel_mask_filepath_and_datapath()

    assert Path(mask_filepath).exists()
    assert mask_datapath == "entry/mask"


def test_get_pixel_mask_filepath_and_datapath_fallback(tmp_path):
    # store a reference to a mask file that does not exist at the stored
    # path, but does exist next to the nexus file (as *.h5) - this is the
    # "odin/ophyd async" fallback the loader compensates for.
    real_mask = tmp_path / "mask_orig.h5"
    build_mask_file(real_mask, "entry/mask")

    nxs = build_eiger_nexus(
        tmp_path / "scan.nxs",
        mask_ref="/nonexistent/dir/mask_orig.h5//entry/mask",
    )
    loader = EigerDataLoader(nxs)

    mask_filepath, mask_datapath = loader.get_pixel_mask_filepath_and_datapath()

    assert Path(mask_filepath) == real_mask
    assert mask_datapath == "entry/mask"


def test_get_mask(nexus_file):
    loader = EigerDataLoader(nexus_file)

    mask = loader.get_mask()

    assert mask.shape == (4, 5)
    assert mask.dtype == bool


def test_get_calibrant_when_present_returns_none(nexus_file):
    # NOTE: get_calibrant() never returns the value it reads on success, so
    # this documents the current (surprising) behaviour rather than "Si".
    loader = EigerDataLoader(nexus_file)

    assert loader.get_calibrant() is None


def test_get_calibrant_when_missing_returns_none(tmp_path, capsys):
    nxs = build_eiger_nexus(tmp_path / "scan.nxs", include_calibrant=False)
    loader = EigerDataLoader(nxs)

    assert loader.get_calibrant() is None
    assert "calibrant" in capsys.readouterr().out


def test_is_background_true(tmp_path):
    nxs = build_eiger_nexus(tmp_path / "scan.nxs", background=1)
    loader = EigerDataLoader(nxs)

    assert loader.is_background() is True


def test_is_background_false(nexus_file):
    loader = EigerDataLoader(nexus_file)

    assert loader.is_background() is False


def test_plan_name(tmp_path):
    nxs = build_eiger_nexus(tmp_path / "scan.nxs", plan_name="calibration_collection")
    loader = EigerDataLoader(nxs)

    assert loader.plan_name == "calibration_collection"
    assert loader.get_plan_name() == "calibration_collection"


# ---------------------------------------------------------------------------
# Eiger500K construction / geometry handling
# ---------------------------------------------------------------------------


def test_eiger500k_with_dict_poni():
    eiger = Eiger500K(poni=PONI_DICT)

    assert eiger.ai is not None
    assert eiger.ai.dist == pytest.approx(0.7)
    assert eiger.pixel1 == PIXEL_SIZE


def test_eiger500k_with_poni_filepath_str(tmp_path):
    poni_path = make_poni_file(tmp_path / "good.poni")

    eiger = Eiger500K(poni=str(poni_path))

    assert eiger.ai is not None
    assert eiger.ai.dist == pytest.approx(0.25)


def test_eiger500k_with_poni_filepath_path_object(tmp_path):
    poni_path = make_poni_file(tmp_path / "good.poni")

    eiger = Eiger500K(poni=poni_path)

    assert eiger.ai is not None


def test_eiger500k_with_settings_poni(tmp_path):
    poni_path = make_poni_file(tmp_path / "good.poni")
    settings = EigerSettings(poni_filepath=str(poni_path))

    eiger = Eiger500K(settings=settings)

    assert eiger.ai is not None
    assert eiger.ai.dist == pytest.approx(0.25)


def test_eiger500k_settings_without_poni_filepath_raises():
    with pytest.raises(FileNotFoundError):
        Eiger500K(settings=EigerSettings())


def test_eiger500k_no_poni_no_settings_has_no_ai():
    eiger = Eiger500K()

    assert eiger.ai is None
    assert eiger.max_shape == DEFAULT_MAX_SHAPE


def test_eiger500k_pixel_size_mismatch_raises(tmp_path):
    mismatched_poni = make_poni_file(tmp_path / "bad.poni", pixel_size=1e-4)

    with pytest.raises(ValueError, match="Pixel size"):
        Eiger500K(poni=str(mismatched_poni))


def test_eiger500k_with_filepath_builds_data_loader_and_process_step_scan(tmp_path):
    nxs = build_eiger_nexus(tmp_path / "scan.nxs", tth=np.array([1.0, 2.0]))

    eiger = Eiger500K(filepath=nxs)

    assert isinstance(eiger.data_loader, EigerDataLoader)
    # exercises the (currently no-op) loop over positions without error
    eiger.process_step_scan()


def test_load_geometry_is_currently_a_noop():
    eiger = Eiger500K(poni=PONI_DICT)

    assert eiger.load_geometry("some.poni") is None


def test_set_calibrant():
    eiger = Eiger500K(poni=PONI_DICT)

    calibrant = eiger.set_calibrant("Si", WAVELENGTH_ANGSTROM)

    assert eiger.calibrant is calibrant
    assert calibrant.wavelength == pytest.approx(WAVELENGTH_ANGSTROM / 1e10)


# ---------------------------------------------------------------------------
# Simulation / integration
# ---------------------------------------------------------------------------


@pytest.fixture
def eiger():
    return Eiger500K(poni=PONI_DICT)


def test_simulate_data(eiger):
    positions_in_tth = np.linspace(1, 20, 4)

    images, ais = eiger.simulate_data(
        positions_in_tth=positions_in_tth,
        calibrant_name="Si",
        wavelength_in_ang=WAVELENGTH_ANGSTROM,
    )

    assert len(images) == len(ais) == 4
    assert images[0].shape == DEFAULT_MAX_SHAPE
    assert np.any(images[0] > 0)
    # rot2 should track the requested two-theta position (in radians)
    assert ais[1].rot2 == pytest.approx(np.deg2rad(positions_in_tth[1]))


def test_simulate_data_reuses_existing_calibrant(eiger):
    eiger.set_calibrant("Si", WAVELENGTH_ANGSTROM)
    calibrant_before = eiger.calibrant

    eiger.simulate_data(
        positions_in_tth=[1.0],
        calibrant_name="Si",
        wavelength_in_ang=WAVELENGTH_ANGSTROM,
    )

    assert eiger.calibrant is calibrant_before


def test_simulate_data_raises_without_ai():
    eiger = Eiger500K()
    eiger.calibrant = eiger.set_calibrant("Si", WAVELENGTH_ANGSTROM)

    with pytest.raises(AttributeError, match="No Azimuthal Integrator"):
        eiger.simulate_data(
            positions_in_tth=[1.0],
            calibrant_name="Si",
            wavelength_in_ang=WAVELENGTH_ANGSTROM,
        )


def test_integrate_images(eiger):
    positions_in_tth = np.linspace(1, 30, 3)
    images, ais = eiger.simulate_data(
        positions_in_tth=positions_in_tth,
        calibrant_name="Si",
        wavelength_in_ang=WAVELENGTH_ANGSTROM,
    )

    x_data, y_data = eiger.integrate_images(images, ais)

    assert len(x_data) == len(y_data) == 10000


def test_simulate_1d_pattern(eiger):
    x_data, y_data = eiger.simulate_1d_pattern(
        positions_in_tth=np.linspace(1, 30, 3),
        calibrant_name="Si",
        wavelength_in_ang=WAVELENGTH_ANGSTROM,
    )

    assert len(x_data) > 0
    assert len(y_data) == len(x_data)


def test_eiger500k_test_method_runs_without_display(monkeypatch, eiger):
    import matplotlib.pyplot as plt

    # `test()` calls plt.show() directly, which raises a UserWarning under
    # the non-interactive Agg backend used in CI; filterwarnings="error"
    # turns that into a hard failure, so patch it out.
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)

    eiger.test()

    plt.close("all")
