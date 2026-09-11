"""Tests for xrpd_toolbox.i15_1.eiger_pyfai.

Two kinds of tests live here:

- Unit tests and Two "system" tests

      pytest tests/test_eiger_pyfai.py -k system -s -v

  or run this file directly with python tests/test_eiger_pyfai.py
"""

import json
import logging
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pyFAI.calibrant import get_calibrant

from xrpd_toolbox.i15_1 import eiger_pyfai
from xrpd_toolbox.i15_1.eiger_500k import DEFAULT_MAX_SHAPE, PIXEL_SIZE, Eiger500K

SI_CALIBRANT = get_calibrant("Si")
SI_CALIBRANT.wavelength = 1e-10


def test_calibrate_single_geometry_from_rings_extracts_once_at_largest_ring_count():
    # extract_cp() *replaces* rather than accumulates control points, using
    # the geometry's current (possibly already-nudged) dist/poni/rot to
    # decide which pixels belong to which ring - re-extracting every round
    # from a progressively refined geometry let small errors compound round
    # to round (verified: several degrees off on closely-spaced Si rings),
    # so only the largest ring count is ever used, in a single extract+
    # refine pass.
    geometry = MagicMock()

    result = eiger_pyfai.calibrate_single_geometry_from_rings(geometry, rings=[5, 7, 9])

    assert result is geometry


def test_calibrate_single_geometry_from_rings_with_fix():
    geometry = MagicMock()

    eiger_pyfai.calibrate_single_geometry_from_rings(geometry, rings=[5], fix=["dist"])

    geometry.geometry_refinement.refine2.assert_called_once_with(fix=["dist"])


def test_calibrate_single_frame_without_rings():
    mock_sg_cls = MagicMock()
    mock_sg = mock_sg_cls.return_value
    mock_sg.geometry_refinement.data = np.zeros((5, 2))

    with patch.object(eiger_pyfai, "SingleGeometry", mock_sg_cls):
        result = eiger_pyfai._calibrate_single_frame(
            "frame_0000_3.0000deg",
            np.zeros((4, 5)),
            3.0,
            SI_CALIBRANT,
            0.25,
            None,
            1.0,
        )

    assert result is mock_sg
    mock_sg.extract_cp.assert_called_once_with(max_rings=None, pts_per_deg=1.0)
    mock_sg.geometry_refinement.refine2.assert_called_once_with()

    _, kwargs = mock_sg_cls.call_args
    assert kwargs["label"] == "frame_0000_3.0000deg"
    assert kwargs["calibrant"] is SI_CALIBRANT
    # must be an actual Eiger500K instance (not the "Eiger500k" name string)
    # or SingleGeometry resolves it via pyFAI's own detector registry to a
    # differently-shaped, wrong detector - see the NOTE in
    # _calibrate_single_frame.
    assert isinstance(kwargs["detector"], Eiger500K)
    assert kwargs["detector"].max_shape == DEFAULT_MAX_SHAPE
    geometry = kwargs["geometry"]
    # the SingleGeometry `detector=` kwarg and the one embedded in
    # `geometry` must be the same object - only the former actually takes
    # effect, but they should never be allowed to disagree.
    assert geometry["detector"] is kwargs["detector"]
    assert geometry["rot2"] == pytest.approx(np.deg2rad(3.0))
    assert geometry["wavelength"] == SI_CALIBRANT.wavelength
    # every frame needs a pos_function or GoniometerRefinement.refine2()
    # crashes later with "'NoneType' object is not callable" - see
    # test_system_calibrate for the real (unmocked) version of this check.
    assert kwargs["pos_function"](3.0) == (3.0,)


def test_calibrate_single_frame_with_int_max_rings():
    mock_sg_cls = MagicMock()
    mock_sg = mock_sg_cls.return_value
    mock_sg.geometry_refinement.data = np.zeros((5, 2))

    with patch.object(eiger_pyfai, "SingleGeometry", mock_sg_cls):
        eiger_pyfai._calibrate_single_frame(
            "frame_0", np.zeros((4, 5)), 3.0, SI_CALIBRANT, 0.25, 7, 2.0
        )

    mock_sg.extract_cp.assert_called_once_with(max_rings=7, pts_per_deg=2.0)


def test_calibrate_single_frame_with_iterable_rings_delegates():
    mock_sg_cls = MagicMock()
    mock_sg = mock_sg_cls.return_value
    mock_sg.geometry_refinement.data = np.zeros((5, 2))

    calls = []
    original = eiger_pyfai.calibrate_single_geometry_from_rings

    def spy(geometry, rings=None, fix=None):
        calls.append((geometry, rings, fix))
        return original(geometry, rings=rings or [], fix=fix)

    with (
        patch.object(eiger_pyfai, "SingleGeometry", mock_sg_cls),
        patch.object(eiger_pyfai, "calibrate_single_geometry_from_rings", spy),
    ):
        result = eiger_pyfai._calibrate_single_frame(
            "frame_0", np.zeros((4, 5)), 3.0, SI_CALIBRANT, 0.25, [5, 7], 1.0
        )

    assert result is mock_sg
    assert len(calls) == 1
    assert calls[0][1] == [5, 7]


def test_calibrate_single_frame_asserts_control_points_present():
    mock_sg_cls = MagicMock()
    mock_sg = mock_sg_cls.return_value
    mock_sg.geometry_refinement.data = None

    with patch.object(eiger_pyfai, "SingleGeometry", mock_sg_cls):
        with pytest.raises(AssertionError):
            eiger_pyfai._calibrate_single_frame(
                "frame_0", np.zeros((4, 5)), 3.0, SI_CALIBRANT, 0.25, None, 1.0
            )


# ---------------------------------------------------------------------------
# _load_goniometer_dir
# ---------------------------------------------------------------------------


def test_load_goniometer_dir_raises_when_both_files_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match=eiger_pyfai.GONIOMETER_SAVE_NAME):
        eiger_pyfai._load_goniometer_dir(tmp_path)


def test_load_goniometer_dir_raises_when_metadata_missing(tmp_path):
    (tmp_path / eiger_pyfai.GONIOMETER_SAVE_NAME).write_text("{}")

    with pytest.raises(FileNotFoundError, match=eiger_pyfai.METADATA_SAVE_NAME):
        eiger_pyfai._load_goniometer_dir(tmp_path)


def test_load_goniometer_dir_success(tmp_path):
    (tmp_path / eiger_pyfai.GONIOMETER_SAVE_NAME).write_text("{}")
    meta = {
        "unit": "2th_deg",
        "npt": 100,
        "wavelength": 1e-11,
        "calibrant": "Si",
        "calib_two_theta_deg": [1.0, 2.0, 3.0],
        "radial_range": None,
    }
    (tmp_path / eiger_pyfai.METADATA_SAVE_NAME).write_text(json.dumps(meta))

    fake_gonio = MagicMock()
    mock_sload = MagicMock(return_value=fake_gonio)

    with patch.object(eiger_pyfai.Goniometer, "sload", mock_sload):
        gonio, loaded_meta = eiger_pyfai._load_goniometer_dir(tmp_path)

    assert gonio is fake_gonio
    assert loaded_meta == meta
    mock_sload.assert_called_once_with(str(tmp_path / eiger_pyfai.GONIOMETER_SAVE_NAME))


# ---------------------------------------------------------------------------
# build_and_save_goniometer
# ---------------------------------------------------------------------------


def _fake_calibrate_single_frame_factory():
    """Returns a drop-in replacement for _calibrate_single_frame that avoids
    any real peak-finding / refinement."""

    def fake(
        label, image, two_theta_deg, calibrant, initial_dist_m, max_rings, pts_per_deg
    ):
        sg = MagicMock()
        sg.label = label
        sg.geometry_refinement = SimpleNamespace(
            dist=initial_dist_m,
            poni1=0.05,
            poni2=0.02,
            rot1=0.0,
            rot2=np.deg2rad(two_theta_deg),
            rot3=0.0,
            data=np.zeros((5, 2)),
        )
        return sg

    return fake


@pytest.fixture
def fake_gonioref():
    gonioref = MagicMock()
    gonioref.single_geometries = {}
    gonioref.chi2.return_value = 0.001

    with (
        patch.object(
            eiger_pyfai, "GoniometerRefinement", MagicMock(return_value=gonioref)
        ),
        patch.object(
            eiger_pyfai,
            "_calibrate_single_frame",
            _fake_calibrate_single_frame_factory(),
        ),
    ):
        yield gonioref


def test_build_and_save_goniometer_explicit_output_dir(tmp_path, fake_gonioref):
    images = np.zeros((3, 4, 5))
    angles = np.array([1.0, 2.0, 3.0])

    gonio_path, meta_path = eiger_pyfai.build_and_save_goniometer(
        nexus_filepath=tmp_path / "scan.nxs",
        images=images,
        angles=angles,
        wavelength_in_angstrom=0.161699,
        output_dir=tmp_path,
    )

    assert gonio_path == str(tmp_path / eiger_pyfai.GONIOMETER_SAVE_NAME)
    assert Path(meta_path).exists()

    meta = json.loads(Path(meta_path).read_text())
    assert meta["calibrant"] == "Si"
    assert meta["unit"] == "2th_deg"
    assert meta["npt"] == 2000
    assert meta["radial_range"] is None
    assert meta["calib_two_theta_deg"] == angles.tolist()
    assert meta["wavelength"] == pytest.approx(0.161699e-10)

    fake_gonioref.refine2.assert_called_once()
    fake_gonioref.chi2.assert_called_once()
    fake_gonioref.save.assert_called_once_with(gonio_path)
    assert len(fake_gonioref.single_geometries) == 3
    assert set(fake_gonioref.single_geometries) == {
        "frame_0000_1.0000deg",
        "frame_0001_2.0000deg",
        "frame_0002_3.0000deg",
    }


def test_build_and_save_goniometer_default_output_dir_is_nexus_parent(
    tmp_path, fake_gonioref
):
    images = np.zeros((2, 4, 5))
    angles = np.array([1.0, 2.0])
    nexus_filepath = tmp_path / "sub" / "scan.nxs"
    nexus_filepath.parent.mkdir()

    gonio_path, meta_path = eiger_pyfai.build_and_save_goniometer(
        nexus_filepath=nexus_filepath,
        images=images,
        angles=angles,
        wavelength_in_angstrom=0.161699,
    )

    assert Path(gonio_path).parent == nexus_filepath.parent
    assert Path(meta_path).parent == nexus_filepath.parent


def test_build_and_save_goniometer_stores_radial_range_and_npt(tmp_path, fake_gonioref):
    images = np.zeros((2, 4, 5))
    angles = np.array([1.0, 2.0])

    _, meta_path = eiger_pyfai.build_and_save_goniometer(
        nexus_filepath=tmp_path / "scan.nxs",
        images=images,
        angles=angles,
        wavelength_in_angstrom=0.161699,
        output_dir=tmp_path,
        radial_range=(0.0, 60.0),
        npt=500,
    )

    meta = json.loads(Path(meta_path).read_text())
    assert meta["radial_range"] == [0.0, 60.0]
    assert meta["npt"] == 500


def test_build_and_save_goniometer_initial_params_seeded_from_first_frame(tmp_path):
    gonioref = MagicMock()
    gonioref.single_geometries = {}
    gonioref.chi2.return_value = 0.0
    mock_gonioref_cls = MagicMock(return_value=gonioref)

    images = np.zeros((2, 4, 5))
    angles = np.array([1.0, 4.0])

    with (
        patch.object(eiger_pyfai, "GoniometerRefinement", mock_gonioref_cls),
        patch.object(
            eiger_pyfai,
            "_calibrate_single_frame",
            _fake_calibrate_single_frame_factory(),
        ),
    ):
        eiger_pyfai.build_and_save_goniometer(
            nexus_filepath=tmp_path / "scan.nxs",
            images=images,
            angles=angles,
            wavelength_in_angstrom=0.161699,
            output_dir=tmp_path,
            initial_dist_m=0.3,
        )

    initial_params = mock_gonioref_cls.call_args.args[0]
    assert initial_params["dist"] == pytest.approx(0.3)
    assert initial_params["rot2_scale"] == 1.0
    assert initial_params["rot2_offset"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# integrate_with_goniometer
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_goniometer_dir():
    fake_gonio = MagicMock()
    fake_gonio.get_ai.side_effect = lambda tth: SimpleNamespace(tth=tth)
    meta = {
        "unit": "2th_deg",
        "npt": 50,
        "wavelength": 1e-11,
        "calibrant": "Si",
        "calib_two_theta_deg": [1.0, 2.0, 3.0],
        "radial_range": None,
    }

    with patch.object(
        eiger_pyfai, "_load_goniometer_dir", lambda d: (fake_gonio, meta)
    ):
        yield fake_gonio, meta


@pytest.fixture
def fake_multigeometry():
    mg_instance = MagicMock()
    mg_instance.integrate1d.return_value = SimpleNamespace(
        radial=np.array([1.0, 2.0, 3.0]),
        intensity=np.array([10.0, 20.0, 30.0]),
        sigma=np.array([0.1, 0.2, 0.3]),
    )
    mock_mg_cls = MagicMock(return_value=mg_instance)

    with patch.object(eiger_pyfai, "MultiGeometry", mock_mg_cls):
        yield mock_mg_cls, mg_instance


def test_integrate_with_goniometer_writes_xy_file(
    tmp_path, fake_goniometer_dir, fake_multigeometry
):
    images = np.zeros((3, 4, 5))
    positions = np.array([1.0, 2.0, 3.0])
    out_path = tmp_path / "out" / "result.xy"

    result_path = eiger_pyfai.integrate_with_goniometer(
        images=images,
        positions=positions,
        goniometer_dir=tmp_path,
        output_xy_filepath=out_path,
    )

    assert result_path == out_path
    assert out_path.exists()
    contents = out_path.read_text()
    assert "1" in contents and "10" in contents


def test_integrate_with_goniometer_creates_parent_directory(
    tmp_path, fake_goniometer_dir, fake_multigeometry
):
    out_path = tmp_path / "does" / "not" / "exist" / "result.xy"

    eiger_pyfai.integrate_with_goniometer(
        images=np.zeros((3, 4, 5)),
        positions=np.array([1.0, 2.0, 3.0]),
        goniometer_dir=tmp_path,
        output_xy_filepath=out_path,
    )

    assert out_path.parent.exists()


def test_integrate_with_goniometer_logs_warning_when_out_of_range(
    tmp_path, fake_goniometer_dir, fake_multigeometry, caplog
):
    positions = np.array([1.0, 2.0, 30.0])  # 30 deg is outside [1, 3]

    with caplog.at_level(logging.WARNING, logger="xrpd_toolbox.i15_1.eiger_pyfai"):
        eiger_pyfai.integrate_with_goniometer(
            images=np.zeros((3, 4, 5)),
            positions=positions,
            goniometer_dir=tmp_path,
            output_xy_filepath=tmp_path / "out.xy",
        )

    assert any("extrapolating" in record.message for record in caplog.records)


def test_integrate_with_goniometer_no_warning_when_in_range(
    tmp_path, fake_goniometer_dir, fake_multigeometry, caplog
):
    positions = np.array([1.0, 2.0, 3.0])

    with caplog.at_level(logging.WARNING, logger="xrpd_toolbox.i15_1.eiger_pyfai"):
        eiger_pyfai.integrate_with_goniometer(
            images=np.zeros((3, 4, 5)),
            positions=positions,
            goniometer_dir=tmp_path,
            output_xy_filepath=tmp_path / "out.xy",
        )

    assert not any("extrapolating" in record.message for record in caplog.records)


def test_integrate_with_goniometer_npt_override(
    tmp_path, fake_goniometer_dir, fake_multigeometry
):
    _, mg_instance = fake_multigeometry

    eiger_pyfai.integrate_with_goniometer(
        images=np.zeros((3, 4, 5)),
        positions=np.array([1.0, 2.0, 3.0]),
        goniometer_dir=tmp_path,
        output_xy_filepath=tmp_path / "out.xy",
        npt=999,
    )

    assert mg_instance.integrate1d.call_args.kwargs["npt"] == 999


def test_integrate_with_goniometer_uses_meta_npt_by_default(
    tmp_path, fake_goniometer_dir, fake_multigeometry
):
    _, mg_instance = fake_multigeometry
    _, meta = fake_goniometer_dir

    eiger_pyfai.integrate_with_goniometer(
        images=np.zeros((3, 4, 5)),
        positions=np.array([1.0, 2.0, 3.0]),
        goniometer_dir=tmp_path,
        output_xy_filepath=tmp_path / "out.xy",
    )

    assert mg_instance.integrate1d.call_args.kwargs["npt"] == meta["npt"]


def test_integrate_with_goniometer_radial_range_from_meta(tmp_path, fake_multigeometry):
    mock_mg_cls, _ = fake_multigeometry
    fake_gonio = MagicMock()
    fake_gonio.get_ai.side_effect = lambda tth: SimpleNamespace(tth=tth)
    meta = {
        "unit": "2th_deg",
        "npt": 50,
        "wavelength": 1e-11,
        "calibrant": "Si",
        "calib_two_theta_deg": [1.0, 2.0, 3.0],
        "radial_range": [0.0, 45.0],
    }

    with patch.object(
        eiger_pyfai, "_load_goniometer_dir", lambda d: (fake_gonio, meta)
    ):
        eiger_pyfai.integrate_with_goniometer(
            images=np.zeros((3, 4, 5)),
            positions=np.array([1.0, 2.0, 3.0]),
            goniometer_dir=tmp_path,
            output_xy_filepath=tmp_path / "out.xy",
        )

    assert mock_mg_cls.call_args.kwargs["radial_range"] == (0.0, 45.0)


def test_integrate_with_goniometer_mask_expanded_per_frame(
    tmp_path, fake_goniometer_dir, fake_multigeometry
):
    _, mg_instance = fake_multigeometry
    mask = np.ones((4, 5), dtype=bool)

    eiger_pyfai.integrate_with_goniometer(
        images=np.zeros((3, 4, 5)),
        positions=np.array([1.0, 2.0, 3.0]),
        goniometer_dir=tmp_path,
        output_xy_filepath=tmp_path / "out.xy",
        mask=mask,
    )

    lst_mask = mg_instance.integrate1d.call_args.kwargs["lst_mask"]
    assert len(lst_mask) == 3
    assert all(np.array_equal(m, mask) for m in lst_mask)


def test_integrate_with_goniometer_mask_none_by_default(
    tmp_path, fake_goniometer_dir, fake_multigeometry
):
    _, mg_instance = fake_multigeometry

    eiger_pyfai.integrate_with_goniometer(
        images=np.zeros((3, 4, 5)),
        positions=np.array([1.0, 2.0, 3.0]),
        goniometer_dir=tmp_path,
        output_xy_filepath=tmp_path / "out.xy",
    )

    assert mg_instance.integrate1d.call_args.kwargs["lst_mask"] is None


def test_integrate_with_goniometer_header_written_when_requested(
    tmp_path, fake_goniometer_dir, fake_multigeometry
):
    out_path = tmp_path / "out.xy"

    eiger_pyfai.integrate_with_goniometer(
        images=np.zeros((3, 4, 5)),
        positions=np.array([1.0, 2.0, 3.0]),
        goniometer_dir=tmp_path,
        output_xy_filepath=out_path,
        save_xy_with_header=True,
    )

    contents = out_path.read_text()
    assert "# goniometer_dir" in contents
    assert "# npt: 50" in contents


def test_integrate_with_goniometer_no_header_by_default(
    tmp_path, fake_goniometer_dir, fake_multigeometry
):
    out_path = tmp_path / "out.xy"

    eiger_pyfai.integrate_with_goniometer(
        images=np.zeros((3, 4, 5)),
        positions=np.array([1.0, 2.0, 3.0]),
        goniometer_dir=tmp_path,
        output_xy_filepath=out_path,
    )

    contents = out_path.read_text()
    assert "# goniometer_dir" not in contents


def test_integrate_with_goniometer_default_error_model_is_azimuthal(
    tmp_path, fake_goniometer_dir, fake_multigeometry
):
    _, mg_instance = fake_multigeometry

    eiger_pyfai.integrate_with_goniometer(
        images=np.zeros((3, 4, 5)),
        positions=np.array([1.0, 2.0, 3.0]),
        goniometer_dir=tmp_path,
        output_xy_filepath=tmp_path / "out.xy",
    )

    assert mg_instance.integrate1d.call_args.kwargs["error_model"] == "azimuthal"


def test_integrate_with_goniometer_error_model_override(
    tmp_path, fake_goniometer_dir, fake_multigeometry
):
    _, mg_instance = fake_multigeometry

    eiger_pyfai.integrate_with_goniometer(
        images=np.zeros((3, 4, 5)),
        positions=np.array([1.0, 2.0, 3.0]),
        goniometer_dir=tmp_path,
        output_xy_filepath=tmp_path / "out.xy",
        error_model="poisson",
    )

    assert mg_instance.integrate1d.call_args.kwargs["error_model"] == "poisson"


def test_integrate_with_goniometer_no_xye_file_by_default(
    tmp_path, fake_goniometer_dir, fake_multigeometry
):
    out_path = tmp_path / "out.xy"

    eiger_pyfai.integrate_with_goniometer(
        images=np.zeros((3, 4, 5)),
        positions=np.array([1.0, 2.0, 3.0]),
        goniometer_dir=tmp_path,
        output_xy_filepath=out_path,
    )

    assert not out_path.with_suffix(".xye").exists()


def test_integrate_with_goniometer_writes_xye_file_when_requested(
    tmp_path, fake_goniometer_dir, fake_multigeometry
):
    out_path = tmp_path / "out.xy"

    eiger_pyfai.integrate_with_goniometer(
        images=np.zeros((3, 4, 5)),
        positions=np.array([1.0, 2.0, 3.0]),
        goniometer_dir=tmp_path,
        output_xy_filepath=out_path,
        save_xye=True,
    )

    xye_path = out_path.with_suffix(".xye")
    assert xye_path.exists()
    tth, intensity, error = np.loadtxt(xye_path, unpack=True)
    assert np.array_equal(tth, [1.0, 2.0, 3.0])
    assert np.array_equal(intensity, [10.0, 20.0, 30.0])
    assert np.array_equal(error, [0.1, 0.2, 0.3])


# ---------------------------------------------------------------------------
# Real, unmocked calibrate / integrate cycle
# ---------------------------------------------------------------------------
#
# No mocking below this point. These build synthetic Si powder-ring frames
# with xrpd_toolbox's own Eiger500K.simulate_data() and run the actual
# calibration / integration code paths.
#
# Historical note: this used to go through pyFAI's own built-in Eiger500k
# detector (max_shape (514, 1030)) instead, as a workaround for
# _calibrate_single_frame hardcoding that detector by name - which silently
# overrode the correctly-shaped detector object passed alongside it (see the
# NOTE in _calibrate_single_frame). That workaround masked a second issue:
# with pyFAI's detector shape/aspect ratio, build_and_save_goniometer's
# joint GoniometerRefinement.refine2() step reproducibly converged to a
# wrong geometry - even seeded at the exact true parameters, it walked away
# because a wrong geometry scored a much lower chi2() against the extracted
# control points, regardless of how many frames or how wide an angular
# spread were used. Once _calibrate_single_frame was fixed to consistently
# use xrpd_toolbox's own (correctly-shaped) Eiger500K detector throughout,
# that degeneracy disappeared too - see
# test_build_and_save_goniometer_recovers_true_geometry below, which is the
# regression test for both.

SYSTEM_TEST_OUTPUT_DIR = Path(__file__).parent / "system_test_output"
SYSTEM_TEST_WAVELENGTH_ANGSTROM = 0.161699


def _true_eiger(dist_m: float = 0.25) -> Eiger500K:
    """A beam-centred Eiger500K at `dist_m`, used as the known-true geometry
    that synthetic calibration/measurement frames are simulated from."""
    rows, cols = DEFAULT_MAX_SHAPE
    poni = {
        "dist": dist_m,
        "poni1": rows / 2 * PIXEL_SIZE,
        "poni2": cols / 2 * PIXEL_SIZE,
        "rot1": 0.0,
        "rot2": 0.0,
        "rot3": 0.0,
        "pixel1": PIXEL_SIZE,
        "pixel2": PIXEL_SIZE,
        "wavelength": SYSTEM_TEST_WAVELENGTH_ANGSTROM / 1e10,
    }
    return Eiger500K(poni=poni)


def _fresh_output_dir(name: str) -> Path:
    output_dir = SYSTEM_TEST_OUTPUT_DIR / name
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    return output_dir


def test_calibrate_goniometer_recovers_input_geometry():
    """Simulate Si calibration frames at a known geometry across an array of
    two-theta angles (Eiger500K.simulate_data()), calibrate a Goniometer
    model against them (build_and_save_goniometer), and check the refined
    parameters land back on the values used to simulate the data."""
    output_dir = _fresh_output_dir("calibration_roundtrip")
    dist_true = 0.25
    eiger = _true_eiger(dist_true)
    rows, cols = DEFAULT_MAX_SHAPE
    poni1_true = rows / 2 * PIXEL_SIZE
    poni2_true = cols / 2 * PIXEL_SIZE

    angles = np.array([5.0, 10.0, 15.0, 20.0, 25.0])
    images, _ = eiger.simulate_data(
        positions_in_tth=angles,
        calibrant_name="Si",
        wavelength_in_ang=SYSTEM_TEST_WAVELENGTH_ANGSTROM,
        resolution=0.05,
    )

    eiger_pyfai.build_and_save_goniometer(
        nexus_filepath=output_dir / "fake_calibration_scan.nxs",
        images=np.array(images),
        angles=angles,
        wavelength_in_angstrom=SYSTEM_TEST_WAVELENGTH_ANGSTROM,
        calibrant_name="Si",
        initial_dist_m=dist_true,
        output_dir=output_dir,
        max_rings=[5, 5, 5, 7, 7, 9, 11, 15, 17],
    )

    gonio, _ = eiger_pyfai._load_goniometer_dir(output_dir)
    fitted = dict(
        zip(eiger_pyfai.GEOMETRY_TRANSFORMATION.param_names, gonio.param, strict=True)
    )

    print(f"\ncalibrated at angles {angles.tolist()} deg")
    print("true vs. refined goniometer parameters:")
    true_values = {
        "dist": dist_true,
        "poni1": poni1_true,
        "poni2": poni2_true,
        "rot1": 0.0,
        "rot2_scale": 1.0,
        "rot2_offset": 0.0,
        "rot3": 0.0,
    }
    for name, value in fitted.items():
        print(f"    {name:>12s}: true={true_values[name]:.6g}  refined={value:.6g}")

    assert fitted["dist"] == pytest.approx(dist_true, abs=1e-3)
    assert fitted["poni1"] == pytest.approx(poni1_true, abs=2e-3)
    assert fitted["poni2"] == pytest.approx(poni2_true, abs=2e-3)
    assert fitted["rot1"] == pytest.approx(0.0, abs=2e-3)
    assert fitted["rot3"] == pytest.approx(0.0, abs=2e-3)
    assert fitted["rot2_scale"] == pytest.approx(1.0, abs=2e-3)
    # two-theta arm's error - a fraction of a degree
    assert fitted["rot2_offset"] == pytest.approx(0.0, abs=np.deg2rad(0.5))


def test_system_calibrate():
    """Real, unmocked calibrate step: fits a Goniometer model to synthetic
    Si calibration frames and writes it to
    tests/system_test_output/calibration/ for inspection.

    Run with: pytest tests/test_eiger_pyfai.py -k system_calibrate -s -v
    """
    output_dir = _fresh_output_dir("calibration")
    eiger = _true_eiger()

    angles = np.array([5.0, 10.0, 15.0, 20.0, 25.0])
    images, _ = eiger.simulate_data(
        positions_in_tth=angles,
        calibrant_name="Si",
        wavelength_in_ang=SYSTEM_TEST_WAVELENGTH_ANGSTROM,
        resolution=0.05,
    )
    images = np.array(images)

    gonio_path, meta_path = eiger_pyfai.build_and_save_goniometer(
        nexus_filepath=output_dir / "fake_calibration_scan.nxs",
        images=images,
        angles=angles,
        wavelength_in_angstrom=SYSTEM_TEST_WAVELENGTH_ANGSTROM,
        calibrant_name="Si",
        initial_dist_m=0.25,
        output_dir=output_dir,
        max_rings=[5, 5, 7, 9],
    )

    gonio, meta = eiger_pyfai._load_goniometer_dir(output_dir)
    fitted = dict(
        zip(eiger_pyfai.GEOMETRY_TRANSFORMATION.param_names, gonio.param, strict=True)
    )

    print(f"\nsystem test (calibrate): wrote {gonio_path}")
    print(f"system test (calibrate): wrote {meta_path}")
    print(f"system test (calibrate): calibrated at angles {angles.tolist()} deg")
    print("system test (calibrate): fitted goniometer parameters:")
    for name, value in fitted.items():
        print(f"    {name:>12s} = {value:.6g}")

    assert Path(gonio_path).exists()
    assert Path(meta_path).exists()
    assert all(np.isfinite(value) for value in fitted.values())
    # dist should be in the right ballpark of the 0.25 m used to simulate
    assert fitted["dist"] == pytest.approx(0.25, abs=0.05)


def test_system_integrate():
    """Real, unmocked integrate step: first runs a (small/fast) real
    calibration to get a goniometer model, then integrates fresh synthetic
    frames against it and writes the resulting pattern to
    tests/system_test_output/integration/fake_scan_eiger.xy for inspection.

    Run with: pytest tests/test_eiger_pyfai.py -k system_integrate -s -v
    """
    output_dir = _fresh_output_dir("integration")
    eiger = _true_eiger()

    calibration_angles = np.array([5.0, 10.0, 15.0, 20.0, 25.0])
    calibration_images, _ = eiger.simulate_data(
        positions_in_tth=calibration_angles,
        calibrant_name="Si",
        wavelength_in_ang=SYSTEM_TEST_WAVELENGTH_ANGSTROM,
        resolution=0.05,
    )
    eiger_pyfai.build_and_save_goniometer(
        nexus_filepath=output_dir / "fake_calibration_scan.nxs",
        images=np.array(calibration_images),
        angles=calibration_angles,
        wavelength_in_angstrom=SYSTEM_TEST_WAVELENGTH_ANGSTROM,
        calibrant_name="Si",
        initial_dist_m=0.25,
        output_dir=output_dir,
        max_rings=[5, 5, 5, 7, 7, 9, 11, 15, 17],
    )

    measurement_angles = np.array([8.0, 18.0])
    measurement_images, _ = eiger.simulate_data(
        positions_in_tth=measurement_angles,
        calibrant_name="Si",
        wavelength_in_ang=SYSTEM_TEST_WAVELENGTH_ANGSTROM,
        resolution=0.05,
    )
    measurement_images = np.array(measurement_images)
    out_xy_filepath = output_dir / "fake_scan_eiger.xy"

    result_path = eiger_pyfai.integrate_with_goniometer(
        images=measurement_images,
        positions=measurement_angles,
        goniometer_dir=output_dir,
        output_xy_filepath=out_xy_filepath,
        save_xy_with_header=True,
    )

    radial, intensity = np.loadtxt(result_path, comments="#", unpack=True)

    print(f"\nsystem test (integrate): wrote {result_path}")
    print(f"system test (integrate): {len(radial)} points")
    print(
        f"system test (integrate): radial range "
        f"[{radial.min():.3f}, {radial.max():.3f}] deg"
    )
    print(
        f"system test (integrate): intensity range "
        f"[{intensity.min():.3g}, {intensity.max():.3g}]"
    )

    from scipy.signal import find_peaks

    peak_idx, _ = find_peaks(intensity, height=intensity.max() * 0.05, distance=5)
    found_peaks = np.sort(radial[peak_idx])
    si_calibrant = get_calibrant("Si")
    si_calibrant.wavelength = SYSTEM_TEST_WAVELENGTH_ANGSTROM / 1e10
    expected_peaks = si_calibrant.get_peaks("2th_deg")
    print(f"system test (integrate): found peaks (deg): {found_peaks[:8].tolist()}")
    print(
        f"system test (integrate): expected Si peaks (deg): "
        f"{expected_peaks[:8].tolist()}"
    )

    assert result_path == out_xy_filepath
    assert result_path.exists()
    assert len(radial) == len(intensity) > 0
    assert np.all(np.isfinite(radial))
    assert np.all(np.isfinite(intensity))
    # the integrated pattern should show Si's real peaks, in the right place
    for expected in expected_peaks[(expected_peaks > radial.min())][:5]:
        assert np.any(np.abs(found_peaks - expected) < 0.1), (
            f"no found peak within 0.1 deg of expected Si peak {expected:.3f}"
        )


if __name__ == "__main__":
    # test_system_calibrate()
    test_system_integrate()
