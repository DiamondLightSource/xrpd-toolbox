"""Tests for xrpd_toolbox.i15_1.eiger_pyfai.

The heavy pyFAI calibration/refinement classes (SingleGeometry,
GoniometerRefinement, Goniometer, MultiGeometry) are mocked out so these
tests exercise xrpd_toolbox's own orchestration logic (what gets called,
with what arguments, what gets written to disk) quickly and deterministically,
rather than relying on a real peak-fitting refinement converging.
"""

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from pyFAI.calibrant import get_calibrant

from xrpd_toolbox.i15_1 import eiger_pyfai as ep

SI_CALIBRANT = get_calibrant("Si")
SI_CALIBRANT.wavelength = 1e-10


# ---------------------------------------------------------------------------
# calibrate_single_geometry_from_rings
# ---------------------------------------------------------------------------


def test_calibrate_single_geometry_from_rings_default_fix():
    geometry = MagicMock()

    result = ep.calibrate_single_geometry_from_rings(geometry, rings=[5, 7, 9])

    assert result is geometry
    assert geometry.extract_cp.call_args_list == [
        ((), {"max_rings": 5}),
        ((), {"max_rings": 7}),
        ((), {"max_rings": 9}),
    ]
    assert geometry.geometry_refinement.refine2.call_args_list == [
        ((), {"fix": []}),
        ((), {"fix": []}),
        ((), {"fix": []}),
    ]


def test_calibrate_single_geometry_from_rings_with_fix():
    geometry = MagicMock()

    ep.calibrate_single_geometry_from_rings(geometry, rings=[5], fix=["dist"])

    geometry.geometry_refinement.refine2.assert_called_once_with(fix=["dist"])


def test_calibrate_single_geometry_from_rings_default_rings_arg():
    geometry = MagicMock()

    ep.calibrate_single_geometry_from_rings(geometry)

    # default `rings` is [5, 5, 5, 7, 7, 9, 11, 15, 17] - 9 values
    assert geometry.extract_cp.call_count == 9


# ---------------------------------------------------------------------------
# _calibrate_single_frame
# ---------------------------------------------------------------------------


def test_calibrate_single_frame_without_rings(monkeypatch):
    mock_sg_cls = MagicMock()
    mock_sg = mock_sg_cls.return_value
    mock_sg.geometry_refinement.data = np.zeros((5, 2))
    monkeypatch.setattr(ep, "SingleGeometry", mock_sg_cls)

    result = ep._calibrate_single_frame(
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
    assert kwargs["detector"] == ep.DETECTOR
    geometry = kwargs["geometry"]
    assert geometry["rot2"] == pytest.approx(np.deg2rad(3.0))
    assert geometry["wavelength"] == SI_CALIBRANT.wavelength


def test_calibrate_single_frame_with_int_max_rings(monkeypatch):
    mock_sg_cls = MagicMock()
    mock_sg = mock_sg_cls.return_value
    mock_sg.geometry_refinement.data = np.zeros((5, 2))
    monkeypatch.setattr(ep, "SingleGeometry", mock_sg_cls)

    ep._calibrate_single_frame(
        "frame_0", np.zeros((4, 5)), 3.0, SI_CALIBRANT, 0.25, 7, 2.0
    )

    mock_sg.extract_cp.assert_called_once_with(max_rings=7, pts_per_deg=2.0)


def test_calibrate_single_frame_with_iterable_rings_delegates(monkeypatch):
    mock_sg_cls = MagicMock()
    mock_sg = mock_sg_cls.return_value
    mock_sg.geometry_refinement.data = np.zeros((5, 2))
    monkeypatch.setattr(ep, "SingleGeometry", mock_sg_cls)

    calls = []
    original = ep.calibrate_single_geometry_from_rings

    def spy(geometry, rings=None, fix=None):
        calls.append((geometry, rings, fix))
        return original(geometry, rings=rings or [], fix=fix)

    monkeypatch.setattr(ep, "calibrate_single_geometry_from_rings", spy)

    result = ep._calibrate_single_frame(
        "frame_0", np.zeros((4, 5)), 3.0, SI_CALIBRANT, 0.25, [5, 7], 1.0
    )

    assert result is mock_sg
    assert len(calls) == 1
    assert calls[0][1] == [5, 7]
    # in the rings branch, extract_cp is called once per ring count and
    # *without* the pts_per_deg kwarg (unlike the non-rings branch)
    assert mock_sg.extract_cp.call_args_list == [
        ((), {"max_rings": 5}),
        ((), {"max_rings": 7}),
    ]
    assert mock_sg.geometry_refinement.refine2.call_count == 2


def test_calibrate_single_frame_asserts_control_points_present(monkeypatch):
    mock_sg_cls = MagicMock()
    mock_sg = mock_sg_cls.return_value
    mock_sg.geometry_refinement.data = None
    monkeypatch.setattr(ep, "SingleGeometry", mock_sg_cls)

    with pytest.raises(AssertionError):
        ep._calibrate_single_frame(
            "frame_0", np.zeros((4, 5)), 3.0, SI_CALIBRANT, 0.25, None, 1.0
        )


# ---------------------------------------------------------------------------
# _load_goniometer_dir
# ---------------------------------------------------------------------------


def test_load_goniometer_dir_raises_when_both_files_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match=ep.GONIOMETER_SAVE_NAME):
        ep._load_goniometer_dir(tmp_path)


def test_load_goniometer_dir_raises_when_metadata_missing(tmp_path):
    (tmp_path / ep.GONIOMETER_SAVE_NAME).write_text("{}")

    with pytest.raises(FileNotFoundError, match=ep.METADATA_SAVE_NAME):
        ep._load_goniometer_dir(tmp_path)


def test_load_goniometer_dir_success(tmp_path, monkeypatch):
    (tmp_path / ep.GONIOMETER_SAVE_NAME).write_text("{}")
    meta = {
        "unit": "2th_deg",
        "npt": 100,
        "wavelength": 1e-11,
        "calibrant": "Si",
        "calib_two_theta_deg": [1.0, 2.0, 3.0],
        "radial_range": None,
    }
    (tmp_path / ep.METADATA_SAVE_NAME).write_text(json.dumps(meta))

    fake_gonio = MagicMock()
    mock_sload = MagicMock(return_value=fake_gonio)
    monkeypatch.setattr(ep.Goniometer, "sload", mock_sload)

    gonio, loaded_meta = ep._load_goniometer_dir(tmp_path)

    assert gonio is fake_gonio
    assert loaded_meta == meta
    mock_sload.assert_called_once_with(str(tmp_path / ep.GONIOMETER_SAVE_NAME))


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
def fake_gonioref(monkeypatch):
    gonioref = MagicMock()
    gonioref.single_geometries = {}
    gonioref.chi2.return_value = 0.001
    monkeypatch.setattr(ep, "GoniometerRefinement", MagicMock(return_value=gonioref))
    monkeypatch.setattr(
        ep, "_calibrate_single_frame", _fake_calibrate_single_frame_factory()
    )
    return gonioref


def test_build_and_save_goniometer_explicit_output_dir(tmp_path, fake_gonioref):
    images = np.zeros((3, 4, 5))
    angles = np.array([1.0, 2.0, 3.0])

    gonio_path, meta_path = ep.build_and_save_goniometer(
        nexus_filepath=tmp_path / "scan.nxs",
        images=images,
        angles=angles,
        wavelength_in_angstrom=0.161699,
        output_dir=tmp_path,
    )

    assert gonio_path == str(tmp_path / ep.GONIOMETER_SAVE_NAME)
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

    gonio_path, meta_path = ep.build_and_save_goniometer(
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

    _, meta_path = ep.build_and_save_goniometer(
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


def test_build_and_save_goniometer_initial_params_seeded_from_first_frame(
    tmp_path, monkeypatch
):
    gonioref = MagicMock()
    gonioref.single_geometries = {}
    gonioref.chi2.return_value = 0.0
    mock_gonioref_cls = MagicMock(return_value=gonioref)
    monkeypatch.setattr(ep, "GoniometerRefinement", mock_gonioref_cls)
    monkeypatch.setattr(
        ep, "_calibrate_single_frame", _fake_calibrate_single_frame_factory()
    )

    images = np.zeros((2, 4, 5))
    angles = np.array([1.0, 4.0])

    ep.build_and_save_goniometer(
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
def fake_goniometer_dir(monkeypatch):
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
    monkeypatch.setattr(ep, "_load_goniometer_dir", lambda d: (fake_gonio, meta))
    return fake_gonio, meta


@pytest.fixture
def fake_multigeometry(monkeypatch):
    mg_instance = MagicMock()
    mg_instance.integrate1d.return_value = SimpleNamespace(
        radial=np.array([1.0, 2.0, 3.0]), intensity=np.array([10.0, 20.0, 30.0])
    )
    mock_mg_cls = MagicMock(return_value=mg_instance)
    monkeypatch.setattr(ep, "MultiGeometry", mock_mg_cls)
    return mock_mg_cls, mg_instance


def test_integrate_with_goniometer_writes_xy_file(
    tmp_path, fake_goniometer_dir, fake_multigeometry
):
    images = np.zeros((3, 4, 5))
    positions = np.array([1.0, 2.0, 3.0])
    out_path = tmp_path / "out" / "result.xy"

    result_path = ep.integrate_with_goniometer(
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

    ep.integrate_with_goniometer(
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
        ep.integrate_with_goniometer(
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
        ep.integrate_with_goniometer(
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

    ep.integrate_with_goniometer(
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

    ep.integrate_with_goniometer(
        images=np.zeros((3, 4, 5)),
        positions=np.array([1.0, 2.0, 3.0]),
        goniometer_dir=tmp_path,
        output_xy_filepath=tmp_path / "out.xy",
    )

    assert mg_instance.integrate1d.call_args.kwargs["npt"] == meta["npt"]


def test_integrate_with_goniometer_radial_range_from_meta(
    tmp_path, monkeypatch, fake_multigeometry
):
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
    monkeypatch.setattr(ep, "_load_goniometer_dir", lambda d: (fake_gonio, meta))

    ep.integrate_with_goniometer(
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

    ep.integrate_with_goniometer(
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

    ep.integrate_with_goniometer(
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

    ep.integrate_with_goniometer(
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

    ep.integrate_with_goniometer(
        images=np.zeros((3, 4, 5)),
        positions=np.array([1.0, 2.0, 3.0]),
        goniometer_dir=tmp_path,
        output_xy_filepath=out_path,
    )

    contents = out_path.read_text()
    assert "# goniometer_dir" not in contents
