import shutil
from pathlib import Path

import numpy as np
import pytest

from xrpd_toolbox.fit_engine.background import ConstantBackground
from xrpd_toolbox.i15_1.sample_alignment import (
    SampleAligner,
    is_flat,
    run_sample_alignment,
    sample_alignment_i15_1,
    sample_alignment_model_builder,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_ALIGNMENT_DATA = (
    REPO_ROOT / "src" / "xrpd_toolbox" / "i15_1" / "sample_alignment_data"
)
TEST_FILE = SAMPLE_ALIGNMENT_DATA / "NIST_Si-95016.csv"

EXPECTED_SAMPLE_ALIGNMENT_CENTRES = {
    "GaIn-94521.csv": 70.02,
    "HKUST1-95018.csv": 71.05,
    "NIST_Si-95016.csv": 84.37,
    "NaCl-95017.csv": 55.39,
    "carbon_black-94519.csv": 36.77,
    "water-94520.csv": 55.23,
}


def test_is_flat_returns_false_for_non_flat_array():
    array = np.array([1, 2, 3, 4, 5])
    assert not is_flat(array)


def test_is_flat_returns_true_for_flat_array():
    array = np.array([1, 1, 1, 1, 1])
    assert is_flat(array)


def test_is_flat_returns_false_for_gaussian_array():
    rng = np.random.default_rng(0)
    x = np.linspace(-5, 5, 201)

    flat = 2.0 + rng.normal(0, 0.02, x.size)
    tilted = 2.0 + 0.5 * x
    peak = 2.0 + np.exp(-(x**2))
    dip = 2.0 - np.exp(-(x**2))

    assert is_flat(flat)
    assert not is_flat(tilted)
    assert not is_flat(peak)
    assert not is_flat(dip)


def test_sample_alignment_builder_from_csv():
    model = sample_alignment_model_builder(str(TEST_FILE), peak_type="tophat")

    assert isinstance(model, SampleAligner)
    assert isinstance(model.background, ConstantBackground)
    assert model.data.x.shape == model.data.y.shape
    assert model.data.x.shape[0] > 0
    assert len(model.sample_and_capillary) > 0

    y_calc = model.calculate_profile()
    assert y_calc.shape == model.data.x.shape
    assert y_calc.dtype == model.data.x.dtype


@pytest.mark.parametrize(
    "csv_file, expected_centre",
    EXPECTED_SAMPLE_ALIGNMENT_CENTRES.items(),
)
def test_run_sample_alignment_returns_centered_model(
    csv_file: Path, expected_centre: float
):
    sample_file = SAMPLE_ALIGNMENT_DATA / csv_file
    model = run_sample_alignment(str(sample_file))

    assert isinstance(model, SampleAligner)
    assert model.centre is not None
    assert len(model.sample_and_capillary) > 0
    assert model.data.x.shape[0] > 0
    assert model.centre == pytest.approx(expected_centre, abs=2)


def test_sample_alignment_saves_plot_into_processed_subfolder(tmp_path):
    csv_copy = tmp_path / "NIST_Si-95016.csv"
    shutil.copy(TEST_FILE, csv_copy)

    sample_alignment_i15_1(csv_copy, save=True, beamline=None)

    processed_dir = tmp_path / "processed" / "NIST_Si-95016"
    expected_plot = processed_dir / "NIST_Si-95016_alignment_fit.png"
    assert processed_dir.is_dir()
    assert expected_plot.exists()
    # nothing should have been written next to the source csv itself, or
    # directly in the flat processed/ folder (that's for shared, non-per-file
    # data like goniometer calibrations)
    assert not (tmp_path / "NIST_Si-95016_alignment_fit.png").exists()
    assert not (tmp_path / "processed" / "NIST_Si-95016_alignment_fit.png").exists()
