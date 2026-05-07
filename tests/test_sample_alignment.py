from pathlib import Path

import pytest

from xrpd_toolbox.fit_engine.background import ConstantBackground
from xrpd_toolbox.i15_1.sample_alignment import (
    SampleAligner,
    run_sample_alignment,
    sample_alignment_builder,
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
    "NaCl-95017.csv": 55.396,
    "carbon_black-94519.csv": 36.77,
    "water-94520.csv": 55.23,
}


def test_sample_alignment_builder_from_csv():
    model = sample_alignment_builder(str(TEST_FILE), peak_type="tophat")

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
