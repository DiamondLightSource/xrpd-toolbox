import os
from pathlib import Path

import pytest

from xrpd_toolbox.core import Parameter
from xrpd_toolbox.fit_engine.background import Background, LinearInterpolationBackground
from xrpd_toolbox.fit_engine.fitting_core import ScatteringData, refine_model
from xrpd_toolbox.fit_engine.profile_calculation import ReitveldRefinement, Structure
from xrpd_toolbox.utils.unit_conversion import beam_energy_to_wavelength

# Paths relative to this test file
TEST_DIR = Path(__file__).parent
REPO_ROOT = TEST_DIR.parent
CIF_DIR = REPO_ROOT / "cifs"
TEST_DATA_DIR = TEST_DIR / "data"
OUTPUT_NAME = "test.yaml"

# Data file - use test data directory
DATA_FILE = TEST_DATA_DIR / "1410696_summed_mythen3.xye"


@pytest.mark.skipif(
    not DATA_FILE.exists(),
    reason="Test data file not available (expected in local dev environment)",
)
def test_refine_silicon():
    si_cif = CIF_DIR / "Si.cif"
    assert si_cif.exists(), f"CIF file not found: {si_cif}"

    si_structure = Structure.load_from_cif(si_cif)

    beam_energy = 15
    wavelength = beam_energy_to_wavelength(beam_energy)

    data = ScatteringData.from_xye(
        str(DATA_FILE),
        x_unit="tth",
        data_type="xray",
        wavelength=Parameter(value=wavelength, refine=False),
    )

    background = LinearInterpolationBackground.estimate(data.x, data.y)

    model = ReitveldRefinement(data=data, background=background, structure=si_structure)

    assert isinstance(model.background, Background)
    model.background.refine_none()

    print(model.get_refinement_parameters())
    model.irf.refine_none()
    model.refine(plot=False)
    model.save(OUTPUT_NAME)
    os.remove(OUTPUT_NAME)


@pytest.mark.skipif(
    not Path(OUTPUT_NAME).exists(),
    reason="Requires test_refine_silicon to run first",
)
def test_load_refinement_and_refine():
    loaded_refinement = ReitveldRefinement.load(OUTPUT_NAME)

    loaded_refinement.calculate_profile()

    refine_model(loaded_refinement)
