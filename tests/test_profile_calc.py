from pathlib import Path

import numpy as np
import pytest

from xrpd_toolbox.core import Parameter
from xrpd_toolbox.fit_engine.background import Background, LinearInterpolationBackground
from xrpd_toolbox.fit_engine.fit_statistics import calculate_rwp
from xrpd_toolbox.fit_engine.fitting_core import ScatteringData, refine_model
from xrpd_toolbox.fit_engine.profile_calculation import ReitveldRefinement, Structure
from xrpd_toolbox.utils.unit_conversion import beam_energy_to_wavelength

# Paths relative to this test file
TEST_DIR = Path(__file__).parent
REPO_ROOT = TEST_DIR.parent
CIF_DIR = REPO_ROOT / "cifs"
TEST_DATA_DIR = TEST_DIR / "data" / "i11"

# Data file - use test data directory
DATA_FILE = TEST_DATA_DIR / "1410696_summed_mythen3.xye"


def lattice_a(model: ReitveldRefinement) -> float:
    assert isinstance(model.structure, Structure)
    return float(model.structure.lattice.a)


def rwp(model: ReitveldRefinement) -> float:
    y_calc = model.calculate_profile()
    return calculate_rwp(y_calc=y_calc, y_obs=model.data.y, y_obs_error=model.data.e)


@pytest.mark.skipif(
    not DATA_FILE.exists(),
    reason="Test data file not available (expected in local dev environment)",
)
def test_refine_silicon_save_reload_and_refine_again(tmp_path: Path):
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

    background = LinearInterpolationBackground.estimate(data.x, data.y, points=5)

    model = ReitveldRefinement(data=data, background=background, structure=si_structure)

    assert isinstance(model.background, Background)
    model.background.refine_none()

    model.irf.refine_none()

    starting_rwp = rwp(model)
    model.refine(plot=False)
    refined_rwp = rwp(model)
    refined_profile = np.asarray(model.calculate_profile())
    refined_a = lattice_a(model)

    assert refined_rwp < starting_rwp
    assert refined_rwp < 0.5

    # Si is 5.4311 A, but the wavelength is the nominal 15 keV rather than a
    # refined one, so the lattice parameter soaks up the difference
    assert refined_a == pytest.approx(5.4311, abs=0.2)

    output_filepath = tmp_path / "refinement.yaml"
    model.save(output_filepath)

    loaded_refinement = ReitveldRefinement.load(output_filepath)

    # reloading should give back exactly the refined model
    assert lattice_a(loaded_refinement) == refined_a
    np.testing.assert_allclose(loaded_refinement.calculate_profile(), refined_profile)

    # it is already refined, so refining again shouldn't make the fit worse
    refine_model(loaded_refinement)

    assert rwp(loaded_refinement) <= refined_rwp + 1e-6
