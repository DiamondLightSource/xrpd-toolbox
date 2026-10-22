import math

import numpy as np
import pytest

from xrpd_toolbox.core import XYEData
from xrpd_toolbox.utils.chemical_formula import ChemicalFormula
from xrpd_toolbox.utils.pdf_utils import gr_background, validate_gr

COMPOSITION = "K2PdCl6"
DENSITY = 2.7


def synthetic_gr(
    peak_r: float, coordination: float, baseline_scale: float = 1.0
) -> XYEData:
    """G(r) with a single peak whose R(r) integral is coordination."""

    chem = ChemicalFormula(formula=COMPOSITION)
    rho = chem.rho(density_g_cm3=DENSITY)
    r = np.arange(0.01, 4.0, 0.01)
    area = coordination

    sigma = 0.05
    rdf = (
        area
        * np.exp(-((r - peak_r) ** 2) / (2 * sigma**2))
        / (sigma * math.sqrt(2 * math.pi))
    )
    # G = (R - 4 pi r^2 rho) / r, with the baseline optionally mis-scaled
    gr = rdf / r + baseline_scale * gr_background(r, rho)

    return XYEData(title="synthetic", x=r, y=gr)


def test_validate_gr_passes_for_correct_gr():
    result = validate_gr(
        composition=COMPOSITION,
        density_g_cm3=DENSITY,
        pdf_data=synthetic_gr(peak_r=2.34, coordination=6),
        expected_peak_position=2.34,
        expected_coordination_number=6,
    )
    assert result.passed
    assert result.coordination_number == pytest.approx(6, abs=0.1)
    assert result.peak_position == pytest.approx(2.34, abs=0.01)


def test_validate_gr_fails_wrong_coordination():
    result = validate_gr(
        composition=COMPOSITION,
        density_g_cm3=DENSITY,
        pdf_data=synthetic_gr(peak_r=2.34, coordination=4),
        expected_peak_position=2.34,
        expected_coordination_number=6,
    )
    assert not result.coordination_ok
    assert result.background_ok
    assert result.peak_position_ok


def test_validate_gr_fails_wrong_peak_position():
    result = validate_gr(
        composition=COMPOSITION,
        density_g_cm3=DENSITY,
        pdf_data=synthetic_gr(peak_r=2.6, coordination=6),
        expected_peak_position=2.34,
    )
    assert not result.peak_position_ok


def test_validate_gr_fails_wrong_background():
    result = validate_gr(
        composition=COMPOSITION,
        density_g_cm3=DENSITY,
        pdf_data=synthetic_gr(peak_r=2.34, coordination=6, baseline_scale=0.5),
        expected_peak_position=2.34,
    )
    assert not result.background_ok
