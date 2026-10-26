import math

import numpy as np

from xrpd_toolbox.core import XYEData
from xrpd_toolbox.utils.chemical_formula import ChemicalFormula
from xrpd_toolbox.utils.pdf_utils import gr_background

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
