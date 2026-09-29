from pyFAI.calibrant import Calibrant
from pyFAI.crystallography.cell import Cell

from xrpd_toolbox import BASE_PATH

WB_FILEPATH = BASE_PATH / "i15_1" / "WB"


def calc_and_save_wb_calibrant(filepath: str | None = None):
    """Calculates and saves the WB calibrant used on i15-1, to a file

    Uses lattice parameters determined on i11 in 2026 at 300 K

    WB is Tunsgten in amorphous boron"""

    wb_filepath = filepath or WB_FILEPATH

    tungsten_wb = Cell.cubic(
        3.165448, lattice_type="I"
    )  # as determined by i11 refinement
    tungsten_wb.save(str(wb_filepath), dmin=0.05)

    return


def load_wb_calibrant(
    wavelength: float | None = None, filepath: str | None = None
) -> Calibrant:
    """Loads the custom calibrant tungsten in amorphous boron,
    that has previously been saved.

    Wavelength is in METERS - just like pyfai"""

    wb_filepath = filepath or WB_FILEPATH

    wb_cal = Calibrant(wavelength=wavelength)
    wb_cal.load_file(filename=f"{wb_filepath}.D")

    return wb_cal
