from pyFAI.calibrant import Calibrant
from pyFAI.crystallography.cell import Cell

from xrpd_toolbox import BASE_PATH

WB_FILEPATH = BASE_PATH / "i15_1" / "WB"


def calc_tungsten_with_amoprhous_boron():

    tungsten_wb = Cell.cubic(3.165448, lattice_type="I")
    tungsten_wb.save(str(WB_FILEPATH), dmin=0.05)


def load_wb_calibrant(wavelength: float | None = None) -> Calibrant:

    wb_cal = Calibrant(wavelength=wavelength)
    wb_cal.load_file(filename=f"{WB_FILEPATH}.D")

    return wb_cal


if __name__ == "__main__":
    calc_tungsten_with_amoprhous_boron()
    load_wb_calibrant()
