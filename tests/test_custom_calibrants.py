from pathlib import Path

from pyFAI.crystallography.calibrant import Calibrant

from xrpd_toolbox.i15_1.custom_calibrants import (
    calc_and_save_wb_calibrant,
    load_wb_calibrant,
)


def test_calc_and_save_wb_calibrant(tmp_path: Path):

    temp_wb_calibrant_file = tmp_path / "tmp_wb"

    calc_and_save_wb_calibrant(
        filepath=str(temp_wb_calibrant_file)
    )  # automatically appends .D

    assert Path(f"{temp_wb_calibrant_file}.D").exists()


def test_load_wb_calibrant(tmp_path: Path):

    temp_wb_calibrant_file = tmp_path / "tmp_wb.D"

    calibrant = load_wb_calibrant(filepath=str(temp_wb_calibrant_file))

    assert isinstance(calibrant, Calibrant)
