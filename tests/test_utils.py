import os

import numpy as np

from xrpd_toolbox.utils.energy import beam_energy_to_wavelength, tth_to_q
from xrpd_toolbox.utils.utils import load_int_array_from_file, normalise_to


def test_normalise_to():
    normalised_array = normalise_to([1, 2, 4], minval=0)
    assert np.array_equal(normalised_array, [0.25, 0.5, 1.0])


def test_tth_to_q():
    tth = 30
    q_in_angstrom = tth_to_q(tth, 1)
    assert np.round(q_in_angstrom, 2) == 3.25


def test_beam_energy_to_wavelength():
    wavelength_in_angstrom = beam_energy_to_wavelength(12.34, unit="kev")
    assert round(wavelength_in_angstrom, 2) == 1.0


def test_load_int_array_from_file():
    test_file = "int_array.txt"

    # Create a temporary file
    with open(test_file, "w") as f:
        for i in range(1, 6):
            f.write(f"{i}\n")

    # Test loading the array
    result = load_int_array_from_file(test_file)
    expected = np.array([1, 2, 3, 4, 5])
    assert np.array_equal(result, expected)

    # Clean up
    os.remove(test_file)
