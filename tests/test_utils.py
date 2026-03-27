import os

import numpy as np
import scipy.integrate as integrate

from xrpd_toolbox.utils.peaks import find_and_fit_peaks, gaussian
from xrpd_toolbox.utils.unit_conversion import beam_energy_to_wavelength, two_theta_to_q
from xrpd_toolbox.utils.utils import (
    get_filenumber_from_nxs,
    get_folder_paths,
    load_int_array_from_file,
    nexus_file_match,
    normalise,
    normalise_to,
)


def test_get_filenumber_from_nxs():
    filedir = "/dls/i11/test/cm12345-1/i11-99999.nxs"
    assert get_filenumber_from_nxs(filedir) == 99999


def test_nexus_file_match():
    filedir = "/dls/i15-1/test/cm12345-1/i15-1-99999.nxs"
    filename = os.path.basename(filedir)
    assert nexus_file_match(filename, beamline="i15-1") is not None


def test_normalise_to():
    normalised_array = normalise_to([1, 2, 4], minval=0)
    assert np.array_equal(normalised_array, [0.25, 0.5, 1.0])


def test_normalise():
    normalised_array = normalise([1, 2, 4])
    assert np.amax(normalised_array) == 1.0
    assert np.amin(normalised_array) == 0.0


def test_gaussian():
    x = np.linspace(0, 10, 100)
    y = gaussian(x, amplitude=22.0, centre=5.0, fwhm=1.0)
    assert len(y) == len(x)
    integral = integrate.simpson(y, x)
    assert np.isclose(integral, 22.0, atol=0.5)


def test_tth_to_q():
    tth = 30
    q_in_angstrom = two_theta_to_q(tth, 1)
    assert np.round(q_in_angstrom, 2) == 3.25


def test_beam_energy_to_wavelength():
    wavelength_in_angstrom = beam_energy_to_wavelength(12.34, unit="kev")
    assert round(wavelength_in_angstrom, 2) == 1.0


def test_load_int_array_from_file_returns_array_when_contains_ints():
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


def test_load_int_array_from_file_returns_none_when_file_doesnt_exist():
    test_file = "non_existent.txt"

    # Test loading the array
    result = load_int_array_from_file(test_file)
    expected = np.array([])
    assert np.array_equal(result, expected)


def test_load_int_array_from_file_returns_none_when_file_empty():
    test_file = "int_array.txt"

    # Create a temporary file
    with open(test_file, "w") as f:
        f.write("")

    # Test loading the array
    result = load_int_array_from_file(test_file)
    expected = np.array([])
    assert np.array_equal(result, expected)

    # Clean up
    os.remove(test_file)


def test_get_folder_paths():
    list_of_paths = get_folder_paths("/")

    assert "/home" in list_of_paths
    assert isinstance(list_of_paths, list)


def test_find_and_fit_peaks_with_one_peak():
    np.random.seed(0)  # For reproducibility
    x = np.linspace(0, 10, 100)
    y = gaussian(x, amplitude=1.0, centre=5.0, fwhm=1.0)

    noise = np.random.normal(0, 0.02, size=y.shape)
    y_noisy = y + noise

    peaks = find_and_fit_peaks(x, y_noisy)
    assert np.isclose(peaks[0].centre, 5.0, atol=0.1)

    print(peaks)


def test_find_and_fit_peaks_with_n_peaks():
    np.random.seed(0)  # For reproducibility
    x = np.linspace(0, 100, 1000)

    y_intensity = np.zeros_like(x)

    for n, peak_cen in enumerate(np.linspace(20, 80, 4)):
        peak_intensity = gaussian(x, amplitude=n, centre=peak_cen, fwhm=1.0)
        y_intensity = y_intensity + peak_intensity

    noise = np.random.normal(0, 0.02, size=y_intensity.shape)
    y_noisy = y_intensity + noise

    peaks = find_and_fit_peaks(x, y_noisy)

    assert len(peaks) == 3


if __name__ == "__main__":
    test_find_and_fit_peaks_with_n_peaks()
