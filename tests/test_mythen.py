import matplotlib.pyplot as plt
import numpy as np

from xrpd_toolbox.i11.mythen import MythenReductionSettings
from xrpd_toolbox.utils.utils import bin_and_propagate_errors, gaussian


def test_mythen_settings():
    settings = MythenReductionSettings(
        active_modules=[1, 2, 3],
        bad_modules=[4, 5],
        bad_channel_masking=True,
        flatfield_filepath="flatfield.h5",
        apply_flatfield=True,
        modules_in_flatfield=[1, 2],
        send_to_ispyb=False,
        rebin_step=2.0,
        default_counter=0,
        edge_bad_channels=10,
        error_calc="internal",
        data_reduction_mode=1,
        bad_channels_filepath="bad_channels.txt",
        angcal_filepath="angcal.txt",
    )

    assert settings.apply_flatfield is True
    assert settings.rebin_step == 2.0


def test_mythen_settings_load_from_toml():
    settings = MythenReductionSettings.load_from_toml(
        "tests/test_data/mythen_settings.toml"
    )
    assert settings.apply_flatfield is True


def test_peak_averaging():
    x1 = np.arange(0, 10, 0.1)
    y1 = gaussian(x1, amp=10.0, cen=5.0, fwhm=1.0, background=0.0)
    e1 = np.sqrt(y1)

    x2 = np.arange(0.01, 10.01, 0.1)
    y2 = gaussian(x2, amp=20.0, cen=5.0, fwhm=1.0, background=0.0)
    e2 = np.sqrt(y2)

    x_combined = np.concatenate((x1, x2))
    y_combined = np.concatenate((y1, y2))
    e_combined = np.concatenate((e1, e2))

    x, y, e = bin_and_propagate_errors(
        x_combined, y_combined, e_combined, rebin_step=0.1, error_calc="internal"
    )

    plt.errorbar(x1, y1, yerr=e1, fmt="o", label="Binned Data with Propagated Errors")
    plt.errorbar(x2, y2, yerr=e2, fmt="o", label="Binned Data with Propagated Errors")
    plt.errorbar(x - 0.05, y, yerr=e, label="Binned Data with Propagated Errors")
    plt.show()


if __name__ == "__main__":
    test_peak_averaging()
