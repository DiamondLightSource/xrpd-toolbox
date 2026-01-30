import numpy as np
import pytest

from xrpd_toolbox.i11.mythen import MythenReductionSettings
from xrpd_toolbox.utils.utils import bin_and_propagate_errors, gaussian


@pytest.fixture
def mythen_settings():
    mythen_settings = MythenReductionSettings(
        active_modules=[1, 2, 3],
        bad_modules=[4, 5],
        bad_channel_masking=True,
        flatfield_filepath="flatfield.h5",
        apply_flatfield=False,
        modules_in_flatfield=[1, 2],
        send_to_ispyb=False,
        rebin_step=0.004,
        default_counter=0,
        edge_bad_channels=10,
        error_calc="internal",
        data_reduction_mode="step_scan",
        bad_channels_filepath="bad_channels.txt",
        angcal_filepath="angcal.txt",
    )

    return mythen_settings


def test_mythen_settings(mythen_settings: MythenReductionSettings):
    assert mythen_settings.data_reduction_mode == "step_scan"
    assert mythen_settings.rebin_step == 0.004


def test_mythen_settings_load_from_toml():
    settings = MythenReductionSettings.load_from_toml(
        "/workspaces/XRPD-Toolbox/examples/i11/mythen3_reduction_config.toml"
    )

    print(settings.rebin_step)


def test_peak_bin_and_propagate_errors():
    x1 = np.arange(0, 10, 0.1)
    y1 = gaussian(x1, amp=10.0, cen=5.0, fwhm=1.0, background=0.1)
    e1 = np.sqrt(y1)

    x2 = np.arange(0.01, 10.01, 0.1)
    y2 = gaussian(x2, amp=20.0, cen=5.0, fwhm=1.0, background=0.1)
    e2 = np.sqrt(y2)

    x_combined = np.concatenate((x1, x2))
    y_combined = np.concatenate((y1, y2))
    e_combined = np.concatenate((e1, e2))

    binned_x, binned_y, binned_e = bin_and_propagate_errors(
        x_combined, y_combined, e_combined, rebin_step=0.1, error_calc="internal"
    )

    assert len(binned_x) == len(binned_y) == len(binned_e)
    assert np.amax(binned_y) > np.amax(y1)
    assert np.amax(binned_y) > np.amax(y2)
