import os
from pathlib import Path

import pytest

from xrpd_toolbox import BASE_PATH
from xrpd_toolbox.core import XYEData
from xrpd_toolbox.i11.mythen import MythenDetector, MythenSettings

CONFIG_FILE = (
    Path(__file__).parent.parent / "config" / "i11" / "mythen3_reduction_config.toml"
)

LEGANCY_CONIG = Path(__file__).parent.parent / "config" / "i11" / "legacy_config.toml"


SI_DATA_FILE = BASE_PATH.parent.parent / "tests" / "data" / "1410696.nxs"


@pytest.fixture
def mythen_settings():
    mythen_settings = MythenSettings(
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
        error_calc="poisson",
        data_reduction_mode="step_scan",
        bad_channels_filepath="bad_channels.txt",
        angcal_filepath="angcal.txt",
    )

    return mythen_settings


def test_mythen_settings(mythen_settings: MythenSettings):
    assert mythen_settings.data_reduction_mode == "step_scan"
    assert mythen_settings.rebin_step == 0.004


def test_mythen_settings_load_from_toml():
    settings = MythenSettings.load_from_toml(CONFIG_FILE)

    assert isinstance(settings, MythenSettings)


def test_mythen_toml_save_load(mythen_settings: MythenSettings):
    file_path = "file.toml"

    mythen_settings.save_to_toml(file_path)
    loaded_mythen_settings = MythenSettings.load_from_toml(file_path)

    assert mythen_settings == loaded_mythen_settings

    os.remove(file_path)


def test_mythen_yaml_save_load(mythen_settings: MythenSettings):
    file_path = "file.yaml"

    mythen_settings.save_to_yaml(file_path)
    loaded_mythen_settings = MythenSettings.load_from_yaml(file_path)

    assert mythen_settings == loaded_mythen_settings

    os.remove(file_path)


def test_mythen_load_fails_when_incorrect_file_extension(
    mythen_settings: MythenSettings,
):
    file_path = "file.txt"

    with pytest.raises(ValueError):
        mythen_settings.save_to_yaml(file_path)

    with pytest.raises(ValueError):
        mythen_settings.save_to_toml(file_path)


def test_mythen_data_reduction():

    detector = MythenDetector(filepath=SI_DATA_FILE, filename_suffix="_test")

    detector.process_step_scan(control=False)

    assert str(detector.output_directory) == str(SI_DATA_FILE.parent / "processed")
    assert Path(detector.xye_filepath_out).exists()
    assert str(Path(detector.xye_filepath_out).parent) == str(
        SI_DATA_FILE.parent / "processed"
    )

    _ = XYEData.from_csv(detector.xye_filepath_out)

    os.remove(detector.xye_filepath_out)


def test_data_reduction_mode_validation():
    mythen_settings = MythenSettings(
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
        data_reduction_mode=0,  # type: ignore
        bad_channels_filepath="bad_channels.txt",
        angcal_filepath="angcal.off",
    )

    assert mythen_settings.data_reduction_mode == "step_scan"
    assert str(mythen_settings.angcal_filepath).endswith(".json")


def test_legacy_toml():
    legacy_setting = MythenSettings.load(LEGANCY_CONIG)
    assert legacy_setting
