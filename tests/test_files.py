import pytest

from xrpd_toolbox.utils.files import (
    get_filenumber_from_filepath,
    get_folder_paths,
    get_instrument_session_from_filepath,
    get_nexus_files,
    nexus_file_match,
)


def test_nexus_file_match_matches_valid_name():
    assert nexus_file_match("i15-1-99999.nxs", beamline="i15-1") is not None


def test_nexus_file_match_rejects_invalid_name():
    assert nexus_file_match("not_a_nexus_file.txt", beamline="i15-1") is None


def test_get_nexus_files_filters_and_sorts(tmp_path):
    for name in ["i15-1-2.nxs", "i15-1-1.nxs", "i15-1-1_processed.nxs"]:
        (tmp_path / name).write_text("")
    (tmp_path / "other.txt").write_text("")

    result = get_nexus_files(tmp_path, beamline="i15-1")

    assert result == [
        str(tmp_path / "i15-1-1.nxs"),
        str(tmp_path / "i15-1-2.nxs"),
    ]


def test_get_nexus_files_returns_empty_list_when_no_match(tmp_path):
    (tmp_path / "unrelated.txt").write_text("")

    assert get_nexus_files(tmp_path, beamline="i15-1") == []


def test_get_nexus_files_custom_exclude(tmp_path):
    for name in ["i15-1-1.nxs", "i15-1-2_skipme.nxs"]:
        (tmp_path / name).write_text("")

    result = get_nexus_files(tmp_path, beamline="i15-1", exclude="skipme")

    assert result == [str(tmp_path / "i15-1-1.nxs")]


def test_get_folder_paths_returns_sorted_paths(tmp_path):
    for name in ["b", "a", "c"]:
        (tmp_path / name).mkdir()

    result = get_folder_paths(str(tmp_path))

    assert result == [
        str(tmp_path / "a"),
        str(tmp_path / "b"),
        str(tmp_path / "c"),
    ]


def test_get_filenumber_from_filepath_returns_last_number():
    assert (
        get_filenumber_from_filepath("/dls/i11/test/cm12345-1/i11-99999.nxs") == 99999
    )


def test_get_filenumber_from_filepath_raises_when_no_number():
    with pytest.raises(ValueError, match="No number found in filename"):
        get_filenumber_from_filepath("/dls/i11/test/cm12345-1/no_number_here.nxs")


def test_get_instrument_session_from_filepath_valid():
    filepath = "/dls/i15-1/data/2024/cm12345-1/i15-1-1.nxs"
    assert get_instrument_session_from_filepath(filepath) == "cm12345-1"


def test_get_instrument_session_from_filepath_raises_without_dls_root():
    with pytest.raises(ValueError, match="does not contain a 'dls' root"):
        get_instrument_session_from_filepath("/not/a/matching/path/file.nxs")


def test_get_instrument_session_from_filepath_raises_when_too_short():
    with pytest.raises(ValueError, match="too short"):
        get_instrument_session_from_filepath("/dls/i15-1/data")


def test_get_instrument_session_from_filepath_raises_on_bad_beamline():
    filepath = "/dls/notabeamline/data/2024/cm12345-1/file.nxs"
    with pytest.raises(ValueError, match="Unexpected BEAMLINE format"):
        get_instrument_session_from_filepath(filepath)


def test_get_instrument_session_from_filepath_raises_on_bad_data_segment():
    filepath = "/dls/i15-1/notdata/2024/cm12345-1/file.nxs"
    with pytest.raises(ValueError, match="Expected 'data' segment"):
        get_instrument_session_from_filepath(filepath)


def test_get_instrument_session_from_filepath_raises_on_bad_year():
    filepath = "/dls/i15-1/data/notayear/cm12345-1/file.nxs"
    with pytest.raises(ValueError, match="Unexpected YEAR format"):
        get_instrument_session_from_filepath(filepath)


def test_get_instrument_session_from_filepath_raises_on_bad_session():
    filepath = "/dls/i15-1/data/2024/notasession/file.nxs"
    with pytest.raises(ValueError, match="Unexpected INSTRUMENT_SESSION format"):
        get_instrument_session_from_filepath(filepath)
