import os
import re
from pathlib import Path


def nexus_file_match(str_to_match, beamline: str = "i15-1"):
    return re.match(f"{beamline}" + r"-+[0-9]+\.nxs", str_to_match)


def get_nexus_files(
    instrument_session_folder: str | Path,
    beamline: str = "i15-1",
    exclude: str = "processed",
) -> list[str]:
    """Get all final data files ending with .nxs in some folder"""

    nexus_files = [
        os.path.join(str(instrument_session_folder), f)
        for f in os.listdir(instrument_session_folder)
        if nexus_file_match(f, beamline) and (exclude not in f)
    ]
    nexus_files.sort()

    return nexus_files


def get_folder_paths(root_folder: str | Path) -> list[str]:
    """get all folder directories within another folder"""

    instrument_session_folders = [
        os.path.join(root_folder, f) for f in os.listdir(root_folder)
    ]
    instrument_session_folders.sort()

    return instrument_session_folders


def get_filenumber_from_filepath(filepath: str) -> int:
    """Try to get a filenumber from the filepath
    Raises if it doesn't fine a filenumber

    """
    filename = Path(filepath).stem

    matches = re.findall(r"\d+", filename)

    if not matches:
        raise ValueError(f"No number found in filename: {filename}")

    return int(matches[-1])


def get_instrument_session_from_filepath(filepath: str) -> str:
    """
    Extract the INSTRUMENT_SESSION component from a file path.

    Expects: /dls/BEAMLINE/data/YEAR/INSTRUMENT_SESSION/...

    Returns the instrument session string, e.g. "cm12345-1"

    Raises if the path doesn't find a instrument sesh.
    """
    parts = Path(filepath).parts

    # Regex for each relevant path segment
    beamline_re = re.compile(r"^[a-zA-Z]\d+(-\d+)?$")
    year_re = re.compile(r"^\d{4}$")
    session_re = re.compile(r"^[a-zA-Z]{2}\d+-\d+$")

    # Find 'dls' as an anchor, then walk forward from there
    try:
        dls_index = next(i for i, p in enumerate(parts) if p.lower() == "dls")
    except StopIteration as e:
        raise ValueError(f"Path does not contain a 'dls' root: {filepath!r}") from e

    try:
        beamline = parts[dls_index + 1]
        data_segment = parts[dls_index + 2]
        year = parts[dls_index + 3]
        instrument_session = parts[dls_index + 4]
    except IndexError as e:
        raise ValueError(
            f"Path is too short to contain expected structure: {filepath!r}, {e}"  # noqa
        ) from e

    if not beamline_re.match(beamline):
        raise ValueError(f"Unexpected BEAMLINE format: {beamline!r}")

    if data_segment.lower() != "data":
        raise ValueError(f"Expected 'data' segment, got: {data_segment!r}")

    if not year_re.match(year):
        raise ValueError(f"Unexpected YEAR format: {year!r}")

    if not session_re.match(instrument_session):
        raise ValueError(
            f"Unexpected INSTRUMENT_SESSION format: {instrument_session!r}"
        )

    return instrument_session
