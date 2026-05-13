from collections.abc import Iterable
from pathlib import Path

import numpy as np
import numpy.typing as npt

# def channel_to_angle(ich, off, r, c, dir=1, p=0.05):
#     """
#     ich: channel number, 0-1280
#     off: module offset, degrees
#     r: radius, mm
#     c: center (in pixel or mm?)
#     dir: direction, 1
#     p: pixel size, mm
#     """
#     # print(off)
#     if r < 0:
#         ich = 1279 - ich
#     return off + np.degrees(
#         c * p / np.abs(r) - dir * np.arctan(p * (ich - c) / np.abs(r))
#     )

# def angle_to_channel(ang, off, r, c, dir=1, p=0.05):
#     ich = (
#         np.tan(dir * (np.radians(ang - off) - c * p / np.abs(r))) * np.abs(r) / p
#         + c
#     )
#     if r > 0:
#         return ich
#     else:
#         return 1279 - ich


def channel_to_angle(
    pixel_number: npt.NDArray[np.int_],
    centre: int | float,
    conv: int | float,
    module_offset: int | float,
    zero_offset: int | float,
) -> np.ndarray:
    module_conversions = pixel_number - centre
    module_conversions = module_conversions * conv
    module_conversions = np.arctan(module_conversions)
    raw_tth = module_offset + np.rad2deg(module_conversions) + zero_offset

    return raw_tth


def channel_to_angle_in_real_units(
    pixel_number: npt.NDArray[np.int_],
    centre: int | float,
    offset: int | float,
    beamline_offset: int | float,
    radius: int | float = 762,
    p: float = 0.05,
) -> np.ndarray:
    """
    pixel_number: channel number, usually 0-1280
    centre: centre (in pixel number - ie 1280/2)
    offset: module offset, degrees
    radius: radius, mm - approx 760
    direction: 1 or -1 depending if module is flipped or not
    p: pixel size, mm = 0.05
    """

    raw_tth = channel_to_angle(
        pixel_number, centre, (p / radius), offset, beamline_offset
    )

    return raw_tth


def calc_intial_module_conv(conv=6.5e-05) -> dict[int, float]:
    module_conv_dict = {}

    for mod in range(28):
        if mod > 13:
            module_conv_dict[mod] = -conv
        else:
            module_conv_dict[mod] = conv

    return module_conv_dict


def paired_modules():
    """
    Given a list of module numbers, return a list of (a, b) pairs such that
    a and b are paired as described: 0-27, 1-26, 2-25, ..., 13-14.
    Only pairs where both a and b are in the input list are returned.
    """

    modules = list(range(28))

    modules = np.array(modules)
    n = modules.max()
    pairs = []
    for m in modules:
        pair = n - m
        if pair in modules and m <= pair:
            pairs.append((int(m), int(pair)))

    pairs = np.array(pairs)

    return pairs


def find_pair(mod: int):
    modules_array = paired_modules()

    row, col = np.where(modules_array == mod)
    if len(row) == 0:
        return None  # value not found
    return modules_array[row[0], 1 - col[0]]


def calc_starting_module_offset(initial_module=0.45, offset=2.5) -> dict[int, float]:
    """Used for calculatign the intial centres of each of the modules"""

    module_pairs = paired_modules()
    module_offsets_dict = {}

    for n, module_pair in enumerate(module_pairs[::-1]):
        print(module_pair)

        ring_2_cen = (n * 5) + initial_module
        ring_1_cen = ring_2_cen + offset

        module_offsets_dict[int(module_pair[1])] = ring_2_cen
        module_offsets_dict[int(module_pair[0])] = ring_1_cen

    print(module_offsets_dict)

    return module_offsets_dict


def calc_starting_module_centre(initial_module=0.45, offset=2.5):
    """Used for calculatign the intial centres of each of the modules"""

    module_pairs = paired_modules()
    module_centres_dict = {}

    for n, module_pair in enumerate(module_pairs[::-1]):
        print(module_pair)

        ring_2_cen = (n * 5) + initial_module
        ring_1_cen = ring_2_cen + offset

        module_centres_dict[int(module_pair[1])] = ring_2_cen
        module_centres_dict[int(module_pair[0])] = ring_1_cen

    print(module_centres_dict)

    return module_centres_dict


def read_config(mythen3_config_filepath: str | Path) -> list[int]:
    """
    reads the config file used by SLSDet and works out what modules are currently active
    """

    enabled_modules_hostnames = []

    with open(mythen3_config_filepath) as file:
        lines = [line.rstrip() for line in file]

    for _, line in enumerate(lines):
        if line.startswith("hostname"):
            enabled_modules_hostnames = line.split()[1::]

    enabled_modules = [
        int(n_mod.rstrip()[-3::]) - 100 for n_mod in enabled_modules_hostnames
    ]

    return enabled_modules


def modules_to_pixels(modules: int | Iterable[int]):
    if isinstance(modules, int):
        pixels = slice(modules * 1280, (modules + 1) * 1280, None)
    elif isinstance(modules, Iterable):
        pixels = np.concatenate([np.arange(i * 1280, (i + 1) * 1280) for i in modules])
    else:
        raise TypeError("Must be int or iterable of ints")

    return pixels
