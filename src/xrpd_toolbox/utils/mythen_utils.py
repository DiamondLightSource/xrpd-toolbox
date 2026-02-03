import numpy as np
import numpy.typing as npt


def channel_to_angle(
    pixel_number: npt.NDArray[np.int_],
    centre: int | float,
    conv: int | float,
    offset: int | float,
    beamline_offset: int | float,
):
    module_conversions = pixel_number - centre
    module_conversions = module_conversions * conv
    module_conversions = np.arctan(module_conversions)
    raw_tth = offset + np.rad2deg(module_conversions) + beamline_offset

    return raw_tth


def channel_to_angle_in_real_units(
    pixel_number: npt.NDArray[np.int_],
    centre: int | float,
    offset: int | float,
    beamline_offset: int | float,
    radius: int | float = 762,
    p: float = 0.05,
):
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


def calc_intial_module_conv(conv=6.5e-05):
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


def calc_starting_module_offset(initial_module=0.45, offset=2.5):
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
