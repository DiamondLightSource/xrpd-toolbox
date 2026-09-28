"""Goniometer models for the i15-1 Eiger on its two-theta arm.

Each model turns the arm's motor position (two_theta, in degrees) into the six
pyFAI parameters for that frame. GEOMETRY_VERSION picks the one that's used.
"""

from pyFAI.goniometer import GeometryTransformation

# numexpr can't call np.deg2rad
TWO_THETA_RAD = "(two_theta * 0.017453292519943295)"


# 1. Rigid arm: only rot1 changes, in proportion to two_theta.

RIGID_GEOMETRY_TRANSFORMATION = GeometryTransformation(
    param_names=["dist", "poni1", "poni2", "rot1_scale", "rot1_offset", "rot2", "rot3"],
    pos_names=["two_theta"],
    dist_expr="dist",
    poni1_expr="poni1",
    poni2_expr="poni2",
    rot1_expr=f"rot1_scale * {TWO_THETA_RAD} + rot1_offset",
    rot2_expr="rot2",
    rot3_expr="rot3",
)


# 2. As 1, but the arm angle needn't be linear in the motor position.

NON_LINEAR_GEOMETRY_TRANSFORMATION = GeometryTransformation(
    param_names=[
        "dist",
        "poni1",
        "poni2",
        "rot1_scale",
        "rot1_quad",
        "rot1_offset",
        "rot2",
        "rot3",
    ],
    pos_names=["two_theta"],
    dist_expr="dist",
    poni1_expr="poni1",
    poni2_expr="poni2",
    rot1_expr=f"rot1_scale * {TWO_THETA_RAD}"
    f" + rot1_quad * {TWO_THETA_RAD} ** 2 + rot1_offset",
    rot2_expr="rot2",
    rot3_expr="rot3",
)


# Models 3-6 treat the detector as a rigid body carried round by the arm, so its
# orientation, detector_ij, is (arm rotation) x (detector mount on the arm). numexpr
# has no matrices, so the elements are written out one by one (row i, column j).
# The mount's row 3, column 2 is always 0, so mount_32 is left out.
#
# pyFAI builds an orientation as R3(rot3) R2(rot2) R1(rot1), so its angles are
#   rot1 = arctan2(-detector_32, detector_33)
#   rot2 = arcsin(detector_31)
#   rot3 = arctan2(detector_21, detector_11)
#
# Turning by `arm` about the unit axis (axis_1, axis_2, axis_3), in the same sense
# as pyFAI's rot1, is cos(arm) I + (1 - cos(arm)) axis axis^T - sin(arm) [axis]x


# 3. The arm's axis leans around the beam by yaw, so the rings drift up or down as
# the arm turns. The detector is mounted on the arm at rot2/rot3.

arm = f"(rot1_scale * {TWO_THETA_RAD} + rot1_quad * {TWO_THETA_RAD} ** 2 + rot1_offset)"
cos_arm = f"cos({arm})"
sin_arm = f"sin({arm})"
one_minus_cos_arm = f"(1 - cos({arm}))"

axis_1 = "cos(yaw)"
axis_2 = "sin(yaw)"

arm_11 = f"({cos_arm} + {one_minus_cos_arm} * {axis_1} ** 2)"
arm_12 = f"({one_minus_cos_arm} * {axis_1} * {axis_2})"
arm_13 = f"(-{sin_arm} * {axis_2})"
arm_21 = f"({one_minus_cos_arm} * {axis_1} * {axis_2})"
arm_22 = f"({cos_arm} + {one_minus_cos_arm} * {axis_2} ** 2)"
arm_23 = f"({sin_arm} * {axis_1})"
arm_31 = f"({sin_arm} * {axis_2})"
arm_32 = f"(-{sin_arm} * {axis_1})"
arm_33 = cos_arm

mount_11 = "(cos(rot3) * cos(rot2))"
mount_12 = "(-sin(rot3))"
mount_13 = "(-cos(rot3) * sin(rot2))"
mount_21 = "(sin(rot3) * cos(rot2))"
mount_22 = "cos(rot3)"
mount_23 = "(-sin(rot3) * sin(rot2))"
mount_31 = "sin(rot2)"
mount_33 = "cos(rot2)"

detector_11 = f"({arm_11} * {mount_11} + {arm_12} * {mount_21} + {arm_13} * {mount_31})"
detector_21 = f"({arm_21} * {mount_11} + {arm_22} * {mount_21} + {arm_23} * {mount_31})"
detector_31 = f"({arm_31} * {mount_11} + {arm_32} * {mount_21} + {arm_33} * {mount_31})"
detector_32 = f"({arm_31} * {mount_12} + {arm_32} * {mount_22})"
detector_33 = f"({arm_31} * {mount_13} + {arm_32} * {mount_23} + {arm_33} * {mount_33})"

YAW_GEOMETRY_TRANSFORMATION = GeometryTransformation(
    param_names=[
        "dist",
        "poni1",
        "poni2",
        "rot1_scale",
        "rot1_quad",
        "rot1_offset",
        "rot2",
        "rot3",
        "yaw",
    ],
    pos_names=["two_theta"],
    dist_expr="dist",
    poni1_expr="poni1",
    poni2_expr="poni2",
    rot1_expr=f"arctan2(-{detector_32}, {detector_33})",
    rot2_expr=f"arcsin({detector_31})",
    rot3_expr=f"arctan2({detector_21}, {detector_11})",
)


# 4. The sample is off the arm's centre of rotation, so dist and the PONI change as
# the arm turns. The detector is mounted with a pitch and roll; a yaw on the arm
# would be the same thing as rot1_offset.

arm = f"(rot1_scale * {TWO_THETA_RAD} + rot1_offset)"
cos_arm = f"cos({arm})"
sin_arm = f"sin({arm})"

mount_11 = "(cos(roll) * cos(pitch))"
mount_12 = "(-sin(roll))"
mount_13 = "(-cos(roll) * sin(pitch))"
mount_21 = "(sin(roll) * cos(pitch))"
mount_22 = "cos(roll)"
mount_23 = "(-sin(roll) * sin(pitch))"
mount_31 = "sin(pitch)"
mount_33 = "cos(pitch)"

# the arm turns about the vertical axis, so the first row of the mount is unchanged
detector_11 = mount_11
detector_12 = mount_12
detector_13 = mount_13
detector_21 = f"({cos_arm} * {mount_21} + {sin_arm} * {mount_31})"
detector_22 = f"({cos_arm} * {mount_22})"
detector_23 = f"({cos_arm} * {mount_23} + {sin_arm} * {mount_33})"
detector_31 = f"(-{sin_arm} * {mount_21} + {cos_arm} * {mount_31})"
detector_32 = f"(-{sin_arm} * {mount_22})"
detector_33 = f"(-{sin_arm} * {mount_23} + {cos_arm} * {mount_33})"

# the sample offset as the detector sees it (its orientation transposed x offset),
# with pyFAI's axis 1 up, 2 across the beam and 3 along it
offset_1 = (
    f"({detector_11} * sample_y + {detector_21} * sample_x + {detector_31} * sample_z)"
)
offset_2 = (
    f"({detector_12} * sample_y + {detector_22} * sample_x + {detector_32} * sample_z)"
)
offset_3 = (
    f"({detector_13} * sample_y + {detector_23} * sample_x + {detector_33} * sample_z)"
)

DISPLACED_GEOMETRY_TRANSFORMATION = GeometryTransformation(
    param_names=[
        "dist",
        "poni1",
        "poni2",
        "rot1_scale",
        "rot1_offset",
        "pitch",
        "roll",
        "sample_x",
        "sample_y",
        "sample_z",
    ],
    pos_names=["two_theta"],
    dist_expr=f"dist - {offset_3}",
    poni1_expr=f"poni1 + {offset_1}",
    poni2_expr=f"poni2 + {offset_2}",
    rot1_expr=f"arctan2(-{detector_32}, {detector_33})",
    rot2_expr=f"arcsin({detector_31})",
    rot3_expr=f"arctan2({detector_21}, {detector_11})",
)


# 5. The arm's axis leans around the beam (axis_lean_beam) and across it
# (axis_lean_x), so some of each two_theta step goes into rot2 and rot3.

arm = f"(rot1_scale * {TWO_THETA_RAD} + rot1_offset)"
cos_arm = f"cos({arm})"
sin_arm = f"sin({arm})"
one_minus_cos_arm = f"(1 - cos({arm}))"

axis_1 = "(cos(axis_lean_beam) * cos(axis_lean_x))"
axis_2 = "(sin(axis_lean_beam) * cos(axis_lean_x))"
axis_3 = "sin(axis_lean_x)"

arm_11 = f"({cos_arm} + {one_minus_cos_arm} * {axis_1} ** 2)"
arm_12 = f"({one_minus_cos_arm} * {axis_1} * {axis_2} + {sin_arm} * {axis_3})"
arm_13 = f"({one_minus_cos_arm} * {axis_1} * {axis_3} - {sin_arm} * {axis_2})"
arm_21 = f"({one_minus_cos_arm} * {axis_1} * {axis_2} - {sin_arm} * {axis_3})"
arm_22 = f"({cos_arm} + {one_minus_cos_arm} * {axis_2} ** 2)"
arm_23 = f"({one_minus_cos_arm} * {axis_2} * {axis_3} + {sin_arm} * {axis_1})"
arm_31 = f"({one_minus_cos_arm} * {axis_1} * {axis_3} + {sin_arm} * {axis_2})"
arm_32 = f"({one_minus_cos_arm} * {axis_2} * {axis_3} - {sin_arm} * {axis_1})"
arm_33 = f"({cos_arm} + {one_minus_cos_arm} * {axis_3} ** 2)"

mount_11 = "(cos(rot3) * cos(rot2))"
mount_12 = "(-sin(rot3))"
mount_13 = "(-cos(rot3) * sin(rot2))"
mount_21 = "(sin(rot3) * cos(rot2))"
mount_22 = "cos(rot3)"
mount_23 = "(-sin(rot3) * sin(rot2))"
mount_31 = "sin(rot2)"
mount_33 = "cos(rot2)"

detector_11 = f"({arm_11} * {mount_11} + {arm_12} * {mount_21} + {arm_13} * {mount_31})"
detector_21 = f"({arm_21} * {mount_11} + {arm_22} * {mount_21} + {arm_23} * {mount_31})"
detector_31 = f"({arm_31} * {mount_11} + {arm_32} * {mount_21} + {arm_33} * {mount_31})"
detector_32 = f"({arm_31} * {mount_12} + {arm_32} * {mount_22})"
detector_33 = f"({arm_31} * {mount_13} + {arm_32} * {mount_23} + {arm_33} * {mount_33})"

TILTED_AXIS_GEOMETRY_TRANSFORMATION = GeometryTransformation(
    param_names=[
        "dist",
        "poni1",
        "poni2",
        "rot1_scale",
        "rot1_offset",
        "rot2",
        "rot3",
        "axis_lean_beam",
        "axis_lean_x",
    ],
    pos_names=["two_theta"],
    dist_expr="dist",
    poni1_expr="poni1",
    poni2_expr="poni2",
    rot1_expr=f"arctan2(-{detector_32}, {detector_33})",
    rot2_expr=f"arcsin({detector_31})",
    rot3_expr=f"arctan2({detector_21}, {detector_11})",
)


# 6. Everything: a non-linear arm turning about a leaning axis, the detector
# mounted at rot2/rot3 and the sample off the centre of rotation. On a short scan
# sample_y looks like poni1 and axis_lean_beam like rot3, so those trade off.

arm = f"(rot1_scale * {TWO_THETA_RAD} + rot1_quad * {TWO_THETA_RAD} ** 2 + rot1_offset)"
cos_arm = f"cos({arm})"
sin_arm = f"sin({arm})"
one_minus_cos_arm = f"(1 - cos({arm}))"

axis_1 = "(cos(axis_lean_beam) * cos(axis_lean_x))"
axis_2 = "(sin(axis_lean_beam) * cos(axis_lean_x))"
axis_3 = "sin(axis_lean_x)"

arm_11 = f"({cos_arm} + {one_minus_cos_arm} * {axis_1} ** 2)"
arm_12 = f"({one_minus_cos_arm} * {axis_1} * {axis_2} + {sin_arm} * {axis_3})"
arm_13 = f"({one_minus_cos_arm} * {axis_1} * {axis_3} - {sin_arm} * {axis_2})"
arm_21 = f"({one_minus_cos_arm} * {axis_1} * {axis_2} - {sin_arm} * {axis_3})"
arm_22 = f"({cos_arm} + {one_minus_cos_arm} * {axis_2} ** 2)"
arm_23 = f"({one_minus_cos_arm} * {axis_2} * {axis_3} + {sin_arm} * {axis_1})"
arm_31 = f"({one_minus_cos_arm} * {axis_1} * {axis_3} + {sin_arm} * {axis_2})"
arm_32 = f"({one_minus_cos_arm} * {axis_2} * {axis_3} - {sin_arm} * {axis_1})"
arm_33 = f"({cos_arm} + {one_minus_cos_arm} * {axis_3} ** 2)"

mount_11 = "(cos(rot3) * cos(rot2))"
mount_12 = "(-sin(rot3))"
mount_13 = "(-cos(rot3) * sin(rot2))"
mount_21 = "(sin(rot3) * cos(rot2))"
mount_22 = "cos(rot3)"
mount_23 = "(-sin(rot3) * sin(rot2))"
mount_31 = "sin(rot2)"
mount_33 = "cos(rot2)"

detector_11 = f"({arm_11} * {mount_11} + {arm_12} * {mount_21} + {arm_13} * {mount_31})"
detector_12 = f"({arm_11} * {mount_12} + {arm_12} * {mount_22})"
detector_13 = f"({arm_11} * {mount_13} + {arm_12} * {mount_23} + {arm_13} * {mount_33})"
detector_21 = f"({arm_21} * {mount_11} + {arm_22} * {mount_21} + {arm_23} * {mount_31})"
detector_22 = f"({arm_21} * {mount_12} + {arm_22} * {mount_22})"
detector_23 = f"({arm_21} * {mount_13} + {arm_22} * {mount_23} + {arm_23} * {mount_33})"
detector_31 = f"({arm_31} * {mount_11} + {arm_32} * {mount_21} + {arm_33} * {mount_31})"
detector_32 = f"({arm_31} * {mount_12} + {arm_32} * {mount_22})"
detector_33 = f"({arm_31} * {mount_13} + {arm_32} * {mount_23} + {arm_33} * {mount_33})"

# the sample offset as the detector sees it (its orientation transposed x offset),
# with pyFAI's axis 1 up, 2 across the beam and 3 along it
offset_1 = (
    f"({detector_11} * sample_y + {detector_21} * sample_x + {detector_31} * sample_z)"
)
offset_2 = (
    f"({detector_12} * sample_y + {detector_22} * sample_x + {detector_32} * sample_z)"
)
offset_3 = (
    f"({detector_13} * sample_y + {detector_23} * sample_x + {detector_33} * sample_z)"
)

FULL_GEOMETRY_TRANSFORMATION = GeometryTransformation(
    param_names=[
        "dist",
        "poni1",
        "poni2",
        "rot1_scale",
        "rot1_quad",
        "rot1_offset",
        "rot2",
        "rot3",
        "axis_lean_beam",
        "axis_lean_x",
        "sample_x",
        "sample_y",
        "sample_z",
    ],
    pos_names=["two_theta"],
    dist_expr=f"dist - {offset_3}",
    poni1_expr=f"poni1 + {offset_1}",
    poni2_expr=f"poni2 + {offset_2}",
    rot1_expr=f"arctan2(-{detector_32}, {detector_33})",
    rot2_expr=f"arcsin({detector_31})",
    rot3_expr=f"arctan2({detector_21}, {detector_11})",
)


# 7. Model 6 cut down to what the i15-1-98680 calibration could pin down. With the
# sample offset modelled the motor's scale comes out exact, so it's fixed at -1 (the
# arm turns the opposite way to pyFAI's rot1). The arm's axis is taken as vertical,
# as its lean fitted to 0, and sample_y is left out because along that axis it
# can't be told apart from poni1.

arm = f"(rot1_offset - {TWO_THETA_RAD})"
cos_arm = f"cos({arm})"
sin_arm = f"sin({arm})"

mount_11 = "(cos(rot3) * cos(rot2))"
mount_12 = "(-sin(rot3))"
mount_13 = "(-cos(rot3) * sin(rot2))"
mount_21 = "(sin(rot3) * cos(rot2))"
mount_22 = "cos(rot3)"
mount_23 = "(-sin(rot3) * sin(rot2))"
mount_31 = "sin(rot2)"
mount_33 = "cos(rot2)"

# the arm turns about the vertical axis, so the first row of the mount is unchanged
detector_11 = mount_11
detector_12 = mount_12
detector_13 = mount_13
detector_21 = f"({cos_arm} * {mount_21} + {sin_arm} * {mount_31})"
detector_22 = f"({cos_arm} * {mount_22})"
detector_23 = f"({cos_arm} * {mount_23} + {sin_arm} * {mount_33})"
detector_31 = f"(-{sin_arm} * {mount_21} + {cos_arm} * {mount_31})"
detector_32 = f"(-{sin_arm} * {mount_22})"
detector_33 = f"(-{sin_arm} * {mount_23} + {cos_arm} * {mount_33})"

# the sample offset as the detector sees it, now only across and along the beam
offset_1 = f"({detector_21} * sample_x + {detector_31} * sample_z)"
offset_2 = f"({detector_22} * sample_x + {detector_32} * sample_z)"
offset_3 = f"({detector_23} * sample_x + {detector_33} * sample_z)"

SIMPLIFIED_GEOMETRY_TRANSFORMATION = GeometryTransformation(
    param_names=[
        "dist",
        "poni1",
        "poni2",
        "rot1_offset",
        "rot2",
        "rot3",
        "sample_x",
        "sample_z",
    ],
    pos_names=["two_theta"],
    dist_expr=f"dist - {offset_3}",
    poni1_expr=f"poni1 + {offset_1}",
    poni2_expr=f"poni2 + {offset_2}",
    rot1_expr=f"arctan2(-{detector_32}, {detector_33})",
    rot2_expr=f"arcsin({detector_31})",
    rot3_expr=f"arctan2({detector_21}, {detector_11})",
)


GEOMETRY_TRANSFORMATIONS = {
    1: RIGID_GEOMETRY_TRANSFORMATION,  # simplest - gets worse at high angles
    2: NON_LINEAR_GEOMETRY_TRANSFORMATION,
    3: YAW_GEOMETRY_TRANSFORMATION,
    4: DISPLACED_GEOMETRY_TRANSFORMATION,
    5: TILTED_AXIS_GEOMETRY_TRANSFORMATION,
    6: FULL_GEOMETRY_TRANSFORMATION,  # complete - overfitting
    7: SIMPLIFIED_GEOMETRY_TRANSFORMATION,  # best - only uses what actually changes in FULL_GEOMETRY_TRANSFORMATION #noqa
}


##############
GEOMETRY_VERSION = 7


if GEOMETRY_VERSION not in GEOMETRY_TRANSFORMATIONS:
    raise ValueError(f"No goniometer model for GEOMETRY_VERSION {GEOMETRY_VERSION}")

GEOMETRY_TRANSFORMATION = GEOMETRY_TRANSFORMATIONS[GEOMETRY_VERSION]
