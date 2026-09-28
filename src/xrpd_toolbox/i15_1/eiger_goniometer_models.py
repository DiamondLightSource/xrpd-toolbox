from pyFAI.goniometer import (
    GeometryTransformation,
)

GEOMETRY_VERSION = 1

# rigid arm swinging horizontally, so only rot1 changes with two_theta
RIGID_GEOMETRY_TRANSFORMATION = GeometryTransformation(
    param_names=["dist", "poni1", "poni2", "rot1_scale", "rot1_offset", "rot2", "rot3"],
    pos_names=["two_theta"],
    dist_expr="dist",
    poni1_expr="poni1",
    poni2_expr="poni2",
    # numexpr has no deg2rad
    rot1_expr="rot1_scale * (two_theta * 0.017453292519943295) + rot1_offset",
    rot2_expr="rot2",
    rot3_expr="rot3",
)


###################################################################

# rigid arm swinging horizontally, so only rot1 changes with two_theta. The
# quadratic term is there as the i15-1 arm angle isn't linear in the motor
# position (halves the model residual on i15-1-98680)
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
    # numexpr has no deg2rad
    rot1_expr="rot1_scale * (two_theta * 0.017453292519943295)"
    " + rot1_quad * (two_theta * 0.017453292519943295) ** 2"
    " + rot1_offset",
    rot2_expr="rot2",
    rot3_expr="rot3",
)


####################################################################

# numexpr has no deg2rad
_TWO_THETA = "(two_theta * 0.017453292519943295)"
# the i15-1 arm angle isn't linear in the motor position
_ARM = f"(rot1_scale * {_TWO_THETA} + rot1_quad * {_TWO_THETA} ** 2 + rot1_offset)"
_ROLL = "(rot3 - yaw)"
_W = f"(cos({_ARM}) * sin({_ROLL}) * cos(rot2) + sin({_ARM}) * sin(rot2))"

# The arm turns about a vertical axis tilted by yaw around the beam, carrying the
# detector mounted at rot2/rot3: R3(yaw) R1(arm) R3(-yaw) R3(rot3) R2(rot2).
# These are pyFAI's rot1/rot2/rot3 read back out of that matrix.
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
    rot1_expr=f"arctan2(sin({_ARM}) * cos({_ROLL}),"
    f" cos({_ARM}) * cos(rot2) + sin({_ARM}) * sin({_ROLL}) * sin(rot2))",
    rot2_expr=f"arcsin(cos({_ARM}) * sin(rot2)"
    f" - sin({_ARM}) * sin({_ROLL}) * cos(rot2))",
    rot3_expr=f"arctan2(sin(yaw) * cos({_ROLL}) * cos(rot2) + cos(yaw) * {_W},"
    f" cos(yaw) * cos({_ROLL}) * cos(rot2) - sin(yaw) * {_W})",
)


if GEOMETRY_VERSION == 1:
    GEOMETRY_TRANSFORMATION = RIGID_GEOMETRY_TRANSFORMATION
elif GEOMETRY_VERSION == 2:
    GEOMETRY_TRANSFORMATION = NON_LINEAR_GEOMETRY_TRANSFORMATION
elif GEOMETRY_VERSION == 3:
    GEOMETRY_TRANSFORMATION = YAW_GEOMETRY_TRANSFORMATION
else:
    raise ValueError("No geometry for that version")
