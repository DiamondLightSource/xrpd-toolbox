from xrpd_toolbox.core import Parameter


def test_parameter_maths():
    a = Parameter(value=3, refine=True)

    x = 1.0 + a
    assert isinstance(x, float)
    assert x == 4.0
