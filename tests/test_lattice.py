import numpy as np
import pytest

from xrpd_toolbox.fit_engine.lattice import (
    CubicLattice,
    HexagonalLattice,
    Lattice,
    MonoclinicLattice,
    OrthorhombicLattice,
    RhombohedralLattice,
    TetragonalLattice,
    TriclinicLattice,
    TrigonalLattice,
    crystal_lattice_factory,
)


def test_cubic_lattice():
    cl = CubicLattice(a=5)
    params = cl.get_refinement_parameters()
    params["a"] = 5.5
    cl.set_refinement_parameters(params)

    params = cl.get_refinement_parameters()

    assert params["a"] == 5.5

    model_dict = cl.model_dump()
    cl2 = Lattice(**model_dict)

    assert isinstance(cl2, Lattice)


def test_lattice_radian_conversions_and_matrix():
    lattice = Lattice(a=3.0, b=4.0, c=5.0, alpha=60.0, beta=70.0, gamma=80.0)

    assert np.isclose(lattice.alpha_radians, np.deg2rad(60.0))
    assert np.isclose(lattice.beta_radians, np.deg2rad(70.0))
    assert np.isclose(lattice.gamma_radians, np.deg2rad(80.0))

    matrix = lattice.matrix
    assert matrix.shape == (3, 3)
    assert np.isclose(matrix[0, 0], 3.0)
    assert matrix[1, 0] == pytest.approx(4.0 * np.cos(np.deg2rad(80.0)))
    assert matrix[1, 1] == pytest.approx(4.0 * np.sin(np.deg2rad(80.0)))
    assert matrix[2, 2] >= 0.0


def test_special_lattice_equalities():
    ortho = OrthorhombicLattice(a=2.0, b=3.0, c=4.0)
    assert ortho.alpha == 90
    assert ortho.beta == 90
    assert ortho.gamma == 90
    assert ortho.a == 2.0
    assert ortho.b == 3.0
    assert ortho.c == 4.0

    tetragonal = TetragonalLattice(a=2.5, c=6.0)
    assert tetragonal.a == tetragonal.b
    assert tetragonal.alpha == 90
    assert tetragonal.gamma == 90

    hexagonal = HexagonalLattice(a=2.0, c=5.0)
    assert hexagonal.a == hexagonal.b
    assert hexagonal.gamma == 120
    assert hexagonal.alpha == 90
    assert hexagonal.beta == 90

    trigonal = TrigonalLattice(a=2.5, c=5.0)
    assert trigonal.a == trigonal.b
    assert trigonal.gamma == 120
    assert trigonal.alpha == 90
    assert trigonal.beta == 90

    rhombo = RhombohedralLattice(a=4.0, alpha=100.0)
    assert rhombo.a == rhombo.b == rhombo.c
    assert rhombo.alpha == rhombo.beta == rhombo.gamma

    mono = MonoclinicLattice(a=3.0, b=4.0, c=5.0, beta=110.0)
    assert mono.alpha == 90
    assert mono.gamma == 90
    assert mono.beta == 110.0


@pytest.mark.parametrize(
    "crystal_class,expected_type",
    [
        ("cubic", CubicLattice),
        ("hexagonal", HexagonalLattice),
        ("monoclinic", MonoclinicLattice),
        ("orthorhombic", OrthorhombicLattice),
        ("rhombohedral", RhombohedralLattice),
        ("tetragonal", TetragonalLattice),
        ("trigonal", TrigonalLattice),
        ("triclinic", TriclinicLattice),
    ],
)
def test_crystal_lattice_factory_returns_expected_class(crystal_class, expected_type):
    lattice_cls = crystal_lattice_factory(crystal_class.upper())
    assert lattice_cls is expected_type


def test_crystal_lattice_factory_raises_for_unknown_class():
    with pytest.raises(ValueError, match="unknown"):
        crystal_lattice_factory("unknown-crystal")
