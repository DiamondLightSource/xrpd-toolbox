import numpy as np

from xrpd_toolbox.core import Parameter
from xrpd_toolbox.fit_engine.fitting_core import RefinementBaseModel

MIN_POSSIBLE_LATTICE = (
    0.53  # Single hydrogen atom is 0.53 angstroms so it can never be smaller than this
)


class LatticeParameter(Parameter):
    bounds: list[float] = [MIN_POSSIBLE_LATTICE, np.inf]


class Lattice(RefinementBaseModel):
    """This decribes the assymetric unit cell lattice -
    a, b, c refer to the length of the unit cell in Angstrom
    alpha, beta, gamma are the angles of the unit cell in degreee"""

    a: Parameter | float  # in Angstrom
    b: Parameter | float
    c: Parameter | float
    alpha: Parameter | float  # in degrees
    beta: Parameter | float  # in degrees
    gamma: Parameter | float  # in degrees

    @property
    def alpha_radians(self):
        return np.deg2rad(self.alpha)

    @property
    def beta_radians(self):
        return np.deg2rad(self.beta)

    @property
    def gamma_radians(self):
        return np.deg2rad(self.gamma)

    @property
    def matrix(self) -> np.ndarray:
        """
        Returns 3x3 lattice matrix (fractional → Cartesian)
        """
        a, b, c = self.a, self.b, self.c
        alpha = self.alpha_radians
        beta = self.beta_radians
        gamma = self.gamma_radians

        # lattice vectors
        a_vec = np.array([a, 0.0, 0.0])
        b_vec = np.array([b * np.cos(gamma), b * np.sin(gamma), 0.0])

        cx = c * np.cos(beta)
        cy = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
        cz = np.sqrt(max(c**2 - cx**2 - cy**2, 0.0))

        c_vec = np.array([cx, cy, cz])

        return np.vstack([a_vec, b_vec, c_vec])


class TriclinicLattice(Lattice):
    """This is the most general case of the lattice
    where a, b, c and alpha, beta, gamma can be any value"""

    def __init__(
        self,
        a: float,
        b: float,
        c: float,
        alpha: float,
        beta: float,
        gamma: float,
        parameterise: bool = True,
        **kwargs,
    ):
        a_param = LatticeParameter(value=a, refine=True)
        b_param = LatticeParameter(value=b, refine=True)
        c_param = LatticeParameter(value=c, refine=True)

        alpha_param = Parameter(value=alpha, refine=True)
        beta_param = Parameter(value=beta, refine=True)
        gamma_param = Parameter(value=gamma, refine=True)

        super().__init__(
            a=a_param,
            b=b_param,
            c=c_param,
            alpha=alpha_param,
            beta=beta_param,
            gamma=gamma_param,
        )


class MonoclinicLattice(Lattice):
    """This is a special case of the lattice where alpha = gamma = 90"""

    def __init__(
        self,
        a: float,
        b: float,
        c: float,
        beta: float,
        parameterise: bool = True,
        **kwargs,
    ):
        a_param = LatticeParameter(value=a, refine=True)
        b_param = LatticeParameter(value=b, refine=True)
        c_param = LatticeParameter(value=c, refine=True)

        beta_param = Parameter(value=beta, refine=True)

        # right_angle = Parameter(value=90, refine=False)

        super().__init__(
            a=a_param,
            b=b_param,
            c=c_param,
            alpha=90,
            beta=beta_param,
            gamma=90,
        )


class OrthorhombicLattice(Lattice):
    """This is a special case of the lattice
    where alpha = beta = gamma = 90"""

    def __init__(
        self, a: float, b: float, c: float, parameterise: bool = True, **kwargs
    ):
        a_param = LatticeParameter(value=a, refine=True)
        b_param = LatticeParameter(value=b, refine=True)
        c_param = LatticeParameter(value=c, refine=True)

        # right_angle = Parameter(value=90, refine=False)

        super().__init__(
            a=a_param,
            b=b_param,
            c=c_param,
            alpha=90,
            beta=90,
            gamma=90,
        )


class TetragonalLattice(Lattice):
    """This is a special case of the lattice
    where a = b and alpha = beta = gamma = 90"""

    def __init__(self, a: float, c: float, parameterise: bool = True, **kwargs):
        a_param = LatticeParameter(value=a, refine=True)
        c_param = LatticeParameter(value=c, refine=True)

        # right_angle = Parameter(value=90, refine=False)

        super().__init__(
            a=a_param,
            b=a_param,
            c=c_param,
            alpha=90,
            beta=90,
            gamma=90,
        )


class TrigonalLattice(Lattice):
    """This is a special case of the lattice
    where a = b and alpha = beta = 90 and gamma = 120"""

    def __init__(self, a: float, c: float, parameterise: bool = True, **kwargs):
        a_param = LatticeParameter(value=a, refine=True)
        c_param = LatticeParameter(value=c, refine=True)
        # right_angle = Parameter(value=90, refine=False)
        # one_twenty_angle = Parameter(value=120, refine=False)

        super().__init__(
            a=a_param,
            b=a_param,
            c=c_param,
            alpha=90,
            beta=90,
            gamma=120,
        )


class HexagonalLattice(Lattice):
    """This is a special case of the lattice
    where a = b and gamma = 120 and alpha = beta = 90"""

    def __init__(self, a: float, c: float, parameterise: bool = True, **kwargs):
        a_param = LatticeParameter(value=a, refine=True)
        c_param = LatticeParameter(value=c, refine=True)
        # right_angle = Parameter(value=90, refine=False)
        # one_twenty_angle = Parameter(value=120, refine=False)

        super().__init__(
            a=a_param,
            b=a_param,
            c=c_param,
            alpha=90,
            beta=90,
            gamma=120,
        )


class RhombohedralLattice(Lattice):
    """This is a special case of the lattice
    where a = b = c and alpha = beta = gamma"""

    def __init__(self, a: float, alpha: float, parameterise: bool = True, **kwargs):
        a_param = LatticeParameter(value=a, refine=True)
        alpha_param = Parameter(value=alpha, refine=True)

        super().__init__(
            a=a_param,
            b=a_param,
            c=a_param,
            alpha=alpha_param,
            beta=alpha_param,
            gamma=alpha_param,
        )


class CubicLattice(Lattice):
    def __init__(self, a: float, parameterise: bool = True, **kwargs):
        a_param = LatticeParameter(value=a, refine=True)

        super().__init__(
            a=a_param,
            b=a_param,
            c=a_param,
            alpha=90,
            beta=90,
            gamma=90,
        )


def crystal_lattice_factory(crystal_class: str):
    class_str = str(crystal_class).lower()

    if class_str == "cubic":
        return CubicLattice
    elif class_str == "hexagonal":
        return HexagonalLattice
    elif class_str == "monoclinic":
        return MonoclinicLattice
    elif class_str == "orthorhombic":
        return OrthorhombicLattice
    elif class_str == "rhombohedral":
        return RhombohedralLattice
    elif class_str == "tetragonal":
        return TetragonalLattice
    elif class_str == "trigonal":
        return TrigonalLattice
    elif class_str == "triclinic":
        return TriclinicLattice
    else:
        raise ValueError(f"{crystal_class} unknown")


if __name__ == "__main__":
    cl = CubicLattice(a=5)
    # cl.parameterise()

    print(cl)

    params = cl.get_refinement_parameters()

    print(params)

    params["a"] = 5.5

    cl.set_refinement_parameters(params)

    print(cl)

    model_dict = cl.model_dump()

    # print(cl.get_refinable_parameters())

    cl2 = Lattice.model_validate(model_dict)

    print(cl2)
