import numpy as np

from xrpd_toolbox.utils.settings import XRPDBaseModel


class Lattice(XRPDBaseModel):
    """This decribes the assymetric unit cell lattice -
    a, b, c refer to the length of the unit cell in Angstrom
    alpha, beta, gamma are the angles of the unit cell in degreee"""

    a: float  # in Angstrom
    b: float
    c: float
    alpha: float  # in degrees
    beta: float  # in degrees
    gamma: float  # in degrees

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
        **kwargs,
    ):
        super().__init__(a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)


class MonoclinicLattice(Lattice):
    """This is a special case of the lattice where alpha = gamma = 90"""

    def __init__(self, a: float, b: float, c: float, beta: float, **kwargs):
        super().__init__(a=a, b=b, c=c, alpha=90.0, beta=beta, gamma=90.0)


class OrthorhombicLattice(Lattice):
    """This is a special case of the lattice
    where alpha = beta = gamma = 90"""

    def __init__(self, a: float, b: float, c: float, **kwargs):
        super().__init__(a=a, b=b, c=c, alpha=90.0, beta=90.0, gamma=90.0)


class TetragonalLattice(Lattice):
    """This is a special case of the lattice
    where a = b and alpha = beta = gamma = 90"""

    def __init__(self, a: float, c: float, **kwargs):
        super().__init__(a=a, b=a, c=c, alpha=90.0, beta=90.0, gamma=90.0)


class TrigonalLattice(Lattice):
    """This is a special case of the lattice
    where a = b and alpha = beta = 90 and gamma = 120"""

    def __init__(self, a: float, c: float, **kwargs):
        super().__init__(a=a, b=a, c=c, alpha=90.0, beta=90.0, gamma=120.0)


class HexagonalLattice(Lattice):
    """This is a special case of the lattice
    where a = b and gamma = 120 and alpha = beta = 90"""

    def __init__(self, a: float, c: float, **kwargs):
        super().__init__(a=a, b=a, c=c, alpha=90.0, beta=90.0, gamma=120.0)


class RhombohedralLattice(Lattice):
    """This is a special case of the lattice
    where a = b = c and alpha = beta = gamma"""

    def __init__(self, a: float, alpha: float, **kwargs):
        super().__init__(a=a, b=a, c=a, alpha=alpha, beta=alpha, gamma=alpha)


class CubicLattice(Lattice):
    """This is a special case of the lattice
    where a = b = c and alpha = beta = gamma = 90"""

    def __init__(self, a: float, **kwargs):
        super().__init__(a=a, b=a, c=a, alpha=90.0, beta=90.0, gamma=90.0)
