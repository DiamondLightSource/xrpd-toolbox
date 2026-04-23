import matplotlib.pyplot as plt
import numpy as np

from xrpd_toolbox.core import XRPDBaseModel


# TODO: Should we store x here too?; it adds a lot of data to the serialisation
class Background(XRPDBaseModel):
    """This describes the background of a profile"""

    x: np.ndarray  # x values to evaluate the background at

    def calculate(self, x: np.ndarray | None = None) -> np.ndarray:
        """Evaluate the background at the given x values"""
        raise NotImplementedError(
            "Must implement calculate method in Background subclass"
        )

    @classmethod
    def estimate(cls, x: np.ndarray, y: np.ndarray) -> "Background":
        """Estimate the background from a profile by taking the minimum value of y"""
        raise NotImplementedError(
            "Must implement estimate method in Background subclass"
        )

    def __call__(self, x: np.ndarray | None = None) -> np.ndarray:
        return self.calculate(x)

    def __len__(self):
        return len(self.x)

    def __add__(self, other):
        if isinstance(other, "Background"):
            other = other.calculate()

        return self.calculate(self.x) + np.asarray(other)

    def __radd__(self, other):
        return np.asarray(other) + self.calculate(self.x)

    def __array__(self):
        return self.calculate()

    def plot(self, show: bool = True):
        plt.plot(self.x, self.calculate(), label=f"{type(self).__name__}")
        if show:
            plt.legend()
            plt.show()


class ConstantBackground(Background):
    """This describes a constant background"""

    value: float

    def calculate(self, x: np.ndarray | None = None) -> np.ndarray:
        if x is None:
            x = self.x
        return np.full_like(x, self.value)

    @classmethod
    def estimate(cls, x: np.ndarray, y: np.ndarray) -> "ConstantBackground":
        """Estimate the background from a profile by taking the minimum value of y"""
        value = np.min(y)
        return cls(x=x, value=value)

    def __float__(self):
        return float(self.value)


class LinearBackground(Background):
    slope: float
    intercept: float

    def calculate(self, x: np.ndarray | None = None) -> np.ndarray:
        if x is None:
            x = self.x
        return self.slope * x + self.intercept

    @classmethod
    def estimate(cls, x: np.ndarray, y: np.ndarray) -> "LinearBackground":
        min_x = np.min(x)
        max_x = np.max(x)

        min_y = np.min(y[x == min_x])
        max_y = np.min(y[x == max_x])

        slope = (max_y - min_y) / (max_x - min_x)
        intercept = min_y - slope * min_x

        return cls(x=x, slope=slope, intercept=intercept)


class ChebyshevBackground(Background):
    """This describes a Chebyshev polynomial background"""

    coefficients: np.ndarray  # coefficients of the Chebyshev polynomial

    def calculate(self, x: np.ndarray | None = None) -> np.ndarray:
        if x is None:
            x = self.x
        return np.polynomial.chebyshev.chebval(x, self.coefficients)

    def add_coefficient(self, new_value: int | float = 0):
        self.coefficients = np.append(self.coefficients, new_value)

    def remove_coefficient(self):
        self.coefficients = self.coefficients[0:-1]

    @classmethod
    def estimate(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        degree: int = 8,
        mask: bool = True,
        mask_step: int = 50,
    ) -> "ChebyshevBackground":
        # this is a very simple estimation method that fits
        # a Chebyshev polynomial to the data

        # select points evenly across the x range to fit the background
        # - prevents peaks dominating the fit

        if mask:
            selections_mask = np.arange(
                0, len(x), len(x) // mask_step
            )  # select every mask_step points (20 by default)

            x_selected = x[selections_mask]
            y_selected = y[selections_mask]

        else:
            x_selected = x
            y_selected = y

        coefficients = np.polynomial.chebyshev.chebfit(
            x_selected, y_selected, deg=degree
        )
        return cls(x=x, coefficients=coefficients)


class LinearInterpolationBackground(Background):
    x_sample: np.ndarray
    y_sample: np.ndarray

    def calculate(self, x: np.ndarray | None = None) -> np.ndarray:
        if x is None:
            x = self.x
        return np.interp(x, self.x_sample, self.y_sample)

    @classmethod
    def estimate(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        points: int = 20,
    ) -> "LinearInterpolationBackground":
        """simple estimate that takes a number of points equal to points,
        spaced evenly across x"""

        # gradient = np.gradient(y, x)
        # plt.plot(x, gradient)
        # plt.show()

        indices = np.arange(
            0, len(x), len(x) // points
        )  # select every mask_step points (20 by default)

        x_sample = x[indices]
        y_sample = y[indices]

        return cls(x=x, x_sample=x_sample, y_sample=y_sample)
