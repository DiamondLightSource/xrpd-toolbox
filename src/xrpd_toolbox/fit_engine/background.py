from __future__ import annotations

from typing import Annotated, Literal

import numpy as np
from pydantic import Field

from xrpd_toolbox.core import (
    Parameter,
    ParameterArray,
    SerialisableNDArray,
)
from xrpd_toolbox.fit_engine.fitting_core import RefinementBaseModel


# TODO: Should we store x here too?; If so store it as private
class Background(RefinementBaseModel):
    """This describes the background of a profile"""

    # _x: np.ndarray

    def calculate(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the background at the given x values"""
        raise NotImplementedError(
            "Must implement calculate method in Background subclass"
        )

    @classmethod
    def estimate(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        **kwargs,
    ) -> Background:
        """Estimate the background from a profile by taking the minimum value of y"""
        raise NotImplementedError(
            "Must implement estimate method in Background subclass"
        )

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.calculate(x)


class ConstantBackground(Background):
    """This describes a constant background"""

    background_type: Literal["ConstantBackground"] = "ConstantBackground"
    value: float | Parameter = Parameter(value=0)

    def calculate(self, x: np.ndarray) -> np.ndarray:
        return np.full_like(x, self.value)

    @classmethod
    def estimate(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        **kwargs,
    ) -> ConstantBackground:
        """Estimate the background from a profile by taking the minimum value of y"""
        value = np.min(y)
        return cls(value=Parameter(value=value))

    def __float__(self):
        return float(self.value)


class LinearBackground(Background):
    background_type: Literal["LinearBackground"] = "LinearBackground"

    slope: float | Parameter = Parameter(value=0)
    intercept: float | Parameter = Parameter(value=0)

    def calculate(self, x: np.ndarray) -> np.ndarray:
        return float(self.slope) * x + float(self.intercept)

    @classmethod
    def estimate(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        **kwargs,
    ) -> LinearBackground:
        min_x = np.min(x)
        max_x = np.max(x)

        min_y = np.min(y[x == min_x])
        max_y = np.min(y[x == max_x])

        slope = (max_y - min_y) / (max_x - min_x)
        intercept = min_y - slope * min_x

        return cls(slope=Parameter(value=slope), intercept=Parameter(value=intercept))


class ChebyshevBackground(Background):
    background_type: Literal["ChebyshevBackground"] = "ChebyshevBackground"

    """This describes a Chebyshev polynomial background - GSAS style"""

    coefficients: (
        SerialisableNDArray | ParameterArray
    )  # coefficients of the Chebyshev polynomial

    def calculate(self, x: np.ndarray) -> np.ndarray:
        return np.polynomial.chebyshev.chebval(x, self.coefficients)

    def add_coefficient(self, new_value: int | float = 0):
        self.coefficients = np.append(self.coefficients, new_value)

    def remove_coefficient(self):
        self.coefficients = self.coefficients[0:-1]  # type: ignore - mypy is just being stupid here

    @classmethod
    def estimate(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        degree: int = 8,
        mask: bool = True,
        mask_step: int = 50,
        **kwargs,
    ) -> ChebyshevBackground:
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
        return cls(coefficients=coefficients)


class LinearInterpolationBackground(Background):
    """This is a FullProfs style linear interpolation background
    - also FullProfs default"""

    background_type: Literal["LinearInterpolationBackground"] = (
        "LinearInterpolationBackground"
    )

    x_sample: SerialisableNDArray = Field(repr=False)
    y_sample: SerialisableNDArray | list[Parameter] | ParameterArray = Field(repr=False)

    def calculate(self, x: np.ndarray) -> np.ndarray:
        return np.interp(x, self.x_sample, self.y_sample)

    @classmethod
    def estimate(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        points: int = 20,
        **kwargs,
    ) -> LinearInterpolationBackground:
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

        y_sample_parameter = ParameterArray.from_array(y_sample, **kwargs)

        return cls(x_sample=x_sample, y_sample=y_sample_parameter)


class SplitBackground(Background):
    """This background is basically two linear backgrounds,
    that are split at some point. Before that point is use one background
    and after that point it uses another"""

    background_type: Literal["SplitBackground"] = "SplitBackground"
    split_index: int
    value1: float | Parameter
    value2: float | Parameter

    @classmethod
    def estimate(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        rtol: float = 0.1,
        **kwargs,
    ) -> SplitBackground:
        y_min = float(np.min(y))
        threshold = 2.0 * np.abs(y_min)

        mask1 = np.isclose(y, y_min, rtol=rtol)
        idx1 = np.flatnonzero(mask1)

        if idx1.size == 0:
            # fallback: nearest to min
            idx1 = np.argsort(np.abs(y - y_min))[:10]

        split_index = int(np.max(idx1)) + 1

        value1 = float(np.mean(y[idx1]))

        valid = y >= threshold

        if np.any(valid):
            candidates = y[valid]
            min2 = float(np.min(candidates))

            mask2 = np.isclose(y, min2, rtol=rtol)
            idx2 = np.flatnonzero(mask2)

            if idx2.size == 0:
                idx2 = np.argsort(np.abs(y - min2))[:10]

            value2 = float(np.mean(y[idx2]))
        else:
            # fallback: no values above threshold
            value2 = value1

        value1 = Parameter(value=value1)
        value2 = Parameter(value=value2)

        return cls(split_index=split_index, value1=value1, value2=value2)

    def calculate(self, x: np.ndarray):
        bg = np.empty_like(x, dtype=float)
        bg[: self.split_index] = float(self.value1)
        bg[self.split_index :] = float(self.value2)
        return bg


BackgroundType = Annotated[
    ConstantBackground
    | LinearBackground
    | LinearInterpolationBackground
    | ChebyshevBackground
    | SplitBackground,
    Field(discriminator="background_type"),
]
