from __future__ import annotations

import json
import math
import tomllib
from numbers import Real
from pathlib import Path
from typing import Annotated, Any, Literal, TypeAlias

import matplotlib.pyplot as plt
import numpy as np
import toml
import yaml
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    model_serializer,
    model_validator,
)

SUPPORTED_FILE_TYPES = [".json", ".toml", ".yaml"]

XUnit: TypeAlias = Literal["tth", "tof", "q", "d"]

DataType: TypeAlias = Literal[
    "xray",
    "lab-xray",
    "tof-neutron",
    "cw-neutron",
]


def to_ndarray(v):
    if isinstance(v, np.ndarray):
        return v
    if isinstance(v, list):
        return np.asarray(v)
    return v


SerialisableNDArray = Annotated[
    np.ndarray,
    BeforeValidator(to_ndarray),
    PlainSerializer(lambda x: x.tolist(), return_type=list),
]

SAFE_GLOBALS = {
    "__builtins__": {},
    "abs": abs,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "sqrt": math.sqrt,
    "log": math.log,
    "exp": math.exp,
}


def safe_pow(a, b):
    if abs(a) > 1e6 or abs(b) > 100:
        raise ValueError("too large")
    return a**b


def safe_exp(x):
    if x > 700:  # float overflow boundary
        raise ValueError("exp too large")
    return math.exp(x)


# replace the power to use safe_pow which should protect against CPU heavy operations
SAFE_GLOBALS["pow"] = safe_pow
SAFE_GLOBALS["exp"] = safe_exp


def evaluate_expression(expr: str, ctx: dict[str, float]) -> float:
    """I realise this uses eval() which is discouraged,
    but the other option is nasty, slower and adds bad complexity.
    With the catches and by only using SAFE_GLOBALS and __builtins__
    - the risk is basically 0"""

    # prevents access to sub classes that could be dangerous to introspection attack
    if "__" in expr or "." in expr:
        raise ValueError("Disallowed expression")

    # prevents stupidly long expression which might take forever to parse
    if len(expr) > 200:
        raise ValueError("Expression too long")

    return float(eval(expr, SAFE_GLOBALS, ctx))


class Parameter(BaseModel, Real):
    value: float | int | str
    refine: bool = True
    bounds: list[float] = [-np.inf, np.inf]

    _name: str | None = None
    _model: Any = None  # back reference to RefinementBaseModel

    def _ctx(self) -> dict[str, float]:
        """
        IMPORTANT:
        Only uses RAW values, never float(v), never recursion.
        """
        return {k: float(v.value) for k, v in self._model._params.items()}  # noqa

    def _compute_value(self) -> float:
        # print(self, self.value, "\n\n")

        if isinstance(self.value, (float, int)):
            return float(self.value)
        elif isinstance(self.value, str):
            return evaluate_expression(self.value, self._ctx())
        else:
            raise Exception()

    # helpers
    def get_other_value(self, other: Parameter | int | float):
        if isinstance(other, Parameter):
            return other.value
        else:
            return other

    def __float__(self):
        return self._compute_value()

    def _expr(self):
        """formats the Parameter so that it shows as either
        a number or _name in the expression"""
        return self._name or str(self.value)

    def _to_expression(self, other, op: str):
        # if isinstance(self.value, (int, float)) and isinstance(other, (int, float)):
        #     return Parameter(value=)

        if isinstance(other, Parameter):
            return Parameter(value=f"({self._expr()} {op} {other._expr()})")  # noqa
        else:
            return Parameter(value=f"({self._expr()} {op} {other})")

    # arithmetic
    def __add__(self, other: int | float | Parameter):
        if isinstance(self.value, (int, float)) and isinstance(other, (int, float)):
            return self.value + other
        else:
            return self._to_expression(other, "+")

    def __radd__(self, other):
        return self._to_expression(other, "+")

    def __sub__(self, other):
        return self._to_expression(other, "-")

    def __rsub__(self, other):
        return Parameter(value=f"({other} - {self._expr()})")

    def __mul__(self, other):
        return self._to_expression(other, "*")

    def __rmul__(self, other):
        return self._to_expression(other, "*")

    def __truediv__(self, other):
        return self._to_expression(other, "/")

    def __rtruediv__(self, other):
        return Parameter(value=f"({other} / {self._expr()})")

    def __floordiv__(self, other):
        return self._to_expression(other, "//")

    def __rfloordiv__(self, other):
        return Parameter(value=f"({other} // {self._expr()})")

    def __mod__(self, other):
        return self._to_expression(other, "%")

    def __rmod__(self, other):
        return Parameter(value=f"({other} % {self._expr()})")

    def __pow__(self, other):
        return self._to_expression(other, "**")

    def __rpow__(self, other):
        return Parameter(value=f"({other} ** {self._expr()})")

    # unary
    def __neg__(self):
        return Parameter(value=f"-({self._expr()})")

    def __abs__(self):
        return Parameter(value=f"abs({self._expr()})")

    # comparisons
    def __eq__(self, other):
        return float(self) == float(other)

    def __lt__(self, other):
        return float(self) < float(other)

    def __le__(self, other):
        return float(self) <= float(other)

    def __gt__(self, other):
        return float(self) > float(other)

    def __ge__(self, other):
        return float(self) >= float(other)

    # Real interface
    def __complex__(self):
        return complex(float(self))

    def __pos__(self):
        return self

    # numeric
    def __int__(self):
        return int(float(self))

    def __trunc__(self):
        return int(float(self))

    def __floor__(self):
        return math.floor(float(self))

    def __ceil__(self):
        return math.ceil(float(self))

    def __round__(self, n=None):
        return round(float(self), n)

    def __array__(self):
        return np.array([self._compute_value()], dtype=float)

    @property
    def real(self):
        return float(self)

    @property
    def imag(self):
        return 0

    def conjugate(self):
        return self

    def __divmod__(self, other):
        return divmod(float(self), float(other))

    def __rdivmod__(self, other):
        return divmod(float(other), float(self))


# class Parameter(BaseModel, Real):
#     """A parameter acts like a value, that can be refined"""

#     value: float
#     refine: bool = Field(default=True)
#     bounds: list[float] = Field(default=[-np.inf, np.inf], repr=False)

#     def set_refine(self):
#         self.refine = True

#     def dont_refine(self):
#         self.refine = False

#     # helpers
#     def get_other_value(self, other):
#         if isinstance(other, Parameter):
#             return other.value
#         return other

#     def _op(self, other, fn):
#         return fn(self.value, self.get_other_value(other))

#     def _rop(self, other, fn):
#         return fn(self.get_other_value(other), self.value)

#     # required conversions
#     def __float__(self):
#         return float(self.value)

#     def __int__(self):
#         return int(self.value)

#     def __complex__(self):
#         return complex(self.value)

#     # Real abstract methods
#     def __trunc__(self):
#         return math.trunc(self.value)

#     def __floor__(self):
#         return math.floor(self.value)

#     def __ceil__(self):
#         return math.ceil(self.value)

#     def __round__(self, ndigits=None):
#         return round(self.value, ndigits)

#     # unary
#     def __neg__(self):
#         return -self.value

#     def __pos__(self):
#         return +self.value

#     def __abs__(self):
#         return abs(self.value)

#     # comparisons
#     def __eq__(self, other):
#         return self.value == self.get_other_value(other)

#     def __lt__(self, other):
#         return self.value < self.get_other_value(other)

#     def __le__(self, other):
#         return self.value <= self.get_other_value(other)

#     def __gt__(self, other):
#         return self.value > self.get_other_value(other)

#     def __ge__(self, other):
#         return self.value >= self.get_other_value(other)

#     # arithmetic
#     def __add__(self, other):
#         return self._op(other, operator.add)

#     def __radd__(self, other):
#         return self._rop(other, operator.add)

#     def __sub__(self, other):
#         return self._op(other, operator.sub)

#     def __rsub__(self, other):
#         return self._rop(other, operator.sub)

#     def __mul__(self, other):
#         return self._op(other, operator.mul)

#     def __rmul__(self, other):
#         return self._rop(other, operator.mul)

#     def __truediv__(self, other):
#         return self._op(other, operator.truediv)

#     def __rtruediv__(self, other):
#         return self._rop(other, operator.truediv)

#     def __floordiv__(self, other):
#         return self._op(other, operator.floordiv)

#     def __rfloordiv__(self, other):
#         return self._rop(other, operator.floordiv)

#     def __mod__(self, other):
#         return self._op(other, operator.mod)

#     def __rmod__(self, other):
#         return self._rop(other, operator.mod)

#     def __pow__(self, other):
#         return self._op(other, operator.pow)

#     def __rpow__(self, other):
#         return self._rop(other, operator.pow)

#     # numpy support
#     def __array__(self, dtype=float):
#         return np.asarray(self.value, dtype=dtype)

#     def __array_priority__(self):
#         return 1000

#     def link(self, other: Parameter):
#         """Use this to constrain two parmeters to be the same value
#         custom_parameter = Parameter(value=5.5, refine=True)
#         lattice.a.link(custom_parameter)
#         """
#         self.value = other.value
#         self.refine = other.refine
#         self.refinable = other.refinable
#         self.set_to = other.__name__


# ParameterLike = int | float | Parameter
# FloatParameterLike = float | Parameter
# IntParameterLike = int | Parameter


class ParameterArray(BaseModel):
    """Parameter array should be use for things that contain an array of coeeficients
    see: Chebychev background"""

    parameter_array: list[Parameter]

    @model_serializer(mode="plain")
    def serialize(self):
        if not self.parameter_array:
            return {}

        out = {
            "value": [],
            "refine": [],
            "lower_bounds": [],
            "upper_bounds": [],
        }

        for p in self.parameter_array:
            out["value"].append(p.value)
            out["refine"].append(p.refine)

            lb, ub = p.bounds
            out["lower_bounds"].append(lb)
            out["upper_bounds"].append(ub)

        return out

    @model_validator(mode="before")
    @classmethod
    def deserialize(cls, data):
        # already in internal format → do nothing
        if "parameter_array" in data:
            return data

        # transposed format → rebuild
        if "value" in data:
            n = len(data["value"])

            params = []
            for i in range(n):
                params.append(
                    Parameter(
                        value=data["value"][i],
                        refine=data.get("refine", [True] * n)[i],
                        bounds=[
                            data["lower_bounds"][i],
                            data["upper_bounds"][i],
                        ],
                    )
                )

            return {"parameter_array": params}

        return data

    def __getitem__(self, key: int | slice) -> Parameter | ParameterArray:
        if isinstance(key, int):
            return self.parameter_array[key]
        elif isinstance(key, slice):
            return ParameterArray(parameter_array=self.parameter_array[key])
        else:
            raise TypeError(f"Invalid key: {key}")

    def __array__(self):
        return np.array([f.value for f in self.parameter_array], dtype=float)

    @classmethod
    def from_array(cls, array: np.ndarray | list[float | int], refine: bool = True):
        return cls(parameter_array=[Parameter(value=f, refine=refine) for f in array])


class XRPDBaseModel(BaseModel):
    """The XRPDBaseModel should be used for anything that you want to be able to easily
    serialised/deserialise from file but wont be used for a refinement."""

    model_config = ConfigDict(
        from_attributes=True, arbitrary_types_allowed=True, validate_assignment=True
    )

    @classmethod
    def load_from_toml(cls, filepath: str | Path):
        with open(filepath, "rb") as file:
            settings_dict = tomllib.load(file)

        return cls.model_validate(settings_dict)

    @classmethod
    def load_from_yaml(cls, filepath: str | Path):
        with open(filepath, "rb") as file:
            settings_dict = yaml.safe_load(file)
        return cls.model_validate(settings_dict)

    @classmethod
    def load_from_json(cls, filepath: str | Path):
        with open(filepath, "rb") as file:
            settings_dict = json.load(file)
        return cls.model_validate(settings_dict)

    @classmethod
    def load(cls, filepath: str | Path):
        file_extension = Path(filepath).suffix

        if file_extension == ".json":
            return cls.load_from_json(filepath)
        elif file_extension == ".yaml":
            return cls.load_from_yaml(filepath)
        elif file_extension == ".toml":
            return cls.load_from_toml(filepath)
        else:
            raise ValueError(f"Filetype must be: {SUPPORTED_FILE_TYPES}")

    def save_to_toml(self, filepath: str | Path) -> None:
        if not str(filepath).endswith(".toml"):
            raise ValueError("file_path name must end with .toml")

        print("Saving configuration to:", filepath)

        config_dict = self.model_dump()

        with open(filepath, "w") as outfile:
            toml.dump(config_dict, outfile)

    def save_to_json(self, filepath: str | Path) -> None:
        if not str(filepath).endswith(".json"):
            raise ValueError("file_path name must end with .json")

        print("Saving configuration to:", filepath)

        config_dict = self.model_dump()

        with open(filepath, "w") as outfile:
            json.dump(config_dict, outfile, indent=2, sort_keys=False)

    def save_to_yaml(self, file_path: str | Path) -> None:
        if not str(file_path).endswith(".yaml"):
            raise ValueError("file_path name must end with .yaml")

        print("Saving configuration to:", file_path)

        config_dict = self.model_dump()

        with open(file_path, "w") as outfile:
            yaml.dump(
                config_dict,
                outfile,
                default_flow_style=None,
                sort_keys=False,
                indent=2,
                explicit_start=True,
            )

    def save(self, filepath: str | Path):
        file_extension = Path(filepath).suffix

        if file_extension == ".json":
            return self.save_to_json(filepath)
        elif file_extension == ".yaml":
            return self.save_to_yaml(filepath)
        elif file_extension == ".toml":
            return self.save_to_toml(filepath)
        else:
            raise ValueError(f"Filetype must be: {SUPPORTED_FILE_TYPES}")

    def __getitem__(self, name):
        if name in type(self).model_fields:
            value = getattr(self, name)
            return value
        else:
            raise ValueError(f"{name} not in {self}")

    def __setitem__(self, name, value):
        setattr(self, name, value)


class XYEData(XRPDBaseModel):
    x: SerialisableNDArray = Field(repr=False)
    y: SerialisableNDArray = Field(repr=False)
    e: SerialisableNDArray | None = Field(default=None, repr=False)
    x_unit: str = "index"
    y_unit: str = "Intensity (Arb. Units)"
    source: str | None = None  # for tracking where the data came from

    @model_validator(mode="after")
    def validate_data(self):
        assert len(self.x) == len(self.y)
        if self.e is not None:
            assert len(self.x) == len(self.e)

        return self

    @classmethod
    def from_csv(cls, filepath: str | Path):
        try:
            x, y, e = np.genfromtxt(str(filepath), unpack=True, dtype=float)
        except ValueError:
            x, y = np.genfromtxt(str(filepath), unpack=True, dtype=float)
            e = None

        return cls(x=x, y=y, e=e, source=str(filepath))


# TODO: Decide whether better to put x_unit into XYEData, remove generic ModelDataVar
# and then get data type from Radiation class -
# but the type of radiation is inherently linked to data... so multi phase/radia stuff
class ScatteringData(XYEData):
    x_unit: XUnit = "tth"  # type: ignore[override]
    data_type: DataType = "xray"
    wavelength: float | Parameter  # for x-ray or CW neutron data

    @model_validator(mode="after")
    def validate_data_units(self):
        if self.data_type == "xray":
            assert self.x_unit != "tof", "x_unit cannot be 'tof' for xray data"

        return self

    def plot(self, show: bool = True):
        plt.figure(figsize=(16, 10))
        plt.errorbar(self.x, self.y, yerr=self.e, fmt="o", label="Data")
        plt.xlabel(f"{self.x_unit}")
        plt.ylabel("Intensity (a.u.)")
        plt.title(f"Scattering Data ({self.data_type})")

        if show:
            plt.legend()
            plt.show()

    # If this doesn't accept the data format:
    # I recommend using POWDLL to convert data to TOPAS style .xye format
    @classmethod
    def from_xye(
        cls,
        filepath: str | Path,
        x_unit: XUnit,
        data_type: DataType,
        wavelength: float | Parameter,
    ) -> ScatteringData:
        """Loads scattering data from a CSV file. The file should have 3 (or 2) columns:
        x, y and optionally e (error)
        Equivalent the TOPAS xye format
        """

        if isinstance(wavelength, Parameter):
            wavelength = wavelength
        else:
            wavelength = Parameter(value=wavelength, refine=False)

        try:
            x, y, e = np.genfromtxt(str(filepath), unpack=True, dtype=float)
        except ValueError:
            x, y = np.genfromtxt(str(filepath), unpack=True, dtype=float)
            e = None

        return cls(
            x=x,
            y=y,
            e=e,
            x_unit=x_unit,
            data_type=data_type,
            wavelength=wavelength,
            source=str(filepath),
        )

    @classmethod
    def from_fullprof(
        cls,
        filepath: str | Path,
        x_unit: XUnit,
        data_type: DataType,
        wavelength: float | Parameter,
    ) -> ScatteringData:
        """Loads scattering data from a .xy or .dat file.
        The file should have 3 columns x, y and error
        Equivalent the fullprof INSTRM=10 format
        """

        if isinstance(wavelength, Parameter):
            wavelength = wavelength
        else:
            wavelength = Parameter(value=wavelength, refine=False)
        x, y, e = np.genfromtxt(
            str(filepath),
            skip_header=1,
            comments="!",
            unpack=True,
            dtype=float,
        )

        return cls(
            x=x,
            y=y,
            e=e,
            x_unit=x_unit,
            data_type=data_type,
            wavelength=wavelength,
            source=str(filepath),
        )


if __name__ == "__main__":
    import numpy as np

    a = Parameter(value=3, refine=True)

    print(a)

    x = 1.0 + Parameter(value=3)

    print(x)
