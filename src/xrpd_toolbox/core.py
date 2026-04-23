from __future__ import annotations

import json
import math
import operator
import tomllib
from numbers import Real
from pathlib import Path
from typing import get_args, get_origin

import numpy as np
import toml
import yaml
from pydantic import BaseModel, computed_field, field_serializer, field_validator

SUPPORTED_FILE_TYPES = [".json", ".toml", ".yaml"]


def annotation_contains_type(annotation, clstype) -> bool:
    if annotation is clstype:
        return True

    origin = get_origin(annotation)
    if origin is None:
        return False

    return any(annotation_contains_type(arg, clstype) for arg in get_args(annotation))


class Parameter(BaseModel, Real):
    value: float
    refine: bool = True
    bounds: list[float] = [-np.inf, np.inf]

    # model_config = {
    #     "arbitrary_types_allowed": False,
    #     "validate_assignment": True,
    # }

    # @field_validator("value", mode="before")
    # @classmethod
    # def coerce_value(cls, v):
    #     if isinstance(v, Parameter):
    #         return v.value
    #     if isinstance(v, dict):
    #         return v.get("value")
    #     return v

    # helpers
    def get_other_value(self, other):
        if isinstance(other, Parameter):
            return other.value
        return other

    def _op(self, other, fn):
        return fn(self.value, self.get_other_value(other))

    def _rop(self, other, fn):
        return fn(self.get_other_value(other), self.value)

    # required conversions
    def __float__(self):
        return float(self.value)

    def __int__(self):
        return int(self.value)

    def __complex__(self):
        return complex(self.value)

    # Real abstract methods
    def __trunc__(self):
        return math.trunc(self.value)

    def __floor__(self):
        return math.floor(self.value)

    def __ceil__(self):
        return math.ceil(self.value)

    def __round__(self, ndigits=None):
        return round(self.value, ndigits)

    # unary
    def __neg__(self):
        return -self.value

    def __pos__(self):
        return +self.value

    def __abs__(self):
        return abs(self.value)

    # comparisons
    def __eq__(self, other):
        return self.value == self.get_other_value(other)

    def __lt__(self, other):
        return self.value < self.get_other_value(other)

    def __le__(self, other):
        return self.value <= self.get_other_value(other)

    def __gt__(self, other):
        return self.value > self.get_other_value(other)

    def __ge__(self, other):
        return self.value >= self.get_other_value(other)

    # arithmetic
    def __add__(self, other):
        return self._op(other, operator.add)

    def __radd__(self, other):
        return self._rop(other, operator.add)

    def __sub__(self, other):
        return self._op(other, operator.sub)

    def __rsub__(self, other):
        return self._rop(other, operator.sub)

    def __mul__(self, other):
        return self._op(other, operator.mul)

    def __rmul__(self, other):
        return self._rop(other, operator.mul)

    def __truediv__(self, other):
        return self._op(other, operator.truediv)

    def __rtruediv__(self, other):
        return self._rop(other, operator.truediv)

    def __floordiv__(self, other):
        return self._op(other, operator.floordiv)

    def __rfloordiv__(self, other):
        return self._rop(other, operator.floordiv)

    def __mod__(self, other):
        return self._op(other, operator.mod)

    def __rmod__(self, other):
        return self._rop(other, operator.mod)

    def __pow__(self, other):
        return self._op(other, operator.pow)

    def __rpow__(self, other):
        return self._rop(other, operator.pow)

    # numpy support
    def __array__(self, dtype=None):
        return np.asarray(self.value, dtype=dtype)

    def __array_priority__(self):
        return 1000

    # misc
    def __repr__(self):
        return f"{{value={self.value}, refine={self.refine}}}"

    def link(self, other: Parameter):
        """Use this to constrain two parmeters to be the same value
        custom_parameter = Parameter(value=5.5, refine=True)
        lattice.a.link(custom_parameter)
        """
        self.value = other.value
        self.refine = other.refine
        self.refinable = other.refinable


ParameterLike = int | float | Parameter
FloatParameterLike = float | Parameter
IntParameterLike = int | Parameter


def to_parameter(v):
    if isinstance(v, Parameter):
        return v
    if isinstance(v, dict):
        return Parameter.model_validate(v)
    if isinstance(v, (int, float)):
        return Parameter(value=float(v))
    raise TypeError("Invalid Parameter input")


class XRPDBaseModel(BaseModel):
    model_config = {
        "arbitrary_types_allowed": True,
    }

    @computed_field
    def name(self) -> str:
        return self.__class__.__name__

    def _serialize_numpy(self, v: np.ndarray) -> list:
        return v.tolist()

    def _serialize_tuple(self, v: tuple) -> list:
        return list(v)

    @field_serializer("*", when_used="always")
    def serialize_special_types(self, value):
        if isinstance(value, np.ndarray):
            return self._serialize_numpy(value)

        if isinstance(value, tuple):
            return self._serialize_tuple(value)

        return value

    @field_validator("*", mode="before")
    @classmethod
    def _coerce_special_types(cls, v, info):
        field = cls.model_fields[info.field_name]
        annotation = field.annotation

        if annotation_contains_type(annotation, np.ndarray) and isinstance(v, list):
            return np.asarray(v)

        return v

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


class RefinementBaseModel(XRPDBaseModel):
    """In the RefinementBaseModel ANYTHING that is a Parameter can be refined.
    eg. Therefore if you set cubic lattice angles to refine, you will break symmetry.
    This requires the user to know what they're doing.
    With great power comes great reposibility"""

    def parameterise(self, refine: bool = False):
        for name, val in type(self).model_fields.items():
            if (
                val.annotation is ParameterLike
                or FloatParameterLike
                or IntParameterLike
            ):
                field = getattr(self, name)

                if not isinstance(field, Parameter):
                    setattr(self, name, Parameter(value=float(field), refine=refine))

    def path_to_string(self, path):
        out = []
        for p in path:
            if isinstance(p, int):
                out[-1] = f"{out[-1]}[{p}]"
            else:
                out.append(p)
        return ".".join(out)

    def iter_parameters(self, prefix=()):
        fields = type(self).model_fields

        for name in fields:
            value = getattr(self, name)
            path = prefix + (name,)

            if isinstance(value, Parameter):
                yield path, value
                continue

            if isinstance(value, RefinementBaseModel):
                yield from value.iter_parameters(path)
                continue

            if isinstance(value, (list, tuple)):
                for i, v in enumerate(value):
                    subpath = path + (i,)

                    if isinstance(v, Parameter):
                        yield subpath, v
                    elif isinstance(v, RefinementBaseModel):
                        yield from v.iter_parameters(subpath)
                continue

            if isinstance(value, dict):
                for k, v in value.items():
                    subpath = path + (k,)

                    if isinstance(v, Parameter):
                        yield subpath, v
                    elif isinstance(v, RefinementBaseModel):
                        yield from v.iter_parameters(subpath)

    def get_refinement_parameters(self):
        result = {}
        seen = set()

        add_seen = seen.add

        for path, p in self.iter_parameters():
            if not p.refine:
                continue

            pid = id(p)
            if pid in seen:
                continue

            add_seen(pid)

            # Only build string here (once)
            key = self.path_to_string(path)
            result[key] = p.value

        return result

    def get_param_by_path(self, path: str) -> Parameter:
        obj = self
        tokens = path.split(".")

        for token in tokens:
            if "[" in token:
                name, rest = token.split("[", 1)
                obj = getattr(obj, name)
                obj = obj[int(rest[:-1])]  # strip trailing ']'
            else:
                obj = getattr(obj, token)

        if not isinstance(obj, Parameter):
            raise TypeError(f"{path} does not resolve to Parameter")

        return obj

        return obj

    def set_refinement_parameters(self, values: dict[str, float]):
        # Build mapping once
        path_map = {}
        seen = set()

        add_seen = seen.add

        for path, p in self.iter_parameters():
            if not p.refine:
                continue

            pid = id(p)
            if pid in seen:
                continue

            add_seen(pid)
            path_map[self.path_to_string(path)] = p

        # Apply updates
        for key, val in values.items():
            path_map[key].value = float(val)

    def refine_none(self):
        for _, param in self.iter_parameters():
            param.refine = False

    def refine_all(self):
        for _, param in self.iter_parameters():
            param.refine = True


if __name__ == "__main__":
    import numpy as np

    a = Parameter(value=3, refine=True)

    print(a)

    x = 1.0 + Parameter(value=3)

    print(x)
