from __future__ import annotations

import ast
import json
import math
import operator as op
import tomllib
from abc import abstractmethod
from numbers import Real
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import toml
import yaml
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    PlainSerializer,
    model_serializer,
    model_validator,
)

SUPPORTED_FILE_TYPES = [".json", ".toml", ".yaml"]


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


# ============================================================
# SAFE AST EVALUATOR (NO eval)
# ============================================================

OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.Mod: op.mod,
    ast.FloorDiv: op.floordiv,
    ast.USub: op.neg,
    ast.UAdd: lambda x: x,
}


def evaluate_binary_operation(
    binary_operation: ast.AST, ctx: dict[str, float]
) -> float:
    if isinstance(binary_operation, ast.Constant):
        value = binary_operation.value

        if isinstance(value, (int, float)):
            return float(value)
        else:
            raise TypeError(f"Unsupported constant type: {type(value)}")

    elif isinstance(binary_operation, ast.Name):
        return float(ctx[binary_operation.id])

    elif isinstance(binary_operation, ast.BinOp):
        return OPS[type(binary_operation.op)](
            evaluate_binary_operation(binary_operation.left, ctx),
            evaluate_binary_operation(binary_operation.right, ctx),
        )

    elif isinstance(binary_operation, ast.UnaryOp):
        return OPS[type(binary_operation.op)](
            evaluate_binary_operation(binary_operation.operand, ctx)
        )

    else:
        raise ValueError("Unsupported expression")


def evaluate_expression(expr: str, ctx: dict[str, float]) -> float:
    ast_expression = ast.parse(expr, mode="eval")

    return evaluate_binary_operation(ast_expression.body, ctx)


# ============================================================
# PARAMETER (FULLY LAZY, NO GLOBAL METHODS)
# ============================================================


class Parameter(BaseModel, Real):
    value: float | int | str
    refine: bool = True
    bounds: list[float] = [-np.inf, np.inf]

    _name: str | None = None
    _model: Any = None  # back-reference to model

    def _ctx(self) -> dict[str, float]:
        """
        IMPORTANT:
        Only uses RAW values, never float(v), never recursion.
        """
        return {k: v.value for k, v in self._model._params.items()}  # noqa

    def _compute_value(self) -> float:
        if isinstance(self.value, (float, int)):
            return float(self.value)
        elif isinstance(self.value, str):
            return evaluate_expression(self.value, self._ctx())
        else:
            raise Exception()

    def __float__(self):
        return self._compute_value()

    def _expr(self):
        """formats the Parameter so that it shows as either
        a number or _name in the expression"""
        return self._name or str(self.value)

    def _to_expression(self, other, op):
        if isinstance(other, Parameter):
            return Parameter(value=f"({self._expr()} {op} {other._expr()})")  # noqa
        else:
            return Parameter(value=f"({self._expr()} {op} {other})")

    # arithmetic
    def __add__(self, other):
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


ParameterLike = int | float | Parameter
FloatParameterLike = float | Parameter
IntParameterLike = int | Parameter


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

    def __getitem__(self, key: str | int | slice):
        if isinstance(key, str) and key in type(self).model_fields:
            return getattr(self, key)
        elif isinstance(key, int):
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
        from_attributes=False, arbitrary_types_allowed=True, validate_assignment=True
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


class RefinementBaseModel(XRPDBaseModel, extra="allow"):
    """In the RefinementBaseModel ANYTHING that is a Parameter can be refined.
    eg. Therefore if you set cubic lattice angles to refine, you will break symmetry.
    This requires the user to know what they're doing.
    With great power comes great reposibility"""

    def parameterise_all(self, refine: bool = False):
        for name, val in type(self).model_fields.items():
            if (
                val.annotation is ParameterLike
                or FloatParameterLike
                or IntParameterLike
            ):
                field = getattr(self, name)

                if not isinstance(field, Parameter):
                    setattr(self, name, Parameter(value=float(field), refine=refine))

    def path_to_string(self, path) -> str:
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

            elif isinstance(value, ParameterArray):
                for i, p in enumerate(value.parameter_array):
                    subpath = path + (i,)
                    yield subpath, p

            elif isinstance(value, RefinementBaseModel):
                yield from value.iter_parameters(path)

            elif isinstance(value, (list, tuple)):
                for i, v in enumerate(value):
                    subpath = path + (i,)

                    if isinstance(v, Parameter):
                        yield subpath, v
                    elif isinstance(v, RefinementBaseModel):
                        yield from v.iter_parameters(subpath)

            elif isinstance(value, dict):
                for k, v in value.items():
                    subpath = path + (k,)

                    if isinstance(v, Parameter):
                        yield subpath, v
                    elif isinstance(v, RefinementBaseModel):
                        yield from v.iter_parameters(subpath)

    def model_post_init(self, __context: Any):
        self._params = self._collect_parameters()
        self._bind_parameters()

    def _collect_parameters(self) -> dict[str, Parameter]:
        params: dict[str, Parameter] = {}

        for name, value in self.__dict__.items():
            if isinstance(value, Parameter):
                params[name] = value

        return params

    def _bind_parameters(self) -> None:
        for name, param in self._params.items():
            param._name = name  # noqa
            param._model = self  # noqa

    # def iter_parameters(self, prefix=()):
    #     """this version is better - but breaks if Paramater aren't fixed"""

    #     stack = deque([(prefix, self)])

    #     while stack:
    #         path, value = stack.pop()

    #         # --- Parameter (fast path, most important) ---
    #         if isinstance(value, Parameter):
    #             yield path, value
    #             continue

    #         # --- Model ---
    #         if isinstance(value, RefinementBaseModel):
    #             for name in value.model_fields:
    #                 stack.append((path + (name,), getattr(value, name)))
    #             continue

    #         # --- Mapping ---
    #         if isinstance(value, dict):
    #             for k, v in value.items():  # type: ignore - this is impossible to not have items # noqa
    #                 stack.append((path + (k,), v))
    #             continue

    #         # --- Iterable containers (lists, tuples, ParameterArray, etc.) ---
    #         if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
    #             # special-case: ParameterArray exposes internal list
    #             if hasattr(value, "parameter_array"):
    #                 value = value.parameter_array

    #             for i, v in enumerate(value):
    #                 stack.append((path + (i,), v))

    def get_refinement_parameters(self) -> dict:
        result = {}
        seen = set()

        for path, p in self.iter_parameters():
            if not p.refine:
                continue

            pid = id(p)

            # this bit can be removed after I have fixed how parameter stores equiv vals
            if pid in seen:
                continue

            seen.add(pid)

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

    def refine_none(self, keep_refined: list[str] | None = None):
        if keep_refined is None:
            keep_refined = []

        for name, param in self.iter_parameters():
            if name[-1] in keep_refined:
                continue
            else:
                param.refine = False

    def refine_all(self, keep_fixed: list[str] | None = None):
        if keep_fixed is None:
            keep_fixed = []

        for name, param in self.iter_parameters():
            if name[-1] in keep_fixed:
                continue
            else:
                param.refine = True


class Model(RefinementBaseModel):
    """A model can be refined by the refiner. It must contain:
    calculate_residual"""

    @abstractmethod
    def calculate_residual(self):
        raise NotImplementedError(
            "Must implmenet calculate_residual for Model subclass"
        )


if __name__ == "__main__":
    import numpy as np

    a = Parameter(value=3, refine=True)

    rf = RefinementBaseModel()

    print(a)

    x = 1.0 + Parameter(value=3)

    print(x)

    rf["a"] = 1

    print(rf.model_dump_json())

    print(rf)
