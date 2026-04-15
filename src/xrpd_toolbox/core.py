from __future__ import annotations

import json
import tomllib
from pathlib import Path

import numpy as np
import toml
import yaml
from pydantic import BaseModel

SUPPORTED_FILE_TYPES = [".json", ".toml", ".yaml"]


class XRPDBaseModel(BaseModel):
    model_config = {
        "arbitrary_types_allowed": True,
        "json_encoders": {np.ndarray: lambda v: v.tolist()},
    }

    @classmethod
    def load_from_toml(cls, filepath: str | Path):
        with open(filepath, "rb") as file:
            settings_dict = tomllib.load(file)

        return cls(**settings_dict)

    @classmethod
    def load_from_yaml(cls, filepath: str | Path):
        with open(filepath, "rb") as file:
            settings_dict = yaml.safe_load(file)
        return cls(**settings_dict)

    @classmethod
    def load_from_json(cls, filepath: str | Path):
        with open(filepath, "rb") as file:
            settings_dict = json.load(file)
        return cls(**settings_dict)

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


class Parameter(XRPDBaseModel):
    value: int | float
    refine: bool = True

    # Conversions
    def __float__(self) -> float:
        return float(self.value)

    def __int__(self) -> int:
        return int(self.value)

    def __array__(self, dtype=None):
        return np.asarray(self.value, dtype=dtype)

    # Internal helper
    def _get_value(self, other: int | float | Parameter) -> int | float:
        return other.value if isinstance(other, Parameter) else other

    # Arithmetic (forward)
    def __add__(self, other: int | float | Parameter) -> Parameter:
        return Parameter(value=self.value + self._get_value(other))

    def __sub__(self, other: int | float | Parameter) -> Parameter:
        return Parameter(value=self.value - self._get_value(other))

    def __mul__(self, other: int | float | Parameter) -> Parameter:
        return Parameter(value=self.value * self._get_value(other))

    def __truediv__(self, other: int | float | Parameter) -> Parameter:
        return Parameter(value=self.value / self._get_value(other))

    def __floordiv__(self, other: int | float | Parameter) -> Parameter:
        return Parameter(value=self.value // self._get_value(other))

    def __mod__(self, other: int | float | Parameter) -> Parameter:
        return Parameter(value=self.value % self._get_value(other))

    def __pow__(self, other: int | float | Parameter) -> Parameter:
        return Parameter(value=self.value ** self._get_value(other))

    # Arithmetic (reverse)
    def __radd__(self, other: int | float | Parameter) -> Parameter:
        return Parameter(value=self._get_value(other) + self.value)

    def __rsub__(self, other: int | float | Parameter) -> Parameter:
        return Parameter(value=self._get_value(other) - self.value)

    def __rmul__(self, other: int | float | Parameter) -> Parameter:
        return Parameter(value=self._get_value(other) * self.value)

    def __rtruediv__(self, other: int | float | Parameter) -> Parameter:
        return Parameter(value=self._get_value(other) / self.value)

    def __rfloordiv__(self, other: int | float | Parameter) -> Parameter:
        return Parameter(value=self._get_value(other) // self.value)

    def __rmod__(self, other: int | float | Parameter) -> Parameter:
        return Parameter(value=self._get_value(other) % self.value)

    def __rpow__(self, other: int | float | Parameter) -> Parameter:
        return Parameter(value=self._get_value(other) ** self.value)

    # In-place operations
    def __iadd__(self, other: int | float | Parameter):
        self.value += self._get_value(other)
        return self

    def __isub__(self, other: int | float | Parameter):
        self.value -= self._get_value(other)
        return self

    def __imul__(self, other: int | float | Parameter):
        self.value *= self._get_value(other)
        return self

    def __itruediv__(self, other: int | float | Parameter):
        self.value /= self._get_value(other)
        return self

    # Unary
    def __neg__(self) -> Parameter:
        return Parameter(value=-self.value)

    def __pos__(self) -> Parameter:
        return Parameter(value=+self.value)

    def __abs__(self) -> Parameter:
        return Parameter(value=abs(self.value))

    # Comparisons
    def __eq__(self, other: object) -> bool:
        if isinstance(other, Parameter):
            return self.value == other.value
        return self.value == other

    def __lt__(self, other: int | float | Parameter) -> bool:
        return self.value < self._get_value(other)

    def __le__(self, other: int | float | Parameter) -> bool:
        return self.value <= self._get_value(other)

    def __gt__(self, other: int | float | Parameter) -> bool:
        return self.value > self._get_value(other)

    def __ge__(self, other: int | float | Parameter) -> bool:
        return self.value >= self._get_value(other)


if __name__ == "__main__":
    import numpy as np

    x = np.array([1, 1, 1]) + Parameter(value=3)

    g = XRPDBaseModel()

    print(x, g)
