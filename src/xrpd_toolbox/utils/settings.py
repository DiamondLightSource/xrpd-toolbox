import json
import tomllib
from pathlib import Path

import toml
import yaml
from pydantic import BaseModel

SUPPORTED_FILE_TYPES = [".json", ".toml", ".yaml"]


class SettingsBase(BaseModel):
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
