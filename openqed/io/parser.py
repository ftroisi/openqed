# Copyright 2024 Francesco Troisi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This module contains the classes to read and dump data to/from yaml or json file."""

import os
import json
from typing import Any
from yaml import load, dump
# Use CLoader and CDumper for faster parsing. If not available, use the default ones. Those
# are available if libyaml is installed (see official PyYAML documentation for more information)
try:
    from yaml import CLoader as YAMLLoader, CDumper as YAMLDumper
except ImportError:
    from yaml import Loader as YAMLLoader, Dumper as YAMLDumper

class Parser():
    """This class handles the parsing of a JSON or YAML file
    
    The parsing of a YAML file is done thanks to the PyYAML library. If you install the libyaml C
    library (see the official PyYAML documentation for more information), CLoader will be used to
    achieve faster parsing.
    """
    def __init__(self, file_name: str) -> None:
        if not os.path.isfile(file_name):
            raise ValueError(f"The file '{file_name}' does not exist in the current working" +
                             f"directory: {os.getcwd()}")
        self.file_name: str = file_name

    @classmethod
    def read_yaml(cls, file_name: str) -> dict[str, Any]:
        """Read a yaml file and return a dictionary with the parameters."""
        data: dict[str, Any] = {}
        with open(file_name, 'r', encoding="UTF-8") as f:
            data: dict[str, Any] = load(f, Loader=YAMLLoader)
            f.close()
        # Convert the keys to lowercase
        return {key.lower(): value for key, value in data.items()}

    @classmethod
    def read_json(cls, file_name: str) -> dict[str, Any]:
        """Read a json file and return a dictionary with the parameters."""
        data: dict[str, Any] = {}
        with open(file_name, 'r', encoding="UTF-8") as f:
            data: dict[str, Any] = json.load(f)
            f.close()
        # Convert the keys to lowercase
        return {key.lower(): value for key, value in data.items()}

class Dumper():
    """This class handles the dumping of some data to a JSON or YAML file
    
    The dumping of a YAML file is done thanks to the PyYAML library. If you install the libyaml C
    library (see the official PyYAML documentation for more information), CDumper will be used to
    anchieve faster parsing.
    """
    def __init__(self, file_name: str | None = None) -> None:
        if file_name is None:
            raise ValueError("No file provided")
        self.file_name: str = file_name

    @classmethod
    def dump_yaml(cls, file_name: str, mode: str, data: dict[str, str]) -> None:
        """Dump the data to a yaml file."""
        try:
            with open(file_name, mode, encoding="UTF-8") as f:
                dump(data, f, Dumper=YAMLDumper)
                f.close()
        except Exception as exc:
            raise ValueError(f"Error while dumping the data to the file: {file_name}") from exc

    @classmethod
    def dump_json(cls, file_name: str, mode: str, data: dict[str, str]) -> None:
        """Dump the data to a json file."""
        try:
            with open(file_name, mode, encoding="UTF-8") as f:
                json.dump(data, f)
                f.close()
        except Exception as exc:
            raise ValueError(f"Error while dumping the data to the file: {file_name}") from exc
