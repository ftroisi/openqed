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

"""This module contains the class and functions to read the input files for OpenQED."""

from typing import TypedDict, Any
import numpy as np
from .parser import Parser

class UserInput(TypedDict):
    """This class defines all possible parameters that can appear in the input file."""
    spacing: float | np.float64

class InputFile(Parser):
    """
    This class read the input file and stores the parameters in a TypedDict
    (for type checking and fast access).

    Args:
        file_name: The name of the input file. Defaults to "input.yml". Note that the file name
            must include the extension. The supported extensions are "yml", "yaml" and "json".

    Raises:
        ValueError: If the file type is not supported.
    """
    def __init__(self, file_name: str = "input.yml") -> None:
        # First, initialize the Parser
        file_type: str = file_name.split(".")[-1].lower()
        super().__init__(file_name)
        # Then, read the file
        if file_type in ("yaml", "yml"):
            raw_data: dict[str, Any] = self.read_yaml(file_name)
        elif file_type == "json":
            raw_data: dict[str, Any] = self.read_json(file_name)
        else:
            raise ValueError(f"File type {file_type} not supported")
        # Convert the raw data to a TypedDict
        self.input: UserInput = self._convert_to_typed_dict(raw_data)

    def _convert_to_typed_dict(self, data: dict[str, Any]) -> UserInput:
        """Convert the dictionary to a TypedDict."""
        return UserInput(spacing=float(data.get("spacing", 0.0)))
