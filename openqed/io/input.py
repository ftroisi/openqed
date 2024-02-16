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

import os
import numpy as np

class InputFile():
    """"""
    def __init__(self, file_name: str | None = None) -> None:
        if file_name is None:
            raise ValueError("No input file provided")
        self.file_name: str = file_name

    def read(self) -> dict[str, str]:
        """Read the input file and return a dictionary with the parameters."""
        with open(self.file_name, 'r', encoding="UTF-8") as f:
            lines: list[str] = f.readlines()
            input_dict: dict[str, str] = {}
            for line in lines:
                if line[0] == '#':
                    continue
                value: list[str] = line.split("\n")[0].split('=')
                # Assign value
                input_dict[value[0].replace(' ', '')] = value[1].replace(' ', '')
            f.close()
        return input_dict
