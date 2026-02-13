# Copyright 2026 Francesco Troisi
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

"""This module contains the base abstract Observable class"""

from abc import ABC, abstractmethod
import os

class Observable(ABC):
    """
    This class is the base abstract Observable class.
    All modules in the `observables` folder must inherit from this class.
    """
    def __init__(self, *, name: str, results_path: str | None = None) -> None:
        """Initializes the Observable

        Args:
            name: The name of the Observable.
            results_path: The path where the results of the Observable will be saved. Note that this should be a folder,
                not a file. If None, the results will not be saved.
        """
        self.name: str = name.strip()
        if len(self.name) == 0:
            raise ValueError("The name of the Observable cannot be empty or contain only whitespace.")
        self.results_path: str | None = results_path
        # Check if the results path exists, if not create it
        if self.results_path is not None and not os.path.exists(self.results_path):
            os.makedirs(self.results_path)

    @abstractmethod
    def compute(self):
        """This method computes the observable."""
        pass
