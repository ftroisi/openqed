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

""" Base test class """

import unittest
from abc import ABC
import time

class BaseOpenqedTest(unittest.TestCase, ABC):
    """ Base test class """
    def setUp(self) -> None:
        self._start_time = time.time()
        self._class_location = __file__

    def tearDown(self) -> None:
        print(f"Test {self.id()} took {time.time() - self._start_time} seconds")