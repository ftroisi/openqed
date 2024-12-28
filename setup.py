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

import inspect
import sys
import setuptools

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements: list[str] = f.read().splitlines()
    f.close()

if (not hasattr(setuptools, "find_namespace_packages") or
    not inspect.ismethod(setuptools.find_namespace_packages)):
    print(f"Your setuptools version:'{setuptools.__version__}' does not support PEP 420 " +
        "(find_namespace_packages). Upgrade it to version >='40.1.0' and repeat install.")
    sys.exit(1)

# Read long description from README.
with open('README.md', "r", encoding="utf-8") as readme_file:
    README: str = readme_file.read()
    readme_file.close()

setuptools.setup(
    name="openqed",
    version="0.1.0",
    description="OpenQED: An open-source library for simulating quantum electrodynamics.",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/ftroisi/openqed",
    author="Francesco Troisi",
    author_email="francesco.troisi98@gmail.com",
    license="Apache-2.0",
    classifiers=[
        "Environment :: Console",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
    ],
    keywords="qed cavity light-matter",
    packages=setuptools.find_packages(include=["openqed"]),
    install_requires=requirements,
    include_package_data=True,
    python_requires=">=3.7",
    extras_require={
        "mpl": ["matplotlib>=3.3"],
        "mpi": ["mpi4py>=3.1"],
    },
    project_urls={
        "Bug Tracker": "https://github.com/ftroisi/openqed/issues",
        "Source Code": "https://github.com/ftroisi/openqed",
    },
    zip_safe=False,
)
