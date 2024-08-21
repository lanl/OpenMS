#
# @ 2023. Triad National Security, LLC. All rights reserved.
#
# This program was produced under U.S. Government contract 89233218CNA000001
# for Los Alamos National Laboratory (LANL), which is operated by Triad
# National Security, LLC for the U.S. Department of Energy/National Nuclear
# Security Administration. All rights in the program are reserved by Triad
# National Security, LLC, and the U.S. Department of Energy/National Nuclear
# Security Administration. The Government is granted for itself and others acting
# on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this
# material to reproduce, prepare derivative works, distribute copies to the
# public, perform publicly and display publicly, and to permit others to do so.
#
# Author: Yu Zhang <zhy@lanl.gov>
#


import os, sys
from setuptools import setup, find_packages
from setuptools import Extension
from setuptools.command.build_ext import build_ext as _build_ext
import subprocess

# required list:
# Function to read the requirements.txt file
def parse_requirements(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line and not line.startswith("#")]

# Parse the requirements from the requirements.txt file
install_requires = parse_requirements("requirements.txt")

class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy

        self.include_dirs.append(numpy.get_include())

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        # Check if CMake build is requested
        if os.getenv('BUILD_LIB', '0') == '1':
            self.build_cmake()
        else:
            print("Skipping CMake build. Set BUILD_LIB=1 to enable.")

    def build_cmake(self):
        # Define the build directory and the install directory
        build_dir = os.path.join(os.getcwd(), "openms/lib/build")
        install_dir = os.path.join(os.getcwd(), "openms/lib/deps/")

        # Ensure the directories exist
        os.makedirs(build_dir, exist_ok=True)
        os.makedirs(install_dir, exist_ok=True)

        # Run CMake with the specified installation prefix
        subprocess.check_call([
            "cmake",
            self.extensions[0].sourcedir,
            f"-DCMAKE_INSTALL_PREFIX={install_dir}"
        ], cwd=build_dir)

        # Build and install the project
        subprocess.check_call(["make"], cwd=build_dir)
        subprocess.check_call(["make", "install"], cwd=build_dir)

    def build_extensions(self):
        # Standard extension build (e.g., Cython or pure Python)
        for ext in self.extensions:
            if isinstance(ext, CMakeExtension):
                continue  # Skip CMake extensions
            _build_ext.build_extensions(self)

setup(
    name="openms",
    version="0.1.0",
    description="An open-source multiscale solver for coupled Maxwell-Schrödinger equations.",
    packages=find_packages(),
    install_requires=install_requires,
    ext_modules=[CMakeExtension("openms_lib", sourcedir="./openms/lib")],
    cmdclass={"build_ext": CMakeBuild},
    extras_require={ # optional package for extra features
        "mpi": ["mpi4py"], # TBA
        "gpu": ["cupy"],   # TBA
        "test": ["pytest"], # for test
        "all": ["mpi4py"]  # TBA
    },
)
