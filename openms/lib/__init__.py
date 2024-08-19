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
# Author:   Yu Zhang    <zhy@lanl.gov>
#         Ilia Mazin <imazin@lanl.gov>
#

r"""
Low-level libraries, including:

- internal Finite-Difference Time-Domain (FDTD) code,
- efficient mathematical algorithms
- backends
"""

from . import backend
from . import boson
from . import constants
from . import hippynn_es_driver
from . import jaxlib
from . import logger
from . import mathlib
from . import misc
from . import scipy_helper

try:
    from . import fdtdc as FDTD

    FDTD_AVAILABLE = True
except ImportError:
    FDTD_AVAILABLE = False
