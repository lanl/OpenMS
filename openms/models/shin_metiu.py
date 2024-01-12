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

from openms.qmd import QuantumDriver
import numpy as np
from math import erf
from openms.lib.misc import eps
import warnings

"""
Shin-Metiu model:
    H=
"""


# TO be finished
class Shin_Metiu(QuantumDriver):
    """
    Class for 1D Shin-Metiu model BO calculation in a real-space grid

    object molecule: molecule object
    integer nx: the number of grid points
    double xmin: lower bound of the 1D space
    double xmax: upper bound of the 1D space
    double L: the distance between two fixed nuclei
    double Rc: the parameter of a moving nucleus
    double Rl: the parameter of a fixed nucleus in the left side
    double Rr: the parameter of a fixed nucleus in the right side
    """

    def __init__(
        self,
        molecule,
        nx=401,
        xmin=-20.0,
        xmax=20.0,
        L=19.0,
        Rc=5.0,
        Rl=4.0,
        Rr=3.1,
        nroots=1,
    ):
        # Initialize model common variables
        super().__init__()

        # self.nroots = nroots
        self.nroots = molecule.nstates

        # Set the grid
        self.nx = nx
        self.xmin = xmin
        self.xmax = xmax

        # Parameters in au
        self.L = L + eps
        self.Rc = Rc
        self.Rl = Rl
        self.Rr = Rr

        self.dx = (self.xmax - self.xmin) / float(self.nx - 1)
        self.H = np.zeros((self.nx, self.nx))
