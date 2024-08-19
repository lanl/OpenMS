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

r"""
Walkers functions (TBA)
-----------------------

This class contains the walkers WF and weights
The walkers are adjusted according to the type of trial WF

"""

import sys
from abc import abstractmethod
import numpy as backend


def initialize_walkers(trial):
    r"""get initial walkers
    # TODO: 1) get initial_wakers for various trials
    # for SD, initial_walkers = trial.psi.copy()
    """

    # N copies of trial
    # shape: Nw x N_NO x N_AO for SD
    initial_walkers = trial.psi.copy()

    # 2) TODO: multiSD

    return initial_walkers


class BaseWalkers(object):
    def __init__(self, **kwargs):
        r"""Base walker class"""
        self.nwalkers = kwargs.get("nwalkers", 100)
        self.verbose = kwargs.get("verbose", 2)

        # C_w
        self.weights = backend.ones(self.nwalkers)
        self.overlap = backend.ones(self.nwalkers)
        self.eloc = backend.zeros(self.nwalkers)

        self.spin_restricted = False
        #print("nwalkers in walkerfunciton is", self.nwalkers)

# walker in so orbital (akin ghf walker)
class  Walkers_so(BaseWalkers):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        trial = kwargs.get("trial", None)
        if trial is None:
            raise Exception("Trial WF must be specified in walker")
        # 1) create walkers
        initial_walkers = initialize_walkers(trial)
        self.phiw = backend.array(
            [initial_walkers] * self.nwalkers, dtype=backend.complex128
        )
        self.spin_restricted = False
