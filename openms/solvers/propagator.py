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

# Time-dependent propagators

r"""
Propagators (or integrators)
============================

Doc TBA.

"""

class PropagatorBase(object):

    def __init__(self, fcn, t0, y0, dt, tmax, order, **kwargs):
        r"""
        t0:
        fcn is a callable function.
        """
        self.dt = dt
        self.fcn = fcn
        self.t0 = t0
        self.y0 = y0
        self.tmax = tmax
        self.order = order

        self.rtol = kwargs.get("rtol", 1.0e-3)
        self.atol = kwargs.get("atol", 1.0e-6)
        self.verbose = kwargs.get("verbose", 1)

    def init(self):
        pass

    def step(self, n=1):
        r"""n integration step (default 1 step)"""
        pass

    def propagate(self):
        pass


class RKN(PropagatorBase):
    r"""Explicit N-orther Runge–Kutta method"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def step(self, n=1):
        r"""n integration step (default 1 step)"""
        pass

    def propagate(self):
        r"""propagate df/dt = fcn(t) function"""

        pass


class iRKN(RKN):
    r"""Implicit N-orther Runge–Kutta method"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def step(self, n=1):
        r"""n integration step (default 1 step)"""
        pass

    def propagate(self, fcn):
        r"""propagate df/dt = fcn(t) function

        fcn is a callable function.
        """

        pass
