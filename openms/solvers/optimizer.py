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
Optimizers
==========


A collection of optimizers, including:

  - Newton-Conjugate-Gradient algorithm (Newton-CG)


Newton-CG method:

Newton’s method is based on fitting the function locally to a quadratic form:

.. math::
    f(x) \approx f(x_0) + \nabla f(x_0) (x-x_0) + \frac{1}{2}(x-x_0)^T H(x_0) (x-x_0)

where :math:`H(x_0)` is the Hessian.

"""

# scipy nonlinear solvers:
#    https://github.com/pv/scipy-work/blob/master/scipy/optimize/nonlin.py

# we may directly use scipy optimization functions




class OptimizeBase(object):
    def __init__(self, *args, **kwargs):
        self.verbose = kwargs.get("verbose", 1)


class NewtonCG(OptimizeBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

# QuasiNewton



# Newton_Krylov


#
# Newton-Raphson method
#
class NewtonRaphson(OptimizeBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)



# conjugate gradient and derivative algorihtms

# newton solver
def newton_solver(matvec):
    r"""
    matvec: functions for computing Ax products
    """
    rtol = 1.e-8
    epsfnc = 1.e-10
