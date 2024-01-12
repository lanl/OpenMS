import os
import unittest
from logging import _SysExcInfoType
from pathlib import Path
from typing import List, Union

import numpy as np
from scipy.integrate import RK45

from openms.qmd.propagator import rk4

CWD = os.getcwd()
NULL = open(os.devnull, "w")
TESTS_ROOT_DIR = Path(__file__).parent.parent.absolute()


def qmd_integrator(f: callable, y: float, t0: float, t: float, dt: float):
    """Integrating function f from t0 to t with an initial value of y with our
       version of RK4.

    :param f: function
    :type f: callable
    :param y: initial value
    :type y: float
    :param t0: initial time
    :type t0: float
    :param t: final time
    :type t: float
    :param dt: time step
    :type dt: float
    :return: integrated value
    :rtype: float
    """
    steps = int((t - t0) / dt)
    for _ in range(steps):
        y = rk4(f, t0, y, dt)
        t0 += dt
    return y


def scipy_integrator(f: callable, y: float, t0: float, t: float, dt: float):
    scipy_rk4 = RK45(f, t0, [y], t, atol=1e90, rtol=1e90, first_step=0.1, max_step=dt)
    while scipy_rk4.status == "running":
        scipy_rk4.step()
    return scipy_rk4.y[0]


# TODO: due to the nature of numerical integrator, we can only use assertAlmostEqual and
# TODO: might have to pass `places=` to make it "almost". Can we improve the precision?
class TestRK4Integrator(unittest.TestCase):
    def real_test_case_1(self):
        # Integrate 3t^2, so the result should be t^3
        f = lambda t, y: 3 * t**2
        t0 = 0
        y = 0
        t = 20
        dt = 0.1
        y_scipy = scipy_integrator(f, y, t0, t, dt)
        y_qmd = qmd_integrator(f, y, t0, t, dt)
        self.assertAlmostEqual(y_scipy, y_qmd)

    def real_test_case_2(self):
        # Integrating from 0 to 1 gives 22 / 7 - pi
        f = lambda t, y: t**4 * (1 - t) ** 4 / (1 + t**2)
        t0 = 0
        y = 0
        t = 1
        dt = 0.1
        y_qmd = qmd_integrator(f, y, t0, t, dt)
        self.assertAlmostEqual(y_qmd, 22 / 7 - np.pi)

    def real_test_case_3(self):
        f = lambda t, y: t**3 + np.sqrt(y)
        t0 = 0
        y = 0
        t = 20
        dt = 0.1
        y_scipy = scipy_integrator(f, y, t0, t, dt)
        y_qmd = qmd_integrator(f, y, t0, t, dt)
        error = (y_scipy - y_qmd) / y_qmd
        # the relative error is about 4.5e-7, so places=6 has to be used
        # the absolute error is about 0.02
        self.assertAlmostEqual(error, 0, places=6)

    def complex_test_case_1(self):
        f = lambda t, y: t**3 + y
        t0 = 0
        y = 1j
        t = 20
        # reduce the time step for smaller error
        dt = 0.01
        y_scipy = scipy_integrator(f, y, t0, t, dt)
        y_qmd = qmd_integrator(f, y, t0, t, dt)
        error = (y_scipy - y_qmd) / y_qmd
        # the relative error is about 1.5e-9
        # the absolute error is about 4.3 + 0.8i
        self.assertAlmostEqual(error, 0)

    def complex_test_case_2(self):
        f = lambda t, y: t + y**2
        t0 = 0
        y = 0j
        t = 20
        dt = 0.01
        y_scipy = scipy_integrator(f, y, t0, t, dt)
        y_qmd = qmd_integrator(f, y, t0, t, dt)
        error = (y_scipy - y_qmd) / y_qmd
        # the relative error is about 1.7e-5
        # the absolute error is about -1.1e-05 + 1e-4i
        self.assertAlmostEqual(error, 0, places=4)
