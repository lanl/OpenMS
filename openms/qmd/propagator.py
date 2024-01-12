from typing import Callable, Union

import numpy as np


def rk4(func: Callable, t0: float, y0: Union[float, np.array], dt: float):
    """Runge-Kutta 4 integrator for an arbitrary univariate function.

    :param func: derivate of the function needs to be integrated. dy/dt
    :type func: Callable
    :param t0: initial time of current step
    :type t0: float
    :param y0: value of the function y at current step
    :type y0: Union[float, numpy.array]
    :param dt: time step
    :type dt: float
    :return: value of function y at t0 + dt
    :rtype: Union[float, numpy.array]
    """
    # TODO: the error of current implementation is slightly larger than scipy's
    # FIXME: the current implementation suffers from much larger error when dt is increased
    k1 = func(t0, y0)
    k2 = func(t0 + dt / 2, y0 + k1 * dt / 2)
    k3 = func(t0 + dt / 2, y0 + k2 * dt / 2)
    k4 = func(t0 + dt, y0 + k3 * dt)
    return y0 + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6
