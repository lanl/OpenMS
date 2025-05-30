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
Variational Quantum State
=========================

(Currently, this is in a separate repo 'qneura')
"""

import numpy
from typing import Any, Optional

class VariationalQuantStateBase(object):
    r"""Base class for variational quantum states :math:`\ket{\Psi(\theta)}`
    which depends on a set of parameters :math:`\theta\equiv \{\theta_i\}`.


    TBA.

    """
    def __init__(self, symm: Optional = None):
        self._symm = symm # if symm is not None else Identity()


class Jastrow(VariationalQuantStateBase):
    r"""Jastrow-type variational quantum states:

    .. math::
        \ket{\Psi_J} = e^{\hat{J}}\ket{\text{HF}}.

    Jastrow wave function :math:`\Psi(s) = \exp(- \sum_{i \neq j} J_{ij} \hat{n}_i \hat{n}_j)`,
    Here :math:`\hat{n}_i` is a psuedo operator, it can the occupation operator for general
    Hubbard-U model or dipole operator for Cavity QED problems, or others (you name/definite it!)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def projection2ci(self, sigma):
        r"""Project the WF onto a CI

        where :math:`sigma` is a CI determinent.
        In practice, we return :math:`log[\Psi(\sigma)]`.

        If :math:`\hat{J}` is diagonal in number operators, then :math:`e^{\hat{J}}\ket{\text{HF}}`
        is also diaognal in the occupation basis:

        .. math::
            \Psi(\sigma) = exp[J(\sigma)]\bra\sigma\ket{\text{HF}}.

        If :math:`\hat{J}` is not diagonal:

        .. math::
            \Psi(\sigma) = \langle\sigma\ket{\Psi_J}
            = \bra{\sigma} \sum_n \frac{1}{n!}\hat{J}^n \ket{\text{HF}}.
        """
        logsigma = 0.0
        #

        return logsigma


    amplitude = projection2ci
