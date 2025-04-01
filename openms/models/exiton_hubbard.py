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

from openms.qmd import QuantumDriver


class ExcitonHubbardU(QuantumDriver):
    r"""Excitonic models in real space :cite:`Slobodkin2020prl`

    The Hamiltonain is:

    .. math::

        H = -\frac{1}{2m_x}\sum_i\nabla_i + \sum_{i<j} V^d_{\sigma^z_i\sigma^z_j}(|r_i-r_j|)
            -\Delta_{DQ}\sum_i \sigma^x_i.

    :math:`i` labels the excitons and :math:`m_x` is the effective in-plane exciton mass.
    Each exciton has two dipolar exciton states: :math:`\ket{u}` and :math:`\ket{d}`.

    The layer-dependent dipolar interaction reads:

    .. math::

        V^d_{\sigma^z_i\sigma^z_j}(r) = & V_{p,p}(r)\delta_{\sigma^z_i\sigma^z_j} +
        V_{p,-p}(r)\delta_{\sigma^z_i,-\sigma^z_j} \\
        V_{p,p}(r) = & \frac{e}{\kappa}\left(\frac{2}{r} - \frac{2}{\sqrt{r^2+d^2}} \right) \\
        V_{p,-p}(r) = & \frac{e}{\kappa}\left(\frac{1}{r} + \frac{1}{\sqrt{r^2+4d^2}}
                      - \frac{2}{\sqrt{r^2+d^2}} \right)

    :math:`\Delta_{DQ}` is the energy gap between the hole-symmetric quadrupolar
    exciton and the dipolar excitons.
    """

    def __init__(self,
        coords, # A, coordinates of excitons
        d=1.0, # A, inter-lay distance
        DQ_gap=30.0, # meV, D-Q gap
        *args, **kwargs):
        r"""initialize excitonic hubbard U model"""

        self.coords = coords
        self.d = d
        self.DQ_gap = DQ_gap
        Nexc = len(coords)

    def quantization(self):
        r"""return the discretized Hamiltonain on a real-space lattice

        i.e., :math:`H(r)\rightarrow H = T + V`.
        """

        pass

    def get_hcore(self):
        r"""Return one-body integral"""
        pass

    def get_v2b(self):
        r"""Return effective two-body integral

        Raise warning when this function is called as we don't
        construct v2b for model; Inform that get_chols should be used!"""
        pass

    def get_chols(self):
        r"""Return chols of the two-body integral"""
        pass
