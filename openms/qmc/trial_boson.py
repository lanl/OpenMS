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
Trial WF for bosonic systems
----------------------------


Identical orbital representation:


A single-permanet, N-boson WF is

.. math::

    \ket{\phi} = \hat{\phi}^\dagger_1\hat{\phi}^\dagger_2\cdots \hat{\phi}^\dagger_N\ket{0}.

using identical-orbital representation :math:`\ket{\phi} = (\hat{\phi}^\dagger)^N\ket{0}`,
where :math:`\hat{\phi}^\dagger = \sum_\alpha c^\dagger_\alpha \phi_\alpha`.
In matrix representation, :math:`\ket{\phi}` is a MxN matrix whose columns are identicial (the unique
column is denoted as :math:`\boldsymbol{\phi}`).

Hence, the overlap is computed as

.. math::

    \bra{\psi}\phi\rangle = per(\boldsymbol{\psi}^T\cdot\boldsymbol{\phi})
                          = N!(\boldsymbol{\psi}^T\cdot\boldsymbol{\phi})^N,

And similarly,

.. math::

    \bra{\psi}\hat{A}\ket{\phi}\rangle =
    N! N (\boldsymbol{\psi}^T\cdot\boldsymbol{A}\cdot \boldsymbol{\phi})
                          (\boldsymbol{\psi}^T\cdot\boldsymbol{\phi})^{N-1},

where :math:`\boldsymbol{A}` is the matrix for :math:`\hat{A}`. The matrix element
of a two-body operator is

.. math::

    \bra{\psi} b^\dagger_\alpha b^\dagger_\beta b_\gamma b_\delta\ket{\phi} =
    N! N(N-1) \phi^*_\alpha\phi^*_\beta \phi_\gamma\phi_\delta
    (\boldsymbol{\psi}^T\cdot\boldsymbol{\phi})^{N-2}.



program overview
----------------

"""


import sys
from abc import abstractmethod
from pyscf.lib import logger
import numpy as backend


class TrialBosonBase(object):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def force_bias(self, walkers):
        r"""Compute the force bias"""
        pass


# ===============================================
# trial WF in the first quantization
# ===============================================


class TrialQ(TrialBosonBase):
    def __init__(self, *args, **kwargs):
        r"""Define Bosonic Trial WF in 1st quantization (coordinate space)"""
        super().__init__(*args, **kwargs)

        self.mass = kwargs.get("mass", 1.0)
        self.freq = kwargs.get("freq", 1.0)
        self.qshift = kwargs.get("qshift", 0.0)
        self.mw = self.mass * self.freq  # m * w
        self.Q = None  # lattice configurations

    def update_Q_shift(self, qshift):
        self.qshift = qshift.copy()

    def gradient(self, Q):
        grad = -self.mw * (Q - self.qshift)
        return grad

    def qvalue(self, Q):
        tmp = Q - self.qshift
        return backend.prod(backend.exp(-0.5 * self.mw * tmp * tmp))

    def laplacian(self, Q):
        r"""compute laplacian

        .. math::

            \nabla^2 = [\omega^2 (Q - Q_{shift})^2 - \omega]\langle Q \rangle
        """
        return (self.mw * (Q - self.qshift)) ** 2.0 - self.mw

    def local_energy(self, Q):
        r"""Compute local energy of phonons in 1st quantization

        .. math::

            E_{loc} = -\frac{1}{2m}\nabla^2 + \frac{m\omega^2}{2}Q^2 - 0.5\omega

        """

        nsites = Q.shape[0]

        kin = -0.5 * backend.sum(self.laplacian(Q)) / self.mass
        pot = 0.5 * self.mw * self.freq * backend.sum(Q * Q)
        etot = kin + pot - 0.5 * self.freq * nsites

        # print("kinetic energy =    ", kin)
        # print("potential energy =  ", pot)
        # print("no. of boson sites =", nsites)

        return etot


class TrialP(TrialBosonBase):
    def __init__(self, *args, **kwargs):
        r"""Define Bosonic Trial WF in 1st quantization (momentum space)"""
        super().__init__(*args, **kwargs)

        self.mass = kwargs.get("mass", 1.0)
        self.freq = kwargs.get("freq", 1.0)
        self.pshift = kwargs.get("pshift", 0.0)
        self.mw = self.mass * self.freq  # m * w
        self.P = None  # lattice momentum

    def gradient(self, P):
        return -1.0 / self.mw * (P - self.pshift)

    def pvalue(self, P):
        r"""
        -1/(m*w) * P
        """
        tmp = P - self.pshift
        return backend.prod(backend.exp(-0.5 / self.mw * tmp * tmp))

    def laplacian(self, P):
        r"""compute laplacian:

        .. math::

            \nabla^2 = [\omega^2 (P-P_{shift})^2 - \omega]\langle P\rangle.
        """
        return ((P - self.pshift) / self.mw) ** 2.0 - 1.0 / self.mw

    def local_energy(self, P):
        r"""Compute local energy of phonons (to be checked)

        .. math::

           E = \frac{P^2}{2m} - \frac{m\omega^2}{2}\nabla^2 -\frac{1}{2}\omega
        """
        nsites = P.shape[0]

        kin = 0.5 / self.mass * backend.sum(P * P)
        pot = -0.5 * self.mw * self.freq * backend.sum(self.laplacian(P))
        etot = kin + pot - 0.5 * self.freq * nsites

        return etot


class TrialCS(object):
    def __init__(self, system, **kwargs):
        r"""Coherent state trial for bosonic systems"""
        self.system = system
        self.name = "coherent"

    def build(self):
        pass


class TrialVLF(object):
    def __init__(self, system, **kwargs):
        r"""Variational LF transformation based trial WF."""
        self.system = system
        self.name = "coherent"

    def build(self):
        pass


# ===============================================
# trial WF in the second quantization
# ===============================================


class TrialIOR(TrialBosonBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def force_bias(self, walkers):
        r"""Compute Force bias in IOR

        TBA.

        return None
        """

if __name__ == "__main__":
    print("test-bosonic trial WF (TBA)")
    phonon_trial = TrialQ(mass=2.0, freq=1.0)
    print(phonon_trial.mass)
    Q = backend.arange(0.0, 10.0, 2.0)
    print("\nLocal energy =", phonon_trial.local_energy(Q))
