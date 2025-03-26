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

**Fock state representation**:

Fock state is :math:`\ket{n_1, n_2, \cdots, n_N}`. Its size
is :math:`M^N`, where :math:`M` is the maximum occupation number
and :math:`N` is the number of modes. So space grows expoentially
with the system size, which is not a good representation for large number of modes.
This representation is implemeted for benchmark other more efficient representations.

**Coherents state representation**:

Coherent state is

.. math::
   \ket{\alpha_i} = e^{-|\alpha_i|^2/2}\sum_n \frac{\alpha^N_i}{n!}\ket{n}.


**Squeezed state representation**:

TBA.

**Identical orbital representation**:

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
import numpy as np



def coherent_state_coeff(n, alpha):
    """ Compute coefficient for Fock state |n> in a coherent state |α> """
    from scipy.special import factorial
    return np.exp(-abs(alpha)**2 / 2) * (alpha**n) / np.sqrt(factorial(n))


class TrialBosonBase(object):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def force_bias(self, walkers):
        r"""Compute the force bias"""
        pass


# ===============================================
# trial WF in the second quantization
# ===============================================

def calc_trial_walker_ovlp(phi_w, psi_T):
    r"""
    Compute bosonic trial_walker overlap
    """
    return backend.dot(phi_w, psi_T.conj())


def calc_trial_walker_ovlp_gf(walkers, trial):
    r"""
    Compute the bosonic Green's function:

    .. math::
         G_{ij} = \frac{ \bra{\Phi_T}  a^\dagger_i a_j \ket{\Phi_W}} {\langle \Phi_T  \ket{\Phi_W}}.

    Matrix Elements in the Fock Basis :math:`\ket{n_1, n_2, \cdots, \cdots, n_N}`:

    .. math::
         & a^\dagger_i \ket{n_1, n_2, \cdots, n_i, \cdots, n_N}
         = \sqrt{n_i + 1} \ket{n_1, n_2, \cdots, n_i + 1, \cdots, n_N} \\
         & a_j \ket{n_1, n_2, \cdots, n_j, \cdots, n_N}
         = \sqrt{n_j} \ket{n_1, n_2, \cdots, n_j-1, \cdots, n_N}

    Hence, the element is

    .. math::
        G_{ij} = \sum_{\boldsymbol{n}, \boldsymbol{n}'} C^*_T(\boldsymbol{n})
                 C_W(\boldsymbol{n}')\sqrt{n'_j(n_i+1)}\delta_{\boldsymbol{n}', \boldsymbol{n}'+e_i-e_j}

    :math:`\delta_{\boldsymbol{n}', \boldsymbol{n}'+e_i-e_j}` means that mode :math:`j` loses
    one particle and mode :math:`i` gains one.

    :param trial_wf: Dictionary mapping Fock states to coefficients for trial wavefunction.
    :param walker_wf: Dictionary mapping Fock states to coefficients for walker wavefunction.
    :param N: Number of bosonic modes.
    :return: Green's function matrix G of shape (N, N).
    """
    # compute ovlp
    ovlp = calc_trial_walker_ovlp(walkers.boson_phiw, trial.boson_psi)

    # compute GF

    pass


class TrivialFock(TrialBosonBase):
    r"""Trial in entire Fock space
    """
    def __init__(self, nmodes, nfock, *args, **kwargs):
        r"""Define Bosonic Trial WF in 1st quantization (coordinate space)"""
        super().__init__(*args, **kwargs)

        self.nfock = nfock
        self.nmodes = nmodes
        self.ndim = self.nfock ** self.nmodes


    def build(self, alpha=None):
        r"""initialize trial wavefunction
        alpha: array of coherent state amplitudes (shape: (N,))
        """

        count = 0
        self.boson_psi = np.zeros(self.ndim)
        for n_vec in backend.ndindex(*(self.nfock, ) * self.nmodes):
            if alpha is not None:
                # Compute coefficient using coherent state expansion
                coeff = np.prod([coherent_state_coeff(n, alpha[i]) for i, n in enumerate(n_vec)])
            else:
                n_vec = tuple(n_vec)
                coeff = np.exp(-sum(n_vec))
            self.boson_psi[count] = coeff
            count +=1
        assert count == self.ndim

    def ovlp_with_walkers_gf(self, walkers):
        r""""Compute the overlap with walkers and walker GFs
        assume the bosonic walkers is stored in walkers.boson_phiw

        """

        return calc_trial_walker_ovlp_gf(walker, self)


class CoherentState(TrialBosonBase):
    r"""Coherent state representation"""
    def __init__(self, nmodes, nfock, *args, **kwargs):
        r"""Define Bosonic Trial WF in 1st quantization (coordinate space)"""
        super().__init__(*args, **kwargs)

        self.nfock = nfock
        self.nmodes = nmodes
        self.ndim = nmodes

    def build(self, alpha):
        assert alpha.shape[0] == self.ndim
        self.boson_psi = alpha


    def ovlp_with_walkers(self, walkers):

        pass




class TrialIOR(TrialBosonBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.nmax = kwargs.get("nmax", 4)

        # in IOR, the trial wavefunction is [nmode, n_exc] matrix
        self.boson_psi = backend.zeros((self.nmodes, self.nmax))

        #




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



# examples

if __name__ == "__main__":
    print("test-bosonic trial WF (TBA)")
    phonon_trial = TrialQ(mass=2.0, freq=1.0)
    print(phonon_trial.mass)
    Q = backend.arange(0.0, 10.0, 2.0)
    print("\nLocal energy =", phonon_trial.local_energy(Q))

    # Test Fock state
    print(f"\n{'='* 25} Test fock state     {'='* 25}\n")
    nmode = 4
    nfock = 3
    alpha = np.random.random(nmode)
    print("alpha is", alpha)
    trial = TrivialFock(nmode, nfock)
    trial.build(alpha=alpha)

    class TempWalker:
        def __init__(self, nwalker, psi):
            self.boson_phiw = np.array([psi + np.random.random(psi.shape) * 0.01 for _ in range(nwalker)] )

    walker = TempWalker(100, trial.boson_psi)
    print("tria.shape = ", trial.boson_psi.shape)
    print("waker.shape =", walker.boson_phiw.shape)
    ovlp = trial.ovlp_with_walkers(walker)

    # Test coherent State
    print(f"\n{'='* 25} Test coherent state {'='* 25}\n")
    trial = CoherentState(nmode, nfock)
    trial.build(alpha=alpha)
    walker = TempWalker(100, trial.boson_psi)


