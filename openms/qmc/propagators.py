import itertools
from pyscf.lib import logger
import numpy as backend
import scipy

from openms.lib.logger import task_title
from abc import abstractmethod, ABC

class PropagatorBase(ABC):

    r"""
    Base propagator class
    """

    def __init__(self, dt=0.01, **kwargs):
        self.dt = dt
        verbose = kwargs.get("verbose", 1)
        stdout = kwargs.get("stdout", 1)

        self.energy_scheme = kwargs.get("energy_scheme", "hybrid")
        self.taylor_order = kwargs.get("taylor_order", 10)
        self.verbose = verbose
        self.time = 0.0

        # intermediate variables
        self.TL_tensor = None
        self.exp_h1e = None
        self.nfields = None
        self.mf_shift = None

    # @abstractmethod
    def build(self, h1e, eri, ltensor, trial):
        r"""Build the propagators and intermediate variables

        Note the Hamiltonain in QMC format (with MF shift) is:

        .. math::

            \hat{H}_{mc} = \hat{T} + \sum_\gamma \langle L_\gamma\rangle \hat{L}_\gamma +
                          \frac{1}{2}\sum_\gamma (\hat{L}_\gamma - \langle \hat{L}_\gamma\rangle)^2 +
                          C - \frac{1}{2}\langle \hat{L}_\gamma \rangle^2.

        where :math:`\langle \hat{L}_{\gamma} \rangle = \sum_{pq} L_{\gamma,pq} \rho^{MF}_{pq}`
        and :math:`\hat{T}` operator is:

        .. math::

            T_{pq} = h_{pq} - \frac{1}{2}\sum_{rr} I_{pqrr}
                   = h_{pq} - \frac{1}{2}\sum_{\gamma r} L_{\gamma,pr}L^*_{\gamma,qr}

        Hence, after extracting the MF shift, the final shifted oei is:

        .. math::

            T^{eff}_{pq} = [T_{pq} - \frac{1}{2}\sum_{\gamma r} L_{\gamma,pr}L^*_{\gamma,qr}]
                         + \sum_{\gamma} \langle \hat{L}_{\gamma}\rangle L_{\gamma, pq}

        """
        self.nfields = ltensor.shape[0]

        shifted_h1e = backend.zeros(h1e.shape)
        rho_mf = trial.psi.dot(trial.psi.T.conj())
        self.mf_shift = 1j * backend.einsum("npq,pq->n", ltensor, rho_mf)

        for p, q in itertools.product(range(h1e.shape[0]), repeat=2):
            shifted_h1e[p, q] = h1e[p, q] - 0.5 * backend.trace(eri[p, :, :, q])
        shifted_h1e = shifted_h1e - backend.einsum(
            "n, npq->pq", self.mf_shift, 1j * ltensor
        )

        self.TL_tensor = backend.einsum("pr, npq->nrq", trial.psi.conj(), ltensor)
        self.exp_h1e = scipy.linalg.expm(-self.dt / 2 * shifted_h1e)
        # logger.debug(self, "norm of shifted_h1e: %15.8f", backend.linalg.norm(shifted_h1e))
        # logger.debug(self, "norm of TL_tensor:   %15.8f", backend.linalg.norm(self.TL_tensor))

    def dump_flags(self):
        r"""dump flags (TBA)
        """
        print(task_title("Flags of propagator"))
        print(f"Time step is       :  {self.dt:.4f}")
        print(task_title(""))


    @abstractmethod
    def propagate_walkers(self, trial, walkers, vbias, eshift):
        pass

    @abstractmethod
    def propagate_walkers_onebody(self, hamiltonians, trial, walkers, eshift):
        pass

    # @abstractmethod
    def propagate_walkers_twobody(self, hamiltonians, trial, walkers, eshift):
        pass


class Phaseless(PropagatorBase):
    r"""
    HS-transformation based AFQMC propagators
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def propagate_walkers_onebody(self, phiw):
        r"""Propgate one-body term"""
        return backend.einsum("pq, zqr->zpr", self.exp_h1e, phiw)

    def propagate_walkers(self, trial, walkers, vbias, ltensor, eshift=0.0):
        r"""
        Eqs 50 - 51 of Ref: https://www.cond-mat.de/events/correl13/manuscripts/zhang.pdf

        Trotter decomposition of the imaginary time propagator:

        .. math::

            e^{-\Delta\tau/2 H_1} e^{-\Delta\tau \sum_\gamma L^2_\gamma /2 } e^{-\Delta\tau H_1/2}

        where the two-body propagator in HS form

        .. math::

            e^{-\Delta\tau L^2_\gamma} \rightarrow  \exp[x\sqrt{-\Delta\tau}L_\gamma]
            = \sum_n \frac{1}{n!} [x\sqrt{-\Delta\tau}L_\gamma]^n
        """
        ovlp = trial.ovlp_with_walkers(walkers)

        # a) 1-body propagator propagation :math:`e^{-dt/2*H1e}`
        walkers.phiw = self.propagate_walkers_onebody(walkers.phiw)

        # b): 2-body propagator propagation :math:`\exp[(x-\bar{x}) * L]`
        # normally distributed AF
        xi = backend.random.normal(0.0, 1.0, self.nfields * walkers.nwalkers)
        xi = xi.reshape(walkers.nwalkers, self.nfields)

        xbar = -backend.sqrt(self.dt) * (1j * 2 * vbias - self.mf_shift)
        xshift = xi - xbar
        # TODO: further improve the efficiency of this part
        eri_op_power = (
            1j * backend.sqrt(self.dt) * backend.einsum("zn, npq->zpq", xshift, ltensor)
        )

        # \sum_n 1/n! (j\sqrt{\Delta\tau) xL)^n
        temp = walkers.phiw.copy()
        for order_i in range(self.taylor_order):
            temp = backend.einsum("zpq, zqr->zpr", eri_op_power, temp) / (order_i + 1.0)
            walkers.phiw += temp

        # c):  1-body propagator propagation e^{-dt/2*H1e}
        walkers.phiw = self.propagate_walkers_onebody(walkers.phiw)
        # walkers.phiw = backend.exp(-self.dt * nuc) * walkers.phiw

        # (x*\bar{x} - \bar{x}^2/2)
        cfb = backend.einsum("zn, zn->z", xi, xbar) - 0.5 * backend.einsum(
            "zn, zn->z", xbar, xbar
        )
        cmf = -backend.sqrt(self.dt) * backend.einsum("zn, n->z", xshift, self.mf_shift)

        # update_weight and apply phaseless approximation
        newovlp = trial.ovlp_with_walkers(walkers)
        self.update_weight(walkers, ovlp, newovlp, cfb, cmf, eshift)

        # logger.debug(self, f"norm of cfb :   {backend.linalg.norm(cfb)}")
        # logger.debug(self, f"norm of cmf :   {backend.linalg.norm(cmf)}")
        # logger.debug(self, f"updated weight: {walkers.weights}")

    def update_weight(self, walkers, ovlp, newovlp, cfb, cmf, eshift=0.0):
        r"""
        Update the walker coefficients using two different schemes.

        a). Hybrid scheme:

        .. math::

              W^{(n+1)} =

        b). Local scheme:

        .. math::

              W^{(n+1)} =
        """

        # TODO: 1) adpat it fully to the propagator class
        # 2) introduce eshift ?

        # be cautious! power of 2 was neglected before.
        ovlp_ratio = (backend.linalg.det(newovlp) / backend.linalg.det(ovlp)) ** 2

        # the hybrid energy scheme
        if self.energy_scheme == "hybrid":
            self.ebound = (2.0 / self.dt) ** 0.5
            ehybrid = -(backend.log(ovlp_ratio) + cfb + cmf) / self.dt
            ehybrid = backend.clip(
                ehybrid.real,
                a_min=-self.ebound,
                a_max=self.ebound,
                out=ehybrid.real,
            )
            walkers.ehybrid = ehybrid if walkers.ehybrid is None else walkers.ehybrid

            importance_func = backend.exp(
                -self.dt * (0.5 * (ehybrid + walkers.ehybrid) - eshift)
            )
            walkers.ehybrid = ehybrid
            phase = (-self.dt * walkers.ehybrid - cfb).imag
            phase_factor = backend.array(
                [max(0, backend.cos(iphase)) for iphase in phase]
            )
            importance_func = backend.abs(importance_func) * phase_factor

        elif self.energy_scheme == "local":
            # The local energy formalism
            ovlp_ratio = ovlp_ratio * backend.exp(cmf)
            phase_factor = backend.array(
                [max(0, backend.cos(backend.angle(iovlp))) for iovlp in ovlp_ratio]
            )
            importance_func = (
                backend.exp(-self.dt * backend.real(walkers.eloc)) * phase_factor
            )

        else:
            raise ValueError(f"scheme {self.energy_scheme} is not available!!!")

        walkers.weights *= importance_func


#
class PhaselessBoson(Phaseless):
    r"""Phaseless propagator for Bosons"""

    def __init__(self, dt, verbose=1):
        super().__init__(dt, verbose=verbose)

    def propagate_walkers(self):
        r"""
        TBA
        """
        raise NotImplementedError(
            "propagate_wakers in PhaselessBoson class is not implemented yet."
        )


class PhaselessElecBoson(Phaseless):
    r"""Phaseless propagator for electron-Boson coupled system

    No matter bosons are in 1st or 2nd quantization, the electron onebody
    and two body samplings are the same. The only difference is the
    way of getting Qalpha which will be handled from the boson importance sampling.
    """

    def __init__(self, dt, verbose=1, **kwargs):
        super().__init__(dt, verbose=verbose, **kwargs)
        self.boson_quantization = kwargs.get("quantization", "first")

    def build(self, h1e, eri, ltensor, trial):
        r"""The QMC Hamiltonian of the coupled electron-boson system is
        (We consider DSE in the general form, we can simply set DSE to be
        zero for the cases without DSE terms)

        .. math::

            \hat{H}_{MC} = & \hat{H}^e_{MC} + \frac{1}{2}\sum_\alpha (\boldsymbol{\lambda}_\alpha\cdot\boldsymbol{D})^2
                         + \sum_\alpha \sqrt{\frac{\omega_\alpha}{2}} (\mathbf{\lambda}_\alpha\cdot\boldsymbol{D})
                         + \sum_\alpha \omega_\alpha b^\dagger_\alpha b_\alpha \\
                         = & \hat{H}^e_{MC} + \frac{1}{2}\sum_\alpha L_{\alpha, pq} L^*_{\alpha, rs}
                         + \sum_\alpha \sqrt{\frac{\omega_\alpha}{2}} (\mathbf{\lambda}_\alpha\cdot\boldsymbol{D})
                         + \sum_\alpha \omega_\alpha b^\dagger_\alpha b_\alpha. \\

        Hence, DSE term naturally adds :math:`N_\alpha` (number of bosonic modes) fields into the chols.

        i.e., :math:`\{L_\gamma\} = \{L_\gamma, L_\alpha\}`, i.e., the new letnsor in this code
        is the combination of electronic chols and :math:`L_\alpha`.

        Bilinear coupling term: the contribution of bilinear coupling term is introduced
        by tracing out the bosonic DOF:

        .. math::

            h^{b}_{pq} = g_{\alpha,pq} \sqrt{\frac{\omega_\alpha}{2}}\langle b^\dagger_\alpha + b_\alpha \rangle
                       = g_{\alpha,pq} \langle \omega_\alpha Q_\alpha\rangle.

        where :math:`g_{\alpha, pq} = \bra{p} \boldsymbol{\lambda}_\alpha\cdot\boldsymbol{D} \ket{q}`
        and :math:`Q_\alpha = \sqrt{\frac{1}{2\omega_\alpha}}(b^\dagger_\alpha + b_\alpha)`.
        """

        super().build(h1e, eri, ltensor, trial)

    def propagate_walkers_two_body(self, walkers, trial):
        r"""Propagate by potential term using discrete HS transform."""
        pass

    def propagate_walkers_onebody(self, walker, system, trial, dt):
        r"""Propgate one-body term:
        including both h1e and bilinear coupling parts

        step 1):  compute shifted_h1e with contribution from bilinear coupling:

        .. math::

            T^{eff}_{pq} = h_{pq} - \frac{1}{2} L_{npr} L^*_{nqr} - \sum_n <L_n> L_{n,pq}
                         - \sum_\alpha g_{\alpha,pq} <Q_\alpha>.

        step 2): update the walker WF due to the effective one-body operator (formally
        same as the bare electronic case).
        """

        # Pseudocode:

        #    nmoade = gmat.shape[0]
        #    Qalpha = walkers.get_Qalpha()
        #    oei = backend.einsum("npq, n->pq", Qalpha, gmat)
        #    shifted_h1e = self.shifted_h1e + oei
        #    self.exp_h1e = scipy.linalg.expm(-self.dt / 2 * shifted_h1e)
        #
        #    walkers.phiw = super().propagate_walkers_onebody(walkers.phiw
        pass
    def propagate_walkers(self, trial, walkers, vbias, eshift=0.0):
        r"""Propagate the walkers function for the coupled electron-boson interactions"""

        # 1) boson propagator
        # walkers.phiw = self.propagate_bosons(self, trials, walkers)

        # Note: since 2-4 are similar to bare case, we may recycle the bare code
        # but with updated ltensors and shifted h1e.

        super().propagate_walkers(trial, walkers, vbias, eshift=eshift)

        # 2) 1-body propagator propagation :math:`e^{-dt/2*H1e}`
        # walkers.phiw = self.propagate_walkers_onebody(walkers.phiw)

        # 3) two-body term
        # walkers.phiw = self.propagate_walkers_twobody(walkers.phiw)

        # 4) one-body term
        # walkers.phiw = self.propagate_walkers_onebody(walkers.phiw)

        # 5) boson propagator
        # walkers.phiw = self.propagate_bosons(self, trials, walkers)

        raise NotImplementedError(
            "propagate_wakers in PhaselessElecBoson class is not implemented yet."
        )
