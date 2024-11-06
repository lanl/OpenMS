import itertools
from pyscf.lib import logger
import numpy as backend
import scipy

from openms.lib.logger import task_title
from abc import abstractmethod, ABC


def propagate_effective_oei(phi, system, bt2, H1diag=False):
    r"""Propagate by the kinetic term by direct matrix multiplication.

    For use with the continuus algorithm and free propagation.

    """
    if is_cupy(bt2[0]):
        import cupy

        assert cupy.is_available()
        einsum = cupy.einsum
    else:
        einsum = backend.einsum
    nup = system.nup
    # Assuming that our walker is in UHF form.
    if H1diag:
        phi[:, :nup] = einsum("ii,ij->ij", bt2[0], phi[:, :nup])
        phi[:, nup:] = einsum("ii,ij->ij", bt2[1], phi[:, nup:])
    else:
        phi[:, :nup] = bt2[0].dot(phi[:, :nup])
        phi[:, nup:] = bt2[1].dot(phi[:, nup:])

    if is_cupy(
        bt2[0]
    ):  # if even one array is a cupy array we should assume the rest is done with cupy
        import cupy

        cupy.cuda.stream.get_current_stream().synchronize()

    return


class PropagatorBase(object):
    r"""Base propagator class

    Basic function of propagator:

        - **build**: Used to build the intermediate variables that are not changing during the random walking.
        - **propagate_walkers**: Main function to propagate the walkers.
        - **propagate_walkers_onebody**: Propagate the one-body term.
        - **propagate_walkers_twobody**: Propagate the two-body term.

    """

    def __init__(self, dt=0.01, **kwargs):
        self.dt = dt
        self.verbose = kwargs.get("verbose", 1)
        self.stdout = kwargs.get("stdout", 1)
        self.energy_scheme = kwargs.get("energy_scheme", "hybrid")
        self.taylor_order = kwargs.get("taylor_order", 10)
        self.time = 0.0

        # intermediate variables
        self.TL_tensor = None
        self.exp_h1e = None
        self.nfields = None
        self.mf_shift = None

    # @abstractmethod
    def build(self, h1e, ltensor, trial, geb=None):
        r"""Build the propagators and intermediate variables

        Note the Hamiltonain in QMC format (with MF shift) is:

        .. math::

           \hat{H}_{mc} = \hat{T} + \sum_\gamma \langle L_\gamma\rangle \hat{L}_\gamma
                        + \frac{1}{2}\sum_\gamma (\hat{L}_\gamma - \langle \hat{L}_\gamma\rangle)^2
                        - \frac{1}{2}\langle \hat{L}_\gamma \rangle^2
                        + C.

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

        # trace_eri = backend.trace(eri, axis1=1, axis2=2)
        trace_eri = backend.einsum('npr,nrq->pq', ltensor.conj(), ltensor)
        shifted_h1e = h1e - 0.5 * trace_eri
        # for p, q in itertools.product(range(h1e.shape[0]), repeat=2):
        #    shifted_h1e[p, q] = h1e[p, q] - 0.5 * backend.trace(eri[p, :, :, q])

        shifted_h1e = shifted_h1e - backend.einsum(
            "n, npq->pq", self.mf_shift, 1j * ltensor
        )

        self.TL_tensor = backend.einsum("pr, npq->nrq", trial.psi.conj(), ltensor)
        self.exp_h1e = scipy.linalg.expm(-self.dt / 2 * shifted_h1e)
        self.shifted_h1e = shifted_h1e

        # logger.debug(self, "norm of shifted_h1e: %15.8f", backend.linalg.norm(shifted_h1e))
        # logger.debug(self, "norm of TL_tensor:   %15.8f", backend.linalg.norm(self.TL_tensor))

    def dump_flags(self):
        r"""dump flags (TBA)"""
        logger.note(self, task_title("Flags of propagator"))
        logger.note(self, f"Time step is            :  {self.dt:.4f}")
        logger.note(self, f"Energy scheme is        :  {self.energy_scheme}")
        logger.note(self, f"Taylor order is         :  {self.taylor_order}")
        # print(task_title(""))

    @abstractmethod
    def propagate_walkers(self, trial, walkers, vbias, ltensor, eshift, verbose=0):
        pass

    @abstractmethod
    def propagate_walkers_onebody(self, phiw):
        pass

    # @abstractmethod
    def propagate_walkers_twobody(self, hamiltonians, trial, walkers, eshift):
        pass


from openms.qmc.estimators import local_eng_boson
from openms.qmc.estimators import local_eng_elec_chol
from openms.qmc.estimators import local_eng_elec_chol_new

class Phaseless(PropagatorBase):
    r"""
    HS-transformation based AFQMC propagators

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def compute_local_energies(self, TL_theta, h1e, vbias, gf):
        r"""compute local energies"""
        # eloc = local_eng_elec_chol_new(h1e, ltensor, gf)
        return local_eng_elec_chol(TL_theta, h1e, vbias, gf)

    def propagate_walkers_onebody(self, phiw):
        r"""Propgate one-body term"""
        return backend.einsum("pq, zqr->zpr", self.exp_h1e, phiw)


    def propagate_walkers_twobody(self, trial, walkers, vbias, ltensor):
        #TODO: improve the efficiency
        r"""Propgate two-body term

        This is the major computational bottleneck.
        TODO: improve the efficiency of this part with a) MPI,
        b) GPU, and/or  c) tensor hypercontraction

        Factors due to the shift in propabalities functions (it comes from the mf shift in force bias):

        .. math::
            x\bar{x} - \frac{1}{2}\bar{x}^2

        With the mean-field shift, the force bias becomes

        .. math::
            F &\rightarrow F - \langle F\rangle \\
            \langle F\rangle & = \sqrt{-\Delta\tau} \text{Tr}[L_\gamma G]

        Hence:

        .. math::
            xF - \frac{1}{2}F^2 & = x(F-\langle F\rangle) - \frac{1}{2}(F- \langle F\rangle)^2
                                +  x\langle F\rangle - F\langle F\rangle + \frac{1}{2} \langle F\rangle^2 \\
            cmf & = (x-F) \langle F\rangle \\
            cbf & = x(F-\langle F\rangle) - \frac{1}{2}(F-\langle F\rangle)^2

        and :math:`\frac{1}{2}\langle F\rangle^2` is the propability shift. cfb is the normalizaiton
        factor :math:`N_I`.

        """

        # two-body propagator propagation :math:`\exp[(x-\bar{x}) * L]`
        # normally distributed AF
        xi = backend.random.normal(0.0, 1.0, self.nfields * walkers.nwalkers)
        xi = xi.reshape(walkers.nwalkers, self.nfields)

        xbar = -backend.sqrt(self.dt) * (1j * 2.0 * vbias - self.mf_shift)
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

        # (x*\bar{x} - \bar{x}^2/2), i.e., factors due to the shift in propabalities
        # functions (it comes from the mf shift in force bias)
        # F -> F - <F>, where F =\sqrt{-dt} Tr[LG] and <L> = Tr[L\rho_{mf}]
        # hence xF - 0.5 F^2 = x(F-<F>) - 0.5(F-<F>)^2 + x<F> - F<F> + 0.5<F>^2
        # so (x-F)<F> --> cmf
        #    x(F-<F>) - 0.5(F-<F>)^2 -- > cfb
        #    0.5 <F>^2 propability shift
        cfb = backend.einsum("zn, zn->z", xi, xbar) - 0.5 * backend.einsum(
            "zn, zn->z", xbar, xbar
        )
        # factors due to MF shift and force bias
        cmf = -backend.sqrt(self.dt) * backend.einsum("zn, n->z", xshift, self.mf_shift)

        # logger.debug(self, f"norm of cfb :   {backend.linalg.norm(cfb)}")
        # logger.debug(self, f"norm of cmf :   {backend.linalg.norm(cmf)}")

        return cfb, cmf

    def propagate_walkers(self, trial, walkers, vbias, ltensor, eshift=0.0, verbose=0):
        r"""
        Eqs 50 - 51 of Ref :cite:`zhang2021jcp`.

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
        cfb, cmf = self.propagate_walkers_twobody(trial, walkers, vbias, ltensor)

        # c):  1-body propagator propagation e^{-dt/2*H1e}
        walkers.phiw = self.propagate_walkers_onebody(walkers.phiw)
        # walkers.phiw = backend.exp(-self.dt * nuc) * walkers.phiw

        # update_weight and apply phaseless approximation
        newovlp = trial.ovlp_with_walkers(walkers)
        self.update_weight(walkers, ovlp, newovlp, cfb, cmf, eshift)

        # logger.debug(self, f"updated weight: {walkers.weights}")

    def update_weight(self, walkers, ovlp, newovlp, cfb, cmf, eshift=0.0):
        r"""
        Update the walker coefficients using two different schemes.

        a). Hybrid scheme:

        .. math::

            W^{(n+1)}_k = W^{(n)}_k \frac{\langle \Psi_T\ket{\psi^{(n+1)}_w}}
                            {\langle \Psi_T\ket{\psi^{(n)}_w}}N_I(\boldsymbol(x)_w)

        We use :math:`R` denotes the overlap ratio, and :math:`N_I=\exp[xF - F^2]`. Hence

        .. math::

            R\times N_I = \exp[\log(R) + xF - F^2]

        b). Local scheme:

        .. math::

            W^{(n+1)} = W^{(n)}_w e^{-\Delta\tau E_{loc}}.

        """

        # TODO: 1) adapt it fully to the propagator class
        # 2) introduce eshift ?

        # be cautious! power of 2 was neglected before.
        ovlp_ratio = (backend.linalg.det(newovlp) / backend.linalg.det(ovlp)) ** 2
        # FIXME: add bosonic ovlp_ratio

        # the hybrid energy scheme
        if self.energy_scheme == "hybrid":
            # print("test-yz: hybrid energy scheme is used")
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
            # print("test-yz: local energy scheme is used")
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

    def __init__(self, dt, **kwargs):
        super().__init__(dt, **kwargs)
        self.e_boson_shift = 0.0

    def propagate_walkers(self, *args, **kwargs):
        r"""
        TBA
        """
        raise NotImplementedError(
            "propagate_wakers in PhaselessBoson class is not implemented yet."
        )

    def build(self, h1b, chol_b, trial):
        r"""Build the propagator and intermediate variables

        Two-body part of bosonic Hamiltonian is:

        .. math::

          H^e_2 = & \frac{1}{2}\sum_{ijkl} V_{ijkl} b^\dagger_i b^\dagger_j b_k b_l \\
                = & \frac{1}{2}\sum_{ijkl} V_{ijkl} b^\dagger_i [b_k b^\dagger_j - \delta_{jk}] b_l \\
                = & \frac{1}{2}\sum_{ijkl} V_{ijkl} b^\dagger_i b_k b^\dagger_j b_l - \sum_{ijk} V_{ikkj} b^\dagger_i b_j \\

        Chols of two-body integrals are :math:`V_{ijkl} = \sum_\gamma L^*_{\gamma,il}L_{\gamma,kj}`.
        Hence, the total bosonic Hamiltonian is:

        .. math::

            H_{ph} = \sum_{ij}(K_{ij} - \frac{1}{2}\sum_{\gamma k} L^*_{\gamma,ij} L_{\gamma, kk})b^\dagger_i b_j
                     + \frac{1}{2}\sum_{\gamma,ijkl}L^*_{\gamma,il} L_{\gamma,kj}  b^\dagger_i b_k b^\dagger_j b_l

        After introduing the mean-field shift, the corresponding MC Hamiltonian becomes

        .. math::

            \hat{H}_{mc} = \hat{T} + \sum_\gamma \langle L_\gamma\rangle \hat{L}_\gamma +
                          \frac{1}{2}\sum_\gamma (\hat{L}_\gamma - \langle \hat{L}_\gamma\rangle)^2 +
                          C - \frac{1}{2}\langle \hat{L}_\gamma \rangle^2.

        where :math:`T_{ij} = K_{ij} - \frac{1}{2}\sum_{\gamma k} L^*_{\gamma,ij} L_{\gamma, kk}`
        and :math:`\langle L_\gamma\rangle = \sum_{ij}L_{\gamma,ij}\rho^{MF}_{ij}`.
        It's obvious that bosonic MC Hamiltonain is formally same as the fermionic one (but their
        statistics are different).
        """

        self.num_bfields = chol_b.shape[0]

        shifted_h1b = backend.zeros(h1b.shape)
        brho_mf = trial.psi.dot(trial.psi.T.conj())
        self.bmf_shift = 1j * backend.einsum("npq,pq->n", chol_b, brho_mf)

        trace_v2b = backend.einsum("nil,njj->il", chol_b.conj(), chol_b)
        shifted_h1b = h1b - 0.5 * trace_v2b  - backend.einsum(
            "n, npq->pq", self.bmf_shift, 1j * chol_b
        )

        self.TL_tensor = backend.einsum("pr, npq->nrq", trial.psi.conj(), chol_b)
        self.exp_h1b = scipy.linalg.expm(-self.dt / 2 * shifted_h1b)
        self.h1b = h1b


    def propagate_walkers_onebody(self, phiw):
        r"""Propagate one-body term"""

        return backend.einsum("pq, zqr->zpr", self.exp_h1e, phiw)

    def propagate_walkers_twobody(self, phiw):
        r"""Propagate Bosonic two-body term"""
        raise NotImplementedError(
            "propagate_wakers in PhaselessBoson class is not implemented yet."
        )


class PhaselessElecBoson(Phaseless):
    r"""Phaseless propagator for electron-Boson coupled system

    No matter bosons are in 1st or 2nd quantization, the electron onebody
    and two body samplings are the same. The only difference is the
    way of getting Qalpha which will be handled from the boson importance sampling.

    Brief introduction to the Theoretical background.

    Here we define :math:`\hat{\boldsymbol{g}}^\alpha = \boldsymbol{\lambda}_\alpha
    \cdot \hat{\boldsymbol{D}} =g^\alpha_{ij}\hat{c}^\dagger_i \hat{c}_j`
    and :math:`\hat{X}_\alpha = \sqrt{\frac{1}{2\omega_\alpha}}(b^\dagger_\alpha + b_\alpha)`.
    The Hamiltonian for the electron-boson is

    .. math::

        \hat{H} = \hat{H}_e + \hat{H}_p + \sqrt{\frac{\omega_\alpha}{2}}
        \hat{\boldsymbol{g}}^\alpha (\hat{b}^\dagger_\alpha + \hat{b}_\alpha)
        + \frac{1}{2}(\hat{\boldsymbol{g}}^\alpha)^2.

    The last DSE term is zero for some systems and :math:`\hat{H}_p =\sum_\alpha
    \omega_\alpha b^\dagger_\alpha b_\alpha`.

    With the decoupling of the bilinear term:

    .. math::

        \omega_\alpha\hat{\boldsymbol{g}}^\alpha \hat{X}_\alpha =
        \frac{1}{2} \omega_\alpha[\hat{\boldsymbol{g}}^\alpha + \hat{X}_\alpha]^2
        - \frac{1}{2} \omega_\alpha(\hat{X}_\alpha)^2
        - \frac{1}{2} \omega_\alpha(\hat{\boldsymbol{g}}^\alpha)^2

    The corresponding MC Hamiltonian is:

    .. math::

         \hat{H}_{mc} = \hat{T}_e + \frac{1}{2}\sum_\gamma \hat{L}^2_{\gamma}
                      + \frac{1}{2}\sum_\alpha
                      [\hat{L}^2_{D, \alpha} + \hat{L}^2_{gX,\alpha}
                      + \hat{L}^2_{g, \alpha} + \hat{L}^2_{X, \alpha}]
                      + \hat{H}_p.

    where,

    .. math::

       \hat{L}_{D, \alpha} = & \hat{\boldsymbol{g}}^\alpha \\
       \hat{L}_{gX,\alpha} = & \sqrt{\omega_\alpha} (\hat{\boldsymbol{g}}^\alpha  + \hat{X}_\alpha) \\
       \hat{L}_{g, \alpha} = & i\sqrt{\omega_\alpha} \hat{\boldsymbol{g}}^\alpha \\
       \hat{L}_{X, \alpha} = & i\sqrt{\omega_\alpha} \hat{\boldsymbol{X}}^\alpha

    If we don't decouple the bilinear term, the MC Hamiltonain is:

    .. math::

         \hat{H}_{mc} = \hat{T}_e + \frac{1}{2}\sum_\gamma \hat{L}^2_{\gamma}
                      + \frac{1}{2}\sum_\alpha \hat{L}^2_{D, \alpha}
                      + \sum_\alpha \omega_\alpha \hat{\boldsymbol{g}}^\alpha \hat{X}_\alpha
                      + \hat{H}_p.
    """

    def __init__(self, dt, **kwargs):
        super().__init__(dt, **kwargs)
        self.boson_quantization = kwargs.get("quantization", "second")
        self.decouple_bilinear = kwargs.get("decouple_bilinear", False)
        self.geb  = None # bilinear coupling term (without decomposition)
        self.e_boson_shift = 0.0

    def dump_flags(self):
        super().dump_flags()
        logger.note(self, f"decoupling bilinear term:  {self.decouple_bilinear}")
        logger.note(self, f"quantization of boson:     {self.boson_quantization}")
        logger.note(self, task_title("")+"\n")

    def build(self, h1e, ltensor, trial, geb=None):
        r"""Build the propagator and intermediate variables

        The QMC Hamiltonian of the coupled electron-boson system is shown above.
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

        Two ways of treating the bilinear coupling term:

        a) without decoupling: Bilinear coupling term: the contribution of bilinear coupling
        term is introduced by tracing out the bosonic DOF:

        .. math::

            h^{b}_{pq} = g_{\alpha,pq} \sqrt{\frac{\omega_\alpha}{2}}\langle b^\dagger_\alpha + b_\alpha \rangle
                       = g_{\alpha,pq} \omega_\alpha \langle \hat{X}_\alpha\rangle.

        where :math:`g_{\alpha, pq} = \bra{p} \boldsymbol{\lambda}_\alpha\cdot\boldsymbol{D} \ket{q}`

        b) decoupling the bilinear term:


        """

        super().build(h1e, ltensor, trial)

        self.geb = geb
        # if we decouple the bilinear term
        if self.decouple_bilinear:
            nmodes = self.system.nmodes
            # add shift due to bilinear term
            # FIXME: how to deal with ltensor, combine everything here
            # or separate the bilinear terms
            rho_mf = trial.psi.dot(trial.psi.T.conj())
            mf_shift = 1j * backend.einsum("npq,pq->n", self.chol_bilinear, rho_mf)
            self.shifted_h1e -= backend.einsum("n, npq->pq", mf_shift, 1j * self.chol_bilinear)


    def compute_local_energies(self, TL_theta, h1e, vbias, Gfs):
        r"""compute local energies"""

        Gf, Gb = Gfs
        # eloc = local_eng_elec_chol_new(h1e, ltensor, Gf)
        eng_e = local_eng_elec_chol(TL_theta, h1e, vbias, Gf)

        # bosonc energy
        #print("Gb =", Gb)
        eb = local_eng_boson(self.system.omega, self.system.nboson_states, Gb)

        # bilinear energy
        nmodes = self.system.nmodes
        boson_size = sum(self.system.nboson_states)
        Hb = backend.zeros((boson_size, boson_size), dtype=backend.complex128)
        zalpha = backend.einsum("npq, zpq->zn", self.geb, Gf)
        idx = 0
        for imode in range(nmodes):
            mdim = self.system.nboson_states[imode]
            a = backend.diag(backend.sqrt(backend.arange(1, mdim)), k = 1)
            h_od = a + a.T
            Hb[idx:idx+mdim, idx:idx+mdim] = h_od * zalpha[imode]
        ep = backend.einsum( "NM,zNM->z", Hb, Gb)
        # print("photon energy, bilinear term are ", eb, ep)
        return eng_e + eb + ep

    def propagate_walkers_twobody_1st(self, walkers, trial):
        r"""Propagate by potential term using discrete HS transform."""
        # Construct random auxilliary field.
        nalpha = delta = self.delta
        nup = system.nup
        soffset = walkers.phiw.shape[0] - system.nbasis
        for i in range(0, system.nbasis):
            self.update_greens_function(walker, trial, i, nup)

            # TODO: Ratio of determinants for the two choices of auxilliary fields
            # probs = calculate_overlap_ratio(walker, delta, trial, i)

            if self.charge:
                probs *= self.charge_factor

            if False:
                const = (
                    -self.gamma
                    * system.g
                    * math.sqrt(2.0 * system.m * system.w0)
                    / system.U
                    * walker.X[i]
                )
                factor = backend.array([backend.exp(const), backend.exp(-const)])
                probs *= factor

            # issues here with complex numbers?
            phaseless_ratio = backend.maximum(probs.real, [0, 0])
            norm = sum(phaseless_ratio)
            r = backend.random.random()

            # Is this necessary?
            # todo : mirror correction
            if norm > 0:
                walker.weight = walker.weight * norm
                if r < phaseless_ratio[0] / norm:
                    xi = 0
                else:
                    xi = 1
                vtup = walkers.phiw[i, :nup] * delta[xi, 0]
                vtdown = walkers.phiw[i + soffset, nup:] * delta[xi, 1]
                walkers.phiw[i, :nup] = walkers.phiw[i, :nup] + vtup
                walkers.phiw[i + soffset, nup:] = (
                    walkers.phiw[i + soffset, nup:] + vtdown
                )
                walker.update_overlap(probs, xi, trial.coeffs)
                if walker.field_configs is not None:
                    walker.field_configs.push(xi)
                walker.update_inverse_overlap(trial, vtup, vtdown, i)
            else:
                walker.weight = 0

    def propagate_walkers_onebody(self, phiw):  # walker, system, trial, dt):
        r"""Propgate one-body term:
        including both h1e and bilinear coupling parts

        Parameters
        ----------
        walker :
            Walker object to be updated. On output we have acted on phi by
            :math:`B_{\Delta\tau/2}` and updated the weight appropriately. Updates inplace.
        system :
            System object.
        trial :
            Trial wavefunction object.


        step 1) is to compute shifted_h1e with contribution from bilinear coupling as

        .. math::

            T^{eff}_{pq} = h_{pq} - \frac{1}{2} L_{npr} L^*_{nqr} - \sum_n \langle L_n\rangle L_{n,pq}
            - \sum_{\alpha} g_{\alpha,pq} \langle z_{\alpha} \rangle.

        where :math:`z_\alpha= Tr[gD]` and :math:`D` is the density matrix, i.e.,
        :math:`z_\alpha` is the displacement

        step 2) is to update the walker WF due to the effective one-body operator (formally
        same as the bare electronic case).

        """

        # print("test-yz: propagate_walkers_onebody in phaselessElecBoson is called")

        # bt2 = [scipy.linalg.expm(-dt*system.T[0]), scipy.linalg.expm(-dt*system.T[1])]
        # propagate_effective_oei(walkers.phiw, system, bt2, H1diag=False)

        # Add the contribution from the bilinear term which requres the
        # updated the bilinear coupling term according to new Q (and bosonic WF)

        # 1) second quantizaiton
        # Pseudocode:
        #    nmoade = gmat.shape[0]
        #    Qalpha = walkers.get_Qalpha()
        #    oei = backend.einsum("npq, n->pq", Qalpha, gmat)
        #    shifted_h1e = self.shifted_h1e + oei
        #    self.exp_h1e = scipy.linalg.expm(-self.dt / 2 * shifted_h1e)
        #

        oei = backend.zeros(self.shifted_h1e.shape)

        if not self.decouple_bilinear and self.geb is not None:
            # print("add bilianr coupling oei")
            zlambda = backend.einsum("pq, Xpq ->X", self.DM, self.geb)
            oei = backend.einsum("X, Xpq->pq", zlambda, self.geb)

        # 2) oei_qed in 1st quantization

        """
        Qalpha = backend.ones(nmode)

        if not False:
            # Tr_Q [D * Q], traceout bosonic DOF
            const = gmat * cmath.sqrt(system.mass * system.freq * 2.0) * dt
            const = const.real
            Qso = [walkers.Q, walkers.Q]

            gmat = backend.zeros_like(h1e)

            # update effective oei
            oei = h1e + gmat * const

            # Veph = [backend.diag( backend.exp(const * Qso[0]) ),backend.diag( backend.exp(const * Qso[1]) )]
            # propagate_effective_oei(walkers.phiw, system, Veph, H1diag=True)
            exp_h1e = [scipy.linalg.expm(-dt * oei[0]), scipy.linalg.expm(-dt * oei[1])]
            # print(walkers.phiw.dtype, walker.X.dtype, const)
            propagate_walkers_one_body(walkers.phiw, exp_h1e)
            # propagate_effective_oei(walkers.phiw, system, TV, H1diag=False)

        # Update inverse overlap
        walker.inverse_overlap(trial)
        # Update walker weight
        ot_new = walker.calc_otrial(trial)

        ratio = ot_new / walker.ot
        phase = cmath.phase(ratio)

        if abs(phase) < 0.5 * math.pi:
            (magn, phase) = cmath.polar(ratio)
            cosine_fac = max(0, math.cos(phase))
            walker.weight *= magn * cosine_fac
            walker.ot = ot_new
        else:
            walker.ot = ot_new
            walker.weight = 0.0
        """

        shifted_h1e = self.shifted_h1e + oei
        self.exp_h1e = scipy.linalg.expm(-self.dt / 2 * shifted_h1e)
        return backend.einsum("pq, zqr->zpr", self.exp_h1e, phiw)

    def propagate_decoupled_bilinear(self, trial, walkers):
        r"""Propagate the bilinear term in decoupled formalism

        Any coupling between the fermions and bosons can be decoupled as

        .. math::

            \hat{O}_f\hat{O}_b = \frac{1}{2}\left[(\hat{O}_f + \hat{O}_b)^2 -
            \hat{O}^2_f - \hat{O}^2_b\right].
        """
        pass

    def propagate_bosons(self, trial, walkers):
        r"""
        boson importance sampling
        """
        if "first" in self.boson_quantization:
            self.propagate_bosons_1st(trial, walkers)
        else:
            self.propagate_bosons_2nd(trial, walkers)

    def propagate_bosons_2nd(self, trial, walkers):
        r"""Boson importance sampling in 2nd quantization
        Ref. :cite:`Purwanto:2004qm`.

        Bosonic Hamiltonian is:

        .. math::

            H_{ph} =\omega_\alpha b^\dagger_\alpha b_\alpha

        Since this part diagonal, we use 1D array to store the diaognal elements
        """

        # print("propagating free bosonic part")

        basis = backend.asarray(
            [backend.arange(mdim) for mdim in self.system.nboson_states]
        )
        waTa = backend.einsum("m, mF->mF", self.system.omega, basis).ravel()

        evol_Hph = backend.exp(-0.5 * self.dt * waTa)
        walkers.phiw_b = backend.einsum( "F, zF->zF", evol_Hph, walkers.phiw_b)

    def propagate_bosons_1st(self, trial, walkers):
        r"""Boson importance sampling in 1st quantization formalism

        DQMC type of algorithm:

        .. math::

            Q^{n+1} = Q^n + R(\sigma) + \frac{\Delta\tau}{m}\frac{\nabla_Q \Psi_T(Q)}{\Psi_T(Q)}

        Here R is a normally distributed random number with variance :math:`\sigma=\sqrt{m\Delta u}`.
        The last term is the drift term.

        i.e., Qnew = Qold + dQ + drift

        TBA.
        """

        mass = 1.0  # self.system.mass
        sqrtdt = backend.sqrt(self.dt / mass)
        return
        phiold = trial.value(walker)

        # Drift+diffusion
        driftold = (self.dt / system.m) * trial.gradient(walker)

        if False:
            Ev = (
                0.5
                * system.m
                * system.w0**2
                * (1.0 - 2.0 * system.g**2 / (system.w0 * system.U))
                * backend.sum(walker.X * walker.X)
            )
            Ev2 = (
                -0.5
                * backend.sqrt(2.0 * system.m * system.w0)
                * system.g
                * backend.sum(walker.X)
            )
            lap = trial.laplacian(walker)
            Ek = 0.5 / (system.m) * backend.sum(lap * lap)
            elocold = Ev + Ev2 + Ek
        else:
            elocold = trial.bosonic_local_energy(walker)

        elocold = backend.real(elocold)

        dX = backend.random.normal(loc=0.0, scale=sqrtdtm, size=(system.nbasis))
        Xnew = walker.X + dX + driftold

        walker.X = Xnew.copy()

        phinew = trial.value(walker)
        lap = trial.laplacian(walker)
        walker.Lap = lap

        # Change weight
        if False:
            Ev = (
                0.5
                * system.m
                * system.w0**2
                * (1.0 - 2.0 * system.g**2 / (system.w0 * system.U))
                * backend.sum(walker.X * walker.X)
            )
            Ev2 = (
                -0.5
                * backend.sqrt(2.0 * system.m * system.w0)
                * system.g
                * backend.sum(walker.X)
            )
            lap = trial.laplacian(walker)
            Ek = 0.5 / (system.m) * backend.sum(lap * lap)
            eloc = Ev + Ev2 + Ek
        else:
            eloc = trial.bosonic_local_energy(walker)

        eloc = backend.real(eloc)
        walker.ot *= phinew / phiold

        walker.weight *= math.exp(
            -0.5 * self.dt * (eloc + elocold - 2 * self.eshift_boson)
        )


    def propagate_bosons_bilinear(self, trial, walkers, dt):

        r"""propagate the twobody bosonic term due to the
        decomposition of bilinear term

        .. math::

            \hat{X}_\alpha = \frac{1}{2\omega_\alpha}(\hat{b}^\dagger + \hat{b}_\alpha)

        """

        # note geb already has sqrt(w/2)
        nmodes = self.system.nmodes
        boson_size = sum(self.system.nboson_states)
        Hb = backend.zeros((boson_size, boson_size), dtype=backend.complex128)

        # compute local bosonic energy
        elocold = 0.0

        if self.decouple_bilinear:
            # print("test-yz: propagating the bilinear term in decoupled formalism")
            # (1+i) \sqrt{w) * X = (1+i)/sqrt(2) (b^\dagger + b)
            idx = 0
            for imode in range(nmodes):
                mdim = self.system.nboson_states[imode]
                a = backend.diag(backend.sqrt(backend.arange(1, mdim)), k = 1)
                h_od = (a + a.T) * (0.5 / self.system.omega[imode]) ** 0.5
                Hb[idx:idx+mdim, idx:idx+mdim] = h_od * self.chol_Xa[imode]

            # didn't apply mean-field shift to the bosonic part
            # so no bias as well.
            xi = backend.random.normal(0.0, 1.0, walkers.nwalkers)
            tau = 1j * backend.sqrt(dt)
            op_power = tau * backend.einsum("z, NM->zNM", xi, Hb)

            temp = walkers.phiw_b.copy()
            for order_i in range(self.taylor_order):
                temp = backend.einsum("zNM, zM->zN", op_power, temp) / (order_i + 1.0)
                walkers.phiw_b += temp

        else:
            # print("test-yz: propagating the bilinear term in product formalism")
            # may move this part into the propagate_boson
            zlambda = backend.einsum("pq, Xpq ->X", self.DM, self.geb)
            idx = 0
            for imode in range(nmodes):
                mdim = self.system.nboson_states[imode]
                a = backend.diag(backend.sqrt(backend.arange(1, mdim)), k = 1)
                h_od = a + a.T
                Hb[idx:idx+mdim, idx:idx+mdim] = h_od * zlambda[imode]

            # exp(-\sqrt{w/2} g c^\dag_i c_j (b^\dag + b)) | n>
            evol_Hep = backend.exp(-dt * Hb)
            walkers.phiw_b = backend.einsum( "NM, zM->zN", evol_Hep, walkers.phiw_b)

        # TODO: incorpoate the bosonic energy in the weight updtes
        # compute bosonic energy
        eloc = 0.0
        # TODO: make 0.5*dt as an input variable into this function so that
        # we can easily do either half or a whole step
        walkers.weights *= backend.exp(-dt * (eloc + elocold - 2.0 * self.e_boson_shift) / 2.0)


    def propagate_walkers(self, trial, walkers, vbias, ltensor, eshift=0.0, verbose=0):
        r"""Propagate the walkers function for the coupled electron-boson interactions"""

        ovlp = trial.ovlp_with_walkers(walkers)
        ovlp_b = trial.bovlp_with_walkers(walkers)

        # 1) boson propagator
        self.propagate_bosons(trial, walkers)

        # TODO: decide whether to store Gf in walkers or walkers
        inv_ovlp = backend.linalg.inv(ovlp)
        theta = backend.einsum("zqp, zpr->zqr", walkers.phiw, inv_ovlp)
        self.Gf = backend.einsum("zqr, pr->zpq", theta, trial.psi.conj())

        self.DM = (
            2.0
            * backend.einsum("z, zpq->pq", walkers.weights, self.Gf)
            / backend.sum(walkers.weights)
        )
        #print("test DM=", backend.trace(self.DM))

        # Note: since 2-4 are similar to bare case, we may recycle the bare code
        # but with updated ltensors and shifted h1e.
        # super().propagate_walkers(trial, walkers, vbias, ltensor, eshift=eshift)

        # 2) 1-body propagator propagation :math:`e^{-dt/2*H1e}`
        walkers.phiw = self.propagate_walkers_onebody(walkers.phiw)

        # 3) two-body term
        # walkers.phiw = self.propagate_walkers_twobody(walkers.phiw)

        ltmp = ltensor.copy()
        # 3b) add the contribution due to the decomposition of bilinear term
        #if self.decouple_bilinear:
        #    ltmp = backend.concatenate((ltmp, self.chol_bilinear), axis=0)
        cfb, cmf = self.propagate_walkers_twobody(trial, walkers, vbias, ltmp)
        del ltmp

        # 3c) two-body bosonic term (we can do split step as well)
        self.propagate_bosons_bilinear(trial, walkers, self.dt)

        # 4) one-body electronic term
        walkers.phiw = self.propagate_walkers_onebody(walkers.phiw)

        # 5) boson propagator
        self.propagate_bosons(trial, walkers)

        # 6) update weights
        newovlp = trial.ovlp_with_walkers(walkers)
        self.update_weight(walkers, ovlp, newovlp, cfb, cmf, eshift)
        if verbose:
            print("test-yz: bosonic WF", abs(backend.sum(walkers.phiw_b, axis=0)) / walkers.nwalkers)
