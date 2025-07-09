import itertools
import time
from openms.lib import logger
import numpy as backend
import scipy

from openms.lib.logger import task_title
from openms.lib import NUMBA_AVAILABLE, QMCLIB_AVAILABLE
from openms.__mpi__ import MPI, original_print
from abc import abstractmethod, ABC


# this function will be moved into lib/boson
def boson_adag_plus_a(nmodes, boson_states, za):
    r"""
    Get matrix representation of :math:`\sum_\alpha z_\alpha(a^\dagger_\alpha + a_\alpha)`
    in the Fock-state basis.

    Parameters
    ----------
    nmodes : int
        Number of bosonic modes.
    boson_states : int or list/1D array of ints
        Number of Fock states for each mode.
        - If an integer, it is assumed each mode has that many states (0..boson_states-1).
        - If a list/array, must have length = nmodes.
    za : 1D array
        Coefficients :math:`z_\alpha` for each mode.

    Returns
    -------
    A : 2D numpy.ndarray (square)
        The matrix representation of
        :math:`\sum_\alpha z_\alpha \bigl(a^\dagger_\alpha + a_\alpha\bigr)`
        in the chosen Fock basis.
    """
    # If boson_states is a single integer, replicate it for each mode
    if isinstance(boson_states, int):
        nboson_states = [boson_states] * nmodes
    else:
        nboson_states = list(boson_states)
        if len(nboson_states) != nmodes:
            raise ValueError("Length of boson_states must match nmodes.")

    nboson_states = backend.array(nboson_states, dtype=int)
    za = backend.array(za)
    if len(za) != nmodes:
        raise ValueError("Length of za must match nmodes.")

    boson_size = sum(nboson_states)
    Hb = backend.zeros((boson_size, boson_size), dtype=backend.complex128)
    idx = 0
    for imode in range(nmodes):
        mdim = nboson_states[imode]
        # off-diaognal term
        a = backend.diag(backend.sqrt(backend.arange(1, mdim)), k=1)
        h_od = a + a.T
        Hb[idx:idx+mdim, idx:idx+mdim] = h_od * za[imode]
        idx += mdim

    return Hb

def propagate_onebody(op, phi):
    r""" Base function for propagating onebody operator

    op: (n, n) array
    phi: (nw, n, m) array

    return:
    phi: (nw, n, m) array

    """

    # The loop is faster than the einsum
    if False: # TODO: determine which method to contract the tensor
        phi = backend.einsum("pq, zqr->zpr", op, phi)
    else:
        for iw in range(phi.shape[0]):
            phi[iw] = backend.dot(op, phi[iw])
    return phi

if NUMBA_AVAILABLE:
    from numba import njit, prange

    @njit(parallel=True, fastmath=True)
    def propagate_onebody_numba(op, phi):
        r""" Base function for propagating onebody operator

        op: (n, n) array
        phi: (nw, n, m) array

        return:
        phi: (nw, n, m) array

        """

        nwalkers = phi.shape[0]
        for iw in prange(nwalkers):
            phi[iw] = op @ phi[iw]
        return phi


    @njit(parallel=True, fastmath=True)
    def propagate_exp_op_numba(phiw, op, order):
        """
        Apply exponential operator via Taylor expansion:

        .. math::
            e^{A}\phi = \sum_n A^n/n! \phi

        Parameters
        ----------
        phiw : np.ndarray of shape (nwalkers, ndim)
            Walker wavefunctions
        op : np.ndarray of shape (nwalkers, ndim, ndim)
            Operator A applied to each walker
        order : int
            Order of Taylor expansion
        """
        nwalkers = phiw.shape[0]

        for iw in prange(nwalkers):
            temp = phiw[iw].copy()
            for i in range(order):
                temp = op[iw] @ temp / (i + 1.0)
                phiw[iw] += temp
        return phiw

    @njit(parallel=True, fastmath=True)
    def compute_GF_base(Ghalf, psi):
        # Ghalfa: (z, q, i)
        # psia: (p, i)

        nw, nao, no = Ghalf.shape
        result = backend.empty((nw, nao, nao), dtype=Ghalf.dtype)

        for z in prange(nw):
            result[z] = backend.dot(psi.astype(backend.complex128), Ghalf[z].T)
        return result


    @njit(parallel=True, fastmath=True)
    def tensor_dot_numba(xshift, ltensor, factor):
        """
        Numba-compatible and parallelized build_HS function.

        Parameters
        ----------
        xshift : (nwalkers, nchol)
            Random numbers for HS transformation.
        ltensor : (nchol, nao, nao)
            Cholesky-decomposed tensor.
        factor : scalar
            Scaling factor (e.g., sqrt(dt)).
        nwalkers : int
            Number of walkers.

        Returns
        -------
        eri_op : (nwalkers, nao, nao)
            Two-body propagator operator.
        """
        nchol, nao, _ = ltensor.shape
        nwalkers = xshift.shape[0]

        # eri_op = xshift @ ltensor.reshape(nchol, nao * nao).astype(backend.complex128)
        # eri_op = factor * eri_op.reshape(nwalkers, nao, nao)
        #  # n, npq-> pq
        #  eri_op = backend.zeros((nwalkers, nao, nao), dtype=backend.complex128)
        #  for w in prange(nwalkers):
        #      for l in range(nchol):
        #          eri_op[w] += xshift[w, l] * ltensor[l]
        #  eri_op *= factor
        #  return eri_op
        reshaped = ltensor.reshape(nchol, -1)
        dot_product = backend.dot(xshift, reshaped)
        result = dot_product.reshape(nwalkers, nao, nao)
        return factor * result

    build_HS_numba = tensor_dot_numba

else:

    def compute_GF_base(Ghalf, psi):
        # Ghalfa: (z, q, i)
        # psia: (p, i)

        # walkers.Ga = backend.einsum("zqi, pi->zpq", walkers.Ghalfa, trial.psia.conj())
        # or
        # temp = np.tensordot(walkers.Ghalfa, trial.psia.conj(), axes=([2], [1]))  # shape: (z, q, p)
        # walkers.Ga = np.transpose(temp, (0, 2, 1))
        nw, nao, no = Ghalf.shape
        result = backend.empty((nw, nao, nao), dtype=Ghalf.dtype)
        for z in range(nw):
            result[z] = backend.dot(psi, Ghalf[z].T)
        return result


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


def build_HS(xshift, ltensor, factor):
    r"""xshift @ ltensor"""

    nchol, nao = ltensor.shape[:-1]
    nwalkers = xshift.shape[0]
    eri_op = factor * backend.dot(xshift, ltensor.reshape(nchol, -1)).reshape(nwalkers, nao, nao)
    return eri_op


def propagate_exp_op(phiw, op, order):
    r"""action of exponential operator on (walker) wavefunction

    .. math::
        \ket{\phi'_w} = e^{A} \ket{\phi_w}

    op in the input is the operator :math:`A`. :math:`\ket{\phi_w}` is the
    walker WF.
    """

    if False:
        temp = phiw.copy()
        for i in range(order):
            temp = backend.einsum("zpq, zqr->zpr", op, temp) / (i + 1.0)
            phiw += temp
    else:
        for iw in range(phiw.shape[0]):
            temp = phiw[iw].copy()
            for i in range(order):
                temp = backend.dot(op[iw], temp) / (i + 1.0)
                phiw[iw] += temp

    return phiw

# TODO: set from configuration and availability
if QMCLIB_AVAILABLE:
    print("Debug: using qmclib kernel")
    from openms.lib import _qmclib
    propagate_onebody_kernel = _qmclib.propagate_onebody_complex
    propagate_HS_kernel = _qmclib.propagate_exp_op_complex
elif NUMBA_AVAILABLE:
    print("Debug: using numba kernel")
    propagate_onebody_kernel = propagate_onebody_numba
    propagate_HS_kernel = propagate_exp_op_numba
else:
    print("Debug: using native kernel")
    propagate_HS_kernel = propagate_exp_op
    propagate_onebody_kernel = propagate_onebody


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
        self.taylor_order = kwargs.get("taylor_order", 6)
        self.bias_bound = kwargs.get("bias_bound", 1.0)
        self.num_fake_fields = kwargs.get("num_fake_fields", 0)
        self.ebound = (2.0 / self.dt) ** 0.5
        self.time = 0.0

        # intermediate variables
        self.TL_tensor = None
        self.exp_h1e = None
        self.nfields = None  # Number of auxiliary fields for fermions
        self.nBfields = None # Number of auxiliary fields for bosons
        self.mf_shift = None

        # variables for collecting wall times
        self.wt_onebody = 0.0  # for one-body term
        self.wt_buildh1e = 0.0  # for building h1e (in e-b interaction)
        self.wt_twobody = 0.0  # for total two body propagation
        self.wt_fbias = 0.0  # for computing bias in two-body propagation
        self.wt_fbias_rescale = 0.0  # for computing bias in two-body propagation
        self.wt_random = 0.0  # for random number generation
        self.wt_hs = 0.0  # for propagating HS term in two-body propagation
        self.wt_chs = 0.0  # for constructing HS term in two-body propagation
        self.wt_phs = 0.0  # for propagating HS term in two-body propagation
        self.wt_ovlp = 0.0  # for computing overlap and walker GF
        self.wt_weight = 0.0  # for updating weights
        self.wt_bilinear = 0.0  # for bilinear term
        self.wt_boson = 0.0  # for free bosonic term

        # other intermediate variables
        self.vbias = None
        self.nfbound = 0  # number of bounding operations

        self.nbarefields = 0

        # debugging flags
        self.debug_mpi = False

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
        t0 = time.time()
        self.nfields = ltensor.shape[0]
        nao = ltensor.shape[1]

        shifted_h1e = backend.zeros(h1e.shape)
        # FIXME: Note: if we spin orbital, the rho_mf's diagonal term is 2
        # while if not (i.e., rhf), the diagonal term is 1
        # need to decide how to deal with the factor of 2 in rhf case
        rho_mf = trial.psi.dot(trial.psi.T.conj()) * 2.0 / trial.ncomponents

        # rho_mf = trial.Gf[0] + trial.Gf[1] # we can also use Gf to get rho_mf
        self.mf_shift = 1j * backend.einsum("npq,pq->n", ltensor, rho_mf)

        # logger.debug(self, f"Debug: psi = {trial.psi}")
        # logger.debug(self, f"Debug: rho_mf = {rho_mf}")
        # logger.debug(self, f"Debug: mf_shift = {self.mf_shift}")

        # shift due to eri
        if self.nbarefields > 0:
            trace_eri = backend.einsum("npr,nrq->pq", ltensor[:self.nbarefields].conj(), ltensor[:self.nbarefields])
        else:
            trace_eri = backend.einsum("npr,nrq->pq", ltensor.conj(), ltensor)
        shifted_h1e = h1e - 0.5 * trace_eri
        # for p, q in itertools.product(range(h1e.shape[0]), repeat=2):
        #    shifted_h1e[p, q] = h1e[p, q] - 0.5 * backend.trace(eri[p, :, :, q])

        if len(h1e.shape) == 3:
            logger.debug(
                self,
                f"Debug: norm of modified h1e[0] = {backend.linalg.norm(shifted_h1e[0])}",
            )
        logger.debug(
            self, f"Debug: norm of modified h1e = {backend.linalg.norm(shifted_h1e)}"
        )

        # extract the mean-field shift
        shifted_h1e = shifted_h1e - backend.einsum(
            "n, npq->pq", self.mf_shift, 1j * ltensor
        )

        self.TL_tensor = backend.einsum("pr, npq->nrq", trial.psi.conj(), ltensor)
        self.exp_h1e = scipy.linalg.expm(-self.dt / 2.0 * shifted_h1e)
        self.shifted_h1e = shifted_h1e

        logger.debug(self, f"Debug: shape of expH1 = {self.exp_h1e.shape}")
        logger.debug(
            self, f"Debug: norm of expH1[0]  = {backend.linalg.norm(self.exp_h1e[0])}"
        )
        # logger.debug(self, "norm of shifted_h1e: %15.8f", backend.linalg.norm(shifted_h1e))
        # logger.debug(self, "norm of TL_tensor:   %15.8f", backend.linalg.norm(self.TL_tensor))
        logger.info(
            self, f"Time for building initial propagator is {time.time()-t0:7.3f}"
        )


    def dump_flags(self):
        r"""dump flags (TBA)"""
        logger.note(self, task_title("Flags of propagator"))
        logger.note(self, f" Propagator:            : {self.__class__.__name__}")
        logger.note(self, f" Time step is           : {self.dt:.4f}")
        logger.note(self, f" Taylor order is        : {self.taylor_order}")
        logger.note(self, f" Energy scheme is       : {self.energy_scheme}")
        logger.note(self, f" Number of AFs          : {self.nfields}")
        logger.note(self, f" Number of bare AFs     : {self.nbarefields}")
        logger.note(self, f" Number of fake AFs     : {self.num_fake_fields}")

        # print(task_title(""))

    @abstractmethod
    def propagate_walkers(self, trial, walkers, ltensor, eshift=0.0, verbose=0):
        pass

    @abstractmethod
    def propagate_walkers_onebody(self, walkers):
        pass

    # @abstractmethod
    def propagate_walkers_twobody(self, hamiltonians, trial, walkers, eshift):
        pass


from openms.qmc.estimators import local_eng_boson
from openms.qmc.estimators import local_eng_eboson
from openms.qmc.estimators import local_energy_SD_RHF
from openms.qmc.estimators import local_energy_SD_UHF, local_eng_elec_chol
from openms.qmc.estimators import local_eng_elec_chol_new


class Phaseless(PropagatorBase):
    r"""
    HS-transformation based AFQMC propagators

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ebound = (2.0 / self.dt) ** 0.5


    def compute_local_energies(self, TL_theta, h1e, vbias, gf):
        r"""compute local energies

        Old funciton, to be deprecated!!!!"""
        # eloc = local_eng_elec_chol_new(h1e, ltensor, gf)
        return local_eng_elec_chol(TL_theta, h1e, vbias, gf)

    # TODO: according to the headers of trail,walker, propagator, and options to
    # select right function of computing energy (may use dict)

    def local_energy(self, h1e, ltensor, walkers, trial, enuc=0.0):
        r"""Compute local energy with UHF
        """

        t0 = time.time()
        # update green function fist
        trial.ovlp_with_walkers_gf(walkers)

        if walkers.ncomponents > 1:
            e1, e2 = local_energy_SD_UHF(trial, walkers)
        else:
            e1, e2 = local_energy_SD_RHF(trial, walkers)
        walkers.eloc = e1 + e2 + enuc

        # etot = backend.sum(walkers.weights * walkers.eloc.real)
        etot = backend.dot(walkers.weights, walkers.eloc.real)
        norm = backend.sum(walkers.weights)
        e1 = backend.dot(walkers.weights, e1.real)
        e2 = backend.dot(walkers.weights, e2.real)

        if walkers._mpi.size > 1:
            # If MPI, gather energy here
            # gather weights, energies aross the nodes
            norm = walkers._mpi.comm.allreduce(norm, op=MPI.SUM)
            e1 = walkers._mpi.comm.allreduce(e1, op=MPI.SUM)
            e2 = walkers._mpi.comm.allreduce(e2, op=MPI.SUM)
            etot = walkers._mpi.comm.allreduce(etot, op=MPI.SUM)

        logger.debug(self, f"Debug: total energy (unnormalized) is {etot}")
        logger.debug(self, f"Debug: e1 (unnormalized) is {e1}")
        logger.debug(self, f"Debug: e2 (unnormalized) is {e2}")
        # logger.debug(self, f"Debug: time of computing energy is {time.time() - t0}")

        ##  weights * eloc
        # energy = energy / backend.sum(walkers.weights)
        return [etot, norm, e1, e2]


    def rescale_fbias(self, fbias):
        r"""
        Apply a bound to the force bias `fbias`, rescaling values
        that exceed the specified maximum bound.

        Parameters:
            fbias (ndarray): The input array to process.

        Returns:
            ndarray: The processed array with values adjusted to respect the bound.
        """
        absf = backend.abs(fbias)
        idx_to_rescale = absf > self.bias_bound
        nonzeros = absf > 1.0e-12

        # Rescale only non-zero elements
        f_rescaled = fbias.copy()
        f_rescaled[nonzeros] /= absf[nonzeros]

        # Apply the rescaling to elements exceeding the max bound
        fbias = backend.where(idx_to_rescale, f_rescaled, fbias)
        # Update the `nfb_trig` attribute
        self.nfbound += backend.sum(idx_to_rescale)
        return fbias


    def propagate_walkers_onebody(self, walkers):
        r"""Propgate one-body term

        Note: the shape of phiwa is [nwalkers, nao, nalpha]
        """
        t0 = time.time()
        # logger.debug(self, f"Debug: phiwa.shape = {walkers.phiwa.shape}")
        # logger.debug(self, f"Debug: phiwb.shape = {walkers.phiwb.shape}")

        walkers.phiwa = propagate_onebody_kernel(self.exp_h1e[0], walkers.phiwa)
        logger.debug(self, f"Debug: norm of phiwa after onebody {backend.linalg.norm(walkers.phiwa):.8f}")

        if walkers.ncomponents > 1:
            walkers.phiwb = propagate_onebody_kernel(self.exp_h1e[1], walkers.phiwb)
            logger.debug(self, f"Debug: norm of phiwb after onebody {backend.linalg.norm(walkers.phiwb):.8f}")
        self.wt_onebody += time.time() - t0
        logger.debug(self, f"Debug: time of propagate onebody: { time.time() - t0}")


    def propagate_HS(self, walkers, ltensor, xshift):
        r"""Propagate the walker WF according to the auxiliary field (HS)"""
        # xshift: [nwalker, nchol]
        # ltensor: [nchol, nao, nao]

        # TODO: further improve the efficiency of this part
        # TODO: may use symmetry in ltensor to imporve the efficiency of this part

        t0 = time.time()
        sqrtdt = 1j * backend.sqrt(self.dt)
        if True: # using CPU
            nchol, nao = ltensor.shape[:-1]
            eri_op = sqrtdt * backend.dot(xshift, ltensor.reshape(nchol, -1)).reshape(walkers.nwalkers, nao, nao)
        else:
            eri_op = sqrtdt * backend.einsum("zn, npq->zpq", xshift, ltensor)
        logger.debug(self, f"Debug: time of construct VHS: {time.time() - t0}")
        self.wt_chs += time.time() - t0

        t0 = time.time()
        # \sum_n 1/n! (j\sqrt{\Delta\tau) xL)^n
        propagate_HS_kernel(walkers.phiwa, eri_op, self.taylor_order)
        if walkers.ncomponents > 1:
            propagate_HS_kernel(walkers.phiwb, eri_op, self.taylor_order)
        logger.debug(self, f"Debug: time of propagating twobody exp operator {time.time() - t0}")
        self.wt_phs += time.time() - t0


    def propagate_walkers_twobody(self, trial, walkers, ltensor):
        # TODO: improve the efficiency
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
        t0 = time.time()

        # a) generate normally distributed AF
        # if we compare with electron-boson case, we need to add nmode * nwalker random number in order to
        # have the same random number in each iteration
        if self.debug_mpi:
            # This part is to make sure MPI version has the same random number as the serial code
            # for debug debug MPI purpose
            if walkers._mpi.rank == 0:
                xi_full = backend.random.normal(0.0, 1.0, (self.nfields + self.num_fake_fields) * walkers.global_nwalkers)
                # print("random numbers = ", xi_full) # YZ: confimed to be same for mpi
            else:
                xi_full = None
            xi_size = (self.nfields + self.num_fake_fields) * walkers.nwalkers
            xi = backend.empty(xi_size, dtype=backend.float64)
            # original_print("local xi size = ", xi_size)
            counts = [(self.nfields + self.num_fake_fields) * nw for nw in walkers.walker_counter]
            displs = [0] + list(backend.cumsum(counts))[:-1]
            walkers._mpi.comm.Scatterv([xi_full, counts, displs, MPI.DOUBLE], xi, root=0)
        else:
            # In practice, we will generate random number locally, without scattering
            xi = backend.random.normal(0.0, 1.0, (self.nfields + self.num_fake_fields) * walkers.nwalkers)

        xi = xi.reshape(walkers.nwalkers, self.nfields + self.num_fake_fields)[:, :self.nfields]
        if self.nfields > self.nbarefields:
            self.xi_bilinear = xi[:, self.nbarefields:self.nfields]

        t1 = time.time()
        self.wt_random += t1 - t0

        # logger.debug(self, f"the random numbers are\n{xi}")

        # b) compute force bias
        # F = \sqrt{-\Delta\tau} <L'> = \sqrt{-\Delta\tau} (<L> - <L>_{MF})  (where L' is the shifted chols)
        #   = j\sqrt{\Delta\tau}(<L> - <L>_{MF})
        # xbar is the F
        self.vbias = trial.get_vbias(walkers, ltensor) # (nwalkers, nfield)
        self.wt_fbias += time.time() - t1
        t1 = time.time()

        xbar = -backend.sqrt(self.dt) * (1j * self.vbias - self.mf_shift)
        xbar = self.rescale_fbias(xbar)  # bound of vbias
        xshift = xi - xbar  # [nwalker, nchol]

        logger.debug(self, f"Debug: mf_shift.shape = {self.mf_shift.shape}")
        logger.debug(self, f"Debug: vbias.shape = {self.vbias.shape}")
        logger.debug(self, f"Debug: norm of vbias = {backend.linalg.norm(self.vbias):.8f}")

        self.wt_fbias_rescale += time.time() - t1
        t1 = time.time()

        # c) compute the factors due to mean-field shift and shift in propabailities
        #
        # (x*\bar{x} - \bar{x}^2/2), i.e., factors due to the shift in propabalities:
        #
        # functions (it comes from the mf shift in force bias)
        # F -> F - <F>, where F =\sqrt{-dt} Tr[LG] and <L> = Tr[L\rho_{mf}]
        # hence xF - 0.5 F^2 = x(F-<F>) - 0.5(F-<F>)^2 + x<F> - F<F> + 0.5<F>^2
        # so (x-F)<F> --> cmf
        #    x(F-<F>) - 0.5(F-<F>)^2 -- > cfb
        #    0.5 <F>^2 propability shift
        cfb = backend.sum((xi - 0.5 * xbar) * xbar, axis=1)
        # factors due to MF shift and force bias
        cmf = -backend.sqrt(self.dt) * backend.einsum("zn, n->z", xshift, self.mf_shift)

        # logger.debug(self, f"norm of cfb :   {backend.linalg.norm(cfb)}")
        # logger.debug(self, f"norm of cmf :   {backend.linalg.norm(cmf)}")

        # d) propagate walkers WF using the auxiliary fields
        self.propagate_HS(walkers, ltensor, xshift)
        logger.debug(
            self,
            f"Debug: norm of phiwa after HS propagation: {backend.linalg.norm(walkers.phiwa)}",
        )
        if walkers.ncomponents > 1:
            logger.debug(
                self,
                f"Debug: norm of phiwb after HS propagation: {backend.linalg.norm(walkers.phiwb)}",
            )
        self.wt_hs += time.time() - t1
        self.wt_twobody += time.time() - t0

        return cfb, cmf

    def propagate_walkers(self, trial, walkers, ltensor, eshift=0.0, verbose=0):
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
        # logger.debug(self, f"\nDebug: entered propagate walkers!")

        # a) compute overlap and update the Green's funciton
        t0 = time.time()
        ovlp = trial.ovlp_with_walkers_gf(walkers)
        #logger.debug(self, f"Debug: trial_walker overlap is {ovlp}")
        #logger.debug(
        #    self,
        #    f"Debug: norm of walker.Ghalfa is {backend.linalg.norm(walkers.Ghalfa)}",
        #)
        #if walkers.ncomponents > 1:
        #    logger.debug(
        #        self,
        #        f"Debug: norm of walker.Ghalfb is {backend.linalg.norm(walkers.Ghalfb)}",
        #    )
        self.wt_ovlp += time.time() - t0

        # b) 1-body propagator propagation :math:`e^{-dt/2*H1e}`
        self.propagate_walkers_onebody(walkers)

        # c): 2-body propagator propagation :math:`\exp[(x-\bar{x}) * L]`
        cfb, cmf = self.propagate_walkers_twobody(trial, walkers, ltensor)

        # d):  1-body propagator propagation e^{-dt/2*H1e}
        self.propagate_walkers_onebody(walkers)
        # walkers.phiw = backend.exp(-self.dt * nuc) * walkers.phiw

        # e) update overlap only
        t0 = time.time()
        newovlp = trial.ovlp_with_walkers(walkers)

        logger.debug(
            self, f"Debug: norm of new overlap is {backend.linalg.norm(newovlp)}"
        )
        self.wt_ovlp += time.time() - t0

        # f) update_weight and apply phaseless approximation
        t0 = time.time()
        self.update_weight(walkers, ovlp, newovlp, cfb, cmf, eshift)
        self.wt_weight += time.time() - t0
        # print(f"Max/min/sum(weights): {backend.max(walkers.weights):.3f} {backend.min(walkers.weights):.3f} {backend.sum(walkers.weights):.3f} ")

        assert not backend.isnan(backend.linalg.norm(walkers.weights)), "NaN detected in walkers.weights"
        # logger.debug(self, f"updated weight: {walkers.weights}\n eshift = {eshift}")


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

        # TODO: 1) adapt it fully to the propagator class?

        ovlp_ratio = (newovlp / ovlp) # ** 2 (this factor only applies to RHF case, has been absorbed in the overlap calculation)
        # logger.debug(self, f"Debug: ovlp_ratio is {ovlp_ratio}")

        # the hybrid energy scheme
        if self.energy_scheme == "hybrid":
            ehybrid = -(backend.log(ovlp_ratio) + cfb + cmf) / self.dt
            ehybrid = backend.clip(
                ehybrid.real,
                a_min=eshift.real - self.ebound,
                a_max=eshift.real + self.ebound,
                out=ehybrid.real,
            )
            # walkers.ehybrid = ehybrid if walkers.ehybrid is None else walkers.ehybrid
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

        # update weights and overlap
        walkers.weights *= importance_func
        walkers.ovlp = newovlp


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
        shifted_h1b = (
            h1b
            - 0.5 * trace_v2b
            - backend.einsum("n, npq->pq", self.bmf_shift, 1j * chol_b)
        )

        self.TL_tensor = backend.einsum("pr, npq->nrq", trial.psi.conj(), chol_b)
        self.exp_h1b = scipy.linalg.expm(-self.dt / 2 * shifted_h1b)
        self.h1b = h1b


    def propagate_walkers_onebody(self, walkers):
        r"""Propagate one-body term"""

        walkers.phiw = backend.einsum("pq, zqr->zpr", self.exp_h1e, walkers.phiw)


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

    **Brief introduction to the Theoretical background**:

    Here we define :math:`\hat{\boldsymbol{g}}_\alpha = \boldsymbol{\lambda}_\alpha
    \cdot \hat{\boldsymbol{D}} =g^\alpha_{ij}\hat{c}^\dagger_i \hat{c}_j`
    and :math:`\hat{X}_\alpha = \sqrt{\frac{1}{2\omega_\alpha}}(b^\dagger_\alpha + b_\alpha)`.
    The Hamiltonian for the electron-boson is

    .. math::

        \hat{H} = \hat{H}_e + \hat{H}_p + \sqrt{\frac{\omega_\alpha}{2}}
        \hat{\boldsymbol{g}}_\alpha (\hat{b}^\dagger_\alpha + \hat{b}_\alpha)
        + \frac{1}{2}(\hat{\boldsymbol{g}}_\alpha)^2.

    The last DSE term is zero for some systems and :math:`\hat{H}_p =\sum_\alpha
    \omega_\alpha (b^\dagger_\alpha b_\alpha + 1/2)`.

    Propagation method A: **Decoupled propagation**
    -----------------------------------------------

    We have different way to decouple the bilinear coupling term:

    **Decomposition scheme #1**

    We can decompose the blinear coupling term as:

    .. math::

        \sqrt{\frac{\omega_\alpha}{2}} \hat{\boldsymbol{g}}_\alpha (\hat{b}^\dagger_\alpha + \hat{b}_\alpha) =
        \frac{A_\alpha}{2} \left[ (\hat{\boldsymbol{g}}_\alpha + \hat{O}_\alpha)^2
        - (\hat{O}_\alpha)^2
        - (\hat{\boldsymbol{g}}_\alpha)^2\right]

    There are different choices of :math:`A_\alpha` and :math:`\hat{O}_\alpha`. Such as

       - :math:`A_\alpha = 1` and :math:`\hat{O}_\alpha = \sqrt{\frac{\omega_\alpha}{2}}(\hat{b}^\dagger_\alpha + \hat{b}_\alpha)`
       - :math:`A_\alpha = \omega_\alpha` and :math:`\hat{O}_\alpha = \sqrt{\frac{1}{2\omega_\alpha}}(\hat{b}^\dagger_\alpha + \hat{b}_\alpha)`
       - :math:`A_\alpha = \sqrt{\frac{\omega_\alpha}{2}}` and :math:`\hat{O}_\alpha = (\hat{b}^\dagger_\alpha + \hat{b}_\alpha)`
       - :math:`\cdots`

    Nevertheless, the corresponding MC Hamiltonian is rewritten as:

    .. math::

         \hat{H}_{mc} = & \hat{T}_e + \frac{1}{2}\sum_\gamma \hat{L}^2_{\gamma}
                      + \frac{1}{2}\sum_\alpha \left[(1-A_\alpha)\hat{g}^2_\alpha +  A_\alpha(\hat{g}_\alpha
                      + \hat{O}_\alpha)^2 - A_\alpha\hat{O}^2_\alpha \right] + \hat{H}_p  \\
                      = & \hat{T}_e + \frac{1}{2}\sum_\gamma \hat{L}^2_{\gamma}
                      + [\hat{L}^2_{g, \alpha} + \hat{L}^2_{eb,\alpha} +  \hat{L}^2_{b, \alpha}]
                      + \hat{H}_p.

    where,

    .. math::

       \hat{L}_{g, \alpha} = & \sqrt{1 - A_\alpha} \hat{\boldsymbol{g}}_\alpha \\
       \hat{L}_{eb,\alpha} = & \sqrt{A_\alpha} (\hat{\boldsymbol{g}}_\alpha  + \hat{O}_\alpha) \\
       \hat{L}_{b, \alpha} = & i\sqrt{A_\alpha} \hat{O}_\alpha

    We further absorb the terms in :math:`\hat{L}_{eb,\alpha}` into :math:`\hat{L}_{g,\alpha}`
    and :math:`\hat{L}_{b, \alpha}`, leading to

    .. math::

       \hat{L}_{g, \alpha} = & (\sqrt{1 - A_\alpha} + \sqrt{A_\alpha}) \hat{\boldsymbol{g}}_\alpha \\
       \hat{L}_{b, \alpha} = & (1+i)\sqrt{A_\alpha} \hat{O}_\alpha

    Hence, :math:`2N_b` auxiliary fields are added to the two-body proapgation.


    Decomposition scheme #2
    ^^^^^^^^^^^^^^^^^^^^^^^

    .. math::
         \hat{F}_\alpha \hat{B}_\alpha = \frac{(\hat{f}_\alpha + \hat{B}_\alpha)^2 - (\hat{f}_\alpha - \hat{B}_\alpha)^2}{4}

    Hence, :math:`\{\frac{\hat{F}_\alpha + \hat{B}_\alpha}{\sqrt{2}}, \frac{i(\hat{F}_\alpha - \hat{B}_\alpha)}{\sqrt{2}} \}`
    auxiliary fields will be generated.


    Decomposition scheme #3 "Direct Hubbard–Stratonovich transformation of bilinear coupling term"
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    .. math::
        e^{-\Delta\tau F_\alpha B_\alpha} = \int dx P(x) e^{i\sqrt{\Delta\tau} F_\alpha}
        e^{i\sqrt{\Delta\tau} B_\alpha}

    In this way, only :math:`N_b` additional auxiliary field is added.


    Propagation method B: **Explicit propagation of bilinear coupling:**
    --------------------------------------------------------------------

    If we don't decouple the bilinear term, the MC Hamiltonain is:

    .. math::

         \hat{H}_{mc} = \hat{T}_e + \frac{1}{2}\sum_\gamma \hat{L}^2_{\gamma}
                      + \frac{1}{2}\sum_\alpha \hat{L}^2_{D, \alpha}
                      + \sum_\alpha \omega_\alpha \hat{\boldsymbol{g}}_\alpha \hat{X}_\alpha
                      + \hat{H}_p.
    where :math:`\hat{L}_{D, \alpha} = \hat{g}_\alpha`. In this case, only :math:`N_b` Auxiliary fields
    are added to the two-body walker propagation. But bilinear coupling term has to be added to the propagation,
    which can be done via partial trace over each subsystem.


    **Propagation in the first quantization:** The advantage of propagation in the first quantization is that
    it avoids the necessity of truncating the bosonic Hilbert space.

    The random walker is:

    .. math::
      \ket{\Psi_k(\tau)} =\ket{\phi_k(\tau), Q_k(\tau)}.


    One-body propagator: the action of :math:`e^{-\Delta\tau\hat{H}_1(Q_b)}` on the random walker is

       - :math:`\ket{Q'} = \ket{Q}`
       - :math:`\ket{\phi'_k} = e^{-\Delta\tau\hat{H}_1}\ket{\phi_k}`
       - :math:`w'_n =\frac{\bra{\Psi_T}\phi'_k, Q'_k\rangle}{\bra{\Psi_T}\phi_k, Q_k\rangle} w_n(\tau)`

    Note the bilinear coupling term is diagonal in the :math:`\ket{Q}` representation, so the one-body propagator
    will leave :math:`\ket{Q}` invariant.

    Two-body propagator: the two-body propagator is independent of bosonic DOF, the :math:`\ket{Q}` will be invariant:

       - :math:`\ket{Q'} = \ket{Q}`
       - :math:`\ket{\phi'_k} = e^{\sqrt{-\Delta\tau}(x-F)(\hat{L}_\gamma - \bar{L}_\gamma)}\ket{\phi_k(\tau)}`.
       - :math:`w' = \frac{\bra{\Psi_T}\phi'_k, Q'_k\rangle}{\bra{\Psi_T}\phi_k, Q_k\rangle}e^{xF - F^2}w(\tau)`

    Bosonic propagator:


    **Propagation in the second quantization:**

    The random walker is:

    .. math::
      \ket{\Psi_k(\tau)} =\ket{\phi_k(\tau), n_k(\tau)}.


    One-body propagator:

       - :math:`\ket{n'_k} =  e^{-\Delta\tau\hat{H}_1} \ket{n_k(\tau)}`
       - :math:`\ket{\phi'_k} = e^{-\Delta\tau\hat{H}_1}\ket{\phi_k}`
       - :math:`w'_n =\frac{\bra{\Psi_T}\phi'_k, Q'_k\rangle}{\bra{\Psi_T}\phi_k, Q_k\rangle} w_n(\tau)`

    Two-body propagator: Same as above.

    Bosonic propagator:



    """

    def __init__(self, dt, **kwargs):
        super().__init__(dt, **kwargs)
        self.boson_quantization = kwargs.get("quantization", "second")
        self.decouple_bilinear = kwargs.get("decouple_bilinear", False)
        self.decouple_scheme = kwargs.get("decouple_scheme", 1)
        self.turnoff_bosons = kwargs.get("turnoff_bosons", False)
        self.geb = None  # bilinear coupling term (without decomposition)
        self.e_boson_shift = 0.0
        self.e_local_boson = 0.0
        self.nbarefields = kwargs.get("nbarefields", 0)


    def dump_flags(self):
        super().dump_flags()
        logger.note(self, f" Decouple bilinear term : {self.decouple_bilinear}")
        logger.note(self, f" Quantization of boson  : {self.boson_quantization}")
        logger.note(self, task_title("") + "\n")


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

        nmodes = self.system.nmodes
        basis = backend.asarray(
            [backend.arange(mdim) for mdim in self.system.nboson_states]
        )
        waTa = backend.einsum("m, mF->mF", self.system.boson_freq, basis).ravel()
        self.Hb = backend.diag(waTa)
        # logger.debug(self, f"Debug: Hb = {self.Hb}")

        # if we decouple the bilinear term
        if self.decouple_bilinear:
            # add shift due to bilinear term
            # FIXME: how to deal with ltensor, combine everything here
            # or separate the bilinear terms

            logger.debug(self, f"\nDebug: build shifted bosonic operators!")
            rho_mf = trial.psi.dot(trial.psi.T.conj())
            if trial.boson_psi.ndim == 1:
                boson_rhomf = backend.outer(trial.boson_psi, trial.boson_psi.T.conj())
            else:
                boson_rhomf = trial.boson_psi.dot(trial.boson_psi.T.conj())

            # logger.debug(self, f"boson_psi.shape = {trial.boson_psi.shape}")
            # logger.debug(self, f"boson_rhomf.shape = {boson_rhomf.shape}")

            self.chol_bilinear_b = self.chol_bilinear[1] # bosonic part

            # 2) construct bosonic MF shift
            nfock = self.system.nboson_states[0]
            # matrix representation of (a^\dag + a) in Fcok space
            Hq = boson_adag_plus_a(1, [nfock], [1.0])
            self.chol_B = backend.einsum("X, nm-> Xnm", self.chol_bilinear_b, Hq)
            self.nBfields = self.chol_B.shape[0]
            if not self.turnoff_bosons:
                assert self.nBfields == (self.nfields - self.nbarefields)
                self.boson_mfshift = 1j * backend.einsum("npq, pq->n", self.chol_B, boson_rhomf)
                self.shifted_Hb = self.Hb - backend.einsum("n, npq->pq", self.boson_mfshift, 1j * self.chol_B)
                logger.debug(self, f" boson_mfshift =\n {self.boson_mfshift}")
            else:
                self.shifted_Hb = self.Hb.copy()

            logger.debug(self, f" Debug: norm of         Hb: {backend.linalg.norm(self.Hb)}")
            logger.debug(self, f" Debug: norm of shifted Hb: {backend.linalg.norm(self.shifted_Hb)}")
            logger.debug(self, f" Debug: chol_B =\n {self.chol_B}")
            logger.debug(self, f" Debug: boson_rhomf =\n {boson_rhomf}")


    def local_energy(self, h1e, ltensor, walkers, trial, enuc=0.0):
        r"""Overwrite the super().local_energy with bosonic and electron-boson
        interacting energies
        """

        # use the super().local_energy to get the local energy of the fermions
        # and only to add the computation of bosonic and electron-boson interacting
        # local energies here

        etot, norm, e1, e2 = super().local_energy(h1e, ltensor[:self.nbarefields], walkers, trial, enuc=enuc)

        # boson energy
        # walkers.boson_Gf = backend.einsum("wi, wj->wij", walkers.boson_phiw, walkers.boson_phiw.conj())
        eb = local_eng_boson(self.system.boson_freq, self.system.nboson_states, walkers.boson_Gf)
        # print(f"Debug: Gfavg = {backend.sum(walkers.boson_Gf, axis=0) / walkers.nwalkers}")
        # print(f"Debug: Gf[0] = {walkers.boson_Gf[0]}")

        self.update_GF(trial, walkers)

        # electron-boson interacting energy
        Gfermions = [walkers.Ga, walkers.Gb] if walkers.ncomponents > 1 else [walkers.Ga, walkers.Ga]
        eg = local_eng_eboson(self.system.boson_freq, self.system.nboson_states, self.geb, Gfermions, walkers.boson_Gf)

        # update the local energy with eb and eg
        if not self.turnoff_bosons:
            walkers.eloc += eb + eg

        # and then update the total energy
        etot = backend.dot(walkers.weights, walkers.eloc.real)
        eb = backend.dot(walkers.weights, eb.real)
        eg = backend.dot(walkers.weights, eg.real)

        logger.debug(self, f"Debug: total energy (unnormalized) is {etot}")
        logger.debug(self, f"Debug: e1 (unnormalized) is {e1}")
        logger.debug(self, f"Debug: e2 (unnormalized) is {e2}")
        logger.debug(self, f"Debug: eb (unnormalized) is {eb}")
        logger.debug(self, f"Debug: eg (unnormalized) is {eg}")

        ##  weights * eloc
        # energy = energy / backend.sum(walkers.weights)
        return [etot, norm, e1, e2, eb, eg]


    # deprecated function
    #def compute_local_energies(self, TL_theta, h1e, vbias, Gfs):
    #    r"""compute local energies"""

    #    Gf, Gb = Gfs
    #    # eloc = local_eng_elec_chol_new(h1e, ltensor, Gf)

    #    eng_e = local_eng_elec_chol(TL_theta, h1e, vbias, Gf)

    #    # bosonc energy
    #    # print("Gb =", Gb)
    #    eb = local_eng_boson(self.system.boson_freq, self.system.nboson_states, Gb)

    #    # bilinear energy
    #    nmodes = self.system.nmodes
    #    boson_size = sum(self.system.nboson_states)
    #    Hb = backend.zeros((boson_size, boson_size), dtype=backend.complex128)
    #    zalpha = backend.einsum("npq, zpq->zn", self.geb, Gf)
    #    idx = 0
    #    for imode in range(nmodes):
    #        mdim = self.system.nboson_states[imode]
    #        a = backend.diag(backend.sqrt(backend.arange(1, mdim)), k=1)
    #        h_od = a + a.T
    #        Hb[idx : idx + mdim, idx : idx + mdim] = h_od * zalpha[imode]
    #    ep = backend.einsum("NM,zNM->z", Hb, Gb)
    #    # print("photon energy, bilinear term are ", eb, ep)
    #    return eng_e + eb + ep



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


    def propagate_walkers_onebody(self, walkers):  # walker, system, trial, dt):
        r"""Propgate one-body term:
        including both h1e and bilinear coupling parts

        .. note::

            we may move the bilinear-mediated oei into the propgate_walkers_bilinear,
            then we can recycle the propagate_walkers_onebody term from the base class,
            need of this one!

        step 1) is to compute shifted_h1e with contribution from bilinear coupling as

        .. math::

            T^{eff}_{pq} = h_{pq} - \frac{1}{2} L_{npr} L^*_{nqr} - \sum_n \langle L_n\rangle L_{n,pq}
            - \sum_{\alpha} g_{\alpha,pq} \langle z_{\alpha} \rangle.

        where :math:`z_\alpha= \text{Tr}[gD]` and :math:`D` is the density matrix, i.e.,
        :math:`z_\alpha` is the displacement

        step 2) is to update the walker WF due to the effective one-body operator (formally
        same as the bare electronic case).

        Parameters
        ----------
        walker :
            Walker object to be updated. On output we have acted on phi by
            :math:`B_{\Delta\tau/2}` and updated the weight appropriately. Updates inplace.
        system :
            System object.
        trial :
            Trial wavefunction object.

        """

        t0 = time.time()
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

        # H_{int} = c^\dag_i c_j \sum_a g_a \langle (b^\dag_a + b_b) \rangle
        oei = backend.zeros(self.shifted_h1e.shape)

        if not self.decouple_bilinear and self.geb is not None:
            # trace over bosonic DOF
            # zlambda = backend.einsum("pq, Xpq ->X", walkers.rho, self.geb)
            oei = backend.einsum("X, Xpq->pq", walkers.Qalpha, self.geb)


        # 2) oei_qed in 1st quantization (TBA)

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

        t0 = time.time()
        if not self.turnoff_bosons:
            # logger.debug(self, f"Debug: shapes of shifted_h1e and oei {self.shifted_h1e.shape} {oei.shape}")
            shifted_h1e = self.shifted_h1e + oei
            self.exp_h1e = scipy.linalg.expm(-self.dt / 2 * shifted_h1e)
        self.wt_buildh1e += time.time() - t0

        # use super() method
        super().propagate_walkers_onebody(walkers)


    def propagate_walkers_bilinear(self, trial, walkers, dt):
        r"""Propagate the bilinear part

        .. math::
           \exp\{-\Delta\tau H_{eb}\}\ket{\phi_w, n} \rightarrow = \sum_k \frac{1}{k!}
           [-\Delta\tau H_{eb}]^n \ket{\phi_w, n}

        For each order:

        .. math::
           [-\Delta\tau H_{eb}]\ket{\phi_w, n} & = \ket{\phi'_w, m} \equiv A_{pq} \ket{\phi_w} \otimes B_{mn} \ket{n}

         Hence:
           [-\Delta\tau H_{eb}]\ket{\phi_w, n} & = \sum_{m} \bra{m}[-\Delta\tau H_{eb}]\ket{\phi_w, n} \ket{m} \\
           & = [g^\alpha \sum_{m} Q^\alpha_{mn} \ket{m}] \ket{\phi_w}
        """

        # g^\alpha_{pq} -> g_{pq, mn} |ket{qi, n} -> \ket{pi, m}

        nmodes = self.system.nmodes
        boson_size = sum(self.system.nboson_states)

        if self.decouple_bilinear:
            logger.debug(self, f"Debug: propagating the bilinear term in decoupled formalism")
        else:
            logger.debug(self, f"Debug: propagating the bilinear term in product formalism")
            #

        # FIXME; include zlambda or not
        zlambda = backend.ones(nmodes)
        zlambda = backend.einsum("pq, Xpq ->X", walkers.rho, self.geb)

        Hb = boson_adag_plus_a(nmodes, self.system.nboson_states, zlambda)
        # logger.debug(self, f"Hb = {Hb}")

        # Qalpha set o 1 tentatively
        # propagate electronic part
        Qalpha = backend.ones(nmodes)
        Qalpha = walkers.Qalpha
        oei = -dt * backend.einsum("X, Xpq->pq", Qalpha, self.geb)
        evol_Hep = -dt * Hb
        #TODO: matrix element of <m|e^{-z_\alpha(a^\dag_\alpha + a_\alpha)}|n> can be analytically evaluated

        # propagate the WF
        temp = walkers.phiwa.copy()
        temp2 = walkers.boson_phiw.copy()
        for i in range(self.taylor_order):
            temp = backend.einsum("pq, zqr->zpr", oei, temp) / (i + 1.0)
            walkers.phiwa += temp
            # bosonic part
            temp2 = backend.einsum("NM, zM->zN", evol_Hep, temp2)
            walkers.boson_phiw += temp2

        if walkers.ncomponents > 1:
            temp = walkers.phiwb.copy()
            for i in range(self.taylor_order):
                temp = backend.einsum("pq, zqr->zpr", oei, temp) / (i + 1.0)
                walkers.phiwb += temp

        """
        walkers.phiwa = propagate_exp_op(walkers.phiwa, oei, self.taylor_order)
        if walkers.ncomponents > 1:
            walkers.phiwb = propagate_exp_op(walkers.phiwb, oei, self.taylor_order)
        """


    def propagate_decoupled_bilinear(self, trial, walkers):
        r"""Propagate the bilinear term in decoupled formalism

        Any coupling between the fermions and bosons can be decoupled as

        .. math::

            \hat{O}_f\hat{O}_b = \frac{1}{2}\left[(\hat{O}_f + \hat{O}_b)^2 -
            \hat{O}^2_f - \hat{O}^2_b\right].
        """
        # decoupled bosonic propagator
        # O_b =

        pass


    def propagate_bosons(self, trial, walkers, dt):
        r"""
        boson importance sampling
        """
        if "first" in self.boson_quantization:
            self.propagate_bosons_1st(trial, walkers, dt)
        else:
            self.propagate_bosons_2nd(trial, walkers, dt)


    def propagate_bosons_2nd(self, trial, walkers, dt):
        r"""Boson importance sampling in 2nd quantization
        Ref. :cite:`Purwanto:2004qm`.

        Bosonic Hamiltonian is:

        .. math::

            H_{ph} =\omega_\alpha b^\dagger_\alpha b_\alpha

        Since this part diagonal, we use 1D array to store the diaognal elements
        """

        logger.debug(self, "Debug: propagating free bosonic part in 2nd quantization")

        """
        # replaced by pre-computed Hb and exp_Hb
        basis = backend.asarray(
            [backend.arange(mdim) for mdim in self.system.nboson_states]
        )
        waTa = backend.einsum("m, mF->mF", self.system.boson_freq, basis).ravel()
        evol_Hb = backend.exp(-dt * waTa)
        """
        if self.decouple_bilinear:
            # H_b -> H_b + <L^B> L_B + <L^B'> L^B'
            evol_Hb = scipy.linalg.expm(-dt * self.shifted_Hb)
        else:
            evol_Hb = scipy.linalg.expm(-dt * self.Hb)
        walkers.boson_phiw = backend.einsum("mn, zn->zm", evol_Hb, walkers.boson_phiw)

        # eloc = local_eng_boson(self.system.boson_freq, self.system.nboson_states, walkers.boson_Gf)
        # walkers.weights *= backend.exp(-dt * eloc.real)
        logger.debug(self, f"Debug: bosonic WF after free boson:  {abs(backend.sum(walkers.boson_phiw, axis=0)) / walkers.nwalkers}")


    def propagate_bosons_1st(self, trial, walkers, dt):
        r"""Boson importance sampling in 1st quantization formalism

        DQMC type of algorithm:

        .. math::

            Q^{n+1} = Q^n + R(\sigma) + \frac{\Delta\tau}{m}\frac{\nabla_Q \Psi_T(Q)}{\Psi_T(Q)}

        Here R is a normally distributed random number with variance :math:`\sigma=\sqrt{m\Delta u}`.
        The last term is the drift term.

        i.e., Qnew = Qold + dQ + drift

        TBA.
        """
        logger.debug(self, "propagating free bosonic part in 1st quantization")

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
        r"""propagate the bilinear fermion-bosonic interaction term
        within two schemes:

        1.decomposition of bilinear term

        .. math::
            \hat{X}_\alpha = \frac{1}{2\omega_\alpha}(\hat{b}^\dagger + \hat{b}_\alpha)

        1. Direct propagation of the product bilinear term:

        .. math::
           \hat{H}_{ep} = \frac{\omega}{2} g^{\alpha}_{pq} c^\dagger_p c_q (a^\dagger_\alpha + a_\alpha).


        The propagation is:

        .. math::
            \ket{\phi'_w, n'} = \frac{\bra{\Psi_T}\psi'_w, n'\rangle}{\bra{\Psi_T}\psi_w, n\rangle}
                                e^{-\Delta\tau H_{eb}}\ket{\psi_w, n}

        where

        .. math::
            \ket{\psi'_w} = \sum_m \ket{m}\bra{m} e^{-\Delta\tau H_{eb}}\ket{\psi_w, n}.

        .. math::
            \ket{n'} = e^{-\Delta\tau H_{eb}}\ket{n}.

        .. math::
            e^{-\Delta\tau\hat{H}_{ep}} \ket{\psi_w, n}
            = \sum_m \ket{m}\bra{m} e^{-\Delta\tau\hat{H}_{ep}} \ket{\psi_w, n}

        """

        # note geb already has sqrt(w/2)
        nmodes = self.system.nmodes
        boson_size = sum(self.system.nboson_states)
        Hb = backend.zeros((boson_size, boson_size), dtype=backend.complex128)

        # compute local bosonic energy
        elocold = self.e_local_boson
        if self.decouple_bilinear:
            logger.debug(self, f"Debug: propagating the bilinear term in decoupled formalism")

            # \sqrt{A) * X = sqrt(w/2) (b^\dagger + b)

            # xi = backend.random.normal(0.0, 1.0, self.nBfields * walkers.nwalkers)
            # xi = xi.reshape(walkers.nwalkers, self.nBfields) # zn (nwalkers, nBfields)
            xi = self.xi_bilinear
            tau = 1j * backend.sqrt(dt)

            # Note the electronic part out of the decomposed bilinear has
            # been concatenated into ltensor

            # compute bosonic force bias
            boson_vbias = trial.get_boson_vbias(walkers, self.chol_B)
            xbar = -backend.sqrt(dt) * (1j * boson_vbias - self.boson_mfshift)
            xbar = self.rescale_fbias(xbar)  # bound of vbias
            xshift = xi - xbar  # [nwalker, nchol]

            #print(f"boson_vbias   = {1j*boson_vbias}")
            #print(f"boson_mfshift = {self.boson_mfshift}")
            #print(f"xbar          = {xbar}")

            op_power = tau * backend.einsum("zn, nNM->zNM", xshift, self.chol_B)
            #op_power = tau * backend.einsum("zn, nNM->zNM", xi, self.chol_B)

            temp = walkers.boson_phiw.copy()
            for order_i in range(self.taylor_order):
                temp = backend.einsum("zNM, zM->zN", op_power, temp) / (order_i + 1.0)
                walkers.boson_phiw += temp

            # compute the cfb, cmf for weights updates
            cfb = backend.einsum("zn, zn->z", xi, xbar) - 0.5 * backend.einsum("zn, zn->z", xbar, xbar)
            # factors due to MF shift and force bias
            cmf = -backend.sqrt(dt) * backend.einsum("zn, n->z", xshift, self.boson_mfshift)
        else:
            # Trace over fermionic DOF to construct the photonic (bilinear part) Hamiltonian
            logger.debug(self, f"Debug: propagating the bilinear term in product formalism")
            # may move this part into the propagate_boson
            zlambda = backend.einsum("pq, Xpq ->X", walkers.rho, self.geb)
            Hb = boson_adag_plus_a(nmodes, self.system.nboson_states, zlambda)

            #TODO: matrix element of <m|e^{-z_\alpha(a^\dag_\alpha + a_\alpha)}|n> can be analytically evaluated
            # exp(-\sqrt{w/2} g c^\dag_i c_j (b^\dag + b)) | n>
            evol_Hep = scipy.linalg.expm(-dt * Hb)
            walkers.boson_phiw = backend.einsum("NM, zM->zN", evol_Hep, walkers.boson_phiw)

        # compute bosonic energy
        logger.debug(self, f"Debug: bosonic WF after bilinear: {abs(backend.sum(walkers.boson_phiw, axis=0)) / walkers.nwalkers}")

        ## update bilinear part of local energy
        Gfermions = [walkers.Ga, walkers.Gb] if walkers.ncomponents > 1 else [walkers.Ga, walkers.Ga]
        eloc = local_eng_eboson(self.system.boson_freq, self.system.nboson_states, self.geb, Gfermions, walkers.boson_Gf)
        self.e_local_boson = eloc

        ##merged into the update_weight()
        #walkers.weights *= backend.exp(
        #   -dt * (eloc.real + elocold.real - 2.0 * self.e_boson_shift) / 2.0
        #)
        # logger.debug(self, f"Debug: old bilinear local energy {backend.dot(elocold, walkers.weights)}")
        # logger.debug(self, f"Debug: new bilinear local energy {backend.dot(eloc, walkers.weights)}")

        if self.decouple_bilinear:
            return cfb, cmf
        else:
            return backend.zeros(walkers.nwalkers), backend.zeros(walkers.nwalkers)


    def update_GF(self, trial, walkers):
        r"""update GF (TODO: move to trial)"""
        # walkers.Ga = backend.einsum("zqi, pi->zpq", walkers.Ghalfa, trial.psia.conj())
        walkers.Ga = compute_GF_base(walkers.Ghalfa, trial.psia.conj())
        if walkers.ncomponents > 1:
            walkers.Gb = compute_GF_base(walkers.Ghalfb, trial.psib.conj())
            # walkers.Gb = backend.einsum("zqi, pi->zpq", walkers.Ghalfb, trial.psib.conj())
            walkers.Gf = walkers.Ga + walkers.Gb
            logger.debug(self, f"walkers.Gb.shape = {walkers.Gb.shape}")
        else:
            walkers.Gf = 2.0 * walkers.Ga
        # logger.debug(self, f"walkers.Gf.shape = {walkers.Gf.shape}")

        if not self.decouple_bilinear:
            walkers.rho = (
                backend.einsum("z, zpq->pq", walkers.weights, walkers.Gf)
                / backend.sum(walkers.weights)
            )
        # logger.debug(self, f"test DM = {backend.trace(walkers.rho)}")

        # get Qalpha in either 1st or 2nd quantization
        walkers.Qalpha = walkers.get_boson_bdag_plus_b(trial, walkers.boson_phiw)
        logger.debug(self, f"Debug: Updated Q value is {walkers.Qalpha}")


    def propagate_walkers(self, trial, walkers, ltensor, eshift=0.0, verbose=0):
        r"""Propagate the walkers function for the coupled electron-boson interactions"""

        # 1) compute overlap and update the Green's funciton
        t0 = time.time()
        ovlp = trial.ovlp_with_walkers_gf(walkers) # fermionic
        if not self.turnoff_bosons:
            boson_ovlp = trial.boson_ovlp_with_walkers(walkers) # bosonic
            ovlp *= boson_ovlp

        if not self.decouple_bilinear:
            # 2) update Fermionic DM and Qalpha for the bilinear terms
            self.update_GF(trial, walkers)

        self.wt_ovlp += time.time() - t0

        #
        # 3) boson propagator (including free and bilinear terms)
        #
        logger.debug(self, f"Debug: bosonic WF before one propagation: {abs(backend.sum(walkers.boson_phiw, axis=0)) / walkers.nwalkers}")

        if not self.turnoff_bosons:
            t0 = time.time()
            self.propagate_bosons(trial, walkers, 0.5 * self.dt)
            self.wt_boson += time.time() - t0

        #
        # 4) Fermionic one-body propagation
        #
        # Note: 4a)-c) are similar to bare case, we may recycle the bare code
        # but with updated ltensors and shifted h1e.
        # super().propagate_walkers(trial, walkers, ltensor, eshift=eshift)
        #
        # 4a) onebody fermionic propagator propagation :math:`e^{-dt/2*H1e}` with effective one-body electronic term
        #
        self.propagate_walkers_onebody(walkers)

        # 4b) two-body fermionic term
        ltmp = ltensor.copy()
        cfb, cmf = self.propagate_walkers_twobody(trial, walkers, ltmp)
        del ltmp

        # 4c) one-body electronic term
        self.propagate_walkers_onebody(walkers)

        # 5) two-body bosonic term (TBA)

        # 6) bilinear term
        if not self.turnoff_bosons:
            t0 = time.time()
            cfb2, cmf2 = self.propagate_bosons_bilinear(trial, walkers, self.dt)
            # self.propagate_walkers_bilinear(trial, walkers, self.dt)
            cfb += cfb2
            cmf += cmf2
            self.wt_bilinear += time.time() - t0

            # 7) one-body_boson propagator
            t0 = time.time()
            self.propagate_bosons(trial, walkers, 0.5 * self.dt)
            self.wt_boson += time.time() - t0
        #print(f"Debug: bosonic WF: {abs(backend.sum(walkers.boson_phiw, axis=0)) / walkers.nwalkers}")

        #
        # 8) update weights
        #
        t0 = time.time()
        newovlp = trial.ovlp_with_walkers(walkers)
        if not self.turnoff_bosons:
            new_boson_ovlp = trial.boson_ovlp_with_walkers(walkers) # bosonic
            logger.debug(self, f"Debug: norm of new overlap is {backend.linalg.norm(newovlp)}")
            logger.debug(self, f"Debug: bosonic overlap is {backend.linalg.norm(new_boson_ovlp)}")
            # logger.debug(self, f"Debug: bosonic overlap is {new_boson_ovlp}")
            newovlp *= new_boson_ovlp

        # print("new       ovlp = ", newovlp)
        # print("new boson ovlp = ", new_boson_ovlp)
        self.wt_ovlp += time.time() - t0

        t0 = time.time()
        self.update_weight(walkers, ovlp, newovlp, cfb, cmf, eshift)
        self.wt_weight += time.time() - t0
        logger.debug(self, f"Debug: updated weight: {backend.linalg.norm(walkers.weights)}\n eshift = {eshift}")

        assert not backend.isnan(backend.linalg.norm(walkers.weights)), "NaN detected in walkers.weights"

# TODO: finite temperature QMC propagators:
