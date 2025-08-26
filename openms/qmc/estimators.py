import numpy as backend
import numpy as np
import scipy
import time
from openms.lib.misc import deprecated
from openms.lib import QMCLIB_AVAILABLE, NUMBA_AVAILABLE


# for each observables, we may save several quantities using a small class
#  to handle the these data

class observables(object):
    def __init__(self, name, *args, **kargs):
        self.name = name

        self._expectation = {}
        self.build()

    def build(self):

        if self.name == "energy":
            # etot = unscaled_etot / total_weights
            self._expectation = {
                "etot": 0.0 + 0.0j,  # scaled total energy sum(weight * e) / sum(weights)
                "total_weights": 0.0 + 0.0j,  # sum(wights)
                "unscaled_etot": 0.0 + 0.0j,  # sum(weight* e), unscaled total energy
                "unscaled_e1": 0.0 + 0.0,  # sum(weight * E1), unscaled one-body energy
                "unscaled_e2": 0.0 + 0.0,  # sum(weight * E2), unscaled two-body energy
            }
         # quantities to be stored for occupation analysis

    def update(self, values):
        r"""update the data
        """
        pass


    @property
    def size(self):
        return len(self._expectation)


    def reset(self):
        r"""reset the value to zero"""
        for key, value in self._expectation:
            if isinstance(v, np.ndarray):
                self._expectation[key] = np.zeros_like(value)
            else:
                self._expectation[key] = 0.0 + 0.0j


def get_wfn(weights, psiw):
    r"""
    Get the wavefunction from QMC walkers:

    .. math::

        \ket{\Psi} = \frac{1}{N} \sum_k w_k \ket{\psi_w}

    where :math:`N` is the normalization factor.

    """
    wfn = backend.einsum("z,zFSak->FSak", weights, psiw)
    norm = backend.linalg.det(backend.einsum("FSaj,FSak->jk", wfn.conj(), wfn))
    norm = backend.prod(norm)
    return wfn / backend.sqrt(norm)


# ---------------------------------
# mixed sparsity - low rank scheme
# ---------------------------------

def _banded_LT_dot_G(L, G, bw, backend):
    """
    Compute LG = L^T @ G using that L is (approximately) banded with half-bandwidth = bw.
    Shapes: L (N,N), G (N,M) -> LG (N,M)
    """
    N, M = G.shape
    LG = backend.zeros((N, M), dtype=backend.result_type(L, G))
    for q in range(N):
        p0 = q - bw if q - bw >= 0 else 0
        p1 = q + bw + 1 if q + bw + 1 <= N else N
        if p1 <= p0:  # empty
            continue
        col = L[p0:p1, q]                 # (len,)
        if col.size == 0:
            continue
        LG[q, :] = backend.dot(col.T, G[p0:p1, :])
    return LG

def exx_mixed_rank_sparsity(
    rltensor,    # (Ngamma, N, N) – unchanged over time
    Ghalf,       # (nwalkers, N, N)
    plan,        # from build_Ltensor_plan
    meta,        # from build_Ltensor_plan
    backend=None
):
    if backend is None:
        import numpy as np
        backend = np

    nwalkers = Ghalf.shape[0]
    Ngamma = meta["Ngamma"]
    blksize = meta["blksize"]

    exx = backend.zeros(nwalkers, dtype=backend.complex128)

    for i in range(nwalkers):
        G = Ghalf[i]  # used as both T and Theta
        acc = backend.zeros((), dtype=backend.complex128)
        for l in range(Ngamma):
            pl = plan[l]
            if pl["type"] == "lowrank":
                # LG = (T @ X) @ (G.T @ Uright)^T
                A = backend.dot(G.T, pl["X"])        # (N,k)
                B = backend.dot(G.T, pl["Uright"])   # (N,k)
                LG = backend.dot(pl["X"], B.T)             # (N,N)
                #LG = backend.dot(A, B.T)             # (N,N)
            else:
                # banded route
                L = rltensor[l]
                LG = _banded_LT_dot_G(L, G, blksize, backend)

            # exx += <LG.ravel(), LG.T.ravel()> = sum(LG * LG.T)
            acc += backend.sum(LG * LG.T)
        exx[i] = 0.5 * acc

    return exx


# -------------------------------
# Gf estimators
# -------------------------------


def bosonic_GF(T, W):
    r"""compute the bosonic green's function

    Parameters
    ----------
    T: ndarray
       Trial WF
    W: ndarray
       Walker WF

    Returns
    -------
    Gf: ndarray
       Walker GF
    Ghalf: ndarray
       Walker half-rotated GF
    """

    # T shape is nfock
    # W shape is (nwalker, nfock)

    # TODO:
    TW = backend.dot(W.T, T.conj())
    Ghalf = backend.dot(scipy.linalg.inv(TW), W.T)
    Gf = backend.dot(T.conj(), Ghalf)
    return Gf, Ghalf


def GF(T, W):
    r"""
    Compute one-body Green's function (Eqns. 65-66 of Ref. :cite:`zhang2021jcp`):

    .. math::

        G_{ij} = & \frac{\bra{\Psi_T} c^\dagger_i c_j \ket{\psi_k}}{\langle \Psi_T \ket{\psi_k}}
               =\left[W(T^{\dagger} W)^{-1} T^{\dagger} \right]_{ji} \\
        G^{1/2}_{ij} = & [W(T^{\dagger} W)^{-1}]_{ij}

    where :math:`T/W` are the matrix associated with the SD
    (trial and walker), respectively :math:`\ket{\Psi_{T/W}}`

    Parameters
    ----------
    T: ndarray
       Trial WF
    W: ndarray
       Walker WF

    Returns
    -------
    Gf: ndarray
       Walker GF
    Ghalf: ndarray
       Walker half-rotated GF
    """

    TW = backend.dot(W.T, T.conj())
    Ghalf = backend.dot(scipy.linalg.inv(TW), W.T)
    Green = backend.dot(T.conj(), Ghalf)

    return Green, Ghalf


def GF_so(T, W, na, nb):
    r"""
    Compute one-body Green's function in SO

    .. math::

        G^\sigma_{ij} = \frac{\bra{\Psi_T} c^\dagger_i c_j \ket{\psi_k}}{\langle \Psi_T \ket{\psi_k}}
               =\left[W^\sigma(T^{\sigma\dagger} W^\sigma)^{-1} T^{\sigma\dagger} \right]_{ji}

    """
    Gfa, Gfa_half = GF(T[:, :na], W[:, :na])
    # print(f"Debug: T.shape = ", T.shape, " na/nb =", na, nb)
    # print(f"Debug: Gfa.shape = {Gfa.shape}")
    if nb > 0:
        Gfb, Gfb_half = GF(T[:, na:], W[:, na:])
        # print(f"Debug: Gfb.shape = {Gfb.shape}")
    else:
        Gfb = backend.zeros(Gfa.shape, dtype=Gfa.dtype)
        Gfb_half = backend.zeros((0, Gfa_half.shape[1]), dtype=Gfa_half.dtype)
    return backend.array([Gfa, Gfb]), [Gfa_half, Gfb_half]


# -----------------------------------------------------------
# energy estimators for coupled electron-boson interactions
# -----------------------------------------------------------

#
# TODO: make a dict to map different combinaiton of trial and walkers
# to specific function for computing energy and other properties
# To do so, we have two steps:
#    1): generate trial_walker header:
#    2): map the trial_walker header to certian function according
# to the dict below
#
#energy_dict = {
#    "SD_trial_rhf_walker": xx,
#    "SD_trial_uhf_walker": xx,
#    "MSD_trial_rhf_walker": xx,
#    "MSD_trial_uhf_walker": xx,
#}

# function to handle different energy measurement case
def measure_energy(trial, walkers, h1e, ltensors, enuc):
    r"""Measure ground state energy based on walker weights and GF

    According to the type of walkers, the energy measurement is
    directed to different functions.
    """

    # TODO:

    pass


# local energy for coupled fermion-boson system

def local_eng_eb_2nd(h1e, chols, geb, freq, Gf, Gb, spin_fac=0.5):
    r"""compute the local enegy of the coupled electron-boson system
    in the second quantizaiton format
    """
    nao = h1e.shape[1]

    E_elec = local_eng_elec_chol_new(h1e, ltensor, Gf, spin_fac)


def local_eng_eb_1st(h1e, eri, gmat, mass, freq, Gf, Q, laplacian, spin_fac=0.5):
    r"""Compute the local energy of coupled electron-boson system
    in the firstquantizaiton format

    Args:

        h1e: one-body integral
        eri: two-body integral
        gmat: electron-boson coupling matrix
        mass: mass of bosons
        freq: frequencies of bosons
        Q: coordinates of bosons
        Gf: green's function
        laplacian: laplacian of bosons

    Return:

       local energy (with different components)
    """

    # assume shape of [spin, nao, nao]
    nao = h1e.shape[1]

    # 1) electronic part
    E_electron = local_eng_elec(h1e, eri, Gf, spin_fac)

    # 2) bosonic part
    E_boson = local_eng_boson(nao, mass, frq, Q)

    # 3) e-boson coupling
    rho = Gf[0].diagonal() + Gf[1].diagonal()
    e_eb = -gmat * backend.sqrt(2.0 * mass * freq) * backend.dot(rho, Q)

    # 4) total energy
    etot = E_electron + E_boson + e_eb

    return [etot, E_electron, E_boson, e_eb]


# -----------------------------
# bosonic energy estimators
# -----------------------------

def e_rh1e_Ghalf(rh1e, Ghalf):
    r"""compute one body energy using rotated_h1e and Ghalf
    """
    if False:
        e1 = backend.einsum("qi, zqi->z", rh1e, Ghalf)
    else:
        nwalkers = Ghalf.shape[0]
        tmp = Ghalf.reshape((nwalkers, -1))
        e1 = backend.dot(tmp, rh1e.ravel())
    return e1


def local_energy_SD_RHF(trial, walkers, enuc = 0.0):
    r"""Compute local energy with half-rotated integrals
    """
    # Ghalfa/b: [nwalkers, nao, na/nb]
    # rh1a/b: [nao, na/nb]

    # e1 = 2.0 * backend.einsum("qi, zqi->z", trial.rh1a, walkers.Ghalfa)
    e1 = 2.0 * e_rh1e_Ghalf(trial.rh1a, walkers.Ghalfa)
    e1 += enuc

    t0 = time.time()
    # coulomb energy
    ecoul = 4.0 * ecoul_rltensor_uhf(trial.rltensora, walkers.Ghalfa)
    t1 = time.time()

    # exchange
    exx = 2.0 * exx_rltensor_Ghalf_kernel(trial.rltensora, walkers.Ghalfa)
    t2 = time.time()
    # print(f"Debug: wall time of ecoul and exx: {t1-t0:.3f} {t2-t1:.3f}")
    e2 = ecoul - exx

    return e1, e2


def ecoul_rltensor_uhf(rltensora, Ghalfa, rltensorb=None, Ghalfb=None):
    r"""Compute Coulomb energy

    Parameters
    ----------

    """

    if False:
        # einsum code
        LG = backend.einsum("nqi, zqi->zn", rltensora, Ghalfa)
        if Ghalfb is not None:
            LG += backend.einsum("nqi, zqi->zn", rltensorb, Ghalfb)
        ecoul = 0.5 * backend.einsum("zn, zn->z", LG, LG)
    else:
        nwalkers = Ghalfa.shape[0]
        nchol = rltensora.shape[0]

        tmpa = Ghalfa.reshape((nwalkers, -1))
        LG = backend.dot(tmpa, rltensora.reshape((nchol, -1)).T)
        if Ghalfb is not None:
            tmpb = Ghalfb.reshape((nwalkers, -1))
            LG += backend.dot(tmpb, rltensorb.reshape((nchol, -1)).T)
        # (nwalkers, nchol)
        ecoul = 0.5 * backend.sum(LG * LG, axis=1)
    return ecoul

ecoul_rltensor_Ghalf = ecoul_rltensor_uhf

def exx_THC_Ghalf(rX, U, Ghalf):
    r"""Compute exchange energy via the THC methods

    Not done yet!!!

    .. math::

        L^\gamma_{\mu\nu} \simeq \sum_P V^\gamma_{\mu P} W_P^\gamma U^\gamma_{\nu P}
                          = \sum_P X_{\gamma, \mu P} U^\gamma_{\nu P}

    where :math:`X_{\gamma, \mu P} = V^\gamma_{\mu P} W^\gamma_P`.
    With THC, :math:`f^\gamma_{ij}` can be further written as

    .. math::
        f^\gamma_{ij} = & \sum_{pr\mu} T_{pi} [(U^\gamma_{p\mu} \sigma^\gamma_{\mu}) U^\gamma_{r\mu}] \Theta_{rj} \\
                      = & \sum_{pr\mu} [T_{pi} X^\gamma_{p\mu}] [U^\gamma_{r\mu} \Theta_{rj}] \\
                      = & \sum_{\mu} A_{\gamma, i\mu} B_{\gamma,j\mu}

    Parameters
    ----------
    rX : :class:`numpy.ndarray`
        Half-rotated HTC tensor
    U : :class:`numpy.ndarray`
        Second THC
    Ghalf : :class:`numpy.ndarray`
        Walker's half-rotated Green's function
        Shape is (nwalkers, nao, nsigma).

    Returns
    -------
    exx : :class:`numpy.ndarray`
        Exchange energy for all walkers.
    """

    # TODO: Double check the code

    # rX is the rotated X, [rX]_{qP} = \sum_p T_{pq} X_{pP}
    nwalkers = Ghalf.shape[0]
    nchol = rltensor.shape[0]
    exx = backend.zeros(nwalkers, dtype=backend.complex128)
    for i in range(nwalkers):
        for l in range(nchol):
            B = backend.dot(U[l].T, Ghalf[i]) # \mu j
            LG = backend.dot(rX[l], B) # ij
            exx[i] += backend.dot(LG.ravel(), LG.T.ravel())
    exx *= 0.5
    return exx


# numba functions
if NUMBA_AVAILABLE:
    from numba import njit, prange

    # @njit(parallel=True, fastmath=True)
    @njit(parallel=True)
    def exx_rltensor_Ghalf_numba(rltensor, Ghalf):
        """
        Parameters
        ----------
        rltensor : np.ndarray (nchol, nao, no)
            Real-space tensor
        Ghalf : np.ndarray (nwalkers, nao, no)
            Half of the Green's function

        Returns
        -------
        exx : np.ndarray (nwalkers,)
            Exchange energy per walker
        """
        nwalkers = Ghalf.shape[0]
        nchol = rltensor.shape[0]
        exx = np.zeros(nwalkers, dtype=np.complex128)

        for i in prange(nwalkers):
            for l in range(nchol):
                # LG = rltensor[l].T @ Ghalf[i]
                LG = rltensor[l].T.astype(np.complex128) @ Ghalf[i]

                LG_flat = LG.ravel()
                LG_T_flat = LG.T.ravel()
                exx[i] += LG_flat @ LG_T_flat

        exx *= 0.5
        return exx


    @njit(parallel=True, fastmath=True)
    def ecoul_rltensor_Ghalf_numba(rltensora, Ghalfa, rltensorb=None, Ghalfb=None):
        """
        Efficient Coulomb energy calculation using reshaped dot products.
        Avoids explicit loops by relying on np.dot().
        """
        nwalkers = Ghalfa.shape[0]
        nchol = rltensora.shape[0]

        ecoul = np.zeros(nwalkers, dtype=np.complex128)

        for iw in prange(nwalkers):
            for l in range(nchol):
                LG = np.dot(Ghalfa[iw].ravel(), rltensora[l].ravel())
                if Ghalfb is not None and rltensorb is not None:
                    LG += np.dot(Ghalfb[iw].ravel(), rltensorb[l].ravel())
                ecoul[iw] += 0.5 * LG

        return ecoul


def exx_rltensor_Ghalf(rltensor, Ghalf):
    r"""Compute exchange contribution for real Choleskies with RHF/UHF trial.

    Note that :math:`G_{pq} = \sum_i T_{pi} \Theta_{qi}`.
    The exchange energy is written as:

    .. math::
        E_X & = \sum_{pqrs} V_{pqrs} G_{ps} G_{qr} \\
            & = \sum_{pqrs,\gamma} L_{\gamma, pr} G_{ps}  L_{\gamma, qs} G_{qr} \\
            & = \sum_{pqrs,\gamma} L_{\gamma, pr} T_{pi} \Theta_{si} L_{\gamma, qs} T_{qj} \Theta_{rj} \\
            & = \sum_{ijrs,\gamma} \tilde{L}_{\gamma, ri} \Theta_{si} \tilde{L}_{\gamma, sj} \Theta_{rj} \\
            & = \sum_{ij,\gamma} f^\gamma_{ij} f^\gamma_{ij}

    where :math:`f^\gamma_{ij} = \tilde{L}_{\gamma, ri}\Theta_{rj}`.


    Parameters
    ----------
    rltensor : :class:`numpy.ndarray`
        Half-rotated cholesky for one spin.
    Ghalf : :class:`numpy.ndarray`
        Walker's half-rotated Green's function
        Shape is (nwalkers, nao, nsigma).

    Returns
    -------
    exx : :class:`numpy.ndarray`
        Exchange energy for all walkers.
    """

    # t0 = time.time()
    # rltensor_real = rltensor.real
    # rltensor_imag = rltensor.imag

    # Ghalf_real = Ghalf.real
    # Ghalf_imag = Ghalf.imag

    # LG_real = backend.einsum('nqi, zqj->znij', rltensor_real, Ghalf_real) - \
    #          backend.einsum('nqi, zqj->znij', rltensor_imag, Ghalf_imag)
    # LG_imag = backend.einsum('nqi, zqj->znij', rltensor_real, Ghalf_imag) + \
    #          backend.einsum('nqi, zqj->znij', rltensor_imag, Ghalf_real)
    # LG = LG_real + 1.0j * LG_imag
    # t1 = time.time() - t0

    if False:
        # another version without separating real and imaginary parts
        # einsum code # creating a big tensor is time/memery consuming
        t0 = time.time()
        LG = backend.einsum('nqi, zqj->znij', rltensor, Ghalf)
        t2 = time.time() - t0
        # print(f"Debug: compare wall time {t1} vs {t2}")
        # Compute exchange contribution
        exx = 0.5 * backend.einsum('znij, znji->z', LG, LG)
    else:
        nwalkers = Ghalf.shape[0]
        nchol = rltensor.shape[0]
        exx = backend.zeros(nwalkers, dtype=backend.complex128)
        for i in range(nwalkers):
            for l in range(nchol):
                LG = backend.dot(rltensor[l].T, Ghalf[i]) # ij
                exx[i] += backend.dot(LG.ravel(), LG.T.ravel())
                # LG = rltensor[l].T @ Ghalf[i]
                # exx[i] += np.sum(LG * LG.T)
        exx *= 0.5

    return exx


if QMCLIB_AVAILABLE:
    from openms.lib import _qmclib
    exx_rltensor_Ghalf_kernel = _qmclib.exx_rltensor_Ghalf_complex
elif NUMBA_AVAILABLE:
    exx_rltensor_Ghalf_kernel = exx_rltensor_Ghalf_numba
else:
    exx_rltensor_Ghalf_kernel = exx_rltensor_Ghalf


@deprecated
def exx_rltensor_Ghalf_chunked(rltensor, Ghalf, comm, MPI, nwalkers, counts, displs):
    r"""Distributed computation of exx energy

    We assume:
       - 1) Ghalf is scattered over all rank and each rank has it's local chunk
       - 2) ltensor is shared across the nodes

    """

    rank = comm.Get_rank()
    nlocal = counts[rank]
    nchol = rltensor.shape[0]

    # Local computation
    # (we can recycle the "exx_rltensor_Ghalf" code for this local computation
    # ------------------------------------------------------
    local_exx = np.zeros(nlocal, dtype=np.complex128)

    for i in range(nlocal):
        for l in range(nchol):
            LG = np.dot(rltensor[l].T, Ghalf[i])  # shape (n, m)
            local_exx[i] += np.dot(LG.ravel(), LG.T.ravel())  # scalar
    local_exx *= 0.5

    # Gather exx from all processes
    # ------------------------------------------------------
    exx = None
    if rank == 0:
        exx = np.zeros(nwalkers, dtype=np.complex128)

    # Note numpy\.complex128 <--> MPI.COMPLEX16
    comm.Gatherv(local_exx, [exx, counts, displs, MPI.COMPLEX16], root=0)

    return exx


def local_energy_SD_UHF(trial, walkers, enuc = 0.0):
    r"""Compute local energy with half-rotated integrals
    """
    # Ghalfa/b: [nwalkers, nao, na/nb]
    # rh1a/b: [nao, na/nb]

    e1 = e_rh1e_Ghalf(trial.rh1a, walkers.Ghalfa)
    e1 += e_rh1e_Ghalf(trial.rh1b, walkers.Ghalfb)
    e1 += enuc

    # coulomb energy
    ecoul = ecoul_rltensor_uhf(trial.rltensora, walkers.Ghalfa, trial.rltensorb, walkers.Ghalfb)

    # exchange
    exx = exx_rltensor_Ghalf_kernel(trial.rltensora, walkers.Ghalfa)
    exx += exx_rltensor_Ghalf_kernel(trial.rltensorb, walkers.Ghalfb)

    e2 = ecoul - exx

    # print(f"Debug: e1 = {e1}")
    # print(f"Debug: ecoul = {ecoul}")
    # print(f"Debug: exx = {exx}")
    # print(f"Debug: e2 = {e2}")
    # print(f"Debug: e_tot = {e1 + e2}")

    return e1, e2

def local_eng_boson_2nd(omega, nboson_states, Gb):
    r"""compute the local bosonic energies with bosonic GF (Gb) in
    compute local energy of bosons in 2nd quantizaiton

    omega: ndarray
    nboson_states: ndarray [nfock, ..., nfock_n]
    Gb: ndarray, bosonic green function
    """
    # bosonc energy
    basis = backend.asarray(
        [backend.arange(mdim) for mdim in nboson_states]
    )

    waTa = backend.einsum("m, mF->mF", omega, basis).ravel()
    eb = backend.einsum("F,zFF->z", waTa, Gb)
    #print("Debug: Bosonic Gf = ", Gb)
    #print("Debug: waTa =       ", waTa)
    #print("eb =                ", eb)
    return eb


def local_eng_eboson(omega, nboson_states, geb, Gfermions, Gboson):
    r"""
    Gfermions: tuple of Fermionic GFs for up and down spin (if available)

    Parameters
    ----------
    omega: 1d array
        frequencies of bosons
    nboson_states: 1d array
        number of Fock state for each bosonic mode
    geb: ndarray
        electron-boson coupling matrix
    Gfermions: ndarray
        Fermionic GFS
    Gboson: ndarray
        bosonic GFs

    Returns
    -------
    eg: ndarray
        electron-boson interacting energy
    """

    nmodes = len(omega)

    zalpha = backend.einsum("npq, zpq->zn", geb, Gfermions[0])
    if Gfermions[1] is not None:
        zalpha += backend.einsum("npq, zpq->zn", geb, Gfermions[1])

    boson_size = sum(nboson_states)
    Hb = backend.zeros((Gboson.shape[0], boson_size, boson_size), dtype=backend.complex128)
    idx = 0
    for imode in range(nmodes):
        mdim = nboson_states[imode]
        a = backend.diag(backend.sqrt(backend.arange(1, mdim)), k=1)
        h_od = a + a.T
        Hb[:, idx:idx+mdim, idx:idx+mdim] += h_od[None, :, :] * zalpha[:, imode][:, None, None]
    eg = backend.einsum("zNM,zNM->z", Hb, Gboson)
    return eg


local_eng_boson = local_eng_boson_2nd


def local_eng_boson_1st(nao, mass, freq, Q):
    r"""Compute local energy of bosons in 1st quantization"""

    kin = -0.5 * backend.sum(Lap) / mass - 0.5 * freq * nao
    pot = 0.5 * freq**2 * mass * backend.sum(Q * Q)
    return kin + pot


# -------------------------------
# electronic energy estimators
# -------------------------------


def local_eng_elec_chol_new(h1e, ltensor, Gf):
    r"""Computing local energy Using L tensor and G

    .. math::

        ej = & 2 I_{pq rs} G_{rs} G_{pq}
           = 2 [L^*_{n, pq} G_{pq}] [L_{n, rs}G_{rs}] \\
        ek = & I_{prqs} * G_{ps} * G_{qr}
           = L^*_{n, pr} * G_{ps} * [L_{n, qs} * G_{qr}]
    """
    vj = backend.einsum("nrs, zrs->zn", ltensor, Gf)
    ej = 2.0 * backend.sum(vj * vj, axis=1)

    # ek is the major bottleneck
    # may replace it with c++ code
    vk = backend.einsum("npr, zps->znrs", ltensor.conj(), Gf)
    ek = backend.einsum("znpr, znrp->z", vk, vk)

    # this version uses less memory
    """
    nchols = ltensor.shape[0]
    nw = Gf.shape[0]
    ek = 0.0
    for iw in range(nw):
        Gtmp = Gf[iw]
        vk = backend.einsum("npr, ps->nrs", ltensor.conj(), Gtmp)
        ek += backend.einsum("npr, nrp->z", vk, vk)
    """

    # one-body term
    e1 = 2.0 * backend.einsum("zpq, pq->z", Gf, h1e)
    energy = e1 + ej - ek
    return energy


def local_eng_elec_chol(TL_theta, h1e, vbias, Gf):
    r"""Compute local energy from oei, eri and GF

    Args:
        Gf: Green function

    .. math::

         E = \sum_{pq\sigma} T_{pq} G_{pq\sigma}
             + \frac{1}{2}\sum_{pqrs\sigma\sigma'} I_{prqs} G_{pr\sigma} G_{qs\sigma'}
             - \frac{1}{2}\sum_{pqrs\sigma} I_{pqrs} G_{ps\sigma} G_{qr\sigma}

    if :math:`L_\gamma` tensor is used
    [PS: need to rotate Ltensor into (nocc, norb) shape since G's shape is (nocc, norb)],

    .. math::

         E = & \sum_{pq\sigma} T_{pq} G_{pq\sigma}
             + \frac{1}{2}\sum_{\gamma,pqrs\sigma} L_{\gamma,ps} L_{\gamma,qr} G_{pr\sigma} G_{qs\sigma'}
             - \frac{1}{2}\sum_{\gamma,pqrs\sigma} L_{\gamma,ps} L_{\gamma,qr} G_{ps\sigma} G_{qr\sigma} \\
           = & \sum_{pq\sigma} T_{pq} G_{pq\sigma}
             + \frac{1}{2}\sum_{\gamma,pq\sigma\sigma'} (L_\gamma G_\sigma)_{pq} (L_\gamma G_{\sigma'})_{pq}
             - \frac{1}{2}\sum_{\gamma,\sigma} [\sum_{pq} L_{\gamma,pq} G_{pq\sigma}]^2

    i.e. the Ecoul is :math:`\left[\frac{\bra{\Psi_T}L\ket{\Psi_w}}{\bra{\Psi_T}\Psi_w\rangle}\right]^2`,
    which is the TL_Theta tensor in the code
    """

    vbias2 = vbias * vbias
    ej = 2.0 * backend.einsum("zn->z", vbias2)
    ek = backend.einsum("znpr, znrp->z", TL_theta, TL_theta)
    e2 = ej - ek

    # approach 1) : most inefficient way
    # e2 = 2.0 * backend.einsum("prqs, zpr, zqs->z", eri, Gf, Gf)
    # e2 -= backend.einsum("prqs, zps, zqr->z", eri, Gf, Gf)

    # approach 3): use normal way without using ltensors
    # vjk = 2.0 * backend.einsum("prqs, zpr->zqs", eri, Gf) # E_coulomb
    # vjk -= backend.einsum("prqs, zps->zqr", eri, Gf)  # exchange
    # e2 = backend.einsum("zqs, zqs->z", vjk, Gf)

    e1 = 2.0 * backend.einsum("zpq, pq->z", Gf, h1e)
    energy = e1 + e2
    return energy


def local_eng_elec(h1e, eri, Gf, spin_fac=0.5):
    r"""Compute local energy of electrons"""

    kin = backend.einsum("zSpq,Spq->z", Gf, h1e) * spin_fac

    # E_coul
    tmp = 2.0 * backend.einsum("prqs,zSpr->zqs", eri, Gf) * spin_fac
    ecoul = backend.einsum("zqs,zSqs->z", tmp, Gf)
    # E_xx
    tmp = backend.einsum("prqs,zSps->zSqr", eri, Gf)
    exx = backend.einsum("zSqs,zSqs->z", tmp, Gf)
    pot = (ecoul - exx) * spin_fac

    return kin + pot


local_eng_elec_spin = local_eng_elec


#

_available_observables = {
    "energy": measure_energy,  # total ground state energy
    # "local_energy": measure_local_energy,  # local energy per site
    # "occupation": measure_occupation,  # Fermionic occupation
    # "boson_occ": measure_bosonic_occupation,  # occupation of boson
}


if __name__ == "__main__":
    import random

    def _hubbard_hamilts_pbc(L, U):
        h1e = backend.zeros((L, L))
        g2e = backend.zeros((L,) * 4)
        for i in range(L):
            h1e[i, (i + 1) % L] = h1e[(i + 1) % L, i] = -1
            g2e[i, i, i, i] = U
        return h1e, g2e

    L = 10
    U = 4.0
    nw = 10  # number of walkers

    h1e, eri = _hubbard_hamilts_pbc(L, U)
    h1e = backend.asarray([h1e for i in range(2)])
    Gf = backend.zeros((nw, 2, L, L))
    for iw in range(nw):
        Gf[iw] = (
            backend.asarray([backend.ones((L, L)) for i in range(2)]) * random.random()
        )

    ee = local_eng_elec(h1e, eri, Gf)
    print("electronic energy is", ee)

    # define phonons
    mass = 0.5
    freq = 0.5
    from openms.qmc.trial_boson import TrialQ

    gmat = backend.random.rand(L, L)

    phonon_trial = TrialQ(mass=mass, freq=freq)

    print(phonon_trial.mass)
    Q = backend.zeros(L)
    for i in range(L):
        Q[i] = 0.1 * i
    print("\nLocal phonon energy =", phonon_trial.local_energy(Q))
