import numpy as backend
import scipy



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


# -------------------------------
# Gf estimators
# -------------------------------


def bosonic_GF(T, W):
    r"""compute the bosonic green's function"""

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
    if nb > 0:
        Gfb, Gfb_half = GF(T[:, na:], W[:, na:])
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
    return eb


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


_available_observables = {
    "energy": measure_energy,  # total ground state energy
    # "occupation": measure_occupation,  # Fermionic occupation
    # "boson_occ": measure_bosonic_occupation,  # occupation of boson
    # "occupation": measure_occupation,
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
