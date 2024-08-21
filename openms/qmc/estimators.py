
import numpy as backend

def GF(T, W):
    r"""
    Compute one-body Green's function (Eqns. 65-66 of Ref. :cite:`zhang2021jcp`):

    .. math::

        G_{ij} = \frac{\bra{\Psi_T} c^\dagger_i c_j \ket{\psi_k}}{\langle Psi_T \ket{\psi_k}}
               =\left[W(T^\dagger W)^{-1} T^\dagger \right]_{ji}

    where :math:`T/W` are the matrix associated with the SD (trial and walker), respectively :math:`\ket{\Psi_{T/W}}`
    """

    #
    TW = backend.dot(W.T, T.conj())
    Ghalf = backend.dot(scipy.linalg.inv(TW), W.T)
    Green = backend.dot(T.conj(), Ghalf)

    return Green, Ghalf

def GF_so(T, W, na, nb):
    r"""
    Compute one-body Green's function in SO
    """
    Gfa, Gfa_half = GF(T[:, :na], W[:, :na])
    if nb > 0:
        Gfb, Gfb_half = GF(T[:, na:], W[:, na:])
    else:
        Gfb = backend.zeros(Gfa.shape, dtype=Gfa.dtype)
        Gfb_half = backend.zeros((0, Gfa_half.shape[1]), dtype=Gfa_half.dtype)
    return backend.array([Gfa, Gfb]), [Gfa_half, Gfb_half]


def local_eng_elec_chol(TL_theta, h1e, eri, vbias, Gf):
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

    i.e. the Ecoul is :math:`\left[\frac{\bra{\Psi_T}L\ket{\Psi_w}{\bra{\Psi_T}\Psi_w\rangle}\right]^2`,
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
    r"""Compute local energy of electrons
    """

    kin = backend.einsum("zSpq,Spq->z", Gf, h1e) * spin_fac

    # E_coul
    tmp = 2.0 * backend.einsum("prqs,zSpr->zqs", eri, Gf) * spin_fac
    ecoul = backend.einsum("zqs,zSqs->z", tmp, Gf)
    # E_xx
    tmp = backend.einsum("prqs,zSps->zSqr", eri, Gf)
    exx = backend.einsum("zSqs,zSqs->z", tmp, Gf)
    pot = (ecoul - exx) * spin_fac

    return kin + pot


