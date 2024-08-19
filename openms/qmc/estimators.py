
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
        Gfb = numpy.zeros(Gfa.shape, dtype=Gfa.dtype)
        Gfb_half = numpy.zeros((0, Gfa_half.shape[1]), dtype=Gfa_half.dtype)
    return numpy.array([Gfa, Gfb]), [Gfa_half, Gfb_half]

