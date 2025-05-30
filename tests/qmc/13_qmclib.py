import unittest
from openms.qmc import QMCLIB_AVAILABLE
import numpy
import numpy as np
import numpy.testing as npt
import time

try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

# NUMBA_AVAILABLE = False

if NUMBA_AVAILABLE:
    from numba import njit, prange
if QMCLIB_AVAILABLE:
    from openms.lib import _qmclib

from openms.qmc.propagators import propagate_onebody
from openms.qmc.propagators import propagate_exp_op
from openms.qmc.estimators import exx_rltensor_Ghalf
from openms.qmc.estimators import ecoul_rltensor_Ghalf
from openms.qmc.trial import trial_walker_ovlp_base
from openms.qmc.trial import trial_walker_ovlp_gf_base
if NUMBA_AVAILABLE:
    from openms.qmc.estimators import exx_rltensor_Ghalf_numba
    from openms.qmc.estimators import ecoul_rltensor_Ghalf_numba
    from openms.qmc.propagators import propagate_onebody_numba
    from openms.qmc.propagators import propagate_exp_op_numba
    from openms.qmc.propagators import tensor_dot_numba
    from openms.qmc.trial import trial_walker_ovlp_base_numba
    from openms.qmc.trial import trial_walker_ovlp_gf_base_numba

# ===================
# propagation
# ===================

# @njit(parallel=True, fastmath=True)
# def propagate_exp_op_numba(phiw, op, order):
#     """
#     Apply exponential operator via Taylor expansion:
#
#     Parameters
#     ----------
#     phiw : np.ndarray of shape (nwalkers, ndim)
#         Walker wavefunctions
#     op : np.ndarray of shape (nwalkers, ndim, ndim)
#         Operator A applied to each walker
#     order : int
#         Order of Taylor expansion
#     """
#     nwalkers = phiw.shape[0]
#
#     for iw in prange(nwalkers):
#         temp = phiw[iw].copy()
#         for i in range(order):
#             temp = op[iw] @ temp / (i + 1.0)
#             phiw[iw] += temp
#     return phiw


# ===================
# exx
# ===================


# @njit(parallel=True, fastmath=True)
# def exx_rltensor_Ghalf_numba(rltensor, Ghalf):
#     """
#     Parameters
#     ----------
#     rltensor : np.ndarray (nchol, a, b)
#         Real-space tensor
#     Ghalf : np.ndarray (nwalkers, b, c)
#         Half of the Green's function
#
#     Returns
#     -------
#     exx : np.ndarray (nwalkers,)
#         Exchange energy per walker
#     """
#     nwalkers = Ghalf.shape[0]
#     nchol = rltensor.shape[0]
#     exx = np.zeros(nwalkers, dtype=np.complex128)
#
#     for i in prange(nwalkers):
#         for l in range(nchol):
#             LG = rltensor[l].T @ Ghalf[i]  # shape (b, c) -> (a, c)
#             LG_flat = LG.ravel()
#             LG_T_flat = LG.T.ravel()
#             exx[i] += LG_flat @ LG_T_flat
#
#     exx *= 0.5
#     return exx


# ===========================
# overlap functions
# ===========================


#@njit(parallel=True, fastmath=True)
#def trial_walker_ovlp_gf_base_numba(phiw, psi):
#    """
#    Compute trial-walker overlaps and Green's function using matrix ops.
#
#    Parameters
#    ----------
#    phiw : ndarray (z, p, i)
#        Walker wavefunctions.
#    psi : ndarray (p, j)
#        Trial wavefunction.
#
#    Returns
#    -------
#    ovlp : ndarray (z, i, j)
#        Overlap matrices.
#    Ghalf : ndarray (z, p, i)
#        Green's function components.
#    """
#    nw = phiw.shape[0]
#    ndim, no = psi.shape
#
#    ovlp = np.empty((nw, no, no), dtype=np.complex128)
#    Ghalf = np.empty((nw, ndim, no), dtype=np.complex128)
#
#    for z in prange(nw):
#        ovlp[z] = phiw[z].T @ psi.conj()
#
#        # inv_ovlp = inv(ovlp[z])
#        inv_ovlp = np.linalg.inv(ovlp[z])
#
#        Ghalf[z] = phiw[z] @ inv_ovlp.T
#
#    return ovlp, Ghalf
#
#
#@njit(parallel=True, fastmath=True)
#def trial_walker_ovlp_base_numba(phiw, psi):
#    nw = phiw.shape[0]
#    no = psi.shape[1]
#
#    ovlp = np.empty((nw, no, no), dtype=np.complex128)
#
#    for z in prange(nw):
#        ovlp[z] = phiw[z].T @ psi.conj()
#    return ovlp




nwalkers, ndim = 200, 100
no = ndim // 2
order = 5
nchol = 200
nsteps = 50

phiw = np.random.rand(nwalkers, ndim, ndim) + 1j * 1.e-8
phiw = phiw[:, :, :no]
# op = np.random.rand(nwalkers, ndim, ndim)
op = np.random.rand(nwalkers, ndim, ndim) + 1j * np.random.rand(nwalkers, ndim, ndim)
one_op = op[0]

rltensor = np.random.rand(nchol, ndim, no) + 1j * np.random.rand(nchol, ndim, no)
Ghalf = np.random.rand(nwalkers, ndim, no) + 1j * np.random.rand(nwalkers, ndim, no)


# Warm-up to trigger JIT (Numba)

if NUMBA_AVAILABLE:
    _ = propagate_onebody_numba(op[0], phiw)
    _ = propagate_exp_op_numba(phiw.copy(), op, order)
    _ = exx_rltensor_Ghalf_numba(rltensor, Ghalf)
    _ = ecoul_rltensor_Ghalf_numba(rltensor, Ghalf)


def format_walltimes(t1, t2, t3):

    ostring1 = f"Wall times: native {t1:.3f}"
    ostring2 = "Speedup:   "
    if NUMBA_AVAILABLE:
        ostring1 += f" numba {t2:.3f}"
        ostring2 += f" native/numba = {t1/t2:.2f}"
    if QMCLIB_AVAILABLE:
        ostring1 += f", qmclib {t3:.3f}"
        ostring2 += f", qmclib/native = {t1/t3:.2f}"
    print(ostring1)
    print(ostring2)


def onebody_test():
    print(f"\n{'*' * 10} test onebody propagation {'*' * 10}")
    wtimes = np.zeros(3)

    phiw1 = phiw.copy()
    phiw2 = phiw.copy()
    phiw3 = phiw.copy()

    t1, t2, t3 = 0.0, 0.0, 0.0
    for i in range(nsteps):
        if i % 10 == 0:
            print(f"step {i}")

        t0 = time.time()
        propagate_onebody(one_op, phiw1)
        t1 += time.time() - t0

        if NUMBA_AVAILABLE:
            t0 = time.time()
            propagate_onebody_numba(one_op, phiw2)
            t2 += time.time() - t0

        if QMCLIB_AVAILABLE:
            t0 = time.time()
            phiw3 = _qmclib.propagate_onebody_complex(one_op, phiw3)
            t3 += time.time() - t0

    # Assertions for correctness
    if NUMBA_AVAILABLE:
        npt.assert_allclose(phiw1, phiw2, rtol=1e-5, atol=1e-8)
    if QMCLIB_AVAILABLE:
        npt.assert_allclose(phiw1, phiw3, rtol=1e-5, atol=1e-8)
    format_walltimes(t1, t2, t3)


def twobody_test():
    """
    data = setup_qmc_data()
    phiw1 = data["phiw"].copy()
    phiw2 = data["phiw"].copy()
    phiw3 = data["phiw"].copy()
    op = data["op"]
    order = data["order"]
    """
    phiw1 = phiw.copy()
    phiw2 = phiw.copy()
    phiw3 = phiw.copy()

    t1, t2, t3 = 0.0, 0.0, 0.0

    print(f"\n{'*' * 10} test two-body propagation {'*' * 10}")
    for i in range(nsteps):
        if i % 10 == 0:
            print(f"step {i}")

        t0 = time.time()
        propagate_exp_op(phiw1, op, order)
        t1 += time.time() - t0

        if NUMBA_AVAILABLE:
            t0 = time.time()
            propagate_exp_op_numba(phiw2, op, order)
            t2 += time.time() - t0

        if QMCLIB_AVAILABLE:
            t0 = time.time()
            _qmclib.propagate_exp_op_complex(phiw3, op, order)
            t3 += time.time() - t0

    # Assertions for correctness
    if NUMBA_AVAILABLE:
        npt.assert_allclose(phiw1, phiw2, rtol=1e-5, atol=1e-8)
    if QMCLIB_AVAILABLE:
        npt.assert_allclose(phiw1, phiw3, rtol=1e-5, atol=1e-8)
    format_walltimes(t1, t2, t3)

def exx_test():

    print(f"\n {'*' * 10}test exx {'*' * 10}")
    t1 = 0.0
    t2 = 0.0
    t3 = 0.0
    for i in range(0, nsteps, 10):
        if i % 10 == 0: print(f"step {i}")
        t0 = time.time()
        exx1 = exx_rltensor_Ghalf(rltensor, Ghalf)
        t1 += time.time() - t0

        if NUMBA_AVAILABLE:
            t0 = time.time()
            exx2 = exx_rltensor_Ghalf_numba(rltensor, Ghalf)
            t2 += time.time() - t0

        if QMCLIB_AVAILABLE:
            t0 = time.time()
            exx3 = _qmclib.exx_rltensor_Ghalf_complex(rltensor, Ghalf)
            t3 += time.time() - t0

    # Assertions for correctness
    if NUMBA_AVAILABLE:
        npt.assert_allclose(exx1, exx2, rtol=1e-5, atol=1e-8)
    if QMCLIB_AVAILABLE:
        npt.assert_allclose(exx1, exx3, rtol=1e-5, atol=1e-8)
    format_walltimes(t1, t2, t3)


def ecoul_test():

    print(f"\n {'*' * 10}test ecoul {'*' * 10}")
    Ghalfa = (np.random.rand(nwalkers, ndim, no) + 1j * np.random.rand(nwalkers, ndim, no)).astype(np.complex128)
    rltensora = (np.random.rand(nchol, ndim, no) + 1j * np.random.rand(nchol, ndim, no)).astype(np.complex128)

    Ghalfb = (np.random.rand(nwalkers, ndim, no) + 1j * np.random.rand(nwalkers, ndim, no)).astype(np.complex128)
    rltensorb = (np.random.rand(nchol, ndim, no) + 1j * np.random.rand(nchol, ndim, no)).astype(np.complex128)

    t1 = 0.0
    t2 = 0.0
    t3 = 0.0
    for i in range(0, nsteps, 10):
        if i % 10 == 0: print(f"step {i}")
        t0 = time.time()
        #ec1 = ecoul_rltensor_Ghalf(rltensora, Ghalfa)
        ec1 = ecoul_rltensor_Ghalf(rltensora, Ghalfa, rltensorb, Ghalfb)
        t1 += time.time() - t0

        if NUMBA_AVAILABLE:
            t0 = time.time()
            ec2 = ecoul_rltensor_Ghalf_numba(rltensor, Ghalf)
            t2 += time.time() - t0

        if QMCLIB_AVAILABLE:
            t0 = time.time()
            #ec3 = _qmclib.ecoul_rltensor_Ghalf_complex(rltensora, Ghalfa)
            ec3 = _qmclib.ecoul_rltensor_uhf_complex(rltensora, Ghalfa, rltensorb, Ghalfb)
            t3 += time.time() - t0

    # Assertions for correctness
    if NUMBA_AVAILABLE:
        npt.assert_allclose(ec1, ec2, rtol=1e-5, atol=1e-8)
    if QMCLIB_AVAILABLE:
        npt.assert_allclose(ec1, ec3, rtol=1e-5, atol=1e-8)
    format_walltimes(t1, t2, t3)
    """

    ecoul1 = ecoul_rltensor_Ghalf(rltensora, Ghalfa)
    ecoul3 = _qmclib.ecoul_rltensor_Ghalf_complex(rltensora, Ghalfa)
    npt.assert_allclose(ecoul1, ecoul3, rtol=1e-5, atol=1e-8)
    print(ecoul1.shape)  # (nwalkers,)
    print(ecoul3.shape)  # (nwalkers,)
    """


def tensordot_test():
    print(f"\n {'*' * 10}test tensordot {'*' * 10}")

    xshift = np.random.randn(nwalkers, nchol) + 1j * np.random.randn(nwalkers, nchol)
    ltensor = np.random.rand(nchol, ndim, ndim) + 1j * np.random.rand(nchol, ndim, ndim)
    sqrtdt = 1j * np.sqrt(0.1)

    t1, t2, t3 = 0.0, 1.e-12, 0.0
    for i in range(nsteps):
        if i % 10 == 0: print(f"step {i}")
        t0 = time.time()
        eri_op1 = sqrtdt * np.dot(xshift, ltensor.reshape(nchol, -1)).reshape(nwalkers, ndim, ndim)
        t1 += time.time() - t0

        #if NUMBA_AVAILABLE:
        #    t0 = time.time()
        #    eri_op2 = tensor_dot_numba(xshift, ltensor, sqrtdt)
        #    t2 += time.time() - t0

        if QMCLIB_AVAILABLE:
            t0 = time.time()
            eri_op3 = _qmclib.tensordot_complex(xshift, ltensor)
            t3 += time.time() - t0

    # Assertions for correctness
    #if NUMBA_AVAILABLE:
    #    npt.assert_allclose(eri_op1, eri_op2, rtol=1e-5, atol=1e-8)
    if QMCLIB_AVAILABLE:
        npt.assert_allclose(eri_op1, eri_op3, rtol=1e-5, atol=1e-8)
    format_walltimes(t1, t2, t3)



def overlap_test():
    print(f"\n {'*' * 10}test ovlp and Ghalf {'*' * 10}")

    phiw = np.random.randn(nwalkers, ndim, no) + 1j * np.random.randn(nwalkers, ndim, no)
    psi = np.random.randn(ndim, no) + 1j * np.random.randn(ndim, no)

    # warm up
    _, _ = trial_walker_ovlp_gf_base_numba(phiw, psi)
    _ = trial_walker_ovlp_base_numba(phiw, psi)

    t1, t2, t3 = 0.0, 0.0, 0.0
    for i in range(nsteps):
        if i % 10 == 0: print(f"step {i}")
        t0 = time.time()
        ovlp1, Ghalf1 = trial_walker_ovlp_gf_base(phiw, psi)
        ovlp1 = trial_walker_ovlp_base(phiw, psi)
        t1 += time.time() - t0

        if NUMBA_AVAILABLE:
            t0 = time.time()
            ovlp2, Ghalf2 = trial_walker_ovlp_gf_base_numba(phiw, psi)
            ovlp2 = trial_walker_ovlp_base_numba(phiw, psi)
            t2 += time.time() - t0

        if QMCLIB_AVAILABLE:
            t0 = time.time()
            ovlp3, Ghalf3 = _qmclib.trial_walker_ovlp_gf_base(phiw, psi)
            ovlp3 = _qmclib.trial_walker_ovlp_base(phiw, psi)
            t3 += time.time() - t0

    # Assertions for correctness
    if NUMBA_AVAILABLE:
        npt.assert_allclose(ovlp1, ovlp2, rtol=1e-5, atol=1e-8)
        npt.assert_allclose(Ghalf1, Ghalf2, rtol=1e-5, atol=1e-8)
    if QMCLIB_AVAILABLE:
        npt.assert_allclose(ovlp1, ovlp3, rtol=1e-5, atol=1e-8)
        npt.assert_allclose(Ghalf1, Ghalf3, rtol=1e-5, atol=1e-8)
    format_walltimes(t1, t2, t3)


@unittest.skipIf(not NUMBA_AVAILABLE and not QMCLIB_AVAILABLE, "Numba and QMCLib not available")
class TestNumba(unittest.TestCase):

    #def test_onebody(self):
    #    onebody_test()

    #def test_twobody(self):
    #    twobody_test()

    #def test_overlap(self):
    #    overlap_test()

    def test_exx(self):
        exx_test()

    #def test_ecoul(self):
    #    ecoul_test()

    #def test_tensordot(self):
    #    tensordot_test()

if __name__ == "__main__":
    unittest.main()
