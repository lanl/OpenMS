import numpy as np
from openms.lib import _qmclib
from openms.qmc import trial
from openms.qmc import propagators
import time

nw = nwalkers = 100
ndim = 100
no = ndim // 2
nsteps = 10

order = 6

op = np.random.rand(ndim, ndim)
phi = np.random.rand(nwalkers, ndim, no)

# =============== one-body propagation ============
phi1 = phi.copy()
phi0 = phi.copy()
_qmclib.propagate_onebody(op, phi1)
phi2 = _qmclib.propagate_onebody_return(op, phi)

# compare with python code
propagators.propagate_onebody(op, phi0)

np.testing.assert_allclose(phi1, phi2, rtol=1e-5, atol=1e-8)
np.testing.assert_allclose(phi0, phi2, rtol=1e-5, atol=1e-8)

# ============= two-body example ================
hs_op = np.random.rand(nwalkers, ndim, ndim)
phiw = np.random.rand(nwalkers, ndim, no)

_qmclib.propagate_exp_op(phiw, hs_op, order)
# `phiw` is now modified in-place

op = (np.random.rand(nw, ndim, ndim) + 1j * np.random.rand(nw, ndim, ndim)).astype(np.complex128)
phiw = (np.random.rand(nw, ndim, no) + 1j * np.random.rand(nw, ndim, no)).astype(np.complex128)

phiw1 = phiw.copy()
phiw2 = phiw.copy()

t0 = time.time()
for _ in range(nsteps):
    _qmclib.propagate_exp_op_complex(phiw1, op, order)
t1 = time.time() - t0

t0 = time.time()
for _ in range(nsteps):
    propagators.propagate_exp_op(phiw2, op, order)
t2 = time.time() - t0
# `phiw` is now updated in-place

np.testing.assert_allclose(phiw1, phiw2, rtol=1e-5, atol=1e-8)
print(f"Wall times of two-body: {t1:.3f} {t2:.3f}  speedup = {t2/t1:.1f}")

#================= Ghalf and ovlp =================
phiw = (np.random.rand(nw, ndim, no) + 1j * np.random.rand(nw, ndim, no)).astype(np.complex128)
psi = (np.random.rand(ndim, no) + 1j * np.random.rand(ndim, no)).astype(np.complex128)

t1 = 0.0
t2 = 0.0
for _ in range(nsteps):
    t0 = time.time()
    ovlp2 = trial.trial_walker_ovlp_base(phiw, psi)
    ovlp3, ghalf1 = trial.trial_walker_ovlp_gf_base(phiw, psi)
    t1 += time.time() - t0

    t0 = time.time()
    ovlp1 = _qmclib.trial_walker_ovlp_base(phiw, psi)
    ovlp4, ghalf2 = _qmclib.trial_walker_ovlp_gf_base(phiw, psi)
    t2 += time.time() - t0

np.testing.assert_allclose(ovlp1, ovlp2, rtol=1e-5, atol=1e-8)
np.testing.assert_allclose(ovlp3, ovlp4, rtol=1e-5, atol=1e-8)
np.testing.assert_allclose(ghalf1, ghalf2, rtol=1e-5, atol=1e-8)
print(f"Wall times of computing overlap and GFs: {t1:.3f}  {t2:.3f} Speedup = {t1/t2:.1f}")


# energy estimator

nchol = 300

rltensor = np.random.rand(nchol, ndim, no) + 1j * 1.e-12 # np.random.rand(nchol, ndim, no)
Ghalf = np.random.rand(nwalkers, ndim, no) + 1j * 1.e-12 # np.random.rand(nwalkers, ndim, no)

from openms.qmc import estimators

t0 = time.time()
exx1 = estimators.exx_rltensor_Ghalf(rltensor, Ghalf)
t1 = time.time() - t0

t0 = time.time()
exx2 = _qmclib.exx_rltensor_Ghalf_complex(rltensor, Ghalf)
t2 = time.time() - t0

t0 = time.time()
exx3 = _qmclib.exx_rltensor_Ghalf(rltensor.real, Ghalf.real)
t3 = time.time() - t0

print(f"Wall times of computing exx: {t1:.3f}  {t2:.3f} {t3:.3f} Speedup = {t1/t2:.1f}")
np.testing.assert_allclose(exx1, exx2, rtol=1e-5, atol=1e-8)
np.testing.assert_allclose(exx1, exx3, rtol=1e-5, atol=1e-8)
