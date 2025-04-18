import numpy
import scipy
import time
from pyscf import gto, scf
from openms.qmc import tools
from openms.qmc.afqmc import AFQMC
from molecules import get_mol
import unittest

try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


def test_cholesky_errors(num_steps=6):

    mol = get_mol(name="C3H4O2", basis="sto3g", verbose=1)

    threshold0 = 1.e-4
    eri = mol.intor('int2e_sph', aosym='s1')
    nao = eri.shape[0]
    # eri_2d = eri.reshape((nao**2, -1))

    results = []
    for j in range(0, num_steps, 2):
        # ltensor in AO
        threshold = threshold0 / 10.0 ** j

        # Block decomposition of eri
        t1 = time.time()
        ltensor1 = tools.chols_blocked(mol, thresh=threshold)
        # ltensor1 = ltensor1.reshape((ltensor1.shape[0], -1))
        t1 = time.time() - t1

        # Full decomposition of eri (which is memory expensive)
        t2 = time.time()
        ltensor2 = tools.chols_full(mol, thresh=threshold)
        # ltensor2 = ltensor2.reshape((ltensor2.shape[0], -1))
        t2 = time.time() - t2

        # Compute errors
        eri_block = numpy.einsum("xpq, xrs->pqrs", ltensor1, ltensor1)
        eri_full = numpy.einsum("xpq, xrs->pqrs", ltensor2, ltensor2)
        err_block = numpy.linalg.norm(eri_block - eri)
        err_full = numpy.linalg.norm(eri_full - eri)

        results.append([
            threshold,
            ltensor1.shape[0], err_block, t1,
            ltensor2.shape[0], err_full, t2
        ])

    print(f"\n{'*' * 70}\n Summary of threshold vs ltensor size and errors\n{'*' * 70}")
    print(f" Threshold  |     block decomposition       |      full decomposition")
    print(f"            |  Nl      error       walltime |  Nl      error      walltime")
    for tmp in results:
        print(f" %9.3e  %5d   %11.5e   %8.4f  %5d   %11.5e   %8.4f" % (tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5], tmp[6]))

def test_chols_oao():
    from functools import reduce
    from pyscf import lo
    from pyscf import ao2mo

    threshold = 1.e-8
    mol = get_mol(basis="sto3g", verbose=1)

    overlap = mol.intor("int1e_ovlp")
    Xmat = lo.orth.lowdin(overlap)
    norb = Xmat.shape[0]

    # get original eri and transform into OAO
    eri = mol.intor('int2e_sph', aosym='s1')  # in AO
    eri = ao2mo.full(eri, Xmat, verbose=0, aosym="s1") # in OAO
    # eri = ao2mo.full(mol, Xmat, verbose=0, aosym="s1") # in OAO
    eri_2d = eri.reshape((norb**2, -1))

    u, s, _ = scipy.linalg.svd(eri_2d)
    idx = (s > threshold)
    ltensor = (u[:,idx] * numpy.sqrt(s[idx])).T
    ltensor = ltensor.reshape(ltensor.shape[0], norb, norb)

    # get chols (truncated)
    # construct eri from truncated chols (in OAO)
    # Note the two ltensor can be different (as they can have different size)
    h1e, ltensor1, nuc = tools.get_h1e_chols(mol, thresh=threshold) # in OAO
    eri_block = numpy.einsum("xpq, xrs->pqrs", ltensor1, ltensor1)

    # check if the two eri close
    error = numpy.linalg.norm(eri - eri_block)
    print(f"Error of cholesky decomposition is : {error}")
    numpy.testing.assert_almost_equal(eri, eri_block, decimal=4, err_msg="ERI do not match the reference value.")


if NUMBA_AVAILABLE:
    @jit(nopython=True, fastmath=True)
    def pack_cholesky_jit(idx_i, idx_j, packed_chol, chol):
        nchol = chol.shape[0]
        nao = chol.shape[-1]
        dim_upper_triangle = nao * (nao + 1) // 2

        for x in range(nchol):
            for i in range(dim_upper_triangle):
                packed_chol[x, i] = chol[x, idx_i[i], idx_j[i]]
        return
else:
    def pack_cholesky_jit(idx_i, idx_j, packed_chol, chol):
        nchol = chol.shape[0]
        nao = chol.shape[-1]
        dim_upper_triangle = nao * (nao + 1) // 2

        for x in range(nchol):
            for i in range(dim_upper_triangle):
                packed_chol[x, i] = chol[x, idx_i[i], idx_j[i]]
        return


def pack_cholesky(idx, chol):
    nchol = chol.shape[0]
    packed_chol = []
    for z in range(nchol):
        packed_chol.append(chol[z][idx])
    packed_chol = numpy.array(packed_chol)
    return packed_chol

def unpack_chol(idx, packed_chol):
    nao = max(idx[0]) + 1
    nchol = packed_chol.shape[0]
    chol = numpy.zeros((nchol, nao, nao), dtype=packed_chol.dtype)
    for z in range(nchol):
        chol[z][idx] = packed_chol[z]
    return chol


def test_pack_triu():
    r""" Test the performance improvement by packing chols
    """

    # mol = get_mol(basis="631g", verbose=1)
    mol = get_mol(basis="ccpvtz", verbose=1)

    t0 = time.time()
    ltensor = tools.chols_blocked(mol, thresh=1.e-6)
    nao = ltensor.shape[-1]
    nchol = ltensor.shape[0]

    print("Time of constructing chols = ", time.time() - t0)
    print(f"nchol = {nchol}  nao = {nao}")

    # for real chols, ltensor is symmetric
    idx = numpy.triu_indices(nao) # (row and col indices)
    ntri = nao * (nao + 1) // 2

    t0 = time.time()
    packed_chol0 = numpy.zeros((nchol, ntri))
    pack_cholesky_jit(idx[0], idx[1], packed_chol0, ltensor)
    print("Time of packing is", time.time() - t0)

    # This way is more efficient
    t0 = time.time()
    packed_chol = pack_cholesky(idx, ltensor)
    print("Time of packing is", time.time() - t0)
    print(numpy.allclose(packed_chol0, packed_chol))

    # TODO: test the efficiency of propagation of two-body terms with packed chol
    nalpha = mol.nelec[0]
    nwalkers = 500
    taylor_order = 6
    dt = 0.01

    # eri_op with full matrix
    phiw1 = numpy.random.random((nwalkers, nao, nalpha)) * (1.0 + 0.1j)
    phiw2 = phiw1.copy()

    xi = numpy.random.normal(0.0, 1.0, nchol * nwalkers)
    xi = xi.reshape(nwalkers, nchol)
    sqrtdt = 1j * numpy.sqrt(dt)


    # compare the tiem of constructing Auxiliary field operator wi/wo packing symmetry
    # (xi, chol) multiplication takes nwaler * nchol * nao * nao float point operations
    t0 = time.time()
    eri_op = sqrtdt * numpy.dot(xi, ltensor.reshape(nchol, -1)).reshape(nwalkers, nao, nao)
    t1 = time.time()
    eri_op = sqrtdt * numpy.dot(xi, packed_chol.reshape(nchol, -1)).reshape(nwalkers, -1)
    eri_op = unpack_chol(idx, eri_op)
    print("Time of constructing eri_op (packed vs non-packed): ", time.time() - t1, t1 - t0)

    wall_times = numpy.zeros(4)
    nsteps = 5
    for k in range(nsteps):
        # compare time of proapgating two-body wi/wo using symmetry in the fields.
        t0 = time.time()
        eri_op = sqrtdt * numpy.dot(xi, ltensor.reshape(nchol, -1)).reshape(nwalkers, nao, nao)
        t1 = time.time()
        for iw in range(phiw1.shape[0]):
            temp = phiw1[iw].copy()
            for i in range(taylor_order):
                temp = numpy.dot(eri_op[iw], temp) / (i + 1.0)
                phiw1[iw] += temp
        t2 = time.time()
        wall_times[0] += (t1 - t0)
        wall_times[1] += (t2 - t1)

        # propagate with uper triangle
        # the taylor expansion operator takes ntaylor * nwalker * nao * nao * nocc operators

        # B * Phi = B_{pq} \Phi_{qj} = B_{p<q} \Phi_{qj} * 2.0
        #                            + B_{qq} \Phi_{qj}
        t2 = time.time()
        eri_op = sqrtdt * numpy.dot(xi, packed_chol.reshape(nchol, -1)).reshape(nwalkers, -1)
        eri_op = unpack_chol(idx, eri_op)
        t3 = time.time()
        for iw in range(phiw2.shape[0]):
            temp = phiw2[iw].copy()
            for i in range(taylor_order):
                # temp = scipy.linalg.blas.dsymm(alpha=1.0, a=eri_op[iw], b=temp, side=0, lower=0) / (i + 1.0)
                temp = numpy.dot(eri_op[iw], temp) / (i + 1.0)
                phiw2[iw] += temp
        t4 = time.time()
        wall_times[2] += (t3 - t2)
        wall_times[3] += (t4 - t3)
    wall_times /= nsteps

    print(f"\n************ compare runtime *************")
    print(f"Get B field operator : {wall_times[0]:.4f}  {wall_times[2]:.4f}")
    print(f"Propagation          : {wall_times[1]:.4f}  {wall_times[3]:.4f}")
    print(f"Total                : {wall_times[0]+wall_times[1]:.4f}  {wall_times[2]+wall_times[3]:.4f}")
    print(numpy.allclose(phiw1, phiw2))


class TestQMC_Chols(unittest.TestCase):

    def test_qmc_truncated_chols(self):
        #TODO: QMC vs different truncation threshold
        test_cholesky_errors()


    def test_chols_oao(self):

        test_chols_oao()


    def test_pack_triu(self):

        test_pack_triu()


if __name__ == '__main__':
    # test_cholesky_errors(num_steps=7)
    unittest.main()
