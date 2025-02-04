import numpy
import scipy
import time
from pyscf import gto
from openms.qmc import tools
import unittest
from openms.qmc.afqmc import AFQMC

def get_mol(basis="631g", verbose=1):

    zshift = 0.0
    atom = f"C   0.00000000   0.00000000    {zshift};\
             O   0.00000000   1.23456800    {zshift};\
             H   0.97075033  -0.54577032    {zshift};\
             C  -1.21509881  -0.80991169    {zshift};\
             H  -1.15288176  -1.89931439    {zshift};\
             C  -2.43440063  -0.19144555    {zshift};\
             H  -3.37262777  -0.75937214    {zshift};\
             O  -2.62194056   1.12501165    {zshift};\
             H  -1.71446384   1.51627790    {zshift}"

    mol = gto.M(
        atom = atom,
        basis=basis,
        unit="Angstrom",
        symmetry=True,
        verbose=verbose,
    )
    return mol

def test_cholesky_errors(num_steps=6):

    mol = get_mol(basis="sto3g", verbose=1)

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

    threshold = 1.e-7
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



class TestQMC_Chols(unittest.TestCase):

    def test_qmc_truncated_chols(self):
        #TODO: QMC vs different truncation threshold
        test_cholesky_errors()


    def test_chols_oao(self):

        test_chols_oao()


if __name__ == '__main__':
    # test_cholesky_errors(num_steps=7)
    unittest.main()
