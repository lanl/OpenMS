
import unittest
from pyscf import gto, scf, df, lib
import scipy
import numpy
import time

def get_mol(basis="sto3g", verbose=3):

    itest = 0
    zshift = itest * 2.0

    atom = f"C   0.00000000   0.00000000    {zshift};\
             O   0.00000000   1.23456800    {zshift};\
             H   0.97075033  -0.54577032    {zshift};\
             C  -1.21509881  -0.80991169    {zshift};\
             H  -1.15288176  -1.89931439    {zshift};\
             C  -2.43440063  -0.19144555    {zshift};\
             H  -3.37262777  -0.75937214    {zshift};\
             O  -2.62194056   1.12501165    {zshift};\
             H  -1.71446384   1.51627790    {zshift};\
             C   0.00000000   0.00000000    {zshift+3.0};\
             O   0.00000000   1.23456800    {zshift+3.0};\
             H   0.97075033  -0.54577032    {zshift+3.0};\
             C  -1.21509881  -0.80991169    {zshift+3.0};\
             H  -1.15288176  -1.89931439    {zshift+3.0};\
             C  -2.43440063  -0.19144555    {zshift+3.0};\
             H  -3.37262777  -0.75937214    {zshift+3.0};\
             O  -2.62194056   1.12501165    {zshift+3.0};\
             H  -1.71446384   1.51627790    {zshift+3.0}"

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
        # basis="sto6g",
        # basis="ccpvdz",
        unit="Angstrom",
        symmetry=True,
        verbose=verbose,
    )

    return mol

def test_chols():

    mol = get_mol()
    mf = scf.RHF(mol)
    mf.kernel()
    dm = mf.make_rdm1()

    # make chols
    nao = mf.mol.nao_nr()
    eri = mf.mol.intor("int2e", aosym="s1")

    eri_2d = eri.reshape((nao**2, -1))
    u, s, v = scipy.linalg.svd(eri_2d)

    threshold = 1.e-1
    idx = s > threshold
    ltensor = u[:,idx] * numpy.sqrt(s[idx])
    ltensor = ltensor.T
    ltensor = ltensor.reshape(ltensor.shape[0], nao, nao)

    # ltensor.shape
    print("ltensor.shape=", ltensor.shape)

    #-----------------------
    # use fll eri, naive way
    #-----------------------
    t1 = time.time()
    vj0 = numpy.einsum("pqrs, rs->pq", eri, dm) # (nao)^4
    vk0 = numpy.einsum("psrq, rs->pq", eri, dm) # (nao)^4
    t1 = time.time() - t1

    #-----------------------
    # pyscf dot_eri_dm
    #-----------------------
    t2 = time.time()
    vj1, vk1 = scf.hf.dot_eri_dm(eri, dm)
    t2 = time.time() - t2

    print(f" vj0 = vj1? ", numpy.allclose(vj0, vj1))
    print(f" vk0 = vk1? ", numpy.allclose(vk0, vk1))

    #-----------------------
    # use chols
    # vj2 = numpy.einsum("npq, nrs, rs->pq", ltensor, ltensor, dm)
    # vk2 = numpy.einsum("npq, nrs, rq->ps", ltensor, ltensor, dm)
    #-----------------------
    # the following is not so efficient
    t3 = time.time()
    tmp = numpy.einsum("npq, pq->n", ltensor, dm) # nchol * nao * nao
    vj2 = numpy.einsum('npq, n->pq', ltensor, tmp) #nchol * nao * nao

    tmp = numpy.einsum("npq, rq->npr", ltensor, dm) # nchol * (nao)^3
    vk2 = numpy.einsum("npr, nrs->ps", tmp, ltensor) # nchol * (nao)^3
    t3 = time.time() - t3

    print(f" vj0 = vj2? ", numpy.linalg.norm(vj0 - vj2))
    print(f" vk0 = vk2? ", numpy.linalg.norm(vk0 - vk2))

    print(" compare run times: ", t1, t2, t3)


def df_scf():
    mol = get_mol(basis='ccpvdz')

    auxbasis = 'ccpvdz-jk-fit'
    auxmol = df.addons.make_auxmol(mol, auxbasis)

    # ints_3c is the 3-center integral tensor (ij|P), where i and j are the
    # indices of AO basis and P is the auxiliary basis
    ints_3c2e = df.incore.aux_e2(mol, auxmol, intor='int3c2e')
    ints_2c2e = auxmol.intor('int2c2e')

    nao = mol.nao
    naux = auxmol.nao

    # Compute the DF coefficients (df_coef) and the DF 2-electron (df_eri)
    df_coef = scipy.linalg.solve(ints_2c2e, ints_3c2e.reshape(nao*nao, naux).T)
    df_coef = df_coef.reshape(naux, nao, nao)

    df_eri = lib.einsum('ijP,Pkl->ijkl', ints_3c2e, df_coef)
    print("ints_3c2e.shape", ints_3c2e.shape)
    print("df_coeff.shape", df_coef.shape)

    # Now check the error of DF integrals wrt the normal ERIs
    eri0 = mol.intor('int2e')
    print(abs(eri0 - df_eri).max())
    print(numpy.linalg.norm(eri0 - df_eri))

    mf0 = scf.RHF(mol)
    mf0.kernel()
    e1 = mf0.e_tot

    mf = scf.RHF(mol).density_fit()
    mf.kernel()
    e2 = mf.e_tot
    print(e1, e2)


def qedhf_cd():
    from openms.mqed import qedhf
    from openms.mqed import scqedhf, vtqedhf

    verbose = 4
    zshift = 0.0
    mol = get_mol(basis="ccpvdz", verbose=verbose)
    mol = get_mol(basis="sto3g", verbose=verbose)
    # print("nao = ", mol.nao_nr())

    nmode = 1
    cavity_freq = numpy.zeros(nmode)
    cavity_mode = numpy.zeros((nmode, 3))
    cavity_freq[0] = 0.5
    cavity_mode[0, :] = 0.1 * numpy.asarray([0, 1, 0])

    energies = []

    qedmf = scqedhf.RHF(mol, xc=None, cavity_mode=cavity_mode.copy(), cavity_freq=cavity_freq.copy())
    qedmf.max_cycle = 100
    qedmf.CD_anyway = True
    qedmf.CD_thresh = 1.e-8
    qedmf.kernel()
    energies.append(qedmf.e_tot)

    # VT
    qedmf = vtqedhf.RHF(mol, xc=None, cavity_mode=cavity_mode.copy(), cavity_freq=cavity_freq.copy())
    qedmf.max_cycle = 100
    qedmf.CD_anyway = True
    qedmf.CD_thresh = 1.e-8
    qedmf.kernel()
    energies.append(qedmf.e_tot)

    # VSQ
    qedmf = vtqedhf.RHF(mol, xc=None, cavity_mode=cavity_mode.copy(), cavity_freq=cavity_freq.copy())
    qedmf.max_cycle = 100
    qedmf.CD_anyway = True
    qedmf.CD_thresh = 1.e-8
    qedmf.qed.squeezed_var = numpy.zeros(nmode)
    qedmf.qed.optimize_vsq = True
    qedmf.kernel()
    e_vsq = qedmf.e_tot
    energies.append(qedmf.e_tot)

    return energies


class TestCD(unittest.TestCase):
    def test_cd(self):
        energies = qedhf_cd()
        refs = numpy.array([-262.115302000, -262.125200258278, -262.125344251655])
        for e0, eref in zip(energies, refs):
            self.assertAlmostEqual(e0, eref, places=6, msg="Etot does not match the reference value.")


if __name__ == '__main__':
    # test_chols()
    # df_scf()
    # qedhf_cd()
    unittest.main()
