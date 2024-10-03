import numpy
from pyscf import gto, scf, ao2mo, cc
from openms.mqed import qedhf
from openms.mqed import scqedhf, vtqedhf
from openms.lib.boson import Photon
import unittest


def hubbard_u_mode(n=10, filling=0.5, U=2.0, t=-1.0, PBC=True):

    mol = gto.M(verbose=3)
    mol.nelectron = int(n * filling)

    # over write mol functions
    mol.incore_anyway = True
    mol.nao_nr = lambda *args: n
    mol.tot_electrons = lambda *args: mol.nelectron

    # hopping
    h1 = numpy.zeros((n, n))
    for i in range(n - 1):
        h1[i, i + 1] = h1[i + 1, i] = t
    if PBC:
        h1[n - 1, 0] = h1[0, n - 1] = t

    # onsite U term
    eri = numpy.zeros((n, n, n, n))
    for i in range(n):
        eri[i, i, i, i] = U

    return mol, h1, eri


class TestCustomHam(unittest.TestCase):
    def test_qed_model(self):

        n = 12

        mol, h1, eri = hubbard_u_mode(n=n)
        # MF
        mf = scf.RHF(mol)
        mf.get_hcore = lambda *args: h1
        mf.get_ovlp = lambda *args: numpy.eye(n)
        mf._eri = ao2mo.restore(8, eri, n)
        mf.kernel()
        e1 = mf.e_tot

        # dipole matrix and quadrupole
        dipole_mat = numpy.zeros((3, n, n))
        q_mat = numpy.zeros((9, n, n))
        for i in range(n):
            dipole_mat[0, i, i] = 1.0

        g0 = 1.e-6
        #-----------------
        # QEDHF with zero coupling
        #-----------------

        nmode = 2
        cavity_freq = numpy.zeros(nmode)
        cavity_mode = numpy.zeros((nmode, 3))
        cavity_freq[:] = 0.5
        cavity_mode[0, :] = numpy.asarray([1, 0, 0])
        gfac = g0 * numpy.ones(nmode)

        qed = Photon(mol, mf=mf, omega=cavity_freq, vec=cavity_mode, gfac=gfac)
        # overwrite the integrals
        qed.get_dipole_ao = lambda *args: dipole_mat
        qed.get_quadrupole_ao = lambda *args: q_mat

        qedmf = qedhf.RHF(mol, qed=qed)
        qedmf.get_bare_hcore = lambda *args: h1
        qedmf.get_ovlp = lambda *args: numpy.eye(n)
        qedmf._eri = ao2mo.restore(8, eri, n)

        qedmf.kernel()
        e2 = qedmf.e_tot
        self.assertAlmostEqual(
            e1, e2, places=6, msg="Etot does not match at zero coupling limit!"
        )

        #-----------------
        # scqedhf
        #-----------------
        nmode = 1
        cavity_freq = numpy.zeros(nmode)
        cavity_mode = numpy.zeros((nmode, 3))
        cavity_freq[:] = 0.5
        cavity_mode[0, :] = numpy.asarray([1, 0, 0])
        gfac = g0 * numpy.ones(nmode)

        qed = Photon(mol, mf=mf, omega=cavity_freq, vec=cavity_mode, gfac=gfac)
        # overwrite the integrals
        qed.get_dipole_ao = lambda *args: dipole_mat
        qed.get_quadrupole_ao = lambda *args: q_mat

        qedmf = scqedhf.RHF(mol, qed=qed)
        qedmf.get_bare_hcore = lambda *args: h1
        qedmf.get_ovlp = lambda *args: numpy.eye(n)
        qedmf._eri = ao2mo.restore(1, eri, n)

        qedmf.kernel()
        e2 = qedmf.e_tot
        self.assertAlmostEqual(
            e1, e2, places=6, msg="Etot does not match at zero coupling limit!"
        )

        #-----------------
        # vt-qedhf
        #-----------------

        qedmf = vtqedhf.RHF(mol, qed=qed)
        qedmf.get_bare_hcore = lambda *args: h1
        qedmf.get_ovlp = lambda *args: numpy.eye(n)
        qedmf._eri = ao2mo.restore(1, eri, n)

        qedmf.kernel()
        e2 = qedmf.e_tot
        self.assertAlmostEqual(
            e1, e2, places=6, msg="Etot does not match at zero coupling limit!"
        )

if __name__ == "__main__":
    unittest.main()
