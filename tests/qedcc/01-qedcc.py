import unittest
import numpy
from pyscf import gto, scf, cc
from openms.mqed import qedhf, ccsd
import copy
import time

def LiH_ref():
   qedccsd_refs = numpy.asarray(\
       [-0.0558287,  -0.05584209, \
       -0.05584289,  -0.05585593,\
       -0.056926,    -0.05693954,\
       -0.05695403,  -0.05697047])
   if not complete:
       qedccsd_refs = numpy.asarray(\
           [-0.05734083, -0.05735435, \
            -0.05735539, -0.05736854, \
            -0.05846713, -0.05848054, \
            -0.05849606, -0.05851241])

   atom = f"Li 0.0    0.0     0.0; F 0.0  0.0   1.5"

   mol = gto.M(
       atom = atom,
       basis="sto3g",
       #basis="cc-pvdz",
       unit="Angstrom",
       symmetry=True,
       verbose=3,
   )
   return mol, qedccsd_refs


class TestQEDCC(unittest.TestCase):

    def test_qedcc_ref(self):
        mol, qedccsd_refs = LiH_ref()

        nmode = 1 # create a zero (second) mode to test the code works for multiple modes
        cavity_freq = numpy.zeros(nmode)
        cavity_mode = numpy.zeros((nmode, 3))
        cavity_freq[0] = 3.0 / 27.211386245988
        cavity_mode[0, :] = 1.e-1 * numpy.asarray([0, 0, 1])

        mf = scf.RHF(mol)
        mf.kernel()

        qedmf = qedhf.RHF(mol, xc=None, cavity_mode=cavity_mode, cavity_freq=cavity_freq,
                         add_nuc_dipole=False)
        qedmf.max_cycle = 500
        qedmf.kernel() #dm0=dm)


        # cc
        qed = copy.copy(qedmf.qed)
        qed.verbose = 3
        qed.kernel()

        # Note this energy is different from the test 01-qedhf as we don't include nuclear_dipole
        # in the coupling

        mycc = cc.CCSD(mf)
        mycc.diis_space = 10
        mycc.kernel()

        e_corrs = []
        for add_U2n in [False, True]:
           for nfock1 in range(1, 3):
              for nfock2 in range(1,3):
                 qed = copy.copy(qedmf.qed)
                 qed.verbgose = 5

                 myqedccsd = ccsd.CCSD(qed, nfock1=nfock1, nfock2=nfock2, add_U2n=add_U2n)
                 myqedccsd.max_cycle = 200
                 myqedccsd.verbose = 3
                 e_tot, e_corr = myqedccsd.kernel()
                 e_corrs.append(e_corr)

        e_corrs = numpy.asarray(e_corrs)

        for e_corr, qedccsd_ref in zip(e_corrs, qedccsd_refs):
            self.assertAlmostEqual(e_corr, qedccsd_ref, places=5,
                msg="Correlation energy does not match the reference value.")

    def test_qedcc2(self):
        r"""Test with without complete basis set assumption (i.e., no Q)"""
        mol, qedccsd_refs = LiH_ref(complete=False)

        nmode = 1 # create a zero (second) mode to test the code works for multiple modes
        cavity_freq = numpy.zeros(nmode)
        cavity_mode = numpy.zeros((nmode, 3))
        cavity_freq[0] = 3.0 / 27.211386245988
        cavity_mode[0, :] = 1.e-1 * numpy.asarray([0, 0, 1])

        mf = scf.RHF(mol)
        mf.kernel()

        qedmf = qedhf.RHF(mol, xc=None, cavity_mode=cavity_mode, cavity_freq=cavity_freq,
                         add_nuc_dipole=False, complete_basis=False)

        qedmf.max_cycle = 500
        qedmf.kernel() #dm0=dm)


        # cc
        qed = copy.copy(qedmf.qed)
        qed.verbose = 3
        qed.kernel()


        mycc = cc.CCSD(mf)
        mycc.diis_space = 10
        mycc.kernel()

        e_corrs = []
        for add_U2n in [False, True]:
           for nfock1 in range(1, 3):
              for nfock2 in range(1,3):
                 qed = copy.copy(qedmf.qed)
                 qed.verbgose = 3

                 myqedccsd = ccsd.CCSD(qed, nfock1=nfock1, nfock2=nfock2, add_U2n=add_U2n)
                 myqedccsd.max_cycle = 500
                 myqedccsd.verbose = 3
                 start_time= time.time()
                 e_tot, e_corr = myqedccsd.kernel()
                 e_corrs.append(e_corr)

        e_corrs = numpy.asarray(e_corrs)

        for e_corr, qedccsd_ref in zip(e_corrs, qedccsd_refs):
            self.assertAlmostEqual(e_corr, qedccsd_ref, places=5,
                msg="Correlation energy does not match the reference value.")


    def test_qedcc_isomer(self):
        pass

if __name__ == '__main__':
    unittest.main()
