import unittest
import numpy
from pyscf import gto, scf, cc
from openms.mqed import qedhf, ccsd
import copy
import time

class TestQEDCC(unittest.TestCase):
   def test_qedcc_ref(self):
      qedccsd_refs = numpy.asarray(\
            [-14.593881582106865, -14.64207137319351])

      qed_energies = []
      for bond in numpy.arange(2.0, 2.21, 0.20):
        atom = f"Li 0.0    0.0     0.0; Li 0.0  0.0 {bond}"

        mol = gto.M(
            atom = atom,
            basis = '631g',
            unit = 'Bohr',
            symmetry=True,
            verbose=3,
        )

        nmode = 1 # create a zero (second) mode to test the code works for multiple modes
        cavity_freq = numpy.zeros(nmode)
        cavity_mode = numpy.zeros((nmode, 3))
        cavity_freq[0] = 3.0 / 27.211386245988
        cavity_mode[0, :] = 1.e-1 * numpy.asarray([1, 1, 1])

        mf = scf.RHF(mol)
        mf.kernel()

        qedmf = qedhf.RHF(mol, xc=None, cavity_mode=cavity_mode, cavity_freq=cavity_freq,
                         add_nuc_dipole=False)
        qedmf.max_cycle = 500
        qedmf.kernel() #dm0=dm)

        # cc
        qed = copy.copy(qedmf.qed)
        qed.verbose = 5
        qed.kernel()

        mycc = cc.CCSD(mf)
        mycc.diis_space = 15
        start_time = time.time()
        mycc.kernel()
        ccsd_time = time.time() - start_time
        print('\nCCSD correlation energy:     ', mycc.e_corr)

        myqedccsd = ccsd.CCSD(qed, nfock1=2, nfock2=2, add_U2n=True)
        myqedccsd.max_cycle = 100
        myqedccsd.verbose = 5

        start_time= time.time()
        e_tot, e_corr = myqedccsd.kernel()
        qed_energies.append(e_tot)
        print('\nQED-CCSD correlation energy: ', e_corr)

      for e_corr, qedccsd_ref in zip(qed_energies, qedccsd_refs):
          self.assertAlmostEqual(e_corr, qedccsd_ref, places=4,
              msg="Correlation energy does not match the reference value.")

if __name__ == '__main__':
    unittest.main()
