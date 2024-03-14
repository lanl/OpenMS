import unittest
import numpy
from pyscf import gto, scf, cc
from openms.mqed import qedhf, ccsd
import copy

class TestQEDCC(unittest.TestCase):
    def test_qedcc_ref(self):
        qedccsd_refs = numpy.asarray(\
                        [-0.330200422188726, -0.3302005593983139, \
                        -0.330203599835614, -0.3302037352141863, \
                        -0.330334640899236, -0.3303348700344901, \
                        -0.330338794656879, -0.3303389721399432])

        atom = f"C   0.00000000   0.00000000  0.00;\
                 O   0.00000000   1.23456800  0.00;\
                 H   0.97075033  -0.54577032  0.00;\
                 C  -1.21509881  -0.80991169  0.00;\
                 H  -1.15288176  -1.89931439  0.00;\
                 C  -2.43440063  -0.19144555  0.00;\
                 H  -3.37262777  -0.75937214  0.00;\
                 O  -2.62194056   1.12501165  0.00;\
                 H  -1.71446384   1.51627790  0.00"

        mol = gto.M(
            atom = atom,
            basis="sto3g",
            #basis="cc-pvdz",
            unit="Angstrom",
            symmetry=True,
            verbose=3,
        )

        nmode = 1
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

        print(f"\nHF energy is    {mf.e_tot:.8f}")
        print(f"QEDHF energy is {qedmf.e_tot:.8f}\n")

        # cc
        qed = copy.copy(qedmf.qed)
        qed.verbose = 5
        qed.kernel()
        print('\nadd_nuc_dipole=', qed.add_nuc_dipole)

        # Note this energy is different from the test 01-qedhf as we don't include nuclear_dipole
        # in the coupling
        """
        self.assertAlmostEqual(qedmf.e_tot, hf_ref, places=6,
        msg="Etot does not match the reference value.")
        """

        mycc = cc.CCSD(mf)
        mycc.diis_space = 10
        mycc.kernel()
        print('\nCCSD correlation energy:     ', mycc.e_corr)

        e_corrs = []
        for add_U2n in [False, True]:
           for nfock1 in range(1, 3):
              for nfock2 in range(1,3):
                 qed = copy.copy(qedmf.qed)
                 qed.verbgose = 5
                 prefix = "QEDCCSD_U2" if add_U2n else "QEDCCSD_U1"
                 prefix += f"{nfock2}_S{nfock1}"
                 print(f"\n ------------Calculating {prefix} energies -------\n")

                 myqedccsd = ccsd.CCSD(qed, nfock1=nfock1, nfock2=nfock2, add_U2n=add_U2n)
                 myqedccsd.max_cycle = 200
                 myqedccsd.verbose = 3
                 e_tot, e_corr = myqedccsd.kernel()
                 e_corrs.append(e_corr)

                 print(f"{prefix} correlation energy = ", myqedccsd.e_corr)
                 print('QED-CCSD electron-photon correlation energy: ', e_corr - mycc.e_corr)
        e_corrs = numpy.asarray(e_corrs)

        for e_corr, qedccsd_ref in zip(e_corrs, qedccsd_refs):
            self.assertAlmostEqual(e_corr, qedccsd_ref, places=5,
                msg="Correlation energy does not match the reference value.")


    def test_qedcc_isomer(self):
        pass

if __name__ == '__main__':
    unittest.main()
