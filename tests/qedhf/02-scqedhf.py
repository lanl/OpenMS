import unittest
import numpy
from pyscf import gto, scf
from openms.mqed import scqedhf as qedhf

class TestSCQEDHF(unittest.TestCase):
    def test_energy_match(self):
        ref = -175.016224162572

        itest = -2
        zshift = itest * 2.5

        atom = f"H          0.86681        0.60144        {5.00000+zshift};\
                 F         -0.86681        0.60144        {5.00000+zshift};\
                 O          0.00000       -0.07579        {5.00000+zshift};\
                 He         0.00000        0.00000        {7.50000+zshift}"

        mol = gto.M(atom=atom, basis="sto3g", unit="Angstrom", symmetry=True, verbose=3)

        nmode = 1
        cavity_freq = numpy.zeros(nmode)
        cavity_mode = numpy.zeros((nmode, 3))
        cavity_freq[0] = 0.5
        cavity_mode[0, :] = 1.e-1 * numpy.asarray([0, 1, 0])
        mol.verbose = 1

        qedmf = qedhf.RHF(mol, xc=None, cavity_mode=cavity_mode, cavity_freq=cavity_freq, add_nuc_dipole=True)
        qedmf.max_cycle = 500
        #qedmf.verbose = 1
        qedmf.init_guess = "hcore"
        qedmf.kernel() #conv_tol=1.e-8)

        self.assertAlmostEqual(qedmf.e_tot, ref, places=6, msg="Etot does not match the reference value.")

if __name__ == '__main__':
    unittest.main()

