import unittest
import numpy
from pyscf import gto, scf
from openms.mqed import scqedhf as qedhf

class TestSCQEDHF(unittest.TestCase):
    def test_energy_match(self):
        ref = -262.142884529976

        atom = f"C   0.00000000   0.00000000    0.0000;\
                 O   0.00000000   1.23456800    0.0000;\
                 H   0.97075033  -0.54577032    0.0000;\
                 C  -1.21509881  -0.80991169    0.0000;\
                 H  -1.15288176  -1.89931439    0.0000;\
                 C  -2.43440063  -0.19144555    0.0000;\
                 H  -3.37262777  -0.75937214    0.0000;\
                 O  -2.62194056   1.12501165    0.0000;\
                 H  -1.71446384   1.51627790    0.0000"

        mol = gto.M(atom=atom, basis="sto3g", unit="Angstrom", symmetry=True, verbose=1)

        nmode = 1
        cavity_freq = numpy.zeros(nmode)
        cavity_mode = numpy.zeros((nmode, 3))
        cavity_freq[0] = 0.5
        cavity_mode[0, :] = 5.e-2 * numpy.asarray([0, 1, 0])

        qedmf = qedhf.RHF(mol, xc=None, cavity_mode=cavity_mode, cavity_freq=cavity_freq, add_nuc_dipole=True)
        qedmf.max_cycle = 500
        qedmf.init_guess = "hcore"
        qedmf.kernel()

        self.assertAlmostEqual(qedmf.e_tot, ref, places=6, msg="Etot does not match the reference value.")

if __name__ == '__main__':
    unittest.main()
