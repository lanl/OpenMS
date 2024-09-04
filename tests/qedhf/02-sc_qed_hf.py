import unittest
import numpy

from pyscf import gto
from openms.mqed import scqedhf

class TestSCQEDHF(unittest.TestCase):

    def test_energy_match(self):

        ref = -175.016224162572

        atom = """
               H          0.86681        0.60144        0.00000
               F         -0.86681        0.60144        0.00000
               O          0.00000       -0.07579        0.00000
               He         0.00000        0.00000        2.50000
               """

        mol = gto.M(atom = atom,
                    basis = "sto-3g",
                    unit = "Angstrom",
                    symmetry = True,
                    verbose = 1)

        nmode = 1
        cavity_freq = numpy.zeros(nmode)
        cavity_freq[0] = 0.5
        cavity_mode = numpy.zeros((nmode, 3))
        cavity_mode[0, :] = 0.1 * numpy.asarray([0, 1, 0])

        qedmf = scqedhf.RHF(mol, omega=cavity_freq, vec=cavity_mode)
        qedmf.init_guess = "hcore"
        qedmf.max_cycle = 500
        qedmf.kernel()

        self.assertAlmostEqual(qedmf.e_tot, ref, places=6, msg="Etot does not match the reference value.")

if __name__ == '__main__':
    unittest.main()
