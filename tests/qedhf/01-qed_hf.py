import numpy
import unittest

from pyscf import gto
from openms.lib import boson
from openms.mqed import qedhf

class TestQEDHF(unittest.TestCase):

    def test_qedhf_cs(self):
        ref = -262.052870927714
        itest = 0
        zshift = itest * 2.0

        atom = f"C    0.00000000    0.00000000    {zshift};\
                 O    0.00000000    1.23456800    {zshift};\
                 H    0.97075033   -0.54577032    {zshift};\
                 C   -1.21509881   -0.80991169    {zshift};\
                 H   -1.15288176   -1.89931439    {zshift};\
                 C   -2.43440063   -0.19144555    {zshift};\
                 H   -3.37262777   -0.75937214    {zshift};\
                 O   -2.62194056    1.12501165    {zshift};\
                 H   -1.71446384    1.51627790    {zshift}"

        mol = gto.M(
            atom = atom,
            basis="sto3g",
            #basis="cc-pvdz",
            unit="Angstrom",
            symmetry=True,
            verbose=1)

        nmode = 2 # create a zero (second) mode to test the code works for multiple modes
        cavity_freq = numpy.zeros(nmode)
        cavity_freq[0] = 0.5
        cavity_mode = numpy.zeros((nmode, 3))
        cavity_mode[0, :] = 0.1 * numpy.asarray([0, 1, 0])

        qed = boson.Photon(mol, omega=cavity_freq, vec=cavity_mode)
        qedmf = qedhf.RHF(mol, qed=qed)

        qedmf.max_cycle = 500
        qedmf.kernel()

        self.assertAlmostEqual(qedmf.e_tot, ref, places=6, msg="Etot does not match the reference value.")

    def test_qedhf_fock(self):
        ref = -262.050746414733
        itest = 0
        zshift = itest * 2.0

        atom = f"C    0.00000000    0.00000000    {zshift};\
                 O    0.00000000    1.23456800    {zshift};\
                 H    0.97075033   -0.54577032    {zshift};\
                 C   -1.21509881   -0.80991169    {zshift};\
                 H   -1.15288176   -1.89931439    {zshift};\
                 C   -2.43440063   -0.19144555    {zshift};\
                 H   -3.37262777   -0.75937214    {zshift};\
                 O   -2.62194056    1.12501165    {zshift};\
                 H   -1.71446384    1.51627790    {zshift}"

        mol = gto.M(
            atom = atom,
            basis="sto3g",
            #basis="cc-pvdz",
            unit="Angstrom",
            symmetry=True,
            verbose=1)

        nmode = 2 # create a zero (second) mode to test the code works for multiple modes
        cavity_freq = numpy.zeros(nmode)
        cavity_freq[0] = 0.5
        cavity_mode = numpy.zeros((nmode, 3))
        cavity_mode[0, :] = 0.1 * numpy.asarray([0, 1, 0])

        qed = boson.Photon(mol, omega=cavity_freq, vec=cavity_mode, use_cs=False)
        qedmf = qedhf.RHF(mol, qed=qed)

        qedmf.max_cycle = 500
        qedmf.kernel()

        self.assertAlmostEqual(qedmf.e_tot, ref, places=6, msg="Etot does not match the reference value.")

if __name__ == '__main__':
    unittest.main()
