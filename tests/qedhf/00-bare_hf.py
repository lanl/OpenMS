import unittest
import numpy

from pyscf import gto, scf

class TestBareHF(unittest.TestCase):

    def test_bare_hf_energy(self):

        e_ref = -262.152117320877

        itest = 0.0
        zshift = itest * 2.0

        atom = f"""
                C    0.00000000    0.00000000    {zshift}
                O    0.00000000    1.23456800    {zshift}
                H    0.97075033   -0.54577032    {zshift}
                C   -1.21509881   -0.80991169    {zshift}
                H   -1.15288176   -1.89931439    {zshift}
                C   -2.43440063   -0.19144555    {zshift}
                H   -3.37262777   -0.75937214    {zshift}
                O   -2.62194056    1.12501165    {zshift}
                H   -1.71446384    1.51627790    {zshift}
                """

        mol = gto.M(atom = atom,
                    basis = "sto3g",
                    #basis = "cc-pvdz",
                    unit = "Angstrom",
                    symmetry = True,
                    verbose = 1)

        mf = scf.RHF(mol)
        mf.max_cycle = 500
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, e_ref, places=6, msg="E_tot does not match the reference value.")

if __name__ == '__main__':
    unittest.main()
