import unittest
import numpy
from pyscf import gto
from openms.mqed import scqedhf, vtqedhf

def get_mol():
    atom = f"C   0.00000000   0.00000000    0.0000;\
             O   0.00000000   1.23456800    0.0000;\
             H   0.97075033  -0.54577032    0.0000;\
             C  -1.21509881  -0.80991169    0.0000;\
             H  -1.15288176  -1.89931439    0.0000;\
             C  -2.43440063  -0.19144555    0.0000;\
             H  -3.37262777  -0.75937214    0.0000;\
             O  -2.62194056   1.12501165    0.0000;\
             H  -1.71446384   1.51627790    0.0000"

    return gto.M(atom=atom, basis="sto3g", unit="Angstrom", symmetry=True, verbose=1)


methods_map = {
 "vtqedhf": vtqedhf,
 "scqedhf": scqedhf,
}

def run_qedhf(method="vtqedhf", falpha=None):

    mol = get_mol()

    # scqehd
    nmode = 1
    cavity_freq = numpy.zeros(nmode)
    cavity_freq[0] = 0.5
    cavity_mode = numpy.zeros((nmode, 3))
    cavity_mode[0, :] = 5.e-2 * numpy.asarray([0, 1, 0])

    if falpha is None:
        qedmf = methods_map[method].RHF(mol, xc=None, cavity_mode=cavity_mode,
                cavity_freq=cavity_freq,
                add_nuc_dipole=True)
    else:
        qedmf = methods_map[method].RHF(mol, xc=None, cavity_mode=cavity_mode,
                cavity_freq=cavity_freq,
                add_nuc_dipole=True,
                couplings_var=[falpha])

    qedmf.max_cycle = 500
    qedmf.init_guess = "hcore"
    qedmf.kernel()
    return qedmf.e_tot

class TestVTQEDHF_f(unittest.TestCase):

    def test_energy_match(self):

        refs = [-262.1271784771804, -262.142884529976]
        etots = []

        for j, falpha in enumerate([0.0, 1.0]):
            e_tot = run_qedhf(method="vtqedhf", falpha=falpha)
            etots.append(e_tot)
            self.assertAlmostEqual(e_tot, refs[j], places=6, msg="Etot does not match the reference value.")

        # f_alpha = 1, should also equation to sc_eng
        sc_eng = run_qedhf(method="scqedhf")
        self.assertAlmostEqual(sc_eng, etots[-1], places=6, msg="Etot does not match the reference value.")

    def test_vtqed_min(self):

        ref = -262.1453751012338
        e_tot = run_qedhf(method="vtqedhf")

        self.assertAlmostEqual(e_tot, ref, places=6, msg="Etot does not match the reference value.")

if __name__ == '__main__':
    unittest.main()
