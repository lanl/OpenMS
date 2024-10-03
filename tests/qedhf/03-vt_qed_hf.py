import unittest
import numpy
from pyscf import gto
from openms.mqed import scqedhf, vtqedhf

def get_mol():
    atom = """
           H          0.86681        0.60144        0.00000
           F         -0.86681        0.60144        0.00000
           O          0.00000       -0.07579        0.00000
           He         0.00000        0.00000        2.50000
           """
    mol = gto.M(atom=atom, basis="sto3g", unit="Angstrom", symmetry=True, verbose=1)
    return mol

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
    cavity_mode[0, :] = 1.e-1 * numpy.asarray([0, 1, 0])

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

        refs = [-174.9935188527, -175.016224162572]
        etots = []

        for j, falpha in enumerate([0.0, 1.0]):
            e_tot = run_qedhf(method="vtqedhf", falpha=falpha)
            etots.append(e_tot)
            self.assertAlmostEqual(e_tot, refs[j], places=6, msg="Etot does not match the reference value.")

        # f_alpha = 1, should also equation to sc_eng
        sc_eng = run_qedhf(method="scqedhf")
        self.assertAlmostEqual(sc_eng, etots[-1], places=6, msg="Etot does not match the reference value.")

    def test_vtqed_min(self):

        ref = -175.0168599150538
        e_tot = run_qedhf(method="vtqedhf")

        self.assertAlmostEqual(e_tot, ref, places=6, msg="Etot does not match the reference value.")

if __name__ == '__main__':
    unittest.main()
