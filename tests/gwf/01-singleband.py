import unittest
import numpy
from pyscf import gto, scf, ao2mo
from openms.gwf import ga_sband as ga


def get_model(n=12, filling=0.5, U=2.0, t=-1.0, PBC=True):

    mol = gto.M(verbose=3)
    mol.nelectron = int(n * filling)

    # over write mol functions
    mol.incore_anyway = True
    mol.nao_nr = lambda *args: n
    mol.tot_electrons = lambda *args: mol.nelectron

    mol.spin = 0
    mol.incore_anyway = True
    mol.build()

    # hopping
    h1 = numpy.zeros((n, n))
    for i in range(n - 1):
        h1[i, i + 1] = h1[i + 1, i] = t

    # add the hopping between site 0 and N-1 if PBC
    if PBC:
        h1[n - 1, 0] = h1[0, n - 1] = t

    # onsite U term
    eri = numpy.zeros((n, n, n, n))
    for i in range(n):
        eri[i, i, i, i] = U

    return mol, h1, eri


class TestGWF(unittest.TestCase):

    def test_single_band(self):
        mol, h1e, eri = get_model(n=12, U=1.0, PBC=False)
        nao = mol.nao_nr()

        gamf = ga.GASCF(mol)
        gamf.get_bare_hcore = lambda *args: h1e
        gamf._eri = ao2mo.restore(1, eri, nao)
        gamf.get_ovlp = lambda *args: numpy.eye(nao)

        gamf.verbose = 4
        gamf.max_cycle = 500
        gamf.kernel()

    def test_energy_vs_U(self):
        pass


if __name__ == '__main__':
    unittest.main()
