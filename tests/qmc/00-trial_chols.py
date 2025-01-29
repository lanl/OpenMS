
import unittest
from openms.qmc.afqmc import AFQMC
from pyscf import gto, mcscf, scf, fci, lo, ao2mo
import numpy


class Test(unittest.TestCase):

    def test_chols(self):

        bond = 1.6
        natoms = 4
        atoms = [("H", i * bond, 0, 0) for i in range(natoms)]
        mol = gto.M(atom=atoms, basis='sto-6g', unit='Bohr', verbose=3)

        energy_scheme = "hybrid"
        num_walkers = 500
        time = 5.0
        uhf = False

        afqmc = AFQMC(mol, dt=0.005, total_time=time, num_walkers=num_walkers,
                      energy_scheme=energy_scheme,
                      uhf=uhf,
                      chol_tresh=1.e-9,
                      verbose=3)

        overlap = mol.intor("int1e_ovlp")
        Xmat = lo.orth.lowdin(overlap)
        eri = mol.intor('int2e_sph', aosym='s1')
        eri = ao2mo.full(eri, Xmat)

        eri2 = numpy.einsum('nij, nkl->ijkl', afqmc.ltensor.conj(), afqmc.ltensor)
        # close = numpy.allclose(afqmc.eri, eri2, rtol=1e-05, atol=1e-08)

        numpy.testing.assert_almost_equal(
            eri,
	    eri2,
            decimal=5,
            err_msg="Etot does not match the reference value.",
        )

        L1 = numpy.trace(eri, axis1=1, axis2=2)
        L2 = numpy.einsum('nik,nkj->ij', afqmc.ltensor.conj(), afqmc.ltensor)
        numpy.testing.assert_almost_equal(
            L1,
	    L2,
            decimal=5,
            err_msg="Etot does not match the reference value.",
        )

    def test_trial(self):
        from openms.qmc.trial import TrialHF

        bond = 1.6
        natoms = 2
        atoms = [("H", i * bond, 0, 0) for i in range(natoms)]
        mol = gto.M(atom=atoms, basis="sto-6g", unit="Bohr", verbose=3)
        trial = TrialHF(mol)
        trial.build()

        from openms.qmc.trial import multiCI
        cas = (2, 2)
        trial = multiCI(mol, cas=cas)


if __name__ == "__main__":
    unittest.main()
