import numpy
import unittest
from pyscf import gto, scf, fci
from openms.qmc.afqmc import AFQMC
from molecules import get_mol
from openms.qmc.tools import analysis_autocorr


def calc_qmc_energy(
    mol,
    time=5.0,
    num_walkers=200,
    uhf=False,
    energy_scheme="hybrid",
    block_decompose_eri=False,
):

    r"""Note the number of walkers here is small, in order to do fast test"""
    propagator_options = {
        # "enable_bp": False,
        "enable_bp": True,
        "bp_length": 3,
    }

    afqmc = AFQMC(
        mol,
        dt=0.005,
        total_time=time,
        num_walkers=num_walkers,
        energy_scheme=energy_scheme,
        uhf=uhf,
        chol_thresh=1.0e-20,
        property_calc_freq=10,
        propagator_options=propagator_options,
        verbose=mol.verbose,
    )

    times, energies = afqmc.kernel()
    return energies

class TestQMCH2(unittest.TestCase):

    def test_qmc_h2(self):
        mean_ref = -1.13981
        std_dev_ref = 0.003
        local_mean_ref = -1.13998
        local_std_dev_ref = 0.003

        bond = 1.6 * 0.5291772
        basis = "sto6g"
        verbose = 3

        mol = get_mol(2, bond, basis=basis, verbose=verbose, name="Hchain")
        qmc_energies = calc_qmc_energy(mol, time=5.0, uhf=False)

        results = analysis_autocorr(qmc_energies[len(qmc_energies)//4:])

        mean, std = results["etot"][0], results["etot_error"][0]
        print(f"mean /std = {mean:12.6f}  {std:12.6f}")

        # exit()
        # mean, std_dev = get_mean_std(qmc_energies)
        # self.assertLess(
        #     abs(mean - mean_ref),
        #     1.0e-3,
        #     msg="E_mean does not match the reference value.",
        # )


if __name__ == "__main__":
    unittest.main()
