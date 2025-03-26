import numpy
import unittest
from pyscf import gto, scf, fci
from openms.qmc.afqmc import AFQMC
from molecules import get_mol


def calc_qmc_energy(
    mol,
    time=6.0,
    num_walkers=100,
    uhf=False,
    energy_scheme="hybrid",
    block_decompose_eri=False,
):
    r"""Note the number of walkers here is small, in order to do fast test"""

    afqmc = AFQMC(
        mol,
        dt=0.005,
        total_time=time,
        num_walkers=num_walkers,
        energy_scheme=energy_scheme,
        uhf=uhf,
        chol_thresh=1.0e-10,
        property_calc_freq=1,
        verbose=mol.verbose,
    )

    times, energies = afqmc.kernel()
    return energies


def get_mean_std(energies, ratio=10):
    # Compute the mean and standard deviation
    # Extract the real parts of the last m elements
    m = max(1, len(energies) // ratio)
    last_m_real = numpy.asarray(energies[-m:]).real
    mean = numpy.mean(last_m_real)
    std_dev = numpy.std(last_m_real)

    return mean, std_dev


def run_fci(mol):
    mf = mol.RHF().run()
    #
    # create an FCI solver based on the SCF object
    #
    umf = mol.UHF().run()
    cisolver = fci.FCI(umf)
    e_uhf_fci = cisolver.kernel()[0]
    print("E(UHF-FCI) = %.12f" % e_uhf_fci)
    return e_uhf_fci


class TestQMCH2(unittest.TestCase):

    def test_qmc_h2(self):
        mean_ref = -2.2027
        std_dev_ref = 0.01
        local_mean_ref = -2.2037
        local_std_dev_ref = 0.01

        bond = 1.6 * 0.5291772
        basis = "sto6g"
        verbose = 1
        # # fci
        mol = get_mol(4, bond, basis=basis, verbose=verbose, name="Hchain")
        fci_energy = run_fci(mol)

        # qmc_uhf
        mol = get_mol(4, bond, basis=basis, verbose=verbose, name="Hchain")
        qmc_energies = calc_qmc_energy(mol, uhf=True)
        mean, std_dev = get_mean_std(qmc_energies)

        mol = get_mol(4, bond, basis=basis, verbose=verbose, name="Hchain")
        qmc_energies2 = calc_qmc_energy(mol, uhf=True, energy_scheme="local")
        mean2, std_dev2 = get_mean_std(qmc_energies2)

        print(f"fci_energies :  {fci_energy}")
        print(f"energy and std are (hybrid):  {mean} {std_dev}")
        print(f"energy and std are (local):   {mean2} {std_dev2}")

        self.assertLess(
            abs(mean - mean_ref),
            1.0e-3,
            msg="E_mean does not match the reference value.",
        )
        self.assertLess(
            abs(mean2 - local_mean_ref),
            1.0e-3,
            msg="E_mean does not match the reference value.",
        )
        self.assertLess(
            std_dev,
            std_dev_ref,
            msg=f"Standard deviation {std_dev:0.6f} is not smaller than the reference value {std_dev_ref:0.6f}.",
        )
        self.assertLess(
            std_dev2,
            local_std_dev_ref,
            msg=f"Standard deviation {std_dev2:0.6f} is not smaller than the reference value {local_std_dev_ref:0.6f}.",
        )


if __name__ == "__main__":
    unittest.main()
