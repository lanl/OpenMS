from openms.qmc.afqmc import AFQMC
from pyscf import gto, mcscf, scf, fci
import numpy
import h5py
import unittest

def calc_qmc_energy(atoms, time=6.0, num_walkers = 500, uhf=False, energy_scheme="hybrid"):

    mol = gto.M(atom=atoms, basis='sto-6g', unit='Bohr', verbose=3)

    afqmc = AFQMC(mol, dt=0.005, total_time=time, num_walkers=num_walkers,
                  energy_scheme=energy_scheme,
                  uhf = uhf,
                  verbose=3)

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


class TestQMCH2(unittest.TestCase):

    def test_qmc_h2(self):
        mean_ref = -1.1369791727
        std_dev_ref = 0.0013134
        local_mean_ref = -1.1370359643
        local_std_dev_ref = 0.0014197667

        bond = 1.6
        natoms = 2
        atoms = [("H", i * bond, 0, 0) for i in range(natoms)]

        qmc_energies = calc_qmc_energy(atoms, uhf=True)
        mean, std_dev = get_mean_std(qmc_energies)

        qmc_energies2 = calc_qmc_energy(atoms, uhf=True, energy_scheme='local')
        mean2, std_dev2 = get_mean_std(qmc_energies2)

        print(f"energy and std are (hybrid):  {mean} {std_dev}")
        print(f"energy and std are (local):   {mean2} {std_dev2}")
        # assert numpy.isclose(mean, mean_ref)
        # assert numpy.isclose(std_dev, std_dev_ref)

        self.assertAlmostEqual(mean, mean_ref, places=6, msg="E_mean does not match the reference value.")
        self.assertAlmostEqual(std_dev, std_dev_ref, places=6, msg="Standard deivation does not match the reference value.")
        self.assertAlmostEqual(mean2, local_mean_ref, places=6, msg="E_mean does not match the reference value.")
        self.assertAlmostEqual(std_dev2, local_std_dev_ref, places=6, msg="Standard deivation does not match the reference value.")

if __name__ == "__main__":
    unittest.main()
