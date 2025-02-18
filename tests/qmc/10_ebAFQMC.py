import unittest
import contextlib
import numpy
from pyscf import gto, scf
from openms.mqed import qedhf
from openms.qmc.afqmc import AFQMC
from openms.lib import boson
from utilities.analysis import get_mean_std
from molecules import get_mol, get_cavity

# Test eb-AFQMC using bare molecule object


def afqmc_energy_vs_lambda(
    mol,
    cavity_freq,
    cavity_mode,
    time=5.0,
    nwalkers=100,
    uhf=False,
    qed=False,
    gmat=None,
    decouple=True,
    verbose=1,
):
    dt = 0.005

    nmode = len(cavity_freq)
    # add num_fake field in order to have same number of random numbers in the comparison
    nfake = 0 if qed else nmode
    propagator_options = {
        "decouple_bilinear": decouple,
        "num_fake_fields": nfake,
    }

    afqmc = AFQMC(
        mol,
        dt=dt,
        total_time=time,
        num_walkers=nwalkers,
        energy_scheme="hybrid",
        uhf=uhf,
        boson_freq=cavity_freq,
        gmat=gmat,
        propagator_options=propagator_options,
        verbose=verbose,
    )
    times, energies = afqmc.kernel()
    return energies

class Test_ebAFQMC(unittest.TestCase):

    # 1) eb-AFQMC (bare molecule) vs eb-AFQMC (qed object)

    def test1(self):
        #
        bond = 2.0
        gfac = 0.1
        time = 5.0
        nwalkers = 500

        mol = get_mol(bond=bond)
        cavity_freq, cavity_mode = get_cavity(1, gfac, pol_axis=2)

        # set gmat
        dip_mat = boson.get_dipole_ao(mol)
        gmat_ao = numpy.einsum("nx, xuv->nuv", cavity_mode, dip_mat)

        energies = afqmc_energy_vs_lambda(
            mol,
            cavity_freq,
            cavity_mode,
            time=time,
            nwalkers=nwalkers,
            qed=True,
            gmat=gmat_ao,
            decouple=False,
        )
        means, stds = get_mean_std(energies)
        print("means = ", means)

        # TODO: comapre with AFQMC using QED object



if __name__ == "__main__":
    unittest.main()
