import unittest
import contextlib
import numpy
from pyscf import gto, scf
from openms.mqed import qedhf
from openms.qmc.afqmc import AFQMC
from utilities.analysis import get_mean_std
from molecules import get_mol, get_cavity

def run_fci(mol, cavity_freq, cavity_mode, nphoton=2):
    from openms.mqed import qedfci as fci
    from openms.lib.boson import get_integrals4fci

    r"""get FCI reference energies"""

    nao = mol.nao_nr()
    nelec = mol.nelectron
    nmode = len(cavity_freq)

    # get integrals
    enuc, ovlp, H1, H2, Hep, rdm1 = get_integrals4fci(mol, cavity_freq, cavity_mode)

    # run fci
    # same number of fock states for each mode
    max_cycle = 100
    verbose = 1
    nboson_states = [nphoton for i in range(nmode)]

    e_fci0, e_fci, civec = fci.kernel(
        H1,
        H2,
        nao,
        nelec,
        nmode,
        nboson_states,
        Hep,
        cavity_freq,
        coherent_state=False,
        ecore=enuc,
        tol=1e-8,
        verbose=verbose,
        max_cycle=max_cycle,
    )

    return e_fci


def afqmc_energy_vs_lambda(
    mol,
    cavity_freq,
    cavity_mode,
    time=5.0,
    nwalkers=100,
    uhf=False,
    qed=False,
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

    if qed:
        qedmf = qedhf.RHF(
            mol,
            xc=None,
            cavity_mode=cavity_mode,
            cavity_freq=cavity_freq,
            nboson_states=3,
        )
        qedmf.max_cycle = 500
        qedmf.kernel()  # dm0=dm)
        print("qed hf energy is", qedmf.e_tot)
        qed = qedmf.qed

        afqmc = AFQMC(
            qed,
            mf=qedmf,
            dt=dt,
            total_time=time,
            num_walkers=nwalkers,
            energy_scheme="hybrid",
            uhf=uhf,
            propagator_options=propagator_options,
            verbose=verbose,
        )

    else:
        afqmc = AFQMC(
            mol,
            dt=dt,
            total_time=time,
            num_walkers=nwalkers,
            energy_scheme="hybrid",
            uhf=uhf,
            propagator_options=propagator_options,
            verbose=verbose,
        )
    times, energies = afqmc.kernel()
    return energies


class TestQMCH2(unittest.TestCase):

    def test_qedafqmc_vs_bare_afqmc(self):

        bond = 1.0
        time = 5.0
        nwalkers = 100
        gfac = 1.e-7
        uhf = True
        uhf = False

        # FCI
        mol = get_mol(bond=bond)
        cavity_freq, cavity_mode = get_cavity(1, gfac)
        fcienergy = run_fci(mol, cavity_freq, cavity_mode, nphoton=2)

        # Test configurations
        # 1) Bare QMC, 2) QMC with direct propagation of bilinear, and 3) decoupled propagation of bilinear
        ntest = 3
        means = numpy.zeros(ntest)
        stds = numpy.zeros(ntest)
        qeds = [False, True, True]
        decouples = [False, False, True]

        # Perform AFQMC calculations
        for itest in range(ntest):
            # with open(f"tmp{itest*10}.log", "w") as fout, contextlib.redirect_stdout(fout):
            mol = get_mol(bond=bond)
            cavity_freq, cavity_mode = get_cavity(1, gfac)
            energies = afqmc_energy_vs_lambda(
                mol,
                cavity_freq,
                cavity_mode,
                time=time,
                uhf=uhf,
                nwalkers=nwalkers,
                qed=qeds[itest],
                decouple=decouples[itest],
            )
            means[itest], stds[itest] = get_mean_std(energies)

        # the random number is not the same (for the decoupling of bilinear term)
        # every time step needs to generate Nw more random number, leading
        # to different results

        # Results comparison
        print(f"\n{'*' * 30} Compare energies {'*' * 30}")
        results = {"FCI Energy:": [fcienergy], "Mean Energies": means, "Standard Deviations": stds}
        for key, values in results.items():
            formatted_row = "   ".join(f"{value:14.7e}" for value in values)
            print(f"{key:20s}: {formatted_row}")

        # Assertions
        self.assertAlmostEqual(
            means[0], means[1], places=2, msg="Mean energy does not match between tests 1 and 2."
        )
        self.assertAlmostEqual(
            means[0], means[2], places=2, msg="Mean energy does not match between tests 1 and 3."
        )
        self.assertAlmostEqual(
            stds[0], stds[1], places=3, msg="Standard deviation does not match between tests 1 and 2."
        )


    def test_qedafqmc_vs_gfac(self):
        # test AFQMC energy vs gfac
        # propagate bilinear directly
        bond = 2.0
        time = 5.0
        nw = 100

        gfacs = numpy.arange(0.0, 0.101, 0.05)
        Ng = len(gfacs)
        diff_refs = []

        fcienergies = numpy.zeros(Ng)
        means = numpy.zeros((Ng, 2))
        stds = numpy.zeros((Ng, 2))

        # 1) propagate the bilinear term directly, 2) decoupled propagation
        decouples = [False, True]
        for i in range(Ng):
            for j in range(2):
                # FCI
                mol = get_mol(bond=bond)
                cavity_freq, cavity_mode = get_cavity(1, gfacs[i])
                fcienergies[i] = run_fci(mol, cavity_freq, cavity_mode, nphoton=2)

                # QMC
                mol = get_mol(bond=bond)
                cavity_freq, cavity_mode = get_cavity(1, gfacs[i])
                energies = afqmc_energy_vs_lambda(
                    mol,
                    cavity_freq,
                    cavity_mode,
                    time=time,
                    nwalkers=nw,
                    qed=True,
                    decouple=decouples[j],
                )
                means[i, j], stds[i, j] = get_mean_std(energies)

        diff_energies = numpy.zeros((Ng, 2))
        for i in range(Ng):
            diff_energies[i, :] = means[i, :] - means[0, :]

        print(f"\n{'*' * 30} Energy vs gfac (direct) {'*' * 30}")
        for i in range(Ng):
            print(
                "%9.4f   %15.10f   %15.10f   %15.10f   %10.6f   %10.6f"
                % (
                    gfacs[i], fcienergies[i], means[i, 0],
                    diff_energies[i, 0], stds[i, 0], fcienergies[i],
                )
            )

        print(f"\n{'*' * 30} Energy vs gfac (decoupled) {'*' * 30}")
        for i in range(Ng):
            print(
                "%9.4f   %15.10f   %15.10f   %15.10f   %10.6f   %10.6f"
                % (
                    gfacs[i], fcienergies[i], means[i, 1],
                    diff_energies[i, 1], stds[i, 1], fcienergies[i],
                )
            )


if __name__ == "__main__":
    unittest.main()
