import unittest
import h5py
import numpy
from pyscf.fci import cistring
from pyscf import fci, gto, mcscf, scf
from openms.qmc.afqmc import AFQMC


def get_hydrogen_chain(natoms, bond, basis="sto3g", verbose=1):

    atoms = [("H", i * bond, 0, 0) for i in range(natoms)]
    mol = gto.M(atom=atoms, basis=basis, unit="Bohr", verbose=verbose)

    return mol

def get_mol(basis="ccpvdz", spin=2, verbose=1):

    mol = gto.M(
        atom=[("N", 0, 0, 0), ("N", (0, 0, 3.0))],
        basis=basis,
        verbose=verbose,
        spin=spin,
        unit="Bohr",
    )
    print(f"Number of AO is {mol.nao_nr()}")
    return mol



class TestQMC_MSD(unittest.TestCase):

    def test_multisd_trial(self):
        r"""Test energy vs number of determinents"""

        numdets = range(1, 101, 10)
        for numdet in numdets:
            print(numdet)

        pass


def get_cas_mo(mol, ncas, neleca, nelecb):

    neleccas = neleca + nelecb

    # mf
    mf = scf.RHF(mol)
    mf.chkfile = "scf.chk"
    ehf = mf.kernel()

    # casscf
    mc = mcscf.CASSCF(mf, ncas, neleccas)
    mc.chkfile = "scf.chk"
    e_tot, e_cas, fcivec, mo, mo_energy = mc.kernel()

    # get largest ci coefficients
    coeff, occa, occb = zip(
        *fci.addons.large_ci(
            fcivec, ncas, (neleca, nelecb), tol=1e-8, return_strs=False
        )
    )

    # na = cistring.num_strings(ncas, neleca)
    # nb = cistring.num_strings(ncas, nelecb)
    # print("na = ", na)
    print("fci shape, size", fcivec.shape, fcivec.size)
    print("len(coeff) ", len(coeff))
    print("occa: ", len(occa))
    print("occa: ", len(occb))

    return coeff, occa, occb


def qmc_msd_ref(time=5.0, nwalkers=100):
    from ipie.utils.from_pyscf import gen_ipie_input_from_pyscf_chk
    from ipie.hamiltonians.generic import Generic as HamGeneric
    from ipie.trial_wavefunction.particle_hole import ParticleHole
    from ipie.walkers.uhf_walkers import UHFWalkersParticleHole
    from ipie.utils.mpi import MPIHandler
    from ipie.qmc.afqmc import AFQMC as AFQMC_ref

    ncas = 6
    neleca = 4
    nelecb = 2
    neleccas = neleca + nelecb

    mol = get_mol(spin=neleca - nelecb, verbose=3)
    coeff, occa, occb = get_cas_mo(mol, ncas, neleca, nelecb)

    print("molecular electroncs =", mol.nelec)
    # write wavefunction to checkpoint file.
    with h5py.File("scf.chk", "r+") as fh5:
        fh5["mcscf/ci_coeffs"] = coeff
        fh5["mcscf/occs_alpha"] = occa
        fh5["mcscf/occs_beta"] = occb

    # prepare input
    gen_ipie_input_from_pyscf_chk("scf.chk", mcscf=True)

    # build Hamiltonian
    with h5py.File("hamiltonian.h5") as fa:
        chol = fa["LXmn"][()]
        h1e = fa["hcore"][()]
        e0 = fa["e0"][()]

    #
    # construct Hamiltonian
    #
    num_chol = chol.shape[0]
    num_basis = chol.shape[1]
    print("chol.shape = ", chol.shape)
    print("number of basis = ", num_basis)

    ham = HamGeneric(
        numpy.array([h1e, h1e]),
        chol.transpose((1, 2, 0)).reshape((num_basis * num_basis, num_chol)),
        e0,
    )

    #
    # construct wavefuntion
    #

    wavefunction = (coeff, occa, occb)
    trial = ParticleHole(
        wavefunction,
        mol.nelec,
        num_basis,
        num_dets_for_props=len(wavefunction[0]),
        verbose=True,
    )

    # --- build trial WF ---
    trial.compute_trial_energy = True
    trial.build()
    trial.half_rotate(ham)

    #
    # build walker
    #

    initial_walker = numpy.hstack([trial.psi0a, trial.psi0b])
    # random_perturbation = numpy.random.random(initial_walker.shape)
    # initial_walker = initial_walker + random_perturbation
    # initial_walker, _ = numpy.linalg.qr(initial_walker)
    walkers = UHFWalkersParticleHole(
        initial_walker,
        mol.nelec[0],
        mol.nelec[1],
        num_basis,
        nwalkers,
        MPIHandler(),
    )

    print("\n Build walkers for the first time")
    walkers.build(trial)


    # --
    # construct afqmc object and run
    # --
    print("\n Construct afqmc object and run")
    afqmc_msd = AFQMC_ref.build(
        mol.nelec,
        ham,
        trial,
        walkers=walkers,
        num_walkers=nwalkers,
        num_steps_per_block=25,
        num_blocks=10,
        timestep=0.005,
        stabilize_freq=5,
        seed=96264512,
        pop_control_freq=5,
        verbose=True,
    )

    # afqmc_msd.run()
    # afqmc_msd.finalise(verbose=True)


def my_qmc_msd(time=5.0, nwalkers=100,
    energy_scheme="hybrid",
    block_decompose_eri=True,
    ):
    from openms.qmc.trial import multiCI
    ncas = 6
    neleca = 4
    nelecb = 2
    neleccas = neleca + nelecb
    verbose = 4

    mol = get_mol(spin=neleca - nelecb, verbose=verbose)
    coeff, occa, occb = get_cas_mo(mol, ncas, neleca, nelecb)

    trial = multiCI(mol, cas=(ncas, neleccas))
    trial.build()
    trial.dump_flags()

    walker_options = {"nwalkers": nwalkers,}

    # create afqmc object
    afqmc = AFQMC(
        mol,
        dt=0.005,
        total_time=time,
        uhf=True,
        trial=trial,
        walker_options = walker_options,
        energy_scheme=energy_scheme,
        property_calc_freq=5,
        block_decompose_eri=block_decompose_eri,
        chol_thresh=1.0e-5,
        verbose=verbose,
    )

    # afqmc.kernel()



def test():
    # Not done yet!
    import contextlib
    #with open("tmp10.log", 'w') as f_out, contextlib.redirect_stdout(f_out):
    #    qmc_msd_ref()
    #with open("tmp20.log", 'w') as f_out, contextlib.redirect_stdout(f_out):
    my_qmc_msd()


if __name__ == "__main__":
    # unittest.main()
    test()
