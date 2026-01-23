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


def qmc_msd(time=5.0, nwalkers=100,
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
    qmc_msd()


if __name__ == "__main__":
    # unittest.main()
    test()
