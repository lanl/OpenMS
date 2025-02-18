import unittest
import contextlib
import numpy
from pyscf import gto, scf
from openms.qmc.afqmc import AFQMC
from openms.lib.boson import Photon
from openms.mqed import qedhf, scqedhf, vtqedhf

from utilities.analysis import get_mean_std


class simple_model(object):
    def __init__(self, **kwargs):
        self.norb = kwargs.get("norb", 2)
        self.nelec = kwargs.get("nelec", 2)
        self.nmode = kwargs.get("nmode", 1)
        self.nfock = kwargs.get("nfock", 2)
        self.t = kwargs.get("t", -1.0)
        self.U = kwargs.get("U", 1.0)
        self.gfac = kwargs.get("gfac", numpy.ones(self.nmode))
        self.vec = kwargs.get("vec", numpy.asarray([[1.0, 0.0, 0.0]] * self.nmode))
        self.omega = kwargs.get("omega", numpy.zeros(self.nmode))

        # set up eri
        self.eri = numpy.zeros((self.norb, self.norb, self.norb, self.norb))
        self.hop = numpy.zeros((self.norb, self.norb))

        idx = numpy.arange(self.norb - 1)
        self.hop[idx, idx + 1] = self.hop[idx + 1, idx] = -1

        idx = range(self.norb)
        self.eri[idx, idx, idx, idx] = self.U

        # dipole matrix and quadrupole
        self.dipole_mat = numpy.zeros((3, self.norb, self.norb))
        self.q_mat = numpy.zeros((9, self.norb, self.norb))
        for i in range(self.norb):
            self.dipole_mat[0, i, i] = 1.0

        print(self.gfac.shape)
        print(self.vec.shape)
        print(self.dipole_mat.shape)

        self.gmat = numpy.einsum("n, nx, xpq-> npq", self.gfac, self.vec, self.dipole_mat)


def get_system():
    nmode = 1
    norb = 4
    nelec = 4
    nfock = 12

    omega = numpy.ones(nmode) * 0.1
    idx = range(norb)
    gfac = 0.1 * numpy.ones(nmode)

    system = simple_model(norb=norb, nelec=nelec, gfac=gfac, omega=omega, nfock=nfock)
    return system

def run_fci_eph():
    from utilities.fci import fci_eph

    system = get_system()

    es = []
    nelecs = [(ia, ib) for ia in range(system.norb + 1) for ib in range(ia + 1)]
    for nelec in nelecs:
        e, c = fci_eph.kernel(
            system.hop,
            system.eri,
            system.norb,
            nelec,
            system.nmode,
            system.nfock,
            system.gmat,
            system.omega,
            tol=1e-10,
            verbose=5,
            nroots=1,
            shift_vac=False,
        )
        print("nelec =", nelec, "E =", e)
        es.append(e)
    es = numpy.hstack(es)
    idx = numpy.argsort(es)
    print(es[idx], "\n")

    print("\nGround state is")
    nelec = nelecs[idx[0]]
    e, c = fci_eph.kernel(
        system.hop,
        system.eri,
        system.norb,
        system.nelec,
        system.nmode,
        system.nfock,
        system.gmat,
        system.omega,
        tol=1e-10,
        verbose=0,
        nroots=1,
        shift_vac=False,
    )
    print("nelec =", nelec, "E =", e)
    return e

#
def run_qmc_eph():
    system = get_system()

    # make custom qed Hamiltonian for afqmc simulation
    mol = gto.M(verbose=3)
    # over write mol functions
    mol.nelectron = int(system.nelec)
    mol.incore_anyway = True
    mol.nao_nr = lambda *args: system.norb
    mol.tot_electrons = lambda *args: mol.nelectron

    #
    qed = Photon(mol, mf=None, omega=system.omega, vec=system.vec, gfac=system.gfac)

    # overwrite the integrals
    qed.get_dipole_ao = lambda *args: system.dipole_mat
    qed.get_quadrupole_ao = lambda *args: system.q_mat

    qedmf = qedhf.RHF(mol, qed=qed)
    qedmf.get_bare_hcore = lambda *args: system.hop
    qedmf.get_ovlp = lambda *args: numpy.eye(system.norb)
    qedmf._eri = system.eri
    qedmf.kernel()

    # run afqmc
    qed = qedmf.qed
    uhf = False,
    time = 5.0
    dt = 0.005
    nwalkers = 100
    propagator_options = {
        "decouple_bilinear": True,
    }

    # TODO/FIXME: custom trial here first for the model system


    # TODO: need to turn off dse or add DSE to FCI

    afqmc = AFQMC(qed, mf=qedmf,
                  dt=dt, total_time=time, num_walkers=nwalkers,
                  energy_scheme="hybrid",
                  uhf = uhf,
                  propagator_options = propagator_options,
                  verbose=4)
    # times, energies = afqmc.kernel()

class TestQMCModel(unittest.TestCase):
    def test_epafqmc_vs_fci(self):
        # e_fci = run_fci_eph()

        # rum ep_afqmc
        e_qmc = run_qmc_eph()


if __name__ == "__main__":
    unittest.main()
