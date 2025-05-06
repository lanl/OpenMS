import numpy
import time
import pytest
import unittest
from pyscf import gto, scf, fci, lo
from openms.qmc.afqmc import AFQMC
from openms.qmc import tools
from molecules import get_mol


def calc_qmc_energy(
    mol,
    time=6.0,
    num_walkers=500,
    uhf=False,
    energy_scheme="hybrid",
    block_decompose_eri=False,
):

    afqmc = AFQMC(
        mol,
        dt=0.005,
        total_time=time,
        num_walkers=num_walkers,
        energy_scheme=energy_scheme,
        uhf=uhf,
        chol_thresh=1.0e-8,
        property_calc_freq=10,
        block_decompose_eri=True,
        verbose=mol.verbose,
    )
    afqmc.propagator.debug_mpi = True

    times, energies = afqmc.kernel()
    return energies


@pytest.mark.mpi
def test_mpi_random(seed=10):
    from openms.__mpi__ import MPI, MPIWrapper, original_print

    mpi_handler = MPIWrapper()
    # Determine local range
    size = mpi_handler.size
    rank = mpi_handler.rank

    # Create a rank-specific seed using SeedSequence
    ss = numpy.random.SeedSequence(seed)
    child_seeds = ss.spawn(size)
    #original_print(f'child_seeds of rank {rank} = {child_seeds[rank]}')
    numpy.random.default_rng(child_seeds[rank])
    xi_local = numpy.random.normal(0.0, 1.0, size=(10,))
    #original_print('local_random:', xi_local)


@pytest.mark.mpi
def test_mpi_estimator():
    r"""
    Test MPI propagator and energy estimator
    """

    from openms.qmc import tools
    from openms.__mpi__ import MPI, MPIWrapper
    from openms.qmc import estimators
    from openms.qmc.trial import trial_walker_ovlp_gf_base
    from openms.qmc.propagators import propagate_exp_op

    #set up mpi
    mpi_handler = MPIWrapper()

    basis = 'ccpvdz'
    threshold = 1.e-8
    nwalkers = 201 # test non-uniform walker distribution

    ## Split walkers across MPI processes
    # ------------------------------------------------------
    # Determine local range
    size = mpi_handler.size
    rank = mpi_handler.rank

    counts = [nwalkers // size + (1 if i < nwalkers % size else 0) for i in range(size)]
    displs = [sum(counts[:i]) for i in range(size)]
    nlocal = counts[rank]

    # build molecule
    mol = get_mol(natoms=20, name="Hchain", basis=basis)
    nocc = mol.nelec[0]
    nao = mol.nao_nr()

    # compute mean-field WF
    mf = scf.RHF(mol)
    mf.kernel()

    # build trial WF
    overlap = mol.intor("int1e_ovlp")
    Xmat = lo.orth.lowdin(overlap)
    Xinv = numpy.linalg.inv(Xmat)
    psia = Xinv.dot(mf.mo_coeff[:, : mol.nelec[0]])
    ltensor = tools.chols_blocked(mol, thresh=threshold)
    nchol = ltensor.shape[0]

    # rotated ltensor : \Psi^T L -> (nchol, nano, nocc)
    rltensor = numpy.einsum("pi, npq->nqi", psia.conj(), ltensor, optimize=True)

    # get local and total walker
    phiw_local = numpy.array([psia for _ in range(nlocal)]) * (1.0 + 1.e-20 * 1j)
    phiw_all = numpy.array([psia for _ in range(nwalkers)]) * (1.0 + 1.e-20 * 1j)

    if rank == 0:
        print("mpi_handler.size =", mpi_handler.size)
        print("psia.shape", psia.shape)
        print("phiw.shape", phiw_local.shape)

    dt = 0.005
    taylor_order = 6
    sqrtdt = 1j * numpy.sqrt(dt)
    #xi = numpy.ones(nchol * nwalkers)

    xi_local = numpy.empty((nlocal, nchol), dtype=numpy.float64)
    if rank == 0:
        xi = numpy.random.normal(0.0, 1.0, (nwalkers, nchol))
    else:
        xi = None

    # Scatterv requires 1D buffers
    if rank == 0:
        sendbuf = [xi.flatten(),
                   (numpy.array(counts) * nchol, numpy.array(displs) * nchol), MPI.DOUBLE]
    else:
        sendbuf = None
    mpi_handler.comm.Scatterv(sendbuf, [xi_local, MPI.DOUBLE], root=0)
    # print("start/end: ", displs[rank], displs[rank] + counts[rank])

    # propagate w.o MPI
    # ------------------------------
    t0 = time.time()
    if rank == 0:
        eri_op = sqrtdt * numpy.dot(xi, ltensor.reshape(nchol, -1)).reshape(nwalkers, nao, nao)
        phiw_all = propagate_exp_op(phiw_all, eri_op, taylor_order)
    prop_time = time.time() - t0

    # propagate with MPI
    # ------------------------------
    # print(f"xi_local.shape: {xi_local.shape}  nlocal = {nlocal}")
    t0 = time.time()
    localop = sqrtdt * numpy.dot(xi_local, ltensor.reshape(nchol, -1)).reshape(nlocal, nao, nao)
    phiw_local = propagate_exp_op(phiw_local, localop, taylor_order)
    prop_time2 = time.time() - t0

    # compute exx w.o MPI
    if rank == 0:
        t0 = time.time()
        _, Ghalf = trial_walker_ovlp_gf_base(phiw_all, psia)
        exx0 = estimators.exx_rltensor_Ghalf(rltensor, Ghalf)
        t1 = time.time() - t0

    # MPI computation of exx
    exx1 = None
    if rank == 0:
        exx1 = numpy.zeros(nwalkers, dtype=numpy.complex128)
    t0 = time.time()
    _, Ghalf_local = trial_walker_ovlp_gf_base(phiw_local, psia)
    local_exx = estimators.exx_rltensor_Ghalf(rltensor, Ghalf_local)
    #exx1 = estimators.exx_rltensor_Ghalf_chunked(rltensor, Ghalf_local, mpi_handler.comm, MPI, nwalkers, counts, displs)
    mpi_handler.comm.Gatherv(local_exx, [exx1, counts, displs, MPI.COMPLEX16], root=0)
    t2 = time.time() - t0

    if rank == 0:
        assert numpy.allclose(exx0, exx1, atol=1e-6), "exx0 and exx1 are not close within 1e-6"
        # assert numpy.allclose(phiw_all, phiw_gathered, atol=1e-8), "exx0 and exx1 are not close within 1e-8"
        print(f"Times of computing exx are:     {t1:.3f} vs {t2: .3f}")
        print(f"\nTimes of propagation      : {prop_time:.3f} vs {prop_time2:.3f}")
        print(f"MPI speedup in energy     : {t1/t2:.2f} (ideal speedup is {size})")
        print(f"MPI speedup in propagation: {prop_time/prop_time2:.2f} (ideal speedup is {size})")


class TestQMC_MPI(unittest.TestCase):

    def test_mpi_estimator(self):
        test_mpi_estimator()

    def test_qmc_MPI(self):

        basis = "ccpvdz",
        basis = "sto3g"

        #mol = get_mol(name="C3H4O2", basis=basis, verbose=4)
        mol = get_mol(natoms=10, name="Hchain", basis=basis, verbose=4)

        # here we make the number of walkers odd to make sure the MPI works with
        # unevenly distributed walkers
        qmc_energies = calc_qmc_energy(mol, uhf=False, time=5.0, num_walkers=201)

        results = tools.analysis_autocorr(qmc_energies, verbose=1)
        qmc_mean, std = results['etot'][0], results['etot_error'][0]
        print("qmc_mean/std =", qmc_mean, std)


if __name__ == "__main__":
    unittest.main()
