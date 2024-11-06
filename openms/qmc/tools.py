
from functools import reduce
import numpy
import time

def get_h1e_chols(mol):

    overlap = mol.intor("int1e_ovlp")
    Xmat = lo.orth.lowdin(overlap)
    norb = Xmat.shape[0]

    # h1e in OAO
    h1ao = scf.hf.get_hcore(mol)
    h1e = reduce(numpy.dot, (Xmat.T, h1ao, Xmat))

    # nuclear energy
    nuc = mol.energy_nuc()

    # get chols from sub block
    chols = chols_blocked(mol, thresh=1.e-6, fac=10)


def chols_blocked(mol, thresh=1.e-6, fac=15):
    nao = mol.nao_nr()
    max_chols = fac * nao

    diag = numpy.zeros(nao * nao)
    Ltensor = numpy.zeros((max_chols, nao * nao))

    indices4bas = [0]
    end_index = 0
    print("\ntest-yz: total number of AOs = ", nao)
    print("test-yz: numbar of basis     = ", mol.nbas, "\n")
    for i in range(0, mol.nbas):
        l = mol.bas_angular(i)
        nc = mol.bas_nctr(i)
        end_index += (2 * l + 1) * nc
        # print("test-yz:  agular of each basis = ", l)
        # print("test-yz:  end index            = ", end_index)
        indices4bas.append(end_index)

    ndiag = 0
    for i in range(0, mol.nbas):
        shls = (i, i + 1, 0, mol.nbas, i, i + 1, 0, mol.nbas)
        buf = mol.intor("int2e_sph", shls_slice=shls)
        di, _, _, _ = buf.shape
        diag[ndiag : ndiag + di * nao] = buf.reshape(di * nao, di * nao).diagonal()
        ndiag += di * nao
    # buf = mol.intor("int2e_sph", shls_slice=shls)

    nu = numpy.argmax(diag)
    delta_max = diag[nu]

    print("\nindex of maximum elements:", nu)
    print("\nmax diagonal element    = ", delta_max, numpy.max(diag))

    if True:
        print("# Generating Cholesky decomposition of ERIs." % max_chols)
        print("# max number of cholesky vectors = %d" % max_chols)
        print("# iteration %5d: delta_max = %f" % (0, delta_max))
    j = nu // nao
    l = nu % nao
    sj = numpy.searchsorted(indices4bas, j)
    sl = numpy.searchsorted(indices4bas, l)
    if indices4bas[sj] != j and j != 0:
        sj -= 1
    if indices4bas[sl] != l and l != 0:
        sl -= 1
    Mapprox = numpy.zeros(nao * nao)
    # ERI[:,jl]
    eri_col = mol.intor("int2e_sph", shls_slice=(0, mol.nbas, 0, mol.nbas, sj, sj + 1, sl, sl + 1))
    cj, cl = max(j - indices4bas[sj], 0), max(l - indices4bas[sl], 0)
    Ltensor[0] = numpy.copy(eri_col[:, :, cj, cl].reshape(nao * nao)) / delta_max**0.5

    nchol = 0
    while abs(delta_max) > thresh:
        # Update cholesky vector
        start = time.time()
        # M'_ii = L_i^x L_i^x
        Mapprox += (Ltensor[nchol] * Ltensor[nchol])
        # D_ii = M_ii - M'_ii
        print("diag.shape. Mapprox.shape", diag.shape, Mapprox.shape)

        delta = diag - Mapprox
        nu = numpy.argmax(numpy.abs(delta))
        delta_max = numpy.abs(delta[nu])
        # Compute ERI chunk.
        # shls_slice computes shells of integrals as determined by the angular
        # momentum of the basis function and the number of contraction
        # coefficients. Need to search for AO index within this shell indexing
        # scheme.
        # AO index.
        j = nu // nao
        l = nu % nao
        # Associated shell index.
        sj = numpy.searchsorted(indices4bas, j)
        sl = numpy.searchsorted(indices4bas, l)
        if indices4bas[sj] != j and j != 0:
            sj -= 1
        if indices4bas[sl] != l and l != 0:
            sl -= 1
        # Compute ERI chunk.
        eri_col = mol.intor(
            "int2e_sph", shls_slice=(0, mol.nbas, 0, mol.nbas, sj, sj + 1, sl, sl + 1)
        )
        # Select correct ERI chunk from shell.
        cj, cl = max(j - indices4bas[sj], 0), max(l - indices4bas[sl], 0)
        Munu0 = eri_col[:, :, cj, cl].reshape(nao * nao)
        # Updated residual = \sum_x L_i^x L_nu^x
        R = numpy.dot(Ltensor[: nchol + 1, nu], Ltensor[: nchol + 1, :])
        Ltensor[nchol + 1] = (Munu0 - R) / (delta_max) ** 0.5

        nchol += 1
        if True:
            step_time = time.time() - start
            info = (nchol, delta_max, step_time)
            print("# iteration %5d: delta_max = %13.8e: time = %13.8e" % info)

        if nchol > max_chols: break

    return Ltensor[:nchol]


def get_mean_std(energies, N=10):
    m = max(1, len(energies) // N)
    last_m_real = numpy.asarray(energies[-m:]).real
    mean = numpy.mean(last_m_real)
    std_dev = numpy.std(last_m_real)
    return mean, std_dev
