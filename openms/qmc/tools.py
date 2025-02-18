
from functools import reduce
import numpy
import scipy
import time


def read_fcidump(fname, norb):
    """
    :param fname: electron integrals dumped by pyscf
    :param norb: number of orbitals
    :return: electron integrals for 2nd quantization with chemist's notation
    """
    eri = numpy.zeros((norb, norb, norb, norb))
    h1e = numpy.zeros((norb, norb))

    with open(fname, "r") as f:
        lines = f.readlines()
        for line, info in enumerate(lines):
            if line < 4:
                continue
            line_content = info.split()
            integral = float(line_content[0])
            p, q, r, s = [int(i_index) for i_index in line_content[1:5]]
            if r != 0:
                # eri[p,q,r,s] is with chemist notation (pq|rs)=(qp|rs)=(pq|sr)=(qp|sr)
                eri[p - 1, q - 1, r - 1, s - 1] = integral
                eri[q - 1, p - 1, r - 1, s - 1] = integral
                eri[p - 1, q - 1, s - 1, r - 1] = integral
                eri[q - 1, p - 1, s - 1, r - 1] = integral
            elif p != 0:
                h1e[p - 1, q - 1] = integral
                h1e[q - 1, p - 1] = integral
            else:
                nuc = integral
    return h1e, eri, nuc



def get_h1e_chols(mol, Xmat=None, thresh=1.e-6):
    r"""
    Calculate the one-electron Hamiltonian (h1e) in the orthogonal atomic orbital (OAO) basis
    and the Cholesky decomposition tensor for a molecular system.

    .. note::
        May move this function to qmc class.

    Parameters
    ----------
    mol : object
        A molecular object used for integral calculations (e.g., PySCF's Mole object).
    Xmat: ndarray
        a matrix use to rotate the h1e and eri (or chols)
    thresh : float, optional
        Threshold for truncation in the Cholesky decomposition. Defaults to 1.e-6.

    Returns
    -------
    h1e : numpy.ndarray
        The one-electron Hamiltonian matrix in the OAO basis.
    chols : numpy.ndarray
        The Cholesky decomposition tensor for the molecular integrals.
    nuc : float
        The nuclear repulsion energy of the molecule.

    """
    from pyscf import scf, lo

    if Xmat is None:
        overlap = mol.intor("int1e_ovlp")
        Xmat = lo.orth.lowdin(overlap)
    norb = Xmat.shape[0]

    # h1e in OAO
    h1ao = scf.hf.get_hcore(mol)
    h1e = reduce(numpy.dot, (Xmat.T, h1ao, Xmat))

    # nuclear energy
    nuc = mol.energy_nuc()

    # get chols from sub block
    ltensors = chols_blocked(mol, thresh=thresh, max_chol_fac=15)

    if mol.verbose > 3:
        print(f"Debug: norm of ltensor in AO = {numpy.linalg.norm(ltensors)}")
    # transfer ltensor into OAO
    for i, chol in enumerate(ltensors):
        ltensors[i] = reduce(numpy.dot, (Xmat.conj().T, chol, Xmat))

    return h1e, ltensors, nuc


def chols_full(mol, eri=None, thresh=1.e-12, aosym="s1"):
    r"""
    Get Cholesky decomposition from the SVD of the full ERI.

    Parameters
    ----------
    mol : object
        Molecular object used for integral calculations.
    eri : numpy.ndarray, optional
        Electron repulsion integrals. If None, will be computed.
    thresh : float, optional
        Threshold for truncation. Defaults to 1.e-12.
    aosym : str, optional
        Symmetry in the ERI tensor. Defaults to "s1".

    Returns
    -------
    numpy.ndarray
        Cholesky tensor (in AO basis) from decomposition.
    """

    if eri is None:
        eri = mol.intor('int2e_sph', aosym='s1')
    nao = eri.shape[0]
    eri = eri.reshape((nao**2, -1))
    u, s, v = scipy.linalg.svd(eri)
    del eri
    # print("u.shape", u.shape)
    # print("v.shape", v.shape)
    # print("s.shape", s.shape)
    # print(s>1.e-12)

    idx = (s > thresh)
    ltensor = (u[:,idx] * numpy.sqrt(s[idx])).T
    ltensor = ltensor.reshape(ltensor.shape[0], nao, nao)

    return ltensor

def chols_blocked(mol, thresh=1.e-6, max_chol_fac=15):
    r"""
    Get modified Cholesky decomposition from the block decomposition of ERI.
    See Ref. :cite:`Henrik:2003rs, Nelson:1977si`.

    Initially, we calcualte the diagonal element, :math:`M_{pp}=(pq|pq)`.

    More details TBA.

    Parameters
    ----------
    mol : object
        Molecular object used for integral calculations.
    thresh : float, optional
        Threshold for truncation. Defaults to 1.e-6.
    max_chol_fac : int, optional
        Factor to control the maximum number of L tensors.

    Returns
    -------
    numpy.ndarray
        Cholesky tensor from decomposition.
    """
    print(f"{'*' * 50}\n Block decomposition of ERI on the fly\n{'*' * 50}")

    nao = mol.nao_nr()
    max_chols = max_chol_fac * nao
    ltensor = numpy.zeros((max_chols, nao * nao))
    diag = numpy.zeros(nao * nao)

    # Compute basis indices
    indices4bas = [0]
    end_index = 0
    for i in range(mol.nbas):
        l = mol.bas_angular(i)
        nc = mol.bas_nctr(i)
        end_index += (2 * l + 1) * nc
        # print("test-yz:  agular of each basis = ", l)
        # print("test-yz:  end index            = ", end_index)
        indices4bas.append(end_index)

    # Compute initial diagonal block
    ndiag = 0
    for i in range(mol.nbas):
        shls = (i, i + 1, 0, mol.nbas, i, i + 1, 0, mol.nbas)
        buf = mol.intor("int2e_sph", shls_slice=shls)
        di = buf.shape[0]
        diag[ndiag:ndiag + di * nao] = buf.reshape(di * nao, di * nao).diagonal()
        ndiag += di * nao

    # Find initial maximum diagonal element
    nu = numpy.argmax(diag)
    delta_max = diag[nu]


    # Compute initial Cholesky vector
    j, l = divmod(nu, nao)
    sj, sl = _find_shell(indices4bas, j, l)

    Mapprox = numpy.zeros(nao * nao)
    # compute eri block
    ltensor[0] = _compute_eri_chunk(mol, nao, sj, sl, indices4bas, j, l, delta_max)

    nchol = 0
    while abs(delta_max) > thresh:
        # Update cholesky vector
        start = time.time()
        # M'_ii = L_i^x L_i^x
        Mapprox += (ltensor[nchol] * ltensor[nchol])
        # D_ii = M_ii - M'_ii

        delta = diag - Mapprox
        nu = numpy.argmax(numpy.abs(delta))
        delta_max = numpy.abs(delta[nu])

        # shls_slice computes shells of integrals as determined by the angular
        # momentum of the basis function and the number of contraction
        # coefficients.

        # Search for AO index within this shell indexing scheme.
        j, l = divmod(nu, nao)
        sj, sl = _find_shell(indices4bas, j, l)

        # Compute ERI chunk and select correct ERI chunk from shell.
        Munu0 = _compute_eri_chunk(mol, nao, sj, sl, indices4bas, j, l, 1.0)

        # Updated residual = \sum_x L_i^x L_nu^x and Cholesky tensor
        R = numpy.dot(ltensor[: nchol + 1, nu], ltensor[: nchol + 1, :])
        ltensor[nchol + 1] = (Munu0 - R) / numpy.sqrt(delta_max)

        nchol += 1
        step_time = time.time() - start
        if mol.verbose > 3:
            print(f"# Iteration {nchol:5d}: delta_max = {delta_max:13.8e}: time = {step_time:13.8e}")

        # Stop if maximum number of Cholesky vectors is reached
        if nchol >= max_chols:
            print("Warning: Maximum number of Cholesky vectors reached.")
            break

    # Reshape and truncate the Cholesky tensor
    ltensor = ltensor[:nchol].reshape(nchol, nao, nao)
    print(f"{'*' * 50}\n Block decomposition of ERI on the fly ... Done!\n{'*' * 50}\n")

    return ltensor


def _find_shell(indices4bas, j, l):
    r"""
    Find shell indices for given AO indices.
    """
    sj = numpy.searchsorted(indices4bas, j, side="right") - 1
    sl = numpy.searchsorted(indices4bas, l, side="right") - 1
    return sj, sl


def _compute_eri_chunk(mol, nao, sj, sl, indices4bas, j, l, delta_max):
    r"""
    Compute the initial Cholesky vector.
    """
    shls = (0, mol.nbas, 0, mol.nbas, sj, sj + 1, sl, sl + 1)
    eri_col = mol.intor("int2e_sph", shls_slice=shls)
    cj, cl = j - indices4bas[sj], l - indices4bas[sl]
    return eri_col[:, :, cj, cl].reshape(nao * nao) / numpy.sqrt(delta_max)

#
# analysis tools
#

def get_mean_std(energies, N=10):
    r"""
    Calculate the mean and standard deviation of the real parts of the last subset of qmc energies.

    Parameters
    ----------
    energies : list or numpy.ndarray
        A list or array of energy values (can be complex).
    N : int, optional
        Factor determining the size of the subset for calculations.
        Defaults to 10. The subset size is `len(energies) // N`, with a minimum size of 1.

    Returns
    -------
    mean : float
      The mean of the real parts of the selected energy subset.
    std_dev : float
      The standard deviation of the real parts of the selected energy subset.
    """
    m = max(1, len(energies) // N)
    last_m_real = numpy.asarray(energies[-m:]).real
    mean = numpy.mean(last_m_real)
    std_dev = numpy.std(last_m_real)
    return mean, std_dev

#import pandas as pd

def autocorr_func(x, normalize=True):
    r"""
    Compute the autocorrelation function (ACF) of a 1D array using the Fast Fourier Transform (FFT).

    The autocorrelation function is computed as:

    .. math::
        \text{ACF}(k) = \frac{1}{(N-k)} \sum_{i=1}^{N-k} (x_i - \bar{x})(x_{i+k} - \bar{x})

    Instead of direct computation, we use the Wiener-Khinchin theorem, which relates the ACF
    to the power spectrum via the Fourier Transform:

    .. math::
        \text{ACF} = \mathcal{F}^{-1}(|\mathcal{F}(x)|^2)

    Parameters
    ----------
    x : ndarray
        Input data array (1D).
    normalize : bool, optional
        If True, normalizes the autocorrelation function such that ACF[0] = 1.

    Returns
    -------
    ndarray
        Autocorrelation function of the input array.
    """
    x = numpy.asarray(x, dtype=numpy.float64)
    if x.ndim != 1:
        raise ValueError("Input must be a 1D array.")

    n = 2 ** int(numpy.ceil(numpy.log2(len(x))))  # Next power of two for FFT efficiency
    x_mean = numpy.mean(x)
    f = numpy.fft.fft(x - x_mean, n=2 * n)
    acf = numpy.fft.ifft(f * numpy.conjugate(f))[: len(x)].real
    acf /= (4 * n)  # Normalization factor

    if normalize:
        acf /= acf[0]  # Normalize to 1 at zero lag

    return acf

def auto_window(autocorr_times, c=5.0):
    r"""
    Determines the optimal window size for summing the autocorrelation function.

    The window size is selected based on the Sokal criterion:

    .. math::
        k < c \cdot \tau_k

    where :math:`\tau_k` is the integrated autocorrelation time at lag :math:`k`, and :math:`c` is a tunable factor.

    Parameters
    ----------
    autocorr_times : ndarray
        Integrated autocorrelation times at each lag.
    c : float, optional
        Factor controlling window size (default is 5.0).

    Returns
    -------
    int
        Optimal truncation index for summing ACF.
    """
    mask = numpy.arange(len(autocorr_times)) < c * autocorr_times
    return numpy.argmin(mask) if numpy.any(mask) else len(autocorr_times) - 1

def get_autocorr_time(y, c=5.0):
    r"""
    Computes the integrated autocorrelation time :cite:`goodman:2010`.

    The integrated autocorrelation time :math:`\tau` is estimated as:

    .. math::
        \tau_k = 2 \sum_{i=1}^{k} \text{ACF}(i) - 1

    The sum is truncated using an automated windowing procedure (`auto_window`).

    Parameters
    ----------
    y : ndarray
        Time series data.
    c : float, optional
        Windowing factor for truncation (default is 5.0).

    Returns
    -------
    float
        Estimated autocorrelation time.
    """
    acf = autocorr_func(y)
    autocorr_times = 2.0 * numpy.cumsum(acf) - 1.0  # Integrated autocorrelation time
    window = auto_window(autocorr_times, c)
    return autocorr_times[window]


# analyze error based on auto correlation function
def analysis_autocorr(y, name="etot", verbose=False):
    r"""
    Perform error analysis on QMC data using the autocorrelation function.

    **Reblocking** is a technique to estimate error bars in correlated data by grouping correlated samples
    into larger blocks to obtain independent estimates.

    The block size is determined by the **autocorrelation time** :math:`\tau`:

    .. math::
        \text{Block Size} = \lceil \tau \rceil

    The blocked estimates for the mean and standard error are computed as:

    .. math::
        \bar{y} = \frac{1}{M} \sum_{i=1}^{M} \bar{y}_i

    .. math::
        \sigma_{\bar{y}} = \frac{\sigma}{\sqrt{M}}

    where:
    - :math:`M` is the number of blocks,
    - :math:`\bar{y}_i` is the mean of block :math:`i`,
    - :math:`\sigma` is the standard deviation of the blocked values.

    Parameters
    ----------
    y : DataFrame
        DataFrame containing QMC calculation results.
    name : str, optional
        Column name to analyze (default is "ETotal").
    verbose : bool, optional
        If True, prints additional debug information.

    Returns
    -------
    dict
        DataDict containing mean, standard error, number of samples, and block size.
    """

    datasize = len(y)
    Nmax = int(numpy.log2(datasize))
    Ndata, autocorr_times = [], []

    for i in range(Nmax):
        n = len(y) // (2**i)
        Ndata.append(n)
        autocorr_times.append(get_autocorr_time(y[:n]))

    if verbose:
        for n, tautocorr in zip(reversed(Ndata), reversed(autocorr_times)):
            print(f"nsamples: {n}, autocorrelation time: {tautocorr}")

    block_size = int(numpy.ceil(autocorr_times[0]))  # Use the estimate with the largest sample size
    nblocks = len(y) // block_size
    yblocked = [numpy.mean(y[i * block_size : (i + 1) * block_size]) for i in range(nblocks)]

    yavg = numpy.mean(yblocked)
    ystd = numpy.std(yblocked) / numpy.sqrt(nblocks)

    results = {
            f"{name}": [yavg],
            f"{name}_error": [ystd],
            f"{name}_nsamp": [nblocks],
            "ac_sampsize": [block_size],
    }
    return results
