#
# @ 2023. Triad National Security, LLC. All rights reserved.
#
# This program was produced under U.S. Government contract 89233218CNA000001
# for Los Alamos National Laboratory (LANL), which is operated by Triad
# National Security, LLC for the U.S. Department of Energy/National Nuclear
# Security Administration. All rights in the program are reserved by Triad
# National Security, LLC, and the U.S. Department of Energy/National Nuclear
# Security Administration. The Government is granted for itself and others acting
# on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this
# material to reproduce, prepare derivative works, distribute copies to the
# public, perform publicly and display publicly, and to permit others to do so.
#
# Author: Yu Zhang <zhy@lanl.gov>
#

r"""
Interface to call C or C++ math libraries.

Library examples: cuTENSOR, NCCL, etc.
"""

import os, sys
import warnings
import ctypes
import h5py

import numpy
from scipy.linalg import lapack
from functools import reduce

from threading import Thread
from multiprocessing import Queue, Process

from pyscf import lib # temporary

from openms.lib import backend
from openms.lib import logger
from openms import __config__

SAFE_EIGH_LINDEP = getattr(__config__, "lib_linalg_helper_safe_eigh_lindep", 1e-15)
DAVIDSON_LINDEP = getattr(__config__, "lib_linalg_helper_davidson_lindep", 1e-14)
DSOLVE_LINDEP = getattr(__config__, "lib_linalg_helper_dsolve_lindep", 1e-15)
MAX_MEMORY = getattr(__config__, "lib_linalg_helper_davidson_max_memory", 2000)  # 2GB
FOLLOW_STATE = getattr(__config__, "lib_linalg_helper_davidson_follow_state", False)

def load_library(libname):
    r"""Load C, C++, Fortran libraries."""
    try:
        _loaderpath = os.path.dirname(__file__)
        return numpy.ctypeslib.load_library(libname, _loaderpath)
    except OSError:
        from openms import __path__ as ext_modules

        for path in ext_modules:
            libpath = os.path.join(path, "lib")
            if os.path.isdir(libpath):
                for files in os.listdir(libpath):
                    if files.startswith(libname):
                        return numpy.ctypeslib.load_library(libname, libpath)
        raise

# load bml lib (todo)
# libbml = lib.load_library('libbml')

# -------------------------
# other math algorithms TBA
# -------------------------

def unitary_transform(U, A):
    r"""
    Unitary transform of :math:`\boldsymbol{A}` with :math:`\boldsymbol{U}`.

    :math:`\boldsymbol{\tilde{A}} = \boldsymbol{U}^\dagger \boldsymbol{A} \boldsymbol{U}`

    Parameters
    ----------
    U : :class:`~numpy.ndarray`
        Unitary transformation matrix, :math:`\boldsymbol{U}`.
    A : :class:`~numpy.ndarray`
        Matrix to transform, :math:`\boldsymbol{A}`.

    Returns
    -------
    :class:`~numpy.ndarray`
        Transformed matrix, :math:`\boldsymbol{\tilde{A}}`.
    """

    Atmp = lib.dot(A, U)
    return lib.dot(U.conj().T, Atmp)


def get_l2_norm(A):
    r"""Return L2 norm."""
    # will be replaced by backend.dot
    return numpy.sqrt(lib.dot(A, A))


def full_cholesky_decomposition(matrix, threshold):
    r"""
    Perform a full Cholesky decomposition using the LAPACK dpstrf function.

    Parameters
    ----------
    matrix : :class:`~numpy.ndarray`
        The matrix M to decompose, such that:
        :math:`P^T M P = L L^T`.
    threshold : float
        Threshold to use in the Cholesky decomposition.

    Return
    ------
    cholesky_vectors : :class:`~numpy.ndarray`
        The Cholesky vectors, :math:`L`.
    pivots : :class:`~numpy.ndarray`
        The vector containing the information in :math:`P`.
    n_vectors : int
        The number of Cholesky vectors.
    """

    dim_ = matrix.shape[0]
    cholesky_vectors = matrix.copy()
    pivots = numpy.zeros(dim_, dtype=numpy.int32)

    # DPSTRF computes the Cholesky factorization with complete pivoting
    # of a real symmetric positive semidefinite matrix.
    cholesky_vectors, pivots, n_vectors, info = \
    lapack.dpstrf(cholesky_vectors, lower=1, tol=threshold)

    if info < 0:
        raise ValueError('Cholesky decomposition failed! Something wrong in call to dpstrf.')

    # Zero upper unreferenced triangle
    for i in range(dim_):
        for j in range(i):
            cholesky_vectors[j, i] = 0

    return cholesky_vectors, pivots, n_vectors


def full_cholesky_orth(S, threshold=1.0e-7):
    r"""
    Full Cholesky orthogonalization.

    Compute :math:`P`, :math:`L`, and the number of linearly independent/
    orthonormal AOs (``n_oao``), resulting from a Full Cholesky decomposition
    of the AO overlap matrix :math:`S` within linear dependency threshold
    ``threshold``:

    .. math::

        P^T S P = L L^T

    or

    .. math::

      (L^{-1} P^T) S (P L^{-T} = I \equiv X^T S X = I \rightarrow S = X^{-T} X^{-1},

    where :math:`X = P L^{-T}`

    .. note::
        Values of ``pivots`` are ``[1, n]``, not ``[0, n-1]``,
        so we correct the index before returning ``P``.

    Parameters
    ----------
    S : :class:`~numpy.ndarray`
        Overlap matrix in the non-orthogonal AO basis.
    threshold : float
        Linear dependency threshold.

    Return
    ------
    L : :class:`~numpy.ndarray`
        The Cholesky vectors, :math:`L`.
    P : :class:`~numpy.ndarray`
        The vector containing the information in :math:`P`.
    """

    n = S.shape[0]

    cholesky_vectors, pivots, n_oao = full_cholesky_decomposition(S, threshold)

    if (n_oao > n or n_oao <= 0 or
        any(p > n for p in pivots) or
        any(p <= 0 for p in pivots)):

        print("Something went wrong when decomposing the AO overlap.")
        print("Did you compile with the wrong type of integers in setup?")
        print("For example, system native BLAS with default 64-bit integers.")
        print("If that is the case, use setup with --int32 or install MKL.")
        raise ValueError("Failed to decompose AO overlap.")

    L = numpy.zeros((n_oao, n_oao))
    P = numpy.zeros((n, n_oao))

    L[:n_oao, :n_oao] = cholesky_vectors[:n_oao, :n_oao]
    del cholesky_vectors

    # note: values of pivots is [1, n], not [0, n-1], so we need correct the index
    for i in range(n_oao):
        P[pivots[i] - 1, i] = 1.0

    return L, P
