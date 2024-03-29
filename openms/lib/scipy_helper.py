
#!/usr/bin/env python
# Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

'''
Extensions to the scipy.linalg module

This file is copied from PySCF.lib.scipy_helper.
'''

import sys
import numpy
from pyscf import lib
from functools import reduce
from openms import __config__

LINEAR_DEP_THRESHOLD = getattr(__config__, 'scf_addons_remove_linear_dep_threshold', 1e-8)
CHOLESKY_THRESHOLD = getattr(__config__, 'scf_addons_cholesky_threshold', 1e-10)
FORCE_PIVOTED_CHOLESKY = getattr(__config__, 'scf_addons_force_cholesky', False)
LINEAR_DEP_TRIGGER = getattr(__config__, 'scf_addons_remove_linear_dep_trigger', 1e-10)


# Numpy/scipy does not seem to have a convenient interface for
# pivoted Cholesky factorization. Newer versions of scipy (>=1.4) provide
# access to the raw lapack function, which is wrapped around here.
# With older versions of scipy, we use our own implementation instead.
try:
    from scipy.linalg.lapack import dpstrf as _dpstrf
except ImportError:
    def _pivoted_cholesky_wrapper(A, tol, lower):
        return pivoted_cholesky_python(A, tol=tol, lower=lower)
else:
    def _pivoted_cholesky_wrapper(A, tol, lower):
        N = A.shape[0]
        assert (A.shape == (N, N))
        L, piv, rank, info = _dpstrf(A, tol=tol, lower=lower)
        if info < 0:
            raise RuntimeError('Pivoted Cholesky factorization failed.')
        if lower:
            L[numpy.triu_indices(N, k=1)] = 0
            L[:, rank:] = 0
        else:
            L[numpy.tril_indices(N, k=-1)] = 0
            L[rank:, :] = 0
        return L, piv-1, rank


def pivoted_cholesky(A, tol=-1.0, lower=False):
    '''
    Performs a Cholesky factorization of A with full pivoting.
    A can be a (singular) positive semidefinite matrix.

    P.T * A * P = L * L.T   if   lower is True
    P.T * A * P = U.T * U   if   lower if False

    Use regular Cholesky factorization for positive definite matrices instead.

    Args:
        A : the matrix to be factorized
        tol : the stopping tolerance (see LAPACK documentation for dpstrf)
        lower : return lower triangular matrix L if true
                return upper triangular matrix U if false

    Returns:
        the factor L or U, the pivot vector (starting with 0), the rank
    '''
    return _pivoted_cholesky_wrapper(A, tol=tol, lower=lower)


def pivoted_cholesky_python(A, tol=-1.0, lower=False):
    '''
    Pedestrian implementation of Cholesky factorization with full column pivoting.
    The LAPACK version should be used instead whenever possible!

    Args:
        A : the positive semidefinite matrix to be factorized
        tol : stopping tolerance
        lower : return the lower or upper diagonal factorization

    Returns:
        the factor, the permutation vector, the rank
    '''
    N = A.shape[0]
    assert (A.shape == (N, N))

    D = numpy.diag(A.real).copy()
    if tol < 0:
        machine_epsilon = numpy.finfo(numpy.double).eps
        tol = N * machine_epsilon * numpy.amax(numpy.diag(A))

    L = numpy.zeros_like(A)
    piv = numpy.arange(N)
    rank = 0
    for k in range(N):
        s = k + numpy.argmax(D[k:])
        piv[k], piv[s] = piv[s], piv[k]
        D[k], D[s] = D[s], D[k]
        L[[k, s], :] = L[[s, k], :]
        if D[k] <= tol:
            break
        rank += 1
        L[k, k] = numpy.sqrt(D[k])
        L[k+1:, k] = (A[piv[k+1:], piv[k]] - numpy.dot(L[k+1:, :k], L[k, :k].conj())) / L[k, k]
        D[k+1:] -= abs(L[k+1:, k]) ** 2

    if lower:
        return L, piv, rank
    else:
        return L.conj().T, piv, rank


# copied from pyscf/scf/addons.py

def canonical_orth_(S, thr=1e-7):
    '''Löwdin's canonical orthogonalization'''
    # Ensure the basis functions are normalized (symmetry-adapted ones are not!)
    normlz = numpy.power(numpy.diag(S), -0.5)
    Snorm = numpy.dot(numpy.diag(normlz), numpy.dot(S, numpy.diag(normlz)))
    # Form vectors for normalized overlap matrix
    Sval, Svec = numpy.linalg.eigh(Snorm)
    X = Svec[:,Sval>=thr] / numpy.sqrt(Sval[Sval>=thr])
    # Plug normalization back in
    X = numpy.dot(numpy.diag(normlz), X)
    return X

def partial_cholesky_orth_(S, canthr=1e-7, cholthr=1e-9):
    '''Partial Cholesky orthogonalization for curing overcompleteness.

    References:

    Susi Lehtola, Curing basis set overcompleteness with pivoted
    Cholesky decompositions, J. Chem. Phys. 151, 241102 (2019),
    doi:10.1063/1.5139948.

    Susi Lehtola, Accurate reproduction of strongly repulsive
    interatomic potentials, Phys. Rev. A 101, 032504 (2020),
    doi:10.1103/PhysRevA.101.032504.
    '''
    # Ensure the basis functions are normalized
    normlz = numpy.power(numpy.diag(S), -0.5)
    Snorm = numpy.dot(numpy.diag(normlz), numpy.dot(S, numpy.diag(normlz)))

    # Sort the basis functions according to the Gershgorin circle
    # theorem so that the Cholesky routine is well-initialized
    odS = numpy.abs(Snorm)
    numpy.fill_diagonal(odS, 0.0)
    odSs = numpy.sum(odS, axis=0)
    sortidx = numpy.argsort(odSs, kind='stable')

    # Run the pivoted Cholesky decomposition
    Ssort = Snorm[numpy.ix_(sortidx, sortidx)].copy()
    c, piv, r_c = pivoted_cholesky(Ssort, tol=cholthr)
    # The functions we're going to use are given by the pivot as
    idx = sortidx[piv[:r_c]]

    # Get the (un-normalized) sub-basis
    Ssub = S[numpy.ix_(idx, idx)].copy()
    # Orthogonalize sub-basis
    Xsub = canonical_orth_(Ssub, thr=canthr)

    # Full X
    X = numpy.zeros((S.shape[0], Xsub.shape[1]), dtype=Xsub.dtype)
    X[idx,:] = Xsub

    return X

def remove_linear_dep_(mf, threshold=LINEAR_DEP_THRESHOLD,
                       lindep=LINEAR_DEP_TRIGGER,
                       cholesky_threshold=CHOLESKY_THRESHOLD,
                       force_pivoted_cholesky=FORCE_PIVOTED_CHOLESKY):
    '''
    Args:
        threshold : float
            The threshold under which the eigenvalues of the overlap matrix are
            discarded to avoid numerical instability.
        lindep : float
            The threshold that triggers the special treatment of the linear
            dependence issue.
    '''
    s = mf.get_ovlp()
    cond = numpy.max(lib.cond(s))
    if cond < 1./lindep and not force_pivoted_cholesky:
        return mf

    logger.info(mf, 'Applying remove_linear_dep_ on SCF object.')
    logger.debug(mf, 'Overlap condition number %g', cond)

    if (cond < 1./numpy.finfo(s.dtype).eps and not force_pivoted_cholesky):
        logger.info(mf, 'Using canonical orthogonalization with threshold {}'.format(threshold))
        mf._eigh = _eigh_with_canonical_orth(threshold)
    else:
        logger.info(mf, 'Using partial Cholesky orthogonalization '
                    '(doi:10.1063/1.5139948, doi:10.1103/PhysRevA.101.032504)')
        logger.info(mf, 'Using threshold {} for pivoted Cholesky'.format(cholesky_threshold))
        logger.info(mf, 'Using threshold {} to orthogonalize the subbasis'.format(threshold))
        mf._eigh = _eigh_with_pivot_cholesky(threshold, cholesky_threshold)
    return mf
remove_linear_dep = remove_linear_dep_

def _eigh_with_canonical_orth(threshold=LINEAR_DEP_THRESHOLD):
    def eigh(h, s):
        x = canonical_orth_(s, threshold)
        xhx = reduce(lib.dot, (x.conj().T, h, x))
        e, c = scipy.linalg.eigh(xhx)
        c = numpy.dot(x, c)
        return e, c
    return eigh

def _eigh_with_pivot_cholesky(threshold=LINEAR_DEP_THRESHOLD,
                              cholesky_threshold=CHOLESKY_THRESHOLD):
    def eigh(h, s):
        x = partial_cholesky_orth_(s, canthr=threshold, cholthr=cholesky_threshold)
        xhx = reduce(lib.dot, (x.conj().T, h, x))
        e, c = scipy.linalg.eigh(xhx)
        c = numpy.dot(x, c)
        return e, c
    return eigh


