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
# Authors:   Yu Zhang    <zhy@lanl.gov>
#          Ilia Mazin <imazin@lanl.gov>
#

r"""
Extensions to the scipy.linalg module.

This file is largely-copied from the PySCF ``lib.scipy_helper.py``
and ``scf.addons.py`` modules.
"""

import numpy

from pyscf.lib import cond, logger
import pyscf.scf.addons as pyscf_addons

from openms import __config__

LINEAR_DEP_THRESHOLD = getattr(__config__, "LINEAR_DEP_THRESHOLD", 1e-8)
CHOLESKY_THRESHOLD = getattr(__config__, "CHOLESKY_THRESHOLD", 1e-10)
FORCE_PIVOTED_CHOLESKY = getattr(__config__, "FORCE_PIVOTED_CHOLESKY", False)
LINEAR_DEP_TRIGGER = getattr(__config__, "LINEAR_DEP_TRIGGER", 1e-10)


# Alias PySCF function
partial_cholesky_orth = pyscf_addons.partial_cholesky_orth_


def remove_linear_dep(mf, threshold=LINEAR_DEP_THRESHOLD,
                       lindep=LINEAR_DEP_TRIGGER,
                       cholesky_threshold=CHOLESKY_THRESHOLD,
                       force_pivoted_cholesky=FORCE_PIVOTED_CHOLESKY):
    r"""
    Removes linear dependence.

    Copy of :external:func:`remove_linear_dep_ <pyscf.scf.addons.remove_linear_dep_>`
    from PySCF, modified to also check condition number of the QED interaction matrix
    for every ``mode``, if ``qed`` is stored in mean-field parameter ``mf``.

    Depending on value of ``max_cond``, may modify the generalized eigenvalue problem
    solver stored in the mean-field object as :meth:`mf._eigh <pyscf.scf.hf.SCF._eigh>`
    to: :func:`~pyscf.scf.addons._eigh_with_canonical_orth` or
    :func:`~pyscf.scf.addons._eigh_with_pivot_cholesky`.
    """

    max_cond = 0.0
    s = mf.get_ovlp()
    scond = numpy.max(cond(s))
    if scond > max_cond: max_cond = scond

    from openms.lib import boson
    if isinstance(mf.qed, boson.Boson):
        gconds = cond(mf.qed.gmat)
        max_gcond = numpy.max(gconds)
        if max_gcond > max_cond: max_cond = max_gcond

    if max_cond < 1./lindep and not force_pivoted_cholesky:
        return mf

    logger.info(mf, 'Applying remove_linear_dep on SCF object.')
    logger.debug(mf, 'Overlap condition number %g', cond)
    for a in range(len(gconds)):
        logger.debug(mf, 'mode #%d gmat condition number %g', ((a+1), gconds[a]))

    if (max_cond < 1./numpy.finfo(s.dtype).eps and not force_pivoted_cholesky):
        logger.info(mf, 'Using canonical orthogonalization with threshold {}'.format(threshold))
        mf._eigh = pyscf_addons._eigh_with_canonical_orth(threshold)
    else:
        logger.info(mf, 'Using partial Cholesky orthogonalization '
                    '(doi:10.1063/1.5139948, doi:10.1103/PhysRevA.101.032504)')
        logger.info(mf, 'Using threshold {} for pivoted Cholesky'.format(cholesky_threshold))
        logger.info(mf, 'Using threshold {} to orthogonalize the subbasis'.format(threshold))
        mf._eigh = pyscf_addons._eigh_with_pivot_cholesky(threshold, cholesky_threshold)
    return mf
