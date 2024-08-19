"""
DIIS
"""

from functools import reduce
import numpy
import scipy.linalg
import scipy.optimize
from pyscf import lib
from pyscf.lib import logger

DEBUG = False

class CDIIS(lib.diis.DIIS):
    """
    General DIIS for optimizing a with function f(a) and gradient df/da.
    Modified from PySCF CDIIS class:
        1) adds :param:`err_func` to constructor function.
        2) :attr:`self.damp` is removed.
        3) :func:`update` checks for variational parameters and their gradients,
           :param:`var` and :param:`var_grad`.
        4) :param:`var_grad` is included in the construction of the error vector
           in :func:`get_err_vec`, if it is provided.

    YZ: It may be better to move the error vector construction into each electronic
        structure solver (as they are different in various solvers). Then we don't
        need this class in the end.
    """

    def __init__(self, mf=None, filename=None, Corth=None, err_func=None):
        lib.diis.DIIS.__init__(self, mf, filename)
        self.rollback = 0
        self.space = 8
        self.Corth = Corth
        self.err_func = err_func

    def update(self, s, d, f, *args, **kwargs):
        var = None
        var_grad = None
        if 'var' in kwargs:
            var = kwargs['var']
        if 'var_grad' in kwargs:
            var_grad = kwargs['var_grad']

        errvec = get_err_vec(s, d, f, self.Corth, var_grad)
        logger.debug1(self, 'diis-norm(errvec)=%g', numpy.linalg.norm(errvec))
        params = numpy.hstack([f.ravel(), var.ravel()])
        xnew = lib.diis.DIIS.update(self, params, xerr=errvec)
        if self.rollback > 0 and len(self._bookkeep) == self.space:
            self._bookkeep = self._bookkeep[-self.rollback:]
        return xnew

    def get_num_vec(self):
        if self.rollback:
            return self._head
        else:
            return len(self._bookkeep)

SCFDIIS = SCF_DIIS = DIIS = CDIIS

def get_err_vec_orig(s, d, f):
    '''error vector = SDF - FDS'''
    if isinstance(f, numpy.ndarray) and f.ndim == 2:
        sdf = reduce(numpy.dot, (s,d,f))
        errvec = (sdf.conj().T - sdf).ravel()

    elif isinstance(f, numpy.ndarray) and f.ndim == 3 and s.ndim == 3:
        errvec = []
        for i in range(f.shape[0]):
            sdf = reduce(numpy.dot, (s[i], d[i], f[i]))
            errvec.append((sdf.conj().T - sdf).ravel())
        errvec = numpy.hstack(errvec)

    elif f.ndim == s.ndim+1 and f.shape[0] == 2:  # for UHF
        errvec = numpy.hstack([
            get_err_vec_orig(s, d[0], f[0]).ravel(),
            get_err_vec_orig(s, d[1], f[1]).ravel()])
    else:
        raise RuntimeError('Unknown SCF DIIS type')
    return errvec

def get_err_vec_orth(s, d, f, Corth):
    '''error vector in orthonormal basis = C.T.conj() (SDF - FDS) C'''
    # Symmetry information to reduce numerical error in DIIS (issue #1524)
    orbsym = getattr(Corth, 'orbsym', None)
    if orbsym is not None:
        sym_forbid = orbsym[:,None] != orbsym

    if isinstance(f, numpy.ndarray) and f.ndim == 2:
        sdf = reduce(numpy.dot, (Corth.conj().T, s, d, f, Corth))
        if orbsym is not None:
            sdf[sym_forbid] = 0
        errvec = (sdf.conj().T - sdf).ravel()

    elif isinstance(f, numpy.ndarray) and f.ndim == 3 and s.ndim == 3:
        errvec = []
        for i in range(f.shape[0]):
            sdf = reduce(numpy.dot, (Corth[i].conj().T, s[i], d[i], f[i], Corth[i]))
            if orbsym is not None:
                sdf[sym_forbid] = 0
            errvec.append((sdf.conj().T - sdf).ravel())
        errvec = numpy.hstack(errvec)

    elif f.ndim == s.ndim+1 and f.shape[0] == 2:  # for UHF
        errvec = numpy.hstack([
            get_err_vec_orth(s, d[0], f[0], Corth[0]).ravel(),
            get_err_vec_orth(s, d[1], f[1], Corth[1]).ravel()])
    else:
        raise RuntimeError('Unknown SCF DIIS type')
    return errvec

def get_err_vec(s, d, f, Corth=None, var_grad=None):
    if Corth is None:
        errors = get_err_vec_orig(s, d, f)
    else:
        errors = get_err_vec_orth(s, d, f, Corth)

    # Include contribution of QED variational parameter to error vector
    if var_grad is not None:
        errors = numpy.hstack([errors, var_grad.ravel()])
    return errors
