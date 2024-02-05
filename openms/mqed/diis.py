

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



# General DIIS for optimizing a with funciton f(a) and gradient df/da

# YZ: It's may be better to move the error vector construction into each electronic
# structure solvers (as they are different in various solvers), and we don't need
# this class in the end.

# modified from pyscf diis

class CDIIS(lib.diis.DIIS):
    def __init__(self, mf=None, filename=None, Corth=None, err_func=None):
        lib.diis.DIIS.__init__(self, mf, filename)
        self.rollback = 0
        self.space = 8
        self.Corth = Corth
        self.err_func = err_func
        #?self._scf = mf
        #?if hasattr(self._scf, 'get_orbsym'): # Symmetry adapted SCF objects
        #?    self.orbsym = mf.get_orbsym(Corth)
        #?    sym_forbid = self.orbsym[:,None] != self.orbsym

    def update(self, s, d, f, *args, **kwargs):
        var = None
        var_grad = None
        if 'var' in kwargs:
            var = kwargs['var']
        if 'var_grad' in kwargs:
            var_grad = kwargs['var_grad']
        errvec = get_err_qed(s, d, f, self.Corth, var, var_grad)
        #if var is not None:
        params = numpy.hstack([f.ravel(), var.ravel()])
        logger.debug1(self, 'diis-norm(errvec)=%g', numpy.linalg.norm(errvec))
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

def get_err_qed(s, d, f, Corth=None, var=None, var_grad=None):
    """get the error vector of qed variational parameters"""
    errors = get_err_vec(s, d, f, Corth)
    if var_grad is not None:
        errors = numpy.hstack([errors, var_grad.ravel()])
    return errors

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

def get_err_vec(s, d, f, Corth=None):
    if Corth is None:
        return get_err_vec_orig(s, d, f)
    else:
        return get_err_vec_orth(s, d, f, Corth)

