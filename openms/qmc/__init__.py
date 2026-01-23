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
A collection of QMC solvers for electron-boson interactions.

"""

#from openms.qmc import qmc
#from openms.qmc import afqmc
#from openms.qmc import trial
#from . import propagator

#def AFQMC(mol, *args):
#    return afqmc.AFQMC(mol, *args)

# symbols used in this folder
# i, j, k, l: molecular orbitals
# p, q, r, s: atomic/spin orbital (either AO or OAO)
# n: index for cholesky
# z: index for walker

# TODO list:
# 1) make the TrialWF classes independent of mf object, only need to pass the mo_coefficients to
#    construct the trial WF.
# 2) MultiSD trial
# 3) remove the dependence on the mol/boson object, make it easier to overwrite the integrals.
#    basically, we only need the integrals (or Hamiltonians) for the propagation.
#    Removing the dependence on the mol/boson object will make it easier to incoporate any system, like many molecules
# 3) QMC for interacting bosons
# 4) Free-projection QMC.

import os

try:
    from openms.lib import _qmclib
    QMCLIB_AVAILABLE = True
except ImportError:
    QMCLIB_AVAILABLE = False

# may move to backend module
try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

# from openms.qmc import bp

# backends that can be used:
_AVAILABLE_BACKENDS = []

if QMCLIB_AVAILABLE:
    _AVAILABLE_BACKENDS.append("qmclib")
if NUMBA_AVAILABLE:
    _AVAILABLE_BACKENDS.append("numba")

# numpy backend is always available
_AVAILABLE_BACKENDS.append("numpy")

# public alias mapping: so we can pass -> canonical name
_BACKEND_ALIASES = {
    "qmclib": "qmclib",
    "cpp": "qmclib",        # <--- alias
    "numba": "numba",
    "numpy": "numpy",
    "np": "numpy",          # <-- alias
}

def _normalize_backend(name: str) -> str:
    name = name.lower()
    if name not in _BACKEND_ALIASES:
        raise ValueError(
            f"Unknown backend '{name}'. "
            f"Valid names: {sorted(_BACKEND_ALIASES.keys())}"
        )
    canonical = _BACKEND_ALIASES[name]
    if canonical not in _AVAILABLE_BACKENDS:
        raise ValueError(
            f"Backend '{name}' (canonical '{canonical}') is not available. "
            f"Available: {_AVAILABLE_BACKENDS}"
        )
    return canonical


# default backend: env var > first available
_env_backend = os.getenv("OPENMS_QMC_BACKEND", "").lower()

if _env_backend:
    _CURRENT_BACKEND = _normalize_backend(_env_backend)
else:
    # priority: qmclib > numba > numpy
    for candidate in ("qmclib", "numba", "numpy"):
        if candidate in _AVAILABLE_BACKENDS:
            _CURRENT_BACKEND = candidate
            break


def get_backend() -> str:
    """Return the canonical name of the currently selected QMC backend."""
    return _CURRENT_BACKEND


def set_backend(name: str) -> None:
    """
    Set the QMC backend explicitly.

    Parameters
    ----------
    name : {'qmclib', 'cpp', 'numba', 'numpy', 'np'}
    """
    global _CURRENT_BACKEND
    _CURRENT_BACKEND = _normalize_backend(name)
