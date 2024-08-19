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

import sys

# For code compatibility in python-2 and python-3
if sys.version_info >= (3,):
    unicode = str

import os
import tempfile
from pyscf import __config__

#
# All parameters initialized before loading openms_conf.py will be overwritten
# by the dynamic importing procedure.
#

DEBUG = getattr(__config__, "DEBUG", False)

#
# Define OpenMS (OMS) variables to default to if not defined by
# PySCF during import.
#

OMS_MAX_MEMORY = int(os.environ.get("OMS_MAX_MEMORY", 4000))  # MB
MAX_MEMORY = getattr(__config__, "MAX_MEMORY", OMS_MAX_MEMORY)

OMS_TMPDIR = os.environ.get('OMS_TMPDIR', tempfile.gettempdir())
TMPDIR = getattr(__config__, "TMPDIR", OMS_TMPDIR)

OMS_ARGPARSE = bool(os.getenv('OMS_ARGPARSE', False))
ARGPARSE = getattr(__config__, "ARGPARSE", OMS_ARGPARSE)

VERBOSE = getattr(__config__, "VERBOSE", 3) # default logger level (logger.NOTE)
UNIT = getattr(__config__, "UNIT", "angstrom")

#
# Define variables redefined multiple times throughout code
#

TIGHT_GRAD_CONV_TOL = getattr(__config__, "scf_hf_kernel_tight_grad_conv_tol", True)
LINEAR_DEP_THRESHOLD = getattr(__config__, 'scf_addons_remove_linear_dep_threshold', 1e-8)
CHOLESKY_THRESHOLD = getattr(__config__, 'scf_addons_cholesky_threshold', 1e-10)
FORCE_PIVOTED_CHOLESKY = getattr(__config__, 'scf_addons_force_cholesky', False)
LINEAR_DEP_TRIGGER = getattr(__config__, 'scf_addons_remove_linear_dep_trigger', 1e-10)

#
# Loading openms_conf.py and overwriting above parameters
#

for conf_file in (os.environ.get("OpenMS_CONFIG_FILE", None),
                  os.environ.get("OMS_CONFIG_FILE", None),
                  os.path.join(os.path.abspath("."), ".openms_conf.py"),
                  os.path.join(os.environ.get("HOME", "."), ".openms_conf.py")):
    if conf_file is not None and os.path.isfile(conf_file):
        break
else:
    conf_file = None

if conf_file is not None:
    if sys.version_info < (3, 0):
        with open(conf_file, 'r') as f:
            exec(f.read())
        del f
    else:
        from importlib.machinery import SourceFileLoader

        SourceFileLoader("openms.__config__", conf_file).exec_module()
        del SourceFileLoader
del (os, sys, tempfile)

#
# All parameters initialized after loading openms_conf.py will be kept in the
# program.
#
