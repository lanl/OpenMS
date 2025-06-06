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

import os, sys

# For code compatibility in python-2 and python-3
if sys.version_info >= (3,):
    unicode = str

import tempfile

#
# All parameters initialized before loading openms_conf.py will be overwritten
# by the dynamic importing procedure.
#

DEBUG = False
# DEBUG = getattr(__config__, "DEBUG", False)

#
# Define OpenMS (OMS) variables to default
#

MAX_MEMORY = int(os.environ.get("OpenMS_MAX_MEMORY", 4000))  # MB
TMPDIR = os.environ.get("TMPDIR", ".")
TMPDIR = os.environ.get("OpenMS_TMPDIR", TMPDIR)
VERBOSE = 3  # default logger level (logger.NOTE)
UNIT = "angstrom"
RGPARSE = bool(os.getenv('ARGPARSE', False))


#MAX_MEMORY = getattr(__config__, "MAX_MEMORY", OMS_MAX_MEMORY)
#TMPDIR = getattr(__config__, "TMPDIR", OMS_TMPDIR)
#ARGPARSE = getattr(__config__, "ARGPARSE", OMS_ARGPARSE)
#VERBOSE = getattr(__config__, "VERBOSE", 3) # default logger level (logger.NOTE)
#UNIT = getattr(__config__, "UNIT", "angstrom")

#
# Define variables redefined multiple times throughout code
#

#TIGHT_GRAD_CONV_TOL = getattr(__config__, "scf_hf_kernel_tight_grad_conv_tol", True)
#LINEAR_DEP_THRESHOLD = getattr(__config__, 'scf_addons_remove_linear_dep_threshold', 1e-8)
#CHOLESKY_THRESHOLD = getattr(__config__, 'scf_addons_cholesky_threshold', 1e-10)
#FORCE_PIVOTED_CHOLESKY = getattr(__config__, 'scf_addons_force_cholesky', False)
#LINEAR_DEP_TRIGGER = getattr(__config__, 'scf_addons_remove_linear_dep_trigger', 1e-10)

#
# Loading openms_conf.py and overwriting above parameters
#
for conf_file in (
    os.environ.get("OpenMS_CONFIG_FILE", None),
    os.environ.get("OMS_CONFIG_FILE", None),
    os.path.join(os.path.abspath("."), ".openms_conf.py"),
    os.path.join(os.environ.get("HOME", "."), ".openms_conf.py"),
):
    if conf_file is not None and os.path.isfile(conf_file):
        break
else:
    conf_file = None

if conf_file is not None:
    if sys.version_info < (3, 0):
        import imp

        imp.load_source("openms.__config__", conf_file)
        del imp
    else:
        from importlib import machinery

        machinery.SourceFileLoader("openms.__config__", conf_file).load_module()
        del machinery
del (os, sys)

#
# All parameters initialized after loading openms_conf.py will be kept in the
# program.
#

class Config:
    def __init__(self):
        self.options = {}

    def add_option(self, key, val):
        self.options[key] = val

    def update_option(self, key, val):
        _val = self.options.get(key)
        if _val is None:
            raise KeyError(f"config option not found: {_val}")
        self.options[key] = val

    def get_option(self, key):
        _val = self.options.get(key)
        if _val is None:
            raise KeyError(f"config option not found: {_val}")
        return _val

    def __str__(self):
        _str = ""
        for k, v in self.options.items():
            _str += f"{k} : {v}\n"
        return _str

config = Config()

# update configurations:
# example
config.add_option("OpenMS_MAX_MEMORY", 4000)  # MB
