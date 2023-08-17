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

#
# All parameters initialized before loading openms_conf.py will be overwritten
# by the dynamic importing procedure.
#

DEBUG = False

MAX_MEMORY = int(os.environ.get("OpenMS_MAX_MEMORY", 4000))  # MB
TMPDIR = os.environ.get("TMPDIR", ".")
TMPDIR = os.environ.get("OpenMS_TMPDIR", TMPDIR)

VERBOSE = 3  # default logger level (logger.NOTE)
UNIT = "angstrom"

#
# Loading openms_conf.py and overwriting above parameters
#
for conf_file in (
    os.environ.get("OpenMS_CONFIG_FILE", None),
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
