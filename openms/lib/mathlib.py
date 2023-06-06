#
# @ 2023. Triad National Security, LLC. All rights reserved.
#
#This program was produced under U.S. Government contract 89233218CNA000001 
# for Los Alamos National Laboratory (LANL), which is operated by Triad 
#National Security, LLC for the U.S. Department of Energy/National Nuclear 
#Security Administration. All rights in the program are reserved by Triad 
#National Security, LLC, and the U.S. Department of Energy/National Nuclear 
#Security Administration. The Government is granted for itself and others acting 
#on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this 
#material to reproduce, prepare derivative works, distribute copies to the 
#public, perform publicly and display publicly, and to permit others to do so.
#
# Author: Yu Zhang <zhy@lanl.gov>
#

#interface to call c or c++ math libs, such as cutensor, NCLL, etc.

import os, sys
import warnings
import ctypes
import numpy
from openms.lib import backend
import h5py
from threading import Thread
from multiprocessing import Queue, Process
from openms import __config__
from openms.lib import logger
import scipy.linalg

# load c, c++, fortran libs
def load_library(libname):
    try:
        _loaderpath = os.path.dirname(__file__)
        return numpy.ctypeslib.load_library(libname, _loaderpath)
    except OSError:
        from openms import __path__ as ext_modules
        for path in ext_modules:
            libpath = os.path.join(path, 'lib')
            if os.path.isdir(libpath):
                for files in os.listdir(libpath):
                    if files.startswith(libname):
                        return numpy.ctypeslib.load_library(libname, libpath)
        raise


#load bml lib (todo)
#libbml = lib.load_library('libbml')

SAFE_EIGH_LINDEP = getattr(__config__, 'lib_linalg_helper_safe_eigh_lindep', 1e-15)
DAVIDSON_LINDEP = getattr(__config__, 'lib_linalg_helper_davidson_lindep', 1e-14)
DSOLVE_LINDEP = getattr(__config__, 'lib_linalg_helper_dsolve_lindep', 1e-15)
MAX_MEMORY = getattr(__config__, 'lib_linalg_helper_davidson_max_memory', 2000)  # 2GB

# other math algorithms TBA

