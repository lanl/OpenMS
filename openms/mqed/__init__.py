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
QED solvers:
   - QED-HF: non-self-consistent HF method for polariton. The converged HF of bare molecules is used to compute the DSE.
   - scQEDHF: sel-consistent HF method for polariton. The DSE and MO are self-consistent updated.
   - VT-QEDHF: variational QED-HF method for light-matter interaction at arbitray coupling strength (i.e., can handle cases beyond the polariton regime) by leveraging the variational transformation.
   - MS-QEDHF: Multiscale QEDHF methods
   - mQED-TDSCF
   - mQED-CC
   - mQED-EOMCC
"""

import openms
from openms import mqed
from .qedhf import *


def HF(mol, xc=None, **kwargs):
    if mol.nelectron == 1 or mol.spin == 0:
        if xc is None:
            return mqed.RHF(mol, **kwargs)
        else:
            return mqed.RKS(mol, xc, **kwargs)
    else:
        raise NotImplementedError
