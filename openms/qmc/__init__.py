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
