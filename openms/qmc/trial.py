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

# QMC module (TBA)

import sys
from pyscf import tools, lo, scf, fci
import numpy as np
import h5py

# files for trial WF and random walkers

class TrialWFBase(object):
    r"""
    Base class for trial wavefunction
    """

    def __init__(self,
                 mol,
                 #ne: Tuple[int, int],
                 #n_mo : int,
                 mf = None,
                 numdets = 1,
                 numdets_props = 1,
                 numdets_chunks = 1,
                 verbose = 1):

        self.mol = mol
        if mf is None:
            mf = scf.RHF(self.mol)
            mf.kernel()
        self.mf = mf

        #self.num_elec = num_elec # number of electrons
        #self.n_mo = n_mo
        # only works for spin-restricted reference at this moment
        #self.nalpha = self.nbeta = self.num_elec // 2

        self.numdets = numdets
        self.numdets_props = numdets_props
        self.numdets_chunks = numdets_chunks

        self.build()

    def build(self):
        r"""
        build initial trial wave function
        """
        pass

# single determinant HF trial wavefunction
class TrialHF(TrialWFBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def build(self):

        overlap = self.mol.intor('int1e_ovlp')
        ao_coeff = lo.orth.lowdin(overlap)
        xinv = np.linalg.inv(ao_coeff)

        self.wf = self.mf.mo_coeff
        self.wf = xinv.dot(self.mf.mo_coeff[:, :self.mol.nelec[0]])


# define walker class

class WalkerBase(object):
    r"""
    Walker Base class
    """
    def __init__(self, trial):

        self.trial = trial

    def build(self):

        pass



