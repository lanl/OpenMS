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

# Basic cavity class

class CavityBase(object):

    def __init__(self, mole_obj=None):

        self.molecule = mole_obj
        self.dip_ov = None

    def cavity_frequency(self):
        '''
        return the cavity frequencies
        '''
        raise NotImplementedError

    def build_cavity(self):
        #implemented in subclass
        raise NotImplementedError




class abinitCavity(CavityBase):

    def __init__(self, driver, cavity_struct=None):
        '''
        driver: method of computing the cavity
        cavity_struct: geometry of the cavity
        '''
        super().__init__()
        
        self.structure = cavity_struct


    def build_cavity(self, dipole=None):

        '''
        update cavity according to the new dipole 
        '''


