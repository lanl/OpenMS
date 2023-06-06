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

from openms.lib import backend

class TDNEGF(object):

    def __init__(self, system, baths=None, dt=0.01, tmax=1.e2):
        '''
        sys:
        bath: [List]
        '''

        self.sys = system
        self.baths = baths
        self.dt = dt
        self.tmax = tmax


    def self_energies(self):
        '''
        evaluate self-energies due to sys-bath coupling

        '''

        return None


    def propagation(self):

        '''
        main driver of propagating the reduced density matrix
        '''

        return None


    def get_currents(self):
        '''
        return currents
        '''

        return None


