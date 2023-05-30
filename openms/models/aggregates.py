# Copyright 2023. Triad National Security, LLC. All rights reserved. 
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Yu Zhang <zhy@lanl.gov>
#

'''
disordered molecular aggregates
'''

import openms.lib.backend as bd
import numpy as np # replace np as bd (TODO)
import random

def linear_spec(elist, state, evals, dip, gamma):
    """
    calculate linear spectrum
    state: list of states to be included in the spec
    evals: eigenvalues of 1 exciton states
    dip1: dipolemomnet of 1 exciton states
    """
    spectrum=np.zeros(len(elist))
    for i,e in enumerate(elist):
        tmp = 0.0
        for j in state: 
            tmp += dip[j]*dip[j]*gamma/((e-evals[j])**2+gamma**2)
        spectrum[i] = tmp
    return spectrum

def tdes(elist, state, evals, dip1, nuv, eng2, dip2, gamma):
    """ 
    state: list of single-exciton states
    evals: eigenvalues of 1 exciton states
    dip1: dipolemomnet of 1 exciton states
    nuv: number of 2-exciton states
    eng2: energy difference of 1->2 transition
    dip2: dipolement of 1->2 transition
    gamma: broadening
    """ 
    ne = len(elist)
    spec2d = np.zeros((ne,ne))
    for m in range(ne): # exitation
        for n in range(ne):  
            spec = 0.0
            Ed = elist[n]
            Ex = elist[m]
            for v1, j in enumerate(state):
                tmp = 0.0
                for uv in range(nuv):
                    deltae = eng2[v1,uv]
                    a = dip2[v1,uv]*dip2[v1,uv]*gamma
                    b = (Ed - deltae)**2 + gamma**2
                    tmp -= a/b
    
                a = dip1[j]*dip1[j]*gamma
                b = (Ed - evals[j])**2 + gamma**2
                tmp += 2.0*a/b
    
                a = dip1[j]*dip1[j]*gamma
                b = (Ex - evals[j])**2 + gamma**2
                tmp = tmp * a/b
                spec += tmp
    
            spec2d[n,m] = spec
        if m % 20 == 0: print('%8.3f percent of 2des is done' % (m/ne*100.0))
        sys.stdout.flush()
    return spec2d

def matvec(A,x):

    y = A @ x

    return y



class DMA(object):
    def __init__(self, 
            Nsite = 1,
            Nexc = 5,
            epsilon = 0.0, 
            hopping = 0.1, 
            sigma = 0.0, 
            zeta = 0.0,
            **kwargs):
        '''
        Hamiltonian:

        H = \sum_i (\epsilon_i +\delta) c^\dag_i c_i + 
            \sum_{i,i+1} (t+\delta_2) [c^\dag_{i+1}c_i + h.c.]

        Nsite : Int
           number of excitons
        Nexc : Int
           number of excited states of interest
        epsilon : float
           onsite energy
        hopping : float
           hopping parameter
        sigma : float
           onsite disorder
        zeta : float
           hopping disorder

        Kwargs:
             TBA

        Examples:
            TBA
        '''

        self.Nsite = Nsite
        self.Nexc = min(Nexc, Nsite)

        self.epsilon = epsilon
        self.hopping = hopping
        self.sigma = sigma
        self.zeta = zeta

        self.A = bd.zeros((Nsite, Nsite))
        self.En = bd.zeros(Nsite)
        self.excdipole1 = None
        self.dipsortidx = None
        self.c0 = 0.80
        self.c1 = 0.50
        self.c2 = 0.30

    def kernel(self):
        '''
        Compute the low-lying states
        '''

        for i in range(self.Nsite):

            self.En[i] = random.gauss(self.epsilon, self.sigma*abs(self.hopping))
            self.A[i][i] = self.En[i]
            if i < self.Nsite - 1:
                self.A[i][i+1] = self.hopping
                self.A[i+1][i] = self.hopping
        
        self.evals, self.evecs = np.linalg.eig(self.A)

        idx = self.evals.argsort()
        self.evals = self.evals[idx]
        self.evecs = self.evecs[:,idx]

    def energies(self):
        '''
        return the lowest Nexc states
        '''
        return self.evals[:self.Nexc]

    def dipole(self):
        '''
        compute the dipole of the lowest Nexc states
        dipole of one-exciton state
        .. math:: d_{mu}= \sum_{i} C_{\mu j} |j>
        '''
        self.excdipole1 = np.zeros(self.Nsite)
        for u in range(self.Nsite):
            for j in range(self.Nsite):
                self.excdipole1[u] += self.evecs[j,u]

    def sortdipole(self):
        '''
        sort dipole and get the index of states with largest dipole
        '''
        if self.excdipole1 is None:
            self.dipole()

        dip1norm = [abs(self.excdipole1[i]) for i in range(self.Nsite)]
        self.dipsortidx = np.asarray(dip1norm).argsort()[::-1]

    # compute the linear absorption spectrum of given listed of states
    def linearabs(self, elist=None, selected=None, gamma = 0.001):

        '''
        compute the linear absorption
        selected: a subset of selected states for spectrum calculations
        '''
        if elist is None:
            raise Exception("elist is None! please specify a list of energies for spectrum!")
        
        if self.excdipole1 is None:
            self.dipole()

        if selected is None:
            spectrum =  linear_spec(elist, range(self.Nexc), self.evals, self.excdipole1, gamma)
        else:
            spectrum =  linear_spec(elist, selected, self.evals, self.excdipole1, gamma)

        return spectrum

    def tdes(self):
        '''
        compute two-dimensional absorption spectral
        '''

        return None


if __name__ == '__main__':
    from openms.lib.backend import NumpyBackend, TorchBackend
    from openms.lib.backend import backend as bd
    from openms.lib.backend import set_backend
    set_backend("numpy")

    model = DMA(100, epsilon=0.0, hopping = 0.1, sigma = 0.01)
    model.kernel()

    print('states:', model.energies())

