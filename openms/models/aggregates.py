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
        self.Nexc = Nexc

        self.epsilon = epsilon
        self.hopping = hopping
        self.sigma = sigma
        self.zeta = zeta

        self.A = bd.zeros((Nsite, Nsite))
        self.En = bd.zeros(Nsite)

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
        '''

        return None

    def linearabs(self):

        '''
        compute the linear absorption
        '''

        return None

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

