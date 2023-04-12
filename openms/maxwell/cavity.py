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


