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

# this function implemens the interface to built-in C FDTD code
from openms import lib

# deprecated as we use swig to interface with the c FDTD code

libpfdtd = lib.load_library('libfdtd')
def _fpointer(name):
    return ctypes.c_void_p(_ctypes.dlsym(libfdtd._handle, name))

class FDTDC(object):

    def __init__(self, args, kwargs):

        # do something
        # take x, y, z from args
        x = 1.0
        y = 1.0
        z = 1.0

        # todo
        self._this = ctypes.POINTER(_CVHFOpt)()

        self.structure_size(x,y,z)

    def structure_size(self, x, y, z):

        #TBA
        return None

    def lattice_size(self,lx, ly, lz):


        return None

    def pml_size(self):

        return None

    def set_default_parameter(self):

        return None

    def memory(self):

        return None

    #def real_space_param(self, a=1.0, WC):
    #
    #    return None

    def make_epsilon(self):

        return None

    def background(self):

        return None

    def make_metal_structure(self):

        return None

    def coefficient(self):

        return None

    def out_epsilon(self, plane="x", value=0, output="epsilon.x"):

        return None


