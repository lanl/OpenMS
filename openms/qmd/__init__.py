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

'''
Molecular Dynamics
==================

Simple usage::
TBA
'''

import openms.lib.backend as bd
import numpy as np # replace np as bd (TODO)
from openms import __config__


# Grabs the global SEED variable and creates the random number generator
SEED = getattr(__config__, 'SEED', None)
rng = np.random.Generator(np.random.PCG64(SEED))

from .tsh import TrajectorySurfaceHopping as SH
from .bomd import BOMD

from .es_driver import QuantumDriver
