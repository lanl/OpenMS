#!/usr/bin/env python
# Copyright 2022-2023 The OpenMS Developers. All Rights Reserved.
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


r"""
Spin dynamics in the presence of spin-spin, hyperfine coupling,and spin-honon interactions
Following techniques are implemented:

  - Cluster correlation expansion (CCE)
  - quantum embedding for normal modes

Theoretical background of CCE
-----------------------------

TBA


Theoretical background of embedding
-----------------------------------

TBA

"""

# use qutip
try:
    import qutip
    QUTIP_AVAILABLE = True
except ImportError as e:
    print(f"Could not import 'qutip': {e}")
    QUTIP_AVAILABLE = False

from .spin import Spin
from .system import SystemSpins

