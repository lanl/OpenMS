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

r"""
.. Molecular Dynamics
.. ==================


Brief introduction to MQC
-------------------------

Born-Huang representation:

.. math::

    \Psi(r,R,t)=\sum^\infty_I \chi_I(R,t)\Phi_I(r;R).

Here, :math:`\Phi_I` signifies the electronic state, commonly depicted in an adiabatic representation, and :math:`\chi(R,t)` is the time-variant nuclear wave function linked to the :math:`I` adiabatic state.

.. math::

    H_e(r,R)\Phi_I(r;R)=E_I (R) \Phi_I(r;R).

Here, :math:`E_I(R)` symbolizes the :math:`I` potential energy surface at a certain nuclear position :math:`{R}` for the :math:`I` adiabatic state. The Born-Huang expansion is theoretically precise only when an unlimited number of electronic states are encompassed. Nonetheless, a good approximation is usually achieved with a handful of electronic states. 

Born-Oppenheimer approximation. There are several alternate versions that simplify the Born-Huang representation by restricting it to a single product. The Born-Oppenheimer approximation (BOA) is one such variant, which constrains the summation in the equation to a single steady-state electronic term:

.. math::

   \Psi(r,R,t)= \chi(R,t)\Phi_I(r;R).

Non-adiabatic dynamics beyond BOA
---------------------------------
With the Born-Oppenhermier approximation, the electronic wave function :math:`\Theta(r,R)` is expanded on the basis of adiabatic BO states, which depend on the electronic coordinates :math:`\mathbf{r}` and the nuclear coordinates :math:`\mathbf{R}(t)` according to

.. math::

    \Theta(\textbf{r},\textbf{R}) = \sum_{n=1}^{N_{st}} c_n(t)\ket{\Phi_{n}(\textbf{r},\textbf{R}(t))}. 

Within MQC scheme, the clear trajectories which are obtained by solving the classical Newton’s equations of motion (EOMs)

.. math::

   M_{A}\frac{d^2\textbf{R}_{A}}{dt^2} = -\nabla_{\textbf{R}_{A}}E(\textbf{R}),

While the eletronic WF is propagated quantum-mechanically (obtained by substituting electronic 
WF into the time-dependent Schr\"{o}dinger equation and keeping only the first-order nonadiabatic coupling terms):

.. math::

   i\hbar\frac{\partial c_{n}(t)}{\partial t} = c_{n}(t)E_{n}(\textbf{R}) - i\hbar\sum_{m}c_{m}(t)\dot{\textbf{R}}\cdot \textbf{d}_{nm}.

Here the orthogonal condition of adiabatic states :math:`\langle \Phi_n |\Phi_m \rangle = \delta_{nm}` is used. :math:`\mathbf{d}_{nm} = \langle \Psi_n|\nabla_{\textbf{R}}|\Psi_{m} \rangle`
is the nonadiabatic derivative coupling term (or nonadiabatic coupling vector, NACR). A key variable in above equation is the time-derivative nonadiabatic coupling scalar (NACT) between two adiabatic states

.. math::

  \dot{\textbf{R}}\cdot \textbf{d}_{nm} = \langle \Psi_{n}|\frac{\partial}{\partial t}|\Psi_{m}\rangle,

which is responsible for the nonadiabatic transitions between different adiabatic states and can be easily calculated with many ab initio methods for excited states.


Brief introduction to CTMQC 
---------------------------

TBA

Brief introduction to XF-based MQC
----------------------------------

TBA

Simple usage:

TBA
"""

import openms.lib.backend as bd
import numpy as np # replace np as bd (TODO)
from openms import __config__


# Grabs the global SEED variable and creates the random number generator
SEED = getattr(__config__, 'SEED', None)
rng = np.random.Generator(np.random.PCG64(SEED))

#from .tsh import TrajectorySurfaceHopping as SH
from .es_driver import QuantumDriver
from .bomd import BOMD

from .es_driver import QuantumDriver
