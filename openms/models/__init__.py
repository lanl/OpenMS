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

r"""
This folder implements the models for electronic structure, including:

1) Nearest TB:

.. math:: 

   H0 = \sum_j \epsilon_j c^\dag_j c_j - t \sum_{j}(c^\dag_{j+1}c_j + h.c.)

2) Su–Schrieffer–Heeger (SSH):

.. math:: 
   :nowrap:

   \begin{align*}
   H &= H0 + H_{int} + H_{ph} &\\
   H_{ph} &=  &\\
   H_{int} &= &
   \end{align*}

3) Hubbard-Holstein model (HHM):

.. math::

   H = -t \sum_{js} (c^\dag_{j+1,s} c_{j,s}+ hc) + U\sum_j n_{ju} n_{jd} \\
   + g\sum_{js} (b^\dag_j + b_j) n_{j,\sigma} + omega\sum_j b^\dag_j b_j.

or

.. math::
   :nowrap:

   \begin{align*}
   H &= H0 + H_U + H_{int} + H_{ph}, &\\
   H_U &= U\sum_j n_{ju} n_{jd} &\\
   H_{int} &= g\sum_{js} (b^\dag_j + b_j) n_{j,\sigma} &\\
   H_{ph}&= \omega\sum_j b^\dag_j b_j. &
   \end{align*}
   
where :math:`s` denotes spin DOF and :math:`n_j = c^\dag_j c_j`.

4) Disordered model for molecular aggregates:

.. math::

   H = \sum_j (\epsilon+\Delta_j) c^\dag_j c_j - \sum_j (t+\Gamma_j) c^\dag_j c_j.

where :math:`\Delta_j` and :math:`\Gamma_j` are random variables.

5) Shin-Metiu Model

.. math::

   H =

"""

from openms.models import hh_model
from openms.models import hubbard
from openms.models import aggregates  # disordered molecular aggregates

from .shin_metiu import Shin_Metiu

