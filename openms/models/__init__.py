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

