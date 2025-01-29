
.. module:: openms.nqs

Neural network quantum states (NQS)
***********************************

.. automodule:: openms.nqs
   :members:
   :undoc-members:
   :show-inheritance:


Theoretical background
======================

The NQS ansatz is:

.. math::
   \ket{\Psi(\boldsymbol{\theta})} = e^{\sum^N_i a_i \sigma_i}\prod^M_j 2\cosh\left(b_j + \sum^N_i W_{ij}\sigma_i\right)\ket{\mathcal{S}},

where :math:`\ket{\mathcal{S}}=\ket{\sigma_1, \cdots, \sigma_N}` represents the :math:`S_z` spin configuration basis.
:math:`\boldsymbol{\theta}=(a_i, b_j, W_{ij})` represents the complex-valued network parameters.
:math:`N` is the number of spin orbital and :math:`M` is the number of hidden units.
The parameters :math:`\boldsymbol{\theta}` are optimized to minimize the variational energy.
This ansatz in above equation can be optimized with VMC techniques. Optimization of the wavefunction ansatz
typically relies on the stochastic reconfiguration (SR) approach :cite:`Sorella:1998vk`.
