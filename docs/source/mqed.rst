Multiscale QED solvers
======================

This is the multisclae QED moduel for polariton chemistry:
(TBA)

Theoretical background of Multsicale QED solvers (mQED)
-------------------------------------------------------

This is a collection of solvers, including mean-field and many-body methods, for solving the QED
Hamiltonian


.. math::

  \hat{H} = & \hat{H}_M + \sum_\alpha \omega_\alpha b^\dagger b + \sum_\alpha \boldsymbol{D}\cdot \boldsymbol{\lambda}_\alpha (b^\dagger+b) + \frac{1}{2}\sum_\alpha (\boldsymbol{D}\cdot \boldsymbol{\lambda}_\alpha)^2\\
          = & \hat{H}_M + \sum_\alpha \omega_\alpha b^\dagger b + \sum_\alpha \lambda_\alpha  \boldsymbol{D}\cdot \boldsymbol{e}_\alpha  (b^\dagger+b)
            + \frac{1}{2}\sum_\alpha (\lambda_\alpha\boldsymbol{D}\cdot \boldsymbol{e}_\alpha)^2

where

- :math:`\lambda_\alpha=\sqrt{\frac{1}{2\omega_\alpha V_\alpha}}`.
  :math:`\lambda_\alpha=\sqrt{\frac{1}{2\omega_\alpha V_\alpha}}` and
  :math:`e_\alpha` are the amplitude and unit vector of the photon mode, respectively.
- :math:`V` is the cavity volume.
- :math:`\omega` is the energy of the photon mode.
- :math:`\hat{H}_M` is the molecular Hamiltonian, including electronic (and nuclear if needed) DOFs.
- :math:`\boldsymbol{D}` is dipole operator.

Program overview
----------------

.. automodule:: openms.mqed
   :members:
   :undoc-members:
   :show-inheritance:

Self-consistent (variational) QED-HF solver
-------------------------------------------

.. automodule:: openms.mqed.scqedhf
   :members:
   :undoc-members:
   :show-inheritance:

Multiscale Hartree-Fock
-----------------------

.. automodule:: openms.mqed.mqed_hf
   :members:
   :undoc-members:
   :show-inheritance:

Multiscale Density Functional Theory
------------------------------------

.. automodule:: openms.mqed.mqed_dft
   :members:
   :undoc-members:
   :show-inheritance:

Multiscale Time-Dependent Self-Consistent Filed
-----------------------------------------------

.. automodule:: openms.mqed.mqed_tdscf
   :members:
   :undoc-members:
   :show-inheritance:

Multiscale Coupled-cluster
--------------------------

.. automodule:: openms.mqed.mqed_cc
   :members:
   :undoc-members:
   :show-inheritance:

Multiscale Equation of Motion Coupled-cluster
---------------------------------------------

.. automodule:: openms.mqed.mqed_eomcc
   :members:
   :undoc-members:
   :show-inheritance:


