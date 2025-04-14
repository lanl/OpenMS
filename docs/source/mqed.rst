
.. module:: openms.mqed

Multiscale QED solvers
**********************

This is the multisclae QED moduel for polariton chemistry:
(TBA)


This is a collection of solvers, including mean-field and many-body methods, for solving the QED
Hamiltonian


.. math::

  \hat{H} = & \hat{H}_M + \sum_\alpha \omega_\alpha b^\dagger b + \sum_\alpha \sqrt{\frac{\omega_\alpha}{2}}\boldsymbol{D}\cdot \boldsymbol{\lambda}_\alpha (b^\dagger+b) + \frac{1}{2}\sum_\alpha (\boldsymbol{D}\cdot \boldsymbol{\lambda}_\alpha)^2\\
          = & \hat{H}_M + \sum_\alpha \omega_\alpha b^\dagger b + \sum_\alpha \sqrt{\frac{\omega_\alpha}{2}}\lambda_\alpha \boldsymbol{D}\cdot \boldsymbol{e}_\alpha  (b^\dagger+b)
            + \frac{1}{2}\sum_\alpha (\lambda_\alpha\boldsymbol{D}\cdot \boldsymbol{e}_\alpha)^2

where

- :math:`\lambda_\alpha=\sqrt{\frac{1}{\epsilon V_\alpha}}` and
  :math:`e_\alpha` are the amplitude and unit vector of the photon mode, respectively.
- :math:`V` is the cavity volume.
- :math:`\omega` is the energy of the photon mode.
- :math:`\hat{H}_M` is the molecular Hamiltonian, including electronic (and nuclear if needed) DOFs.
- :math:`\boldsymbol{D}` is dipole operator.


.. automodule:: openms.mqed
   :members:
   :undoc-members:
   :show-inheritance:


.. toctree::
   :maxdepth: 1

   mqed_qedhf
   mqed_scqedhf
   mqed_vtqedhf
   mqed_ms.rst
   mqed_qedcc

