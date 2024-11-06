
******************
Gutzwiller solvers
******************

Theoretical background
======================

Gutzwiller Wavefunction is defiend as

.. math::
    \ket{\Psi_G} = \mathcal{P} \ket{\Psi_0}

where :math:`\mathcal{P}` is the local projector operator that improves the
noninteracting wave function :math:`\ket{\Psi_0}` according to the on-site
interaction by modifying the weight of local electronic configurations:

.. math::
    \mathcal{P} =& \prod_I \mathcal{P}_I \\
    \mathcal{P}_I =  &\sum_{\Gamma\Gamma'} \lambda_{I,\Gamma\Gamma'} \ket{I,\Gamma}\bra{I,\Gamma'}

where :math:`I` is the index of site. :math:`\ket{I,\Gamma}` denotes the local
configurations of site :math:`I`.

Gutzwiller constraints:

.. math::
    \bra{\Psi_0} \mathcal{P}^\dagger \mathcal{P}_I \ket{\Psi_0}  & = 1,\\
    \bra{\Psi_0} \mathcal{P}^\dagger \mathcal{P}_I \hat{n}_I \ket{\Psi_0}  & = \bra{\Psi_0} \hat{n}_I\ket{\Psi_0},\\

where :math:`\hat{n}_I` is the local single-particle density-matrix operator.

.. automodule:: openms.gwf
   :members:
   :undoc-members:
   :show-inheritance:


Symbols used in the documents:

  - :math:`\mathcal{N}` is the number of sites
  - :math:`N_I` is the number of orbitals in site :math:`I`.
  - :math:`I, J` represent the indices of the fragments (or sites) of the system
  - :math:`\alpha, \beta, \gamma, \sigma` denote the fermionic orbital.
  - :math:`\hat{T}_{IJ}`  is the hopping operator between two sites :math:`I, J` and
    :math:`T_{I\alpha J\beta}` is corresponding hopping integral
  - :math:`\hat{H}^{loc}_I` represents the local Hamiltonain of site :math:`I`.
  - :math:`c_{Ip}, c^\dagger_{Iq}` represents the original fermonic operator
  - :math:`f_{Ia}` represents the auxiliary operator



Single-band Gutzwiller method
=============================

.. automodule:: openms.gwf.ga_sband
   :members:
   :undoc-members:
   :show-inheritance:

Multi-band Gutzwiller method for local correlation
==================================================

.. automodule:: openms.gwf.ga_local
   :members:
   :undoc-members:
   :show-inheritance:

Multi-band Gutzwiller method for nonlocal correlation
=====================================================

.. automodule:: openms.gwf.ga_nonlocal
   :members:
   :undoc-members:
   :show-inheritance:

Multi-band Gutzwiller method for electron-boson interactions
============================================================

.. automodule:: openms.gwf.ga_eph
   :members:
   :undoc-members:
   :show-inheritance:
