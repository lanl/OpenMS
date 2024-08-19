*************************
:mod:`lib`: **Libraries**
*************************

.. automodule:: openms.lib
   :members:
   :undoc-members:
   :show-inheritance:

#########################
:mod:`lib.fdtd`: **FDTD**
#########################

.. automodule:: lib.fdtd
   :members:
   :undoc-members:
   :show-inheritance:

##################
:mod:`lib.backend`
##################

.. automodule:: lib.backend
   :members:
   :undoc-members:
   :show-inheritance:

###################################
:mod:`lib.boson`: **Boson classes**
###################################

This module contains the definition of the :class:`~lib.boson.Boson`
superclass and subclasses: :class:`~lib.boson.Photon`,
:class:`~lib.boson.Phonon` (WIP, not implemented).

Dipole/quadrupole moment functions
==================================

.. autofunction:: lib.boson.get_dipole_ao
.. autofunction:: lib.boson.get_quadrupole_ao

:class:`~lib.boson.Boson` class definition
==========================================

:class:`~lib.boson.Boson` is largely a template class, functionality meant to
be implemented within subclasses.

.. autoclass:: lib.boson.Boson
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: update_cs, update_boson_coeff, get_dse_hcore, get_dse_jk, get_q_lambda_ao, get_polarized_quadrupole_ao, get_quadrupole_ao, get_gmat_ao, get_polarized_dipole_ao, get_dipole_ao, get_geb_ao

:class:`~lib.boson.Photon` class definition
===========================================

.. autoclass:: lib.boson.Photon
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: fock, g_aint, g_fock, get_I, get_gmat_so, get_mos, get_omega, gint, hf_energy, kernel, mfG, tmat

:class:`~lib.boson.Phonon` class definition
===========================================

.. autoclass:: lib.boson.Phonon
   :members:
   :undoc-members:
   :show-inheritance:

############################
:mod:`lib.hippynn_es_driver`
############################

.. automodule:: lib.hippynn_es_driver
   :members:
   :undoc-members:
   :show-inheritance:

#################
:mod:`lib.logger`
#################

.. automodule:: lib.logger
   :members:
   :undoc-members:
   :show-inheritance:

##################
:mod:`lib.mathlib`
##################

.. automodule:: lib.mathlib
   :members:
   :undoc-members:
   :show-inheritance:

###############
:mod:`lib.misc`
###############

.. automodule:: lib.misc
   :members:
   :undoc-members:
   :show-inheritance:

#######################
:mod:`lib.scipy_helper`
#######################

.. automodule:: lib.scipy_helper
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: remove_linear_dep_
