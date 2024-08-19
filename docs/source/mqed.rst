**************************************
:mod:`mqed`: **Molecular QED solvers**
**************************************

.. automodule:: openms.mqed
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: HF

########################################
:mod:`~mqed.qedhf`: **QED Hartree-Fock**
########################################

Theory and Background
=====================

Pauli-Fierz Hamiltonian
-----------------------

The light-matter Hamiltonian of molecular quantum electrodynamics (mQED)
is the Pauli-Fierz (PF) Hamiltonian:

.. math::
   \hat{H}_{\tt{PF}} &= \hat{H}_{\tt{e}} + \hat{H}_{\tt{p}}
                      + \hat{H}_{\tt{ep}} + \hat{H}_{\tt{DSE}} \\
                     &= \hat{H}_{\tt{e}} + \sum_\al \left[
                        \hat{H}_{\tt{p}}^{\al}
                        + \hat{H}_{\tt{ep}}^{\al}
                        + \hat{H}_{\tt{DSE}}^{\al} \right]
where:

- :math:`\hat{H}_{\tt{p}}` is the photonic Hamiltonian,
- :math:`\hat{H}_{\tt{ep}}` is the bilinear electron-photon Hamiltonian,
- :math:`\hat{H}_{\tt{DSE}}` is the dipole self-energy (DSE) Hamiltonian

which are dependent on the modes of the photon, :math:`\al`. The matter
(electronic) Hamiltonian, :math:`\hat{H}_{\tt{e}}`, is:

.. math::
   \hat{H}_{\tt{e}} = \sum_{pq} \h{p}{q} \cf{p} \af{q}
                      + \frac{1}{2} \sum_{pqrs}
                        \v{pq}{rs} \cf{p} \cf{q} \af{s} \af{r}
where:

- :math:`\cf{}` and :math:`\af{}` are fermionic single-particle
  creation and annihilation operators, and
- :math:`\{pqrs\}` are general electron orbital indices.

------------------------------------------------------------------------------------

Expanded, the terms of the PF Hamiltonian are:

.. math::
   \hat{H}_{\tt{PF}} &= \hat{H}_e + \sum_\al
                        \left[
                        \om_\al \cb{\al} \ab{\al}
                        + \sqrt{\frac{\om_\al}{2}}\bm{\la}_\al
                          \cdot \hat{D} (\cb{\al} + \ab{\al})
                        + \frac{1}{2} (\bm{\la}_\al \cdot \hat{D})^2
                        \right] \\
                     &= \hat{H}_e + \sum_\al
                        \left[
                        \om_\al \cb{\al} \ab{\al}
                        + \sqrt{\frac{\om_\al}{2}} \bm{e}_{\al}
                          \cdot \la_\al \cdot \hat{D} (\cb{\al} + \ab{\al})
                        + \frac{1}{2} (\bm{e}_{\al} \cdot \la_\al
                          \cdot \hat{D})^2
                        \right]
where:

- :math:`\cb{\al}` and :math:`\ab{\al}` are bosonic
  creation and annihilation operators of photon mode :math:`\al`,
- :math:`\om_\al` is the frequency of the photon mode,
- :math:`\la_\al = \sqrt{\frac{1}{\epsilon V_\al}}` is the
  amplitude/coupling strength of the photon mode,
- :math:`\bm{e}_\al` is the unit vector of the photon mode,
- :math:`V` is the cavity volume, and
- :math:`\hat{D}` is the molecular dipole operator
  (electronic DOFs, can also include nuclear DOFs).

QED-HF Wavefunction
-------------------

The QED-HF reference wavefunction ansatz is:

.. math::

   \ket{\Psi_{\tt{QED-HF}}} = \ket{\Psi_{\tt{HF}}}
                              \otimes \ket{0_{\tt{p}}}
where :math:`\ket{\Psi_{\tt{HF}}}` is the non-QED/HF wavefunction and
:math:`\ket{0_{\tt{p}}}` are zero photon states:

.. math::
   \ket{0_{\tt{p}}} &= \prod_\al \left[
                       \sum_{n} \ket{n} \right] \\
                    &= \prod_\al \left[
                       \bm{C}^n_{\al} \sum_{n}
                       (\cb{\al})^n \ket{0} \right]
in which each photon mode :math:`\al` is expressed in terms of
:math:`n`-normalized photon number states, :math:`\ket{n}`, each of
which are defined in terms of the photon vacuum state, :math:`\ket{0}`.

The QED-HF energy can be evaluated self-consistently after modifying the HF
one- and two-electon integrals, as detailed in :meth:`~mqed.qedhf.get_hcore`
and :meth:`~mqed.qedhf.get_jk`. These two functions make calls to functions
:meth:`~lib.boson.Photon.get_dse_hcore` and :meth:`~lib.boson.Photon.get_dse_jk`,
respectively.

Coherent-State Representation
-----------------------------

The coherent-state (CS) representation is achieved by transforming the
PF Hamiltonian above:

.. math::
   \ket{\Psi_{\tt{CS-QED-HF}}} &= \prod_\al
                                  e^{z_\al (\ab{\al} - \cb{\al})}
                                  \ket{\Psi_{\tt{QED-HF}}} \\
                               &= \prod_\al
                                  e^{z_\al \ab{\al} - {z^{*}_{\al}} \cb{\al}}
                                  \ket{\Psi_{\tt{QED-HF}}} \\
                               &= \bm{U_z} \ket{\Psi_{\tt{QED-HF}}}
where :math:`z_\al` denotes the displacement due to the coupling of
mode :math:`\al` with the electrons of the molecular system:

.. math::
   z_{\al} = \sum_\al \frac{\la_\al \cdot \mel*{\mu}{\hat{D}}{\nu}}
                           {\sqrt{2 \om_\al}}

Consequently, :math:`\bm{U_z}` also transforms the original PF Hamiltonian,
:math:`\bm{U}_{\bm{z}} \hat{H}_{\tt{PF}} \bm{U}^\dagger_{\bm{z}}`, to form
CS Hamiltonian, :math:`\hat{H}_{CS}`:

.. math::
   \hat{H}_{CS} = \hat{H}_{\tt{e}}
                &+ \sum_\al \om_\al \cb{\al} \ab{\al} \\
                &- \sum_\al \sqrt{\frac{\om_\al}{2}}
                            \mel*{\mu}{\bm{\la}_\al
                            \cdot (\hat{D} - \ev*{\hat{D}}_{\mu\nu})}{\nu}
                            (\cb{\al} + \ab{\al}) \\
                &+ \sum_\al \frac{1}{2}
                   \mel*{\mu}{[\bm{\la}_\al
                            \cdot (\hat{D} - \ev*{\hat{D}}_{\mu\nu})]^2}{\nu}

In the CS representation, the QED-HF energy is subject to an DSE-dependent
energy difference:

.. math::
   E_{\tt{QEDHF}} = E_{HF}
                  + \frac{1}{2} \sum_\al
                    \mel*{\mu}
                         {[ \bm{\la}_\al \cdot (\hat{D} - \ev*{\hat{D}}_{\mu\nu})]^2}
                         {\nu}
which is can also be added by modifiying the one-electron integrals. This is
explained in more detail here: :meth:`~lib.boson.Photon.get_dse_hcore`.

:class:`~mqed.qedhf.RHF` class definition
=========================================

.. autofunction:: mqed.qedhf.kernel
.. autofunction:: mqed.qedhf.get_fock

.. autoclass:: mqed.qedhf.RHF
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: kernel, get_fock, get_h1e_DO, grad_var_params, norm_var_params, init_var_params, pre_update_var_params, update_var_params

###################################
:mod:`~mqed.scqedhf`: **SC-QED-HF**
###################################

Theory and Background
=====================

WIP

Polaron Transformation
----------------------

WIP

:class:`RHF <mqed.scqedhf.SCRHF>` class definition
==================================================

.. autofunction:: mqed.scqedhf.kernel
.. autofunction:: mqed.scqedhf.get_orbitals_from_rao
.. autofunction:: mqed.scqedhf.cholesky_diag_fock_rao
.. autofunction:: mqed.scqedhf.get_reduced_overlap

.. autoclass:: mqed.scqedhf.SCRHF
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: kernel, get_fock

###################################
:mod:`~mqed.vtqedhf`: **VT-QED-HF**
###################################

Theory and Background
=====================

WIP

Variational Transformation
--------------------------

WIP

:class:`RHF <mqed.scqedhf.VTRHF>` class definition
==================================================

.. autoclass:: mqed.vtqedhf.VTRHF
   :members:
   :undoc-members:
   :show-inheritance:

****************************************
:mod:`mqed`: **Multiscale mQED solvers**
****************************************

This is the multiscale QED module for polariton chemistry. WIP.

####################################################
:class:`~mqed.ms_qedhf.MSRHF`: **Multiscale QED-HF**
####################################################

.. autoclass:: mqed.ms_qedhf.MSRHF
   :members:
   :undoc-members:
   :show-inheritance:
