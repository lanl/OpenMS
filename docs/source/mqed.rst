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
   \hat{H}_{\textt{PF}} &= \hat{H}_{\text{e}} + \hat{H}_{\text{p}}
                         + \hat{H}_{\text{ep}} + \hat{H}_{\text{DSE}} \\
                        &= \hat{H}_{\text{e}} + \sum_{\alpha}
                           \left[ \hat{H}_{\text{p}}^{\alpha}
                                  + \hat{H}_{\text{ep}}^{\alpha}
                                  + \hat{H}_{\text{DSE}}^{\alpha}
                           \right]
where:

- :math:`\hat{H}_{\text{p}}` is the photonic Hamiltonian,
- :math:`\hat{H}_{\text{ep}}` is the bilinear electron-photon Hamiltonian,
- :math:`\hat{H}_{\text{DSE}}` is the dipole self-energy (DSE) Hamiltonian

which are dependent on the modes of the photon, :math:`\alpha`. The matter
(electronic) Hamiltonian, :math:`\hat{H}_{\text{e}}`, is:

.. math::
   \hat{H}_{\text{e}} = \sum_{pq} h_{p}^{q} \hat{a}^{\dagger}_{p} \hat{a}_{q}
                        + \frac{1}{2} \sum_{pqrs} v_{pq}^{rs}
                          \hat{a}^{\dagger}_{p} \hat{a}^{\dagger}_{q}
                          \hat{a}_{s} \hat{a}_{r}
where:

- :math:`\hat{a}^{\dagger}` and :math:`\hat{a}` are fermionic single-particle
  creation and annihilation operators, and
- :math:`\{pqrs\}` are general electron orbital indices.

------------------------------------------------------------------------------------

Expanded, the terms of the PF Hamiltonian are:

.. math::
   \hat{H}_{\text{PF}} &= \hat{H}_{e} + \sum_{\alpha}
                          \left[ \omega_{\alpha}
                                 \hat{b}^{\dagger}_{\alpha}
                                 \hat{b}_{\alpha}
                                 + \sqrt{ \frac{\omega_{\alpha}}{2} }
                                   \boldsymbol{\lambda}_{\alpha} \cdot \hat{D}
                                   ( \hat{b}^{\dagger}_{\alpha} + \hat{b}_{\alpha} )
                                 + \frac{1}{2} ( \boldsymbol{\lambda}_{\alpha} \cdot \hat{D} )^2
                          \right] \\
                       &= \hat{H}_{e} + \sum_{\alpha}
                          \left[ \omega_{\alpha}
                                 \hat{b}^{\dagger}_{\alpha}
                                 \hat{b}_{\alpha}
                                 + \sqrt{ \frac{\omega_{\alpha}}{2} }
                                   \boldsymbol{e}_{\alpha} \cdot {\lambda}_{\alpha} \cdot \hat{D}
                                   ( \hat{b}^{\dagger}_{\alpha} + \hat{b}_{\alpha} )
                                 + \frac{1}{2} ( \boldsymbol{e}_{\alpha} \cdot {\lambda}_{\alpha}
                                                 \cdot \hat{D} )^2
                          \right] \\
where:

- :math:`\hat{b}^{\dagger}_{\alpha}` and :math:`\hat{b}_{\alpha}` are bosonic
  creation and annihilation operators of photon mode :math:`\alpha`,
- :math:`\omega_{\alpha}` is the frequency of the photon mode,
- :math:`\lambda_{\alpha} = \sqrt{ \frac{1}{\epsilon V_{\alpha}} }` is the
  amplitude/coupling strength of the photon mode,
- :math:`\boldsymbol{e}_{\alpha}` is the unit vector of the photon mode,
- :math:`V` is the cavity volume, and
- :math:`\hat{D}` is the molecular dipole operator
  (electronic DOFs, can also include nuclear DOFs).

QED-HF Wavefunction
-------------------

The QED-HF reference wavefunction ansatz is:

.. math::

   \ket{\Psi_{\text{QED-HF}}} = \ket{\Psi_{\text{HF}}}
                                \otimes \ket{0_{\text{p}}}
where :math:`\ket{\Psi_{\text{HF}}}` is the non-QED/HF wavefunction and
:math:`\ket{0_{\text{p}}}` are zero photon states:

.. math::
   \ket{0_{\text{p}}} &= \prod_{\alpha}
                         \left[ \sum_{n} \ket{n}
                         \right] \\
                      &= \prod_{\alpha}
                         \left[ \boldsymbol{C}^{n}_{\alpha}
                                \sum_{n} ( \hat{b}^{\dagger}_{\alpha} )^{n}
                                \ket{0}
                         \right]
in which each photon mode :math:`\alpha` is expressed in terms of
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
   \ket{\Psi_{\text{CS-QED-HF}}} &= \prod_{\alpha}
                                    e^{z_{\alpha}
                                       ( \hat{b}_{\alpha}
                                         - \hat{b}^{\dagger}_{\alpha} )
                                      } \ket{\Psi_{\text{QED-HF}}} \\
                                 &= \prod_{\alpha}
                                    e^{z_{\alpha} \hat{b}_{\alpha}
                                       - {z^{*}_{\alpha}} \hat{b}^{\dagger}_{\alpha}
                                      } \ket{\Psi_{\text{QED-HF}}} \\
                                 &= \boldsymbol{U_{z}} \ket{\Psi_{\text{QED-HF}}}
where :math:`z_{\alpha}` denotes the displacement due to the coupling of
mode :math:`\alpha` with the electrons of the molecular system:

.. math::
   z_{\alpha} = \sum_{\alpha} \frac{\lambda_{\alpha} \cdot \mel*{\mu}{\hat{D}}{\nu}}
                                   {\sqrt{ 2 \omega_{\alpha} }}

Consequently, :math:`\boldsymbol{U_z}` also transforms the original PF Hamiltonian,
:math:`\boldsymbol{U}_{\boldsymbol{z}} \hat{H}_{\text{PF}}`
:math:`\boldsymbol{U}^{\dagger}_{\boldsymbol{z}}`, to form CS Hamiltonian,
:math:`\hat{H}_{CS}`:

.. math::
   \hat{H}_{CS} = \hat{H}_{\text{e}}
                  &+ \sum_{\alpha} \omega_{\alpha}
                     \hat{b}^{\dagger}_{\alpha} \hat{b}_{\alpha} \\
                  &- \sum_{\alpha} \sqrt{ \frac{\omega_{\alpha}}{2} }
                     \mel*{\mu}{\boldsymbol{\lambda}_{\alpha}
                                \cdot ( \hat{D}
                                        - \ev*{\hat{D}}_{\mu\nu} )}{\nu}
                     ( \hat{b}^{\dagger}_{\alpha} + \hat{b}_{\alpha} ) \\
                  &+ \sum_{\alpha} \frac{1}{2}
                     \mel*{\mu}{[ \boldsymbol{\lambda}_{\alpha}
                                  \cdot ( \hat{D}
                                          - \ev*{\hat{D}}_{\mu\nu} ) ]^2}{\nu}

In the CS representation, the QED-HF energy is subject to an DSE-dependent
energy difference:

.. math::
   E_{\text{QEDHF}} = E_{HF}
                      + \frac{1}{2} \sum_{\alpha}
                        \mel*{\mu}
                             {[ \boldsymbol{\lambda}_{\alpha}
                                \cdot ( \hat{D}
                                        - \ev*{\hat{D}}_{\mu\nu}) ]^2}{\nu}

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

:class:`RHF <mqed.scqedhf.RHF>` class definition
================================================

.. autofunction:: mqed.scqedhf.kernel
.. autofunction:: mqed.scqedhf.get_orbitals_from_rao
.. autofunction:: mqed.scqedhf.cholesky_diag_fock_rao
.. autofunction:: mqed.scqedhf.get_reduced_overlap

.. autoclass:: mqed.scqedhf.RHF
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

:class:`RHF <mqed.scqedhf.RHF>` class definition
================================================

.. autoclass:: mqed.vtqedhf.RHF
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
