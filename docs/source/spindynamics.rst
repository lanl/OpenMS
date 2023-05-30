Spin Dynamics in spin and phonon baths
======================================

This is a module for studying the quantum dynamics of spins, in the presence of both spin and phonon baths..

Hamiltonian
-----------

The package allows one to simulate the dynamics of a central spin or multiple central spins interacting with spin and/or honon bath through the following Hamiltonian:

.. math::
    \hat H = \hat H_S + \hat H_{SB} + \hat H_{B}

Where :math:`\hat H_S` is the Hamiltonian of the central spin(s).
:math:`\hat H_{SB}=\hat H_{SP} + \hat H_{SN}` denotes interactions between central spin(s) and spin/phonon bath.
:math:`\hat H_B=\hat H_{P} + \hat H_{N}` are intrinsic spin/phonon bath interactions. 
For a single central spin, this corresponds to the following
Hamiltonian:

.. math::

        &\hat H_S = \mathbf{SDS} + \mathbf{B\gamma}_{S}\mathbf{S} \\
        &\hat H_{SN} = \sum_i \mathbf{S}\mathbf{A}_i\mathbf{I}_i \\
        &\hat H_{SP} = \sum_{\alpha} \mathbf{B} g^S_\alpha \mathbf{S} Q_\alpha \\ 
        &\hat H_{P} = \sum_{\alpha} \Omega_\alpha b^{+}_\alpha b_\alpha \\
        &\hat H_{N} = \sum_i{\mathbf{I}_i\mathbf{P}_i \mathbf{I}_i +
                      \mathbf{B}\mathbf{\gamma}_i\mathbf{I}_i} +
                      \sum_{i<j} \mathbf{I}_i\mathbf{J}_{ij}\mathbf{I}_j

Where :math:`\mathbf{S}=(\hat{S}_x, \hat{S}_y, \hat{S}_z)` are the spin operators of the central spin,
:math:`\mathbf{I}=(\hat{I}_x, \hat{I}_y, \hat{I}_z)`  are the bath spin operators,
:math:`Q_\alpha=(\hat{b}^{+} + \hat b_\alpha)``  are the bath phonon operators,
and :math:`\mathbf{B}=(B_x,B_y,B_z)` is an external applied magnetic field.

If several central spins are considered, the central spin Hamiltonian is modified as following:

.. math::

    \hat H_S = \sum_i (\mathbf{S_i D_i S_i} + \mathbf{B\gamma}_{S_i}\mathbf{S_i} + \sum_{i<j}\mathbf{S_i K_{ij} S_j})

And the spin-bath Hamiltonians become:

.. math::

    &\hat H_{SN} = \sum_{i,j} \mathbf{S}_i \mathbf{A}_{ij} \mathbf{I}_j, \\
    &\hat H_{SP} = \sum_{i\alpha}[ \mathbf{B}g^S_{i\alpha}\mathbf{S}  +  \sum_{ij,\alpha} \mathbf{S}_i \mathbf{g}^K_{ij,\alpha} \mathbf{S}_j]  Q_\alpha.

The interactions are described by the following tensors
that are either required to be input by user or can be generated
by the package itself.

- :math:`\mathbf{D}` (:math:`\mathbf{P}`)  is the self-interaction tensor of the central spin (bath spin).
  For the electron spin, the tensor corresponds to the zero-field splitting (ZFS) tensor.
  For nuclear spins corresponds to the quadrupole interactions tensor.
- :math:`\mathbf{\gamma}_i` is the magnetic field interaction tensor of the
  :math:`i`-spin describing the interaction of the spin and the external magnetic field :math:`B`.
- :math:`\mathbf{A}` is the interaction tensor between central and bath spins.
  In the case of the nuclear spin bath, it corresponds to the hyperfine couplings.
- :math:`\mathbf{J}` is the interaction tensor between bath spins.
- :math:`\mathbf{K}` is the interaction tensor between central spins.
- :math:`\mathbf{g}^K` is the phonon-mediated interaction tensor between central spins :math:`\frac{d\mathbf{K}_i}{dQ_\alpha}`.
- :math:`\mathbf{g}^S` is the phonon-mediated interaction tensor between central spins and external field :math:`\frac{d\gamma_i}{dQ_\alpha}`.


Implementation
--------------

.. automodule:: openms.spindy
   :members:
   :undoc-members:
   :show-inheritance:

