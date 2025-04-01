
Quantum Monte Carlo Simulations
-------------------------------

There are different ways to setup a QMC calculation. The basic ingredients of the QMC calculations
are Hamiltonian (integrals), trial WF and propagator.

.. 1. Diffusion QMC

Setup a molecule
^^^^^^^^^^^^^^^^

>>> bond = 1.5
>>> atom = f"Li 0.0    0.0     0.0; H 0.0  0.0 {bond}"
>>> mol = gto.M(
>>>     atom = atom,
>>>     basis=basis,
>>>     #basis="cc-pvdz",
>>>     unit="Angstrom",
>>>     symmetry=True,
>>>     verbose=3,
>>> )


Prepare trial WF
^^^^^^^^^^^^^^^^

The trial WF can be a signle-determinant (SD) or multi-determinants (MSD).

Example of SD trial WF:

TBA.


Example of MDS trial WF:

TBA.

Example of spin-unrestricted trail WF:

TBA.

Example of GHF trial:

TBA.



QMC for fermionic system
^^^^^^^^^^^^^^^^^^^^^^^^
If trial is not specified, a defaul trial WF of RHF will be used:

>>> afqmc = AFQMC(mol, dt=0.005, total_time=time, num_walkers=num_walkers,
>>>               energy_scheme=energy_scheme,
>>>               uhf = uhf,
>>>               verbose=3)
>>>
>>> times, energies = afqmc.kernel()


QMC for bosonic system
^^^^^^^^^^^^^^^^^^^^^^

TBA.

QMC for electron-boson interaction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TBA.
