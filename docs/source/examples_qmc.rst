
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
>>>     basis="cc-pvdz",
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

QMC with custom Hamiltonians
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. We can setup a custom system with any number of electrons and orbitals

>>> import numpy
>>> from pyscf import gto, scf
>>>
>>> # define a custom oei and eri
>>> def get_h1_eri(n, U=2.0, t= -1.0, PBC=False):
>>>
>>>     # hopping
>>>     h1 = numpy.zeros((n, n))
>>>     for i in range(n - 1):
>>>         h1[i, i + 1] = h1[i + 1, i] = t
>>>     if PBC:
>>>         h1[n - 1, 0] = h1[0, n - 1] = t
>>>
>>>     # onsite U term
>>>     eri = numpy.zeros((n, n, n, n))
>>>     for i in range(n):
>>>         eri[i, i, i, i] = U
>>>
>>>     return h1, eri

# 1. We can setup a custom system with any number of electrons and orbi|tals

>>>
>>> n = 12
>>> filling = 0.5
>>> mol = gto.M(verbose=3)
>>> mol.nelectron = int(n * filling)
>>> mol.incore_anyway = True
>>> mol.nao_nr = lambda *args: n
>>> mol.tot_electrons = lambda *args: mol.nelectron
>>>
>>> # define custom oei and eri
>>> h1, eri = get_h1_eri(n)
>>>

2. Setup a mean-field type trial Wavefunction

>>> from pyscf import ao2mo
>>> from openms.qmc.trial import TrialHF
>>>
>>> mf = scf.RHF(mol)
>>> mf.max_cycle = 500
>>> mf.get_hcore = lambda *args: h1
>>> mf.get_ovlp = lambda *args: numpy.eye(n)
>>> mf._eri = ao2mo.restore(8, eri, n)
>>> mf.kernel()
>>>
>>> # setup trial
>>> trial = TrialHF(mol, mf=mf)
>>>

3. Setup a AFQMC object

>>>
>>> # TBA.
>>>
