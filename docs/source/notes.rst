
TODO list
=========

  - QEDHF methods with DF
  - QEDCC with DF and different references
  - GA methods for non-local correlaitons
  - GA methods for electron-boson interaction
  - QED-AFQMC with decoupled propagation
  - QED-AFQMC with different trials (MSD and different ansatz for photon)
  - QED-AFQMC with efficient tensor contraction backends
  - MPI version of FDTD and its frequency domain solver
  - Coupled trajectory NAMD
  - Efficient and scalable tensor contraction backends
  - AFQMC for many interacting bosons.

Changelog
=========

.. code release rules (or the rule of updating tags)
.. we have three digits in version number: major, minor and micro (x.x.x):
.. 1) Update the micro number if there are sigificant enhancements/fixes;
.. 2) Update minor number if there are new features
.. 3) Update major number when the minor hits 10 (e.g., 0.9.x -> 1.0.0).

(whats-new-0-2-0)=

v0.2.0 (unreleased)
-------------------

**New features**

  - VSQ-QEDHF method
  - QEDHF methods with multiple photon basis
  - GA method for single-band
  - AFQMC for fermionic system: SD trial, RHF/UHF trial and walkers
  - AFQMC for electron-boson interactions (correlated electrons coupled to non-interacting bosons)

**Enhancements:**

  - QEDHF methods with multiple photon modes and photon basis.
  - QEDHF with double electronic excitation in the coupled e-p excitation operator.

**Bug fixes:**

  - multiple bug fixes.

---

(whats-new-0-1-0)=

v0.1.0
------

**New features**

  - QEDHF method: QEDHF with Fock/CS representation;
  - SC-QEDHF method
  - VT-QEDHF method
  - QED-CC method
  - FDTD method for Maxwell's equations
  - Different backends (Torch, Numpy, TiledArray)
