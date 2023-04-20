# OpenMS:

Open Source Multiscale solvers for coupled Maxwell-Schrödinger equations in Open quantum environments. Open means open source and open quantum system

## LANL-C23019 

Copyright 2023. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.

Here, Open = open quantum system, open source, open science, etc., you name it.

## Authors:

[Yu Zhang](mailto:zhy@lanl.gov)

## Features:

1) open quantum system dynamics (Liouville von Neumann equations)
2) quantum transport and dissipation
3) FDTD solver for complex EM environment
4) Multiscale solver for polariton chemistry (TBA)
5) Green's function embedded method for polariton (TBA)
6) coupled FDTD-Liouville von Neumann equations
7) interface to other packages for electronic structure, such as DFTB+, Pyscf;

Electronic structure solvers include:
a) HF
b) TDSCF
c) Coupled-cluster
d) model systems (SSH, etc.)
e) ...

Open quantum system solvers include:
Maxwell equation solvers:
a) frequency domain
c) time-domain (fdtd)
c) mode decomposition
d) ..

## Citation:

TBA


## Installation

### Install lib

```
  mkdir build && cd build
  cmake ../
  make
  make install
```

## Documentation

https://openms-lmi.readthedocs.io/en/latest/

