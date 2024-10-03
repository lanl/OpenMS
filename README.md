# OpenMS:

Open Source Multiscale solvers for coupled Maxwell-Schrödinger equations in Open quantum environments. Open means open source and open quantum system

-----------------------------------------------
[![Documentation Status](https://readthedocs.org/projects/openms-lmi/badge/?version=latest)](https://openms-lmi.readthedocs.io/en/latest/?badge=latest)
[![GitHub release](https://img.shields.io/github/release/lanl/openms/all.svg)](https://github.com/lanl/OpenMS/releases)

[![Build Status](https://github.com/lanl/openms/actions/workflows/ci.yml/badge.svg)](https://github.com/lanl/OpenMS/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/lanl/OpenMS/graph/badge.svg?token=2CBUTFR93Y)](https://codecov.io/gh/lanl/OpenMS)


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

## Test

All the tests are written as unitttest cases. Use the following script to run all the tests (TBA)

```
python -m unittest discover -s tests -p 'test_*.py'
```

## Documentation

Details of installation instruction, usage, APIs, and examples can be
found in https://openms-lmi.readthedocs.io/en/latest/ or from local builds

html:
```
  cd docs
  make html
```

pdf:
```
  make latexpdf
```
