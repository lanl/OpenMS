#
# @ 2023. Triad National Security, LLC. All rights reserved.
#
# This program was produced under U.S. Government contract 89233218CNA000001
# for Los Alamos National Laboratory (LANL), which is operated by Triad
# National Security, LLC for the U.S. Department of Energy/National Nuclear
# Security Administration. All rights in the program are reserved by Triad
# National Security, LLC, and the U.S. Department of Energy/National Nuclear
# Security Administration. The Government is granted for itself and others acting
# on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this
# material to reproduce, prepare derivative works, distribute copies to the
# public, perform publicly and display publicly, and to permit others to do so.
#
# Authors:  Yu Zhang    <zhy@lanl.gov>
#         Ilia Mazin <imazin@lanl.gov>
#

r"""
This is a collection of solvers, including mean-field and many-body methods,
for solving the molecular quantum electrodynamics (mQED) Hamiltonian.

:mod:`mqed.qedhf`: **QED-HF**
  Basic QED Hartree-Fock (HF) method. Converged non-QED/HF calculation
  is used to compute QED dipole self-energy (DSE) mediated contributions.

:mod:`mqed.scqedhf`: **SC-QED-HF**
  Strongly-coupled QED-HF method. Polaron transformation is performed for
  all photon modes, the DSE contributions and molecular orbitals are
  self-consistently updated.

:mod:`mqed.vtqedhf`: **VT-QED-HF**
  Variational transformation QED-HF method. The degree of the polaron
  transformation is variationally-optimized for each photon mode to
  handle all coupling strengths.

**MS-QED-HF**
  Multi-scale QED-HF methods.

**QED-TD-SCF**
  WIP.

**QED-CC**
  WIP.

**QED-EOM-CC**
  WIP.
"""

#from . import ccsd
#from . import diis
#from . import ms_qedhf
#from . import qedcc_equations
from . import qedhf
from . import scqedhf
from . import vtqedhf


def HF(mol, qed=None, xc=None, **kwargs):
    if mol.nelectron == 1 or mol.spin == 0:
        if xc is None:
            return qedhf.RHF(mol, qed, **kwargs)
        else:
            return qedhf.RKS(mol, qed, xc, **kwargs)
    else:
        raise NotImplementedError
