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
# Author: Yu Zhang <zhy@lanl.gov>
#

r"""
Qutip-based Spin dynamics module.

Qutip-based Spin dynamics module for studying the spin-relaxation
in the presence of spin-spin, hyperfine coupling, and spin-honon
interactions.

Following techniques are implemented:

  - Cluster correlation expansion (CCE)
  - quantum embedding for normal modes

  
Theoretical background of CCE
-----------------------------
TBA


Theoretical background of embedding
-----------------------------------
TBA
"""

from . import bath
from . import gtensorlibs
from . import spin
from . import system

# use qutip
try:
    import qutip
    QUTIP_AVAILABLE = True

except ImportError as e:
    print(f"Could not import 'qutip': {e}")
    QUTIP_AVAILABLE = False
