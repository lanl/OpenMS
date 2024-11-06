#
# Author: Yu Zhang <zhy@lanl.gov>
#

"""
Quantum backends: 
qiskit/openfermion/...

"""

try:
    import qiskit
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

try:
    import openfermion
    OPENFERMION_AVAILABLE = True
except ImportError:
    OPENFERMION_AVAILABLE = False

# set backend


