from openms.lib.misc import Molecule
from openms.lib.hippynn_es_driver import NNDriver
from openms import models
from openms import qmd
import random

# For model calculation, set mass of a fictitious particle of the model.

# Random seed is fixed for test purpose to check if the same result is reproduced.
random.seed(10)

# Define the target system.
# geom = """
# H         -7.09082        0.08303        0.82683
# H         -6.97113       -0.75979       -0.82683
# C         -6.47665       -0.25966        0.00000
# C         -5.15413       -0.07185       -0.00000
# H         -4.65965        0.42828        0.82683
# H         -4.53996       -0.41453       -0.82683
# """
geom = """
H  0.53805  -1.44768   0.24329
H  0.56218   0.24173  -0.43274
C -0.0406   -0.57662  -0.0499
C -1.35746  -0.51769   0.04993
H -1.96062  -1.33579   0.43271
H -1.93595   0.35348  -0.2433
"""
nstates = 2
mol = Molecule(
    atom=geom,
    basis="sto3g",
    charge=0,
    lmodel=True,
    ndim=3,
    nstates=nstates,
    unit="Angstrom",
)

print("molecule was built!")
# Set QM method.
# qm = NNDriver(mol, 1, "assets/ml_model2")
qm = NNDriver(mol, 1, "../tests/assets/models/3_states")
qm.get_Z()

# qm.get_data(molecule=mol, base_dir='./', adiastates=range(nstates), dt=1.0, istep=0)
# for i in range(mol.nstates):
#    print(f"energy of state {i} is {mol.states[i].energy}")


# Determine MD method.
md = qmd.BOMD(molecule=mol, nsteps=10000, dt=0.4, unit_dt="au", init_state=1)


# Execute the simulation.
md.kernel(qm=qm)
