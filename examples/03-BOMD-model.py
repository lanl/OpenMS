from openms.lib.misc import Molecule
from openms import models
from openms import qmd
import random

# For model calculation, set mass of a fictitious particle of the model.

# Random seed is fixed for test purpose to check if the same result is reproduced.
random.seed(10)

# Define the target system.
geom = """
H       -4.0     0.0   0.0
"""

nstates = 2
mol = Molecule(
    atom=geom,
    basis="sto3g",
    charge=1,
    lmodel=True,
    ndim=1,
    nstates=nstates,
    unit="Bohr",
)
# mol = Molecule(geometry=geom, ndim=1, nstates=2, ndof=1, unit_pos='au', l_model=True)

print("molecule was built!")
# Set QM method.
qm = models.Shin_Metiu(molecule=mol)

# qm.get_data(molecule=mol, base_dir='./', adiastates=range(nstates), dt=1.0, istep=0)
# for i in range(mol.nstates):
#    print(f"energy of state {i} is {mol.states[i].energy}")


# Determine MD method.
md = qmd.BOMD(molecule=mol, nsteps=2890, dt=0.5, unit_dt="au", init_state=1)


# Execute the simulation.
md.kernel(qm=qm)
