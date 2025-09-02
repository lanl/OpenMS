
import numpy
from pyscf import gto, scf

def get_h1_eri(n, U=2.0, t= -1.0, PBC=False):

    # hopping
    h1 = numpy.zeros((n, n))
    for i in range(n - 1):
        h1[i, i + 1] = h1[i + 1, i] = t
    if PBC:
        h1[n - 1, 0] = h1[0, n - 1] = t

    # onsite U term
    eri = numpy.zeros((n, n, n, n))
    for i in range(n):
        eri[i, i, i, i] = U

    return h1, eri

# 1. We can setup a custom system with any number of electrons and orbi|tals

n = 12
filling = 0.5
mol = gto.M(verbose=3)
mol.nelectron = int(n * filling)
mol.incore_anyway = True
mol.nao_nr = lambda *args: n
mol.tot_electrons = lambda *args: mol.nelectron

U = 0.0
# define custom oei and eri
h1, eri = get_h1_eri(n, U=U)

# 2. Setup a mean-field type trial Wavefunction
from pyscf import ao2mo
from openms.qmc.trial import TrialHF

mf = scf.RHF(mol)
mf.max_cycle = 500
mf.get_hcore = lambda *args: h1
mf.get_ovlp = lambda *args: numpy.eye(n)
mf._eri = ao2mo.restore(8, eri, n)
mf.kernel()

trial = TrialHF(mol, mf=mf)

# 3. Create a AFQMC object
from openms.qmc.afqmc import AFQMC

time = 5.0
num_walkers = 100

hcore = mf.get_hcore()

def get_integrals():
    ncomponents = 1
    ltensor = numpy.array([numpy.eye(n) * numpy.sqrt(U)])
    h1e = numpy.array([hcore for _ in range(ncomponents)])
    return h1e, ltensor

# FIXME: not working with custom integrals/mol objects
afqmc = AFQMC(
    mol,
    mf=mf,
    dt=0.005,
    total_time=time,
    num_walkers=num_walkers,
    energy_scheme="local",
    chol_thresh=1.0e-10,
    property_calc_freq=1,
    integrals_func=get_integrals,
    verbose=4, #mol.verbose,
)

afqmc.nuc_energy = 0.0
afqmc.kernel()
