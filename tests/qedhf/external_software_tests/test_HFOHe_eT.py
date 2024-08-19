import numpy

from pyscf import gto
from openms.mqed import qedhf

def qedhf_compute(nboson=None, freq=None, cplng=None, init_guess=None, cs=None, basis=None):

    atom = f"""
            H     0.86681000     0.60144000     5.00000000
            F    -0.86681000     0.60144000     5.00000000
            O     0.00000000    -0.07579000     5.00000000
            He    0.00000000     0.00000000     7.50000000
            """

    mol = gto.M(atom = atom,
                #basis = "sto-3g",
                basis = "cc-pvdz",
                unit = "Angstrom",
                verbose = 4)

    nmode = 1
    cavity_freq = numpy.zeros(nmode)
    cavity_freq[0] = freq
    cavity_mode = numpy.zeros((nmode, 3))
    cavity_mode[0, :] = cplng * numpy.asarray([0, 0, 1])

    qedmf = qedhf.RHF(mol, omega=cavity_freq, vec=cavity_mode, nboson=nboson, use_cs=cs, complete_basis=basis)
    if init_guess is not None:
        qedmf.init_guess = init_guess
    #qedmf.conv_tol = 1e-10
    #qedmf.conv_tol_grad = 1e-10
    qedmf.kernel()
    return qedmf.e_tot

fock_energies = []
cs_energies = []
boson_range = range(0,8,1)

for n in boson_range:
    fock_energies.append(qedhf_compute(nboson=n, freq=0.5, cplng=0.05, init_guess="atom", cs=False, basis = False))
    #fock_energies.append(qedhf_compute(nboson=n, freq=0.5, cplng=0.05, init_guess="atom", cs=False, basis = True))
    cs_energies.append(qedhf_compute(nboson=n, freq=0.5, cplng=0.05, init_guess="atom", cs=True, basis = False))
    #cs_energies.append(qedhf_compute(nboson=n, freq=0.5, cplng=0.05, init_guess="atom", cs=True, basis = True))

print ("\n\nFOCK STATE\n----------")
for n in boson_range:
    print (f"{fock_energies[n]:.12f}")

print ("\nCOHERENT STATE\n--------------")
for n in boson_range:
    print (f"{cs_energies[n]:.12f}")
print ("\n")
