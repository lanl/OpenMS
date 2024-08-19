import numpy

from pyscf import gto
from openms.mqed import qedhf
from openms.mqed import scqedhf

fock_energies = []
cs_energies = []

test_range = range(0,8,1)

for b_num in test_range:

    itest = 0.0
    zshift = itest * 2.0

    atom = f"""
            H    0.00000000    0.00000000    {zshift}
            F    0.00000000    0.91700000    {zshift}
            """

    mol = gto.M(atom = atom,
                #basis = "sto3g",
                basis = "cc-pvqz",
                unit = "Angstrom",
                symmetry = True,
                #charge = 1, # example calculation runs on cation, which is not supported
                verbose = 1)

    nmode = 1
    cavity_freq = numpy.zeros(nmode)
    cavity_freq[0] = 2.0
    cavity_mode = numpy.zeros((nmode, 3))
    cavity_mode[0, :] = 0.05 * numpy.asarray([0, 1, 0])

    #if nmode > 1:
    #    cavity_freq[1] = 0.5
    #    cavity_mode[1, :] = 0.1 * numpy.asarray([0, 1, 0])

    qedmf = qedhf.RHF(mol, omega=cavity_freq, vec=cavity_mode, nboson=b_num)
    qedmf.max_cycle = 500
    #qedmf.init_guess = "hcore"
    qedmf.kernel()
    fock_energies.append(qedmf.e_tot)

for i in fock_energies:
    print (f"{i:.12f}")
