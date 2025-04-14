import numpy

from pyscf import gto
from openms.mqed import scqedhf


def scqedhf_compute(atom=None, guess_dm=None, basis_set=None, eta=None, nboson=None, freq=None, cplng=None):

    mol = gto.M(atom = atom,
                basis = basis_set,
                unit = "Angstrom",
                verbose = 4)

    nmode = 1
    cavity_freq = numpy.zeros(nmode)
    cavity_freq[0] = freq
    cavity_mode = numpy.zeros((nmode, 3))
    cavity_mode[0, :] = cplng * numpy.asarray([0, 1, 0])

    qedmf = scqedhf.RHF(mol, omega=cavity_freq, vec=cavity_mode, nboson_states=nboson)
    qedmf.max_cycle = 500
    qedmf.init_guess = "hcore"
    #qedmf.conv_tol = 1e-10
    #qedmf.second_order_eta_step = True

    if guess_dm is not None:
        qedmf.kernel(dm0=guess_dm, init_params=eta)
    else:
        qedmf.kernel()

    dm = qedmf.make_rdm1()
    eta_val = qedmf.eta
    e_tot = qedmf.e_tot
    e_boson = qedmf.qed.e_boson
    cycles = qedmf.cycles

    return dm, eta_val, e_tot, e_boson, cycles


if __name__ == "__main__":

    atm = f"""
           C   0.00000000   0.00000000    0.0000
           O   0.00000000   1.23456800    0.0000
           H   0.97075033  -0.54577032    0.0000
           C  -1.21509881  -0.80991169    0.0000
           H  -1.15288176  -1.89931439    0.0000
           C  -2.43440063  -0.19144555    0.0000
           H  -3.37262777  -0.75937214    0.0000
           O  -2.62194056   1.12501165    0.0000
           H  -1.71446384   1.51627790    0.0000
           """

    # atm = f"""
    #        H     0.86681     0.60144     5.00000
    #        F    -0.86681     0.60144     5.00000
    #        O     0.00000    -0.07579     5.00000
    #        He    0.00000     0.00000     7.50000
    #        """

    #basis = "sto-3g"
    #basis = "6-31g"
    basis = "cc-pvdz"

    #boson_range = range(3,4)
    #boson_range = range(1,5)
    boson_range = range(1,10)

    coupling = 0.10


    # QED-HF (Fock and CS)
    #----------------------------
    #dm0 = None
    #hf_energies = []
    #cs_energies = []

    #for n in boson_range:
    #    dm0, etot_cs, eboson_cs = qedhf_compute(atom=atm, dm0=dm0, basis_set=basis, nboson=n, freq=0.5, cplng=coupling, cs=True)
    #    __, etot, eboson = qedhf_compute(atom=atm, dm0=dm0, basis_set=basis, nboson=n, freq=0.5, cplng=coupling)
    #    dm0 = None
    #    hf_energies.append(etot)
    #    cs_energies.append(etot_cs)


    # SC-QED-HF
    #----------------------------
    dm0 = None
    eta_val = None
    sc_energies = []
    sc_cycles = []
    sc_diffs = []
    e_prev = 0.0

    for n in boson_range:
        dm0, eta_val, etot_sc, eboson_sc, cycle_sc = scqedhf_compute(atom=atm, guess_dm=dm0, basis_set=basis, eta=eta_val, nboson=n, freq=0.5, cplng=coupling)
        #__, __, etot_sc, eboson_sc, cycle_sc = scqedhf_compute(atom=atm, guess_dm=dm0, basis_set=basis, eta=eta_val, nboson=n, freq=0.5, cplng=coupling)
        sc_energies.append(etot_sc)
        sc_cycles.append(cycle_sc)

        e_diff = e_prev - etot_sc
        sc_diffs.append(e_diff)
        e_prev = etot_sc

    sc_cycles = numpy.asarray(sc_cycles)
    sc_diffs = numpy.asarray(sc_diffs)
    print (f"NUMBER OF CYCLES =\n{sc_cycles}\n")
    print (f"ENERGY DIFFERENCES =\n{sc_diffs}\n")

    print ("\n\nSC-QEDHF ENERGIES\n----------")
    for n in boson_range:
        print (f"{sc_energies[n-1]:.12f}")
    print ("\n")
