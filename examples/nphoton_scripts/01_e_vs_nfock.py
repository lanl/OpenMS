import h5py
import numpy as np

from pyscf import gto
from openms.mqed import qedhf


# -------------------------------------
# Molecule setup
# -------------------------------------
def get_mol(basis="cc-pvdz"):
    atom = f"C   0.00000000   0.00000000    0.0000;\
             O   0.00000000   1.23456800    0.0000;\
             H   0.97075033  -0.54577032    0.0000;\
             C  -1.21509881  -0.80991169    0.0000;\
             H  -1.15288176  -1.89931439    0.0000;\
             C  -2.43440063  -0.19144555    0.0000;\
             H  -3.37262777  -0.75937214    0.0000;\
             O  -2.62194056   1.12501165    0.0000;\
             H  -1.71446384   1.51627790    0.0000"

    return gto.Mole(atom=atom, basis=basis, unit="Angstrom", symmetry=True, verbose=1)


# -------------------------------------
# QEDHF kernel
# -------------------------------------
def run_qedhf(gfac=0.0, nfock=1, coherent_state=False, var_params=None):

    mol = get_mol(basis="cc-pvdz")
    mol.verbose = 3
    mol.build()

    nmode = 1
    cavity_freq = np.zeros(nmode)
    cavity_mode = np.zeros((nmode, 3))
    cavity_freq[0] = 0.01
    cavity_mode[0, :] = gfac * np.asarray([0, 1, 0])

    qedmf = qedhf.RHF(mol, cavity_mode=cavity_mode,
                           cavity_freq=cavity_freq,
                           add_nuc_dipole=True,
                           nboson_states=nfock,
                           use_cs=coherent_state)

    qedmf.max_cycle = 1000
    qedmf.init_guess = "hcore"

    if var_params is not None:
        dm = var_params[0]
        qedmf.kernel(dm0=dm)
    else:
        qedmf.kernel()

    # Energies
    e_tot = qedmf.e_tot
    e_boson = qedmf.qed.e_boson
    dm = qedmf.make_rdm1()

    del qedmf
    return e_tot, e_boson, dm


if __name__ == "__main__":

    # -------------------
    # scan lambda
    # -------------------
    lmax = 0.5
    lskip = lmax / 100.
    gfacs = np.arange(0.0, lmax + lskip, lskip)


    cs_en = []
    cs_boson_en = []
    fock_en = []
    fock_boson_en = []
    nfocks = []

    # Run QEDHF with increasing coupling
    for la in gfacs:

        print (f"\nCURRENT LAMBDA VALUE: {la:4f}\n\n")

        # Run coherent state
        etot_cs, eboson_cs, cs_dm = run_qedhf(gfac=la, coherent_state=True)

        # Save energies to list
        cs_en.append(etot_cs)
        cs_boson_en.append(eboson_cs)


        # Initialize nFock, ediff criteria and list of Fock energies
        nf = 0
        ediff = 1000.
        nfock_en = []
        nfock_boson = []

        # Save coherent-state density matrix
        var = [cs_dm]

        # Run QED-HF with increasing nfock
        while ediff >= 1.0e-6:

            # Increase number of Fock states
            nf += 1

            # Run Fock-state
            etot, eboson, fock_dm = run_qedhf(gfac=la, nfock=nf, var_params=var)

            # Save energies to list
            nfock_en.append(etot)
            nfock_boson.append(eboson)

            # Update energy difference
            ediff = np.abs(etot_cs - etot)


        print (f"\nFOCK AND COHERENT STATE CONVERGED!\n" + \
               f"GFAC={la:4f}  NFOCK={nf}\n\n")

        # Ediff criteria met
        fock_en.append(nfock_en)
        fock_boson_en.append(nfock_boson)
        nfocks.append(nf)


    # -------------------
    # write data to file
    # -------------------
    with h5py.File("qedhf_e_vs_nfock.h5", "w") as fa:
        fa["gfacs"] = gfacs
        fa["cs_en"] = cs_en
        fa["cs_boson_en"] = cs_boson_en
        fa["nfock"] = nfocks

        for n, d in enumerate(fock_en):
            fa.create_dataset(name=f"fock_en_gfac{n*lskip:2f}", data=d)

        for n, d in enumerate(fock_boson_en):
            fa.create_dataset(name=f"fock_boson_en_gfac{n*lskip:2f}", data=d)
