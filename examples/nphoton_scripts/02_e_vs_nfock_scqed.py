import h5py
import numpy as np

from pyscf import gto
from openms.mqed import scqedhf


# -------------------------------------
# Molecule setup
# -------------------------------------
def get_mol(basis):

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
# SC-QEDHF kernel
# -------------------------------------
def run_scqedhf(gfac=0.0, nfock=1, freq=0.5, guess_dm=None, guess_eta=None):

    #basis_set = "sto-3g"
    #basis_set = "6-31g"
    basis_set = "cc-pvdz"

    mol = get_mol(basis_set)
    mol.verbose = 3
    mol.build()

    nmode = 1
    cavity_freq = np.zeros(nmode)
    cavity_mode = np.zeros((nmode, 3))
    cavity_freq[0] = freq
    cavity_mode[0, :] = gfac * np.asarray([0, 1, 0])

    scqedmf = scqedhf.RHF(mol, cavity_mode=cavity_mode,
                               cavity_freq=cavity_freq,
                               add_nuc_dipole=True,
                               nboson_states=nfock)
    scqedmf.max_cycle = 1000
    scqedmf.init_guess = "hcore"

    if guess_dm is not None:
        if guess_eta is not None:
            scqedmf.kernel(dm0=guess_dm, init_params=guess_eta)
        else:
            scqedmf.kernel(dm0=guess_dm)
    else:
        scqedmf.kernel()

    # Energies
    e_tot = scqedmf.e_tot
    e_boson = scqedmf.qed.e_boson

    dm = None
    eta_val = None

    # Only create guess if previous calculation converged
    conv = scqedmf.converged
    if conv == True:
        dm = scqedmf.make_rdm1()
        eta_val = scqedmf.eta

    del scqedmf
    return dm, eta_val, e_tot, e_boson, conv


if __name__ == "__main__":

    # -------------------
    # scan lambda
    # -------------------
    lmax = 0.5
    lskip = lmax / 100.

    ## coupling scan
    lstart = 0.0
    lend = 0.5
    gfacs = np.arange(lstart, lend + lskip, lskip)
    nfocks = []

    # frequency
    omega = 0.5

    # Append energies to file
    with h5py.File(f"scqedhf_e_vs_nfock_data.h5", "w") as fa:

        sc_dm = None
        eta_val = None
        prev_nfock = None

        # Run SC-QEDHF with increasing coupling
        for la in gfacs:
            print (f"\nCURRENT LAMBDA VALUE: {la:4f}\n")

            nf = 0
            if prev_nfock is not None and prev_nfock > 4:
                nf = int(0.75 * prev_nfock) - 1

            e_prev = 0.0
            e_diff = 1000.

            etot = []
            eboson = []
            nfocks = []

            # Run SC-QEDHF with increasing nFock
            while e_diff >= 1.0e-6:

                # Increase number of Fock states
                nf += 1

                # Run SC-QEDHF
                sc_dm, eta_val, e_tot, e_boson, conv = run_scqedhf(gfac=la, nfock=nf, freq=omega, guess_dm=sc_dm, guess_eta=eta_val)

                # Save energies to list
                etot.append(e_tot)
                eboson.append(e_boson)
                nfocks.append(nf)

                # Update energy difference if SCF converges
                if conv == True:
                    e_diff = np.abs(e_prev - e_tot)
                    e_prev = 0.0 + e_tot

            # Ediff criteria met
            print (f"\nSC-QEDHF CONVERGED W/ NUMBER OF FOCK STATES!\n" + \
                   f"GFAC={la:4f}  NFOCK={nf}\n\n")

            ## Reset guess for next coupling
            prev_nfock = nf
            #sc_dm = None
            #eta_val = None

            # -------------------
            # write data to file
            fa[f"en_gfac_{la}"] = etot
            fa[f"boson_en_gfac_{la}"] = eboson
            fa[f"nfock_{la}"] = nfocks
