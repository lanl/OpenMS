import time
from datetime import datetime
import matplotlib.pyplot as plt

from pyscf import gto, scf, fci
from openms.qmc import tools
from openms.qmc.afqmc import AFQMC
import numpy


def get_hydrogen_chain(N=5, bond=1.6, basis="sto3g", verbose=3):

    atom = [("H", 1.6 * i, 0, 0) for i in range(0, N)]

    mol = gto.M(
        atom=atom,
        basis=basis,
        verbose=verbose,
        unit="Bohr",
    )
    return mol


def run_qmc(
    natoms=10,
    use_so=True,
    nwalkers=15,
    num_steps=50,
    scheme="hybrid",
    time_step=0.005,
    verbose=3,
):

    mol = get_hydrogen_chain(N=natoms, basis="sto-6g", verbose=verbose)
    # mf = mf_calc(mol)

    total_time = time_step * num_steps

    afqmc = AFQMC(
        mol,
        dt=time_step,
        total_time=total_time,
        num_walkers=nwalkers,
        energy_scheme=scheme,
        property_calc_freq=5,
        use_so=use_so,
        verbose=verbose,
    )

    afqmc.dump_flags()

    t0 = time.time()
    times, energies = afqmc.kernel()
    wt = time.time() - t0
    mean, std_dev = tools.get_mean_std(energies)
    print(f"energy and std are (hybrid):  {mean} {std_dev}")
    return mean, wt


def plot_energies_wt(nlist, energies, walltimes, fname=None):

    fig, ax1 = plt.subplots()

    # Plot energy on the first y-axis
    ax1.plot(nlist, energies[0], "-^", color="blue", label="RHF")
    ax1.plot(nlist, energies[1], "-o", color="blue", label="UHF")
    ax1.set_xlabel("System Size")
    ax1.set_ylabel("Energy", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    # Add a legend for the energy plots
    ax1.legend(loc="upper left")

    # Create a second y-axis for walltimes
    ax2 = ax1.twinx()
    ax2.plot(nlist, walltimes[0], "-^", color="red", label="RHF Walltime")
    ax2.plot(nlist, walltimes[1], "-o", color="orange", label="UHF Walltime")
    ax2.set_ylabel("Walltime (seconds)", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    # Add a legend for the walltime plots
    ax2.legend(loc="upper right")

    # Title and layout
    # plt.title("Energy and Walltime vs System Size")
    fig.tight_layout()

    # Save the plot
    figname = "walltime_vs_energy_vs_N.pdf" if fname is None else fname
    plt.savefig(figname)

    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    ax.plot(nlist, energies[0], "-^", corlor="blue", label="RHF")
    ax.plot(nlist, energies[1], "-o", corlor="blue",  label="UHF")
    ax.plot(nlist, walltimes[0], "-^", color="red", label="RHF")
    ax.plot(nlist, walltimes[1], "-o", color="red",  label="UHF")
    ax.set_xlabel("system size")
    plt.savefig("walltime_vs_N.pdf")
    # plt.show()
    """


def benchmark_walltime(current_data):

    nwalkers = 200
    nlist = range(10, 51, 10)
    energies = numpy.zeros((2, len(nlist)))
    walltimes = numpy.zeros((2, len(nlist)))
    for i, natoms in enumerate(nlist):
        energy, wt = run_qmc(
            natoms=natoms, use_so=False, nwalkers=nwalkers, num_steps=100
        )
        energies[0, i] = energy
        walltimes[0, i] = wt

        energy, wt = run_qmc(
            natoms=natoms, use_so=True, nwalkers=nwalkers, num_steps=100
        )
        energies[1, i] = energy
        walltimes[1, i] = wt

    plot_energies_wt(
        nlist, energies, walltimes, fname=f"wall_time_vs_size_{current_date}.pdf"
    )


def run_fci(mol):
    mf = mol.RHF().run()
    #
    # create an FCI solver based on the SCF object
    #
    cisolver = fci.FCI(mf)
    e_rhf_fci = cisolver.kernel()[0]
    print("E(FCI) = %.12f" % e_rhf_fci)
    return e_rhf_fci


def benchmark_timestep(current_date):
    r"""Compare two energy schemes with different time steps"""

    nwalkers = 500
    num_steps = 2000
    natoms = 10

    verbose = 4
    mol = get_hydrogen_chain(N=natoms, basis="sto-6g", verbose=verbose)
    fci_energy = run_fci(mol)

    # we fix steps
    time_steps = numpy.arange(0.005, 0.0251, 0.005)
    nt = len(time_steps)
    energies = numpy.zeros((2, nt))

    for i, time_step in enumerate(time_steps):
        energies[0, i], _ = run_qmc(
            natoms=natoms,
            use_so=True,
            nwalkers=nwalkers,
            num_steps=num_steps,
            time_step=time_step,
            scheme="hybrid",
            verbose=verbose,
        )
        energies[1, i], _ = run_qmc(
            natoms=natoms,
            use_so=True,
            nwalkers=nwalkers,
            num_steps=num_steps,
            time_step=time_step,
            scheme="local",
            verbose=verbose,
        )

    # plot energies
    fig, ax = plt.subplots()
    # time = backend.arange(0, 5, 0.)
    ax.plot(time_steps, [fci_energy] * nt, "--", color="black", label="FCI")
    ax.plot(time_steps, energies[0], "-s", color="blue", label="Hybrid")
    ax.plot(time_steps, energies[0], "-o", color="red", label="local")
    ax.set_ylabel("Ground state energy")
    ax.set_xlabel("Time step")
    fname = f"Energy_H{natoms}_vs_timestep_{current_date}.pdf"
    plt.savefig(fname)


if __name__ == "__main__":
    current_date = datetime.now().date()
    print("Today's date is:", current_date)
    benchmark_timestep(current_date)
