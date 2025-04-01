#
# @ 2023. Triad National Security, LLC. All rights reserved.
#
# This program was produced under U.S. Government contract 89233218CNA000001
# for Los Alamos National Laboratory (LANL), which is operated by Triad
# National Security, LLC for the U.S. Department of Energy/National Nuclear
# Security Administration. All rights in the program are reserved by Triad
# National Security, LLC, and the U.S. Department of Energy/National Nuclear
# Security Administration. The Government is granted for itself and others acting
# on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this
# material to reproduce, prepare derivative works, distribute copies to the
# public, perform publicly and display publicly, and to permit others to do so.
#
# Author: Yu Zhang <zhy@lanl.gov>
#

r"""
Disordered molecular aggregates.
"""

import openms.lib.backend as bd
import numpy as np  # replace np as bd (TODO)
import random


def linear_spec(elist, state, evals, dip, gamma):
    r"""
    calculate linear spectrum
    state: list of states to be included in the spec
    evals: eigenvalues of 1 exciton states
    dip1: dipolemomnet of 1 exciton states
    """
    spectrum = np.zeros(len(elist))
    for i, e in enumerate(elist):
        tmp = 0.0
        for j in state:
            tmp += dip[j] * dip[j] * gamma / ((e - evals[j]) ** 2 + gamma**2)
        spectrum[i] = tmp
    return spectrum


def tdes(elist, state, evals, dip1, nuv, eng2, dip2, gamma):
    r"""
    state: list of single-exciton states
    evals: eigenvalues of 1 exciton states
    dip1: dipolemomnet of 1 exciton states
    nuv: number of 2-exciton states
    eng2: energy difference of 1->2 transition
    dip2: dipolement of 1->2 transition
    gamma: broadening
    """
    ne = len(elist)
    spec2d = np.zeros((ne, ne))
    for m in range(ne):  # exitation
        for n in range(ne):
            spec = 0.0
            Ed = elist[n]
            Ex = elist[m]
            for v1, j in enumerate(state):
                tmp = 0.0
                for uv in range(nuv):
                    deltae = eng2[v1, uv]
                    a = dip2[v1, uv] * dip2[v1, uv] * gamma
                    b = (Ed - deltae) ** 2 + gamma**2
                    tmp -= a / b

                a = dip1[j] * dip1[j] * gamma
                b = (Ed - evals[j]) ** 2 + gamma**2
                tmp += 2.0 * a / b

                a = dip1[j] * dip1[j] * gamma
                b = (Ex - evals[j]) ** 2 + gamma**2
                tmp = tmp * a / b
                spec += tmp

            spec2d[n, m] = spec
        if m % 20 == 0:
            print("%8.3f percent of 2des is done" % (m / ne * 100.0))
        sys.stdout.flush()
    return spec2d


def matvec(A, x):
    y = A @ x

    return y


# disordered molecular aggregates class
class DMA(object):
    r"""
    Disordered molecular aggregates class.
    """

    def __init__(
        self, Nsite=1, Nexc=5,
        epsilon=0.0, hopping=0.1, sigma=0.0, zeta=0.0, **kwargs
    ):
        r"""
        Hamiltonian:

        H = \sum_i (\epsilon_i +\delta) c^\dag_i c_i +
            \sum_{i,i+1} (t+\delta_2) [c^\dag_{i+1}c_i + h.c.]

        Nsite : Int
           number of excitons
        Nexc : Int
           number of excited states of interest
        epsilon : float
           onsite energy
        hopping : float
           hopping parameter
        sigma : float
           onsite disorder
        zeta : float
           hopping disorder

        Kwargs:
             TBA

        Examples:
            TBA
        """

        self.Nsite = Nsite
        self.Nexc = min(Nexc, Nsite)

        self.epsilon = epsilon
        self.hopping = hopping
        self.sigma = sigma
        self.zeta = zeta

        self.A = bd.zeros((Nsite, Nsite))
        self.En = bd.zeros(Nsite)
        self.excdipole1 = None
        self.dipsortidx = None
        self.c0 = 0.80
        self.c1 = 0.50
        self.c2 = 0.30

    def kernel(self):
        r"""
        Compute the low-lying states.
        """

        for i in range(self.Nsite):
            self.En[i] = random.gauss(self.epsilon, self.sigma * abs(self.hopping))
            self.A[i][i] = self.En[i]
            if i < self.Nsite - 1:
                self.A[i][i + 1] = self.hopping
                self.A[i + 1][i] = self.hopping

        self.evals, self.evecs = np.linalg.eig(self.A)

        idx = self.evals.argsort()
        self.evals = self.evals[idx]
        self.evecs = self.evecs[:, idx]


    def energies(self):
        r"""Return the lowest Nexc states."""
        return self.evals[: self.Nexc]


    def dipole(self):
        r"""
        Compute the dipole of the lowest Nexc states.

        dipole of one-exciton state :math:`d_{mu}= \sum_{i} C_{\mu j} |j>`
        """

        self.excdipole1 = np.zeros(self.Nsite)
        for u in range(self.Nsite):
            for j in range(self.Nsite):
                self.excdipole1[u] += self.evecs[j, u]


    def sortdipole(self):
        r"""
        Sort dipole and get the index of states with largest dipole.
        """

        if self.excdipole1 is None:
            self.dipole()

        dip1norm = [abs(self.excdipole1[i]) for i in range(self.Nsite)]
        self.dipsortidx = np.asarray(dip1norm).argsort()[::-1]

        # print(dipsortidx)
        maxdip = max(dip1norm)
        self.totdip = np.dot(self.excdipole1, self.excdipole1)
        print(
            "\n maximum and total dipole",
            maxdip,
            self.totdip,
            maxdip**2 / self.totdip,
            "\n",
        )

        print("------sorted dipole------")
        self.dip_cutoff = 0.1 * maxdip
        for i in self.dipsortidx:
            if abs(self.excdipole1[i]) >= self.dip_cutoff:
                print(i, abs(self.excdipole1[i]))
        print("dipcutoff=", self.dip_cutoff, "\n")


    def spdfselection(self):
        r"""
        1)  check nodes (state type)  of each eigenstates
        2) select dominant exciton transitions

        J. Chem. Phys. 128, 084706 (2008)
        s-like atomic states: they consist of mainly one peak with no node within the localization segment
        p-like atomic states: They have a well defined node within localization segments and occur in pairs
        with s-like states. Each pair forms an sp doublet localized on the same chain segment.
        """

        N = self.Nsite
        selected_dip = 0.0
        local_gs = []
        if self.dipsortidx is None:
            self.sortdipole()

        for u in self.dipsortidx:
            tmp = 0.0
            for j in range(N):
                tmp += self.evecs[j, u] * abs(self.evecs[j, u])
            if abs(tmp) >= self.c0:
                print("state, sum_j(phi_jv|phi_jv|)", u, tmp)
                selected_dip += self.excdipole1[u] ** 2
                local_gs.append(u)

        print("local ground (s) states", local_gs, "\n")
        print("selected_dip (s)/totdip", selected_dip / self.totdip, "\n")

        # Frourier transform evecs (TODO)
        evecs_ft = np.zeros((N, N))
        for u in range(N):
            for j in range(N):
                evecs_ft[j, u] = 0.0
                # for k in range(N):
                #    evecs_ft[j,u] += evecs[k,u] * sin(k/N)

        # select local excited states
        local_ex = []
        for u in local_gs:
            for v in range(N):
                if v == u:
                    continue
                # if v in local_ex: continue
                if v in local_ex or v in local_gs:
                    continue
                if abs(self.excdipole1[v]) < self.dip_cutoff:
                    continue

                tmp = 0.0
                for j in range(N):
                    tmp += self.evecs[j, u] * abs(self.evecs[j, v])
                if abs(tmp) >= self.c1:
                    # print('state, sum_j(phi_jv|phi_jv|)', u, v, tmp)
                    selected_dip += self.excdipole1[v] ** 2
                    local_ex.append(v)

        print("local excited states", local_ex, "\n")
        print("selected_dip (s + p)/totdip", selected_dip / self.totdip, "\n")

        local_states = local_gs + local_ex

        # -------------------------------------------------------------------------------
        print("state    node    dipole moment")
        # not done yet

    # compute the linear absorption spectrum of given listed of states
    def linearabs(self, elist=None, selected=None, gamma=0.001):
        r"""
        compute the linear absorption
        selected: a subset of selected states for spectrum calculations
        """

        if elist is None:
            raise Exception(
                "elist is None! please specify a list of energies for spectrum!"
            )

        if self.excdipole1 is None:
            self.dipole()

        if selected is None:
            spectrum = linear_spec(
                elist, range(self.Nexc), self.evals, self.excdipole1, gamma
            )
        else:
            spectrum = linear_spec(elist, selected, self.evals, self.excdipole1, gamma)

        return spectrum


    def tdes(self):
        """Compute two-dimensional absorption spectra."""
        return None


if __name__ == "__main__":
    from openms.lib.backend import NumpyBackend, TorchBackend
    from openms.lib.backend import backend as bd
    from openms.lib.backend import set_backend

    set_backend("numpy")

    model = DMA(100, epsilon=0.0, hopping=0.1, sigma=0.01)
    model.kernel()

    # specify lower and upper bounds for spectrum calculations
    ebot = min(model.evals) - 0.1
    etop = max(model.evals) + 0.1
    elist = np.arange(ebot, etop, 0.001)
    model.sortdipole()
    model.spdfselection()

    gamma = 9.0 / 1000.0  # 9 meV
    spectrum = model.linearabs(elist, gamma=gamma)

    # print('states:', model.energies())
    # for i, e in enumerate(elist):
    #    print("%f   %e" %(elist[i], spectrum[i]))
