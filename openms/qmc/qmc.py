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

import sys
from pyscf import tools, lo, scf, fci
import numpy as np
import scipy
import itertools
import logging

from openms.qmc.trial import TrialHF
from pyscf.lib import logger

def read_fcidump(fname, norb):
    """
    :param fname: electron integrals dumped by pyscf
    :param norb: number of orbitals
    :return: electron integrals for 2nd quantization with chemist's notation
    """
    eri = np.zeros((norb, norb, norb, norb))
    h1e = np.zeros((norb, norb))

    with open(fname, "r") as f:
        lines = f.readlines()
        for line, info in enumerate(lines):
            if line < 4:
                continue
            line_content = info.split()
            integral = float(line_content[0])
            p, q, r, s = [int(i_index) for i_index in line_content[1:5]]
            if r != 0:
                # eri[p,q,r,s] is with chemist notation (pq|rs)=(qp|rs)=(pq|sr)=(qp|sr)
                eri[p-1, q-1, r-1, s-1] = integral
                eri[q-1, p-1, r-1, s-1] = integral
                eri[p-1, q-1, s-1, r-1] = integral
                eri[q-1, p-1, s-1, r-1] = integral
            elif p != 0:
                h1e[p-1, q-1] = integral
                h1e[q-1, p-1] = integral
            else:
                nuc = integral
    return h1e, eri, nuc


class QMCbase(object):
    r"""
    Basic QMC class
    """
    def __init__(self,
        system, # or molecule
        mf = None,
        dt = 0.005,
        nsteps = 25,
        total_time = 5.0,
        num_walkers = 100,
        renorm_freq = 5,
        random_seed = 1,
        taylor_order = 6,
        energy_scheme = None,
        batched = False,
        *args, **kwargs):
        r"""

        Args:

           system:      (or molecule) that contains the information about
                        number of electrons, orbitals, Hamiltonain, etc.
           propagator:  propagator class that deals with the way of propagating walkers.
           walker:      Walkers used for open ended random walk.
           renorm_freq: renormalization frequency
           nblocks:     Number of blocks
           nsteps:      Number of steps per block
        """

        self.system = self.mol = system

        # propagator params
        self.dt = dt
        self.total_time = total_time
        self.propagator = None
        self.nsteps = nsteps   #
        self.nblocks = 500     #
        self.pop_control_freq = 5                 # population control frequency
        self.pop_control_method = "pair_brach"    # populaiton control method
        self.eq_time = 2.0                        # time of equilibration phase
        self.eq_steps = int(self.eq_time/self.dt) # Number of time steps for the equilibration phase
        self.stablize_freq = 5                    # Frequency of stablization(re-normalization) steps
        self.energy_scheme = energy_scheme
        self.verbose = 1
        self.stdout = sys.stdout

        self.trial = None


        # walker parameters
        # TODO: move these variables into walker object
        self.walker = None
        self.__dict__.update(kwargs)

        self.taylor_order = taylor_order
        self.num_walkers = num_walkers
        self.renorm_freq = renorm_freq
        self.random_seed = random_seed
        self.walker_coeff = None
        self.walker_tensors = None

        self.mf_shift = None
        self.print_freq = 10

        self.hybrid_energy = None

        self.batched = batched

        self.build() # setup calculations


    def build(self):
        r"""
        Build up the afqmc calculations
        """
        # set up trial wavefunction
        logger.info(self, "\n========  Initialize Trial WF and Walker  ======== \n")
        if self.trial is None:
            self.trial = TrialHF(self.mol)

        # set up walkers
        # TODO: move this into walker class
        temp = self.trial.wf.copy()
        self.walker_tensors = np.array([temp] * self.num_walkers, dtype=np.complex128)
        self.walker_coeff = np.array([1.] * self.num_walkers)


    def get_integrals(self):
        r"""
        TODO:
        return oei and eri in MO
        """

        overlap = self.mol.intor('int1e_ovlp')
        self.ao_coeff = lo.orth.lowdin(overlap)
        norb = self.ao_coeff.shape[0]

        import tempfile
        ftmp = tempfile.NamedTemporaryFile()
        tools.fcidump.from_mo(self.mol, ftmp.name, self.ao_coeff)
        h1e, eri, self.nuc_energy = read_fcidump(ftmp.name, norb)

        # Cholesky decomposition of eri
        eri_2d = eri.reshape((norb**2, -1))
        u, s, v = scipy.linalg.svd(eri_2d)
        ltensor = u * np.sqrt(s)
        ltensor = ltensor.T
        ltensor = ltensor.reshape(ltensor.shape[0], norb, norb)
        self.nfields = ltensor.shape[0]

        return h1e, eri, ltensor


    def propagation(self, h1e, xbar, ltensor):
        pass

    def measure_observables(self, operator):
        observables = None
        return observables

    def walker_trial_overlap(self):
        return np.einsum('pr, zpq->zrq', self.trial.wf.conj(), self.walker_tensors)

    def renormalization(self):
        r"""
        Renormalizaiton and orthogonaization of walkers
        """

        ortho_walkers = np.zeros_like(self.walker_tensors)
        for idx in range(self.walker_tensors.shape[0]):
            ortho_walkers[idx] = np.linalg.qr(self.walker_tensors[idx])[0]
        self.walker_tensors = ortho_walkers

    def local_energy(self, ltheta, h1e, eri, trace_ltheta, gf):
        tmp = 2. * np.einsum("prqs, zpr->zqs", eri, gf)
        tmp -= np.einsum("prqs, zps->zqr", eri, gf)
        e2 = np.einsum("zqs, zqs->z", tmp, gf)

        e1 = 2 * np.einsum("zpq, pq->z", gf, h1e)
        energy = e1 + e2 + self.nuc_energy
        return energy

    def update_weight(self, overlap, cfb, cmf, local_energy, time):
        r"""
        Update the walker coefficients
        """
        newoverlap = self.walker_trial_overlap()
        # be cautious! power of 2 was neglected before.
        overlap_ratio = (np.linalg.det(newoverlap) / np.linalg.det(overlap))**2

        # the hybrid energy scheme
        if self.energy_scheme == "hybrid":
            self.ebound = (2.0 / self.dt) ** 0.5
            hybrid_energy = -(np.log(overlap_ratio) + cfb + cmf) / self.dt
            hybrid_energy = np.clip(hybrid_energy.real, a_min=-self.ebound, a_max=self.ebound, out=hybrid_energy.real)
            self.hybrid_energy = hybrid_energy if self.hybrid_energy is None else self.hybrid_energy

            importance_func = np.exp(-self.dt * 0.5 * (hybrid_energy + self.hybrid_energy))
            self.hybrid_energy = hybrid_energy
            phase = (-self.dt * self.hybrid_energy-cfb).imag
            phase_factor = np.array([max(0, np.cos(iphase)) for iphase in phase])
            importance_func = np.abs(importance_func) * phase_factor

        elif self.energy_scheme == "local":
            # The local energy formalism
            overlap_ratio = overlap_ratio * np.exp(cmf)
            phase_factor = np.array([max(0, np.cos(np.angle(iovlp))) for iovlp in overlap_ratio])
            importance_func = np.exp(-self.dt * np.real(local_energy)) * phase_factor

        else:
            raise ValueError(f'scheme {self.energy_scheme} is not available!!!')

        self.walker_coeff *= importance_func

    def kernel(self, trial_wf=None):
        r"""
        trial_wf: trial wavefunction
        TBA
        """

        np.random.seed(self.random_seed)

        logger.info(self, "\n======== get integrals ========")
        h1e, eri, ltensor = self.get_integrals()
        shifted_h1e = np.zeros(h1e.shape)
        rho_mf = self.trial.wf.dot(self.trial.wf.T.conj())
        self.mf_shift = 1j * np.einsum("npq,pq->n", ltensor, rho_mf)

        for p, q in itertools.product(range(h1e.shape[0]), repeat=2):
            shifted_h1e[p, q] = h1e[p, q] - 0.5 * np.trace(eri[p, :, :, q])
        shifted_h1e = shifted_h1e - np.einsum("n, npq->pq", self.mf_shift, 1j*ltensor)

        self.precomputed_ltensor = np.einsum("pr, npq->nrq", self.trial.wf.conj(), ltensor)

        time = 0.0
        energy_list = []
        time_list = []
        while time <= self.total_time:
            dump_result = (int(time/self.dt) % self.print_freq  == 0)

            # pre-processing: prepare walker tensor
            overlap = self.walker_trial_overlap()
            inv_overlap = np.linalg.inv(overlap)
            theta = np.einsum("zqp, zpr->zqr", self.walker_tensors, inv_overlap)

            gf = np.einsum("zqr, pr->zpq", theta, self.trial.wf.conj())
            ltheta = np.einsum('npq, zqr->znpr', self.precomputed_ltensor, theta)
            trace_ltheta = np.einsum('znpp->zn', ltheta)

            # compute local energy for each walker
            local_energy = self.local_energy(ltheta, h1e, eri, trace_ltheta, gf)
            energy = np.sum([self.walker_coeff[i]*local_energy[i] for i in range(len(local_energy))])
            energy = energy / np.sum(self.walker_coeff)

            # imaginary time propagation
            xbar = -np.sqrt(self.dt) * (1j * 2 * trace_ltheta - self.mf_shift)
            cfb, cmf = self.propagation(shifted_h1e, xbar, ltensor)
            self.update_weight(overlap, cfb, cmf, local_energy, time)

            # re-orthogonalization
            if int(time / self.dt) == self.renorm_freq:
                self.renormalization()

            # print energy and time
            if dump_result:
                time_list.append(time)
                energy_list.append(energy)
                logger.info(self, f" Time: {time:9.3f}    Energy: {energy:15.8f}")

            time += self.dt

        return time_list, energy_list
