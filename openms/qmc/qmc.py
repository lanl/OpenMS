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
import numpy as backend
import scipy
import itertools
import logging

from openms.qmc.trial import TrialHF
from . import generic_walkers as gwalker
from pyscf.lib import logger

def read_fcidump(fname, norb):
    """
    :param fname: electron integrals dumped by pyscf
    :param norb: number of orbitals
    :return: electron integrals for 2nd quantization with chemist's notation
    """
    eri = backend.zeros((norb, norb, norb, norb))
    h1e = backend.zeros((norb, norb))

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
                eri[p - 1, q - 1, r - 1, s - 1] = integral
                eri[q - 1, p - 1, r - 1, s - 1] = integral
                eri[p - 1, q - 1, s - 1, r - 1] = integral
                eri[q - 1, p - 1, s - 1, r - 1] = integral
            elif p != 0:
                h1e[p - 1, q - 1] = integral
                h1e[q - 1, p - 1] = integral
            else:
                nuc = integral
    return h1e, eri, nuc


class QMCbase(object):
    r"""
    Basic QMC class
    """

    def __init__(
        self,
        system,  # or molecule
        mf=None,
        dt=0.005,
        nsteps=25,
        total_time=5.0,
        num_walkers=100,
        renorm_freq=5,
        random_seed=1,
        taylor_order=6,
        energy_scheme=None,
        batched=False,
        *args,
        **kwargs,
    ):
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
        self.nsteps = nsteps  #
        self.nblocks = 500  #
        self.pop_control_freq = 5  # population control frequency
        self.pop_control_method = "pair_brach"  # populaiton control method
        self.eq_time = 2.0  # time of equilibration phase
        self.eq_steps = int(
            self.eq_time / self.dt
        )  # Number of time steps for the equilibration phase
        self.stablize_freq = 5  # Frequency of stablization(re-normalization) steps
        self.energy_scheme = energy_scheme
        self.verbose = 1
        self.stdout = sys.stdout

        self.trial = None

        # walker parameters
        # TODO: move these variables into walker object
        self.__dict__.update(kwargs)
        self.taylor_order = taylor_order
        self.num_walkers = num_walkers
        self.renorm_freq = renorm_freq
        self.random_seed = random_seed

        # waker_tensors/coeff are moved into walkers class (phiw, weights) respectively
        # self.walker_coeff = None
        # self.walker_tensors = None

        self.mf_shift = None
        self.print_freq = 10

        self.hybrid_energy = None

        self.batched = batched

        self.build()  # setup calculations

    def build(self):
        r"""
        Build up the afqmc calculations
        """
        # set up trial wavefunction
        logger.info(self, "\n========  Initialize Trial WF and Walker  ======== \n")
        if self.trial is None:
            self.trial = TrialHF(self.mol)
            self.spin_fac = 1.0

        # set up walkers
        # moved this into walker class
        # temp = self.trial.wf.copy()
        # self.walker_tensors = backend.array([temp] * self.num_walkers, dtype=backend.complex128)
        # self.walker_coeff = backend.array([1.] * self.num_walkers)
        self.walkers = gwalker.Walkers_so(nwalkers=self.num_walkers, trial=self.trial)

    def get_integrals(self):
        r"""return oei and eri in MO"""

        overlap = self.mol.intor("int1e_ovlp")
        self.ao_coeff = lo.orth.lowdin(overlap)
        norb = self.ao_coeff.shape[0]

        import tempfile

        ftmp = tempfile.NamedTemporaryFile()
        tools.fcidump.from_mo(self.mol, ftmp.name, self.ao_coeff)
        h1e, eri, self.nuc_energy = read_fcidump(ftmp.name, norb)

        # Cholesky decomposition of eri
        eri_2d = eri.reshape((norb**2, -1))
        u, s, v = scipy.linalg.svd(eri_2d)
        ltensor = u * backend.sqrt(s)
        ltensor = ltensor.T
        ltensor = ltensor.reshape(ltensor.shape[0], norb, norb)
        self.nfields = ltensor.shape[0]

        return h1e, eri, ltensor

    def propagation(self, walkers, xbar, ltensor):
        pass

    def measure_observables(self, operator):
        observables = None
        return observables

    def walker_trial_overlap(self):
        r"""
        Compute the overlap between trial and walkers:

        .. math::

            \langle \Psi_T \ket{\Psi_w} = det[S]

        where

        .. math::

            S = C^*_{\psi_T} C_{\psi_w}

        and :math:`C_{\psi_T}` and :math:`C_{\psi_w}` are the coefficient matrices
        of Trial and Walkers, respectively.
        """
        return backend.einsum("pr, zpq->zrq", self.trial.psi.conj(), self.walkers.phiw)

    def renormalization(self):
        r"""
        Renormalizaiton and orthogonaization of walkers
        """

        ortho_walkers = backend.zeros_like(self.walkers.phiw)
        for idx in range(self.walkers.phiw.shape[0]):
            ortho_walkers[idx] = backend.linalg.qr(self.walkers.phiw[idx])[0]
        self.walkers.phiw = ortho_walkers

    def local_energy_spin(self, h1e, eri, G1p):
        r"""Compute local energy

        .. math::

             E = \sum_{pq\sigma} T_{pq} G_{pq\sigma}
                 + \frac{1}{2}\sum_{pqrs\sigma\sigma'} I_{prqs} G_{pr\sigma} G_{qs\sigma'}
                 - \frac{1}{2}\sum_{pqrs\sigma} I_{pqrs} G_{ps\sigma} G_{qr\sigma}
        """
        # E_coul
        tmp = 2.0 * backend.einsum("prqs,zSpr->zqs", eri, G1p) * self.spin_fac
        ecoul = backend.einsum("zqs,zSqs->z", tmp, G1p)
        # E_xx
        tmp = backend.einsum("prqs,zSps->zSqr", eri, G1p)
        exx = backend.einsum("zSqs,zSqs->z", tmp, G1p)
        e2 = (ecoul - exx) * self.spin_fac

        e1 = 2 * backend.einsum("zSpq,pq->z", G1p, h1e) * self.spin_fac

        energy = e1 + e2 + self.nuc_energy
        return energy

    def local_energy(self, TL_theta, h1e, eri, vbias, gf):
        r"""Compute local energy from oei, eri and GF

        Args:
            gf: green function

        .. math::

             E = \sum_{pq\sigma} T_{pq} G_{pq\sigma}
                 + \frac{1}{2}\sum_{pqrs\sigma\sigma'} I_{prqs} G_{pr\sigma} G_{qs\sigma'}
                 - \frac{1}{2}\sum_{pqrs\sigma} I_{pqrs} G_{ps\sigma} G_{qr\sigma}

        if :math:`L_\gamma` tensor is used
        [PS: need to rotate Ltensor into (nocc, norb) shape since G's shape is (nocc, norb)],

        .. math::

             E = & \sum_{pq\sigma} T_{pq} G_{pq\sigma}
                 + \frac{1}{2}\sum_{\gamma,pqrs\sigma} L_{\gamma,ps} L_{\gamma,qr} G_{pr\sigma} G_{qs\sigma'}
                 - \frac{1}{2}\sum_{\gamma,pqrs\sigma} L_{\gamma,ps} L_{\gamma,qr} G_{ps\sigma} G_{qr\sigma} \\
               = & \sum_{pq\sigma} T_{pq} G_{pq\sigma}
                 + \frac{1}{2}\sum_{\gamma,pq\sigma\sigma'} (L_\gamma G_\sigma)_{pq} (L_\gamma G_{\sigma'})_{pq}
                 - \frac{1}{2}\sum_{\gamma,\sigma} [\sum_{pq} L_{\gamma,pq} G_{pq\sigma}]^2

        i.e. the Ecoul is :math:`\left[\frac{\bra{\Psi_T}L\ket{\Psi_w}{\bra{\Psi_T}\Psi_w\rangle}\right]^2`,
        which is the TL_Theta tensor in the code
        """
        # approach 0) : most efficient way to compute the energy: use Ltensors instead of eri
        vbias2 = vbias * vbias
        ej = 2.0 * backend.einsum("zn->z", vbias2)
        ek = backend.einsum("znpr, znrp->z", TL_theta, TL_theta)
        e2 = ej - ek

        # approach 1) : most inefficient way
        # e2 = 2.0 * backend.einsum("prqs, zpr, zqs->z", eri, gf, gf)
        # e2 -= backend.einsum("prqs, zps, zqr->z", eri, gf, gf)

        # approach 3): use normal way without using ltensors
        # vjk = 2.0 * backend.einsum("prqs, zpr->zqs", eri, gf) # E_coulomb
        # vjk -= backend.einsum("prqs, zps->zqr", eri, gf)  # exchange
        # e2 = backend.einsum("zqs, zqs->z", vjk, gf)

        e1 = 2 * backend.einsum("zpq, pq->z", gf, h1e)
        energy = e1 + e2 + self.nuc_energy
        return energy

    def update_weight(self, overlap, cfb, cmf, local_energy, time):
        r"""
        Update the walker coefficients using two different schemes.

        a). Hybrid scheme:

        .. math::

              W^{(n+1)} =

        b). Local scheme:

        .. math::

              W^{(n+1)} =
        """
        newoverlap = self.walker_trial_overlap()
        # be cautious! power of 2 was neglected before.
        overlap_ratio = (
            backend.linalg.det(newoverlap) / backend.linalg.det(overlap)
        ) ** 2

        # the hybrid energy scheme
        if self.energy_scheme == "hybrid":
            self.ebound = (2.0 / self.dt) ** 0.5
            hybrid_energy = -(backend.log(overlap_ratio) + cfb + cmf) / self.dt
            hybrid_energy = backend.clip(
                hybrid_energy.real,
                a_min=-self.ebound,
                a_max=self.ebound,
                out=hybrid_energy.real,
            )
            self.hybrid_energy = (
                hybrid_energy if self.hybrid_energy is None else self.hybrid_energy
            )

            importance_func = backend.exp(
                -self.dt * 0.5 * (hybrid_energy + self.hybrid_energy)
            )
            self.hybrid_energy = hybrid_energy
            phase = (-self.dt * self.hybrid_energy - cfb).imag
            phase_factor = backend.array(
                [max(0, backend.cos(iphase)) for iphase in phase]
            )
            importance_func = backend.abs(importance_func) * phase_factor

        elif self.energy_scheme == "local":
            # The local energy formalism
            overlap_ratio = overlap_ratio * backend.exp(cmf)
            phase_factor = backend.array(
                [max(0, backend.cos(backend.angle(iovlp))) for iovlp in overlap_ratio]
            )
            importance_func = (
                backend.exp(-self.dt * backend.real(local_energy)) * phase_factor
            )

        else:
            raise ValueError(f"scheme {self.energy_scheme} is not available!!!")

        self.walkers.weights *= importance_func

    def kernel(self, trial_wf=None):
        r"""main function for QMC time-stepping

        trial_wf: trial wavefunction

        TODO: move the force_bias calculation into separate function

        Green's function is:

        .. math::

            G_{pq} = [\psi_w (\Psi^\dagger_T \psi_w)^{-1} \Psi^\dagger_T]_{qp}

        TL_tensor (precomputed) is:

        .. math::

            TL_{pq} = (\Psi^\dagger_T L_\gamma)_{pq}

        And :math:`\Theta_w` is:

        .. math::

            \Theta_w = \psi_w (\Psi^\dagger_T\psi_w)^{-1} = \psi_w S^{-1}_w

        where :math:`S_w` is the walker-trial overlap.

        Then :math:`(TL)\Theta_k` determines the force bias:

        .. math::

           F_\gamma = \sqrt{-\Delta\tau} \sum_\sigma [(TL)\Theta_w]
        """

        backend.random.seed(self.random_seed)

        logger.info(self, "\n======== get integrals ========")
        h1e, eri, ltensor = self.get_integrals()

        # moved scipy.linalg.expm(-self.dt/2 * h1e) into build propagator and re-used it in propagation
        self.build_propagator(h1e, eri, ltensor)

        time = 0.0
        energy_list = []
        time_list = []
        while time <= self.total_time:
            dump_result = int(time / self.dt) % self.print_freq == 0

            # compute the force_bias (TODO: re-arrange the following code into separate function for force_bias)
            # pre-processing: prepare walker tensor
            overlap = self.walker_trial_overlap()
            inv_overlap = backend.linalg.inv(overlap)

            theta = backend.einsum("zqp, zpr->zqr", self.walkers.phiw, inv_overlap)
            if dump_result:
                logger.debug(
                    self,
                    "\nnorm of walker overlap: %15.8f",
                    backend.linalg.norm(overlap),
                )

            gf = backend.einsum("zqr, pr->zpq", theta, self.trial.psi.conj())
            # :math:`(\Psi_T L_{\gamma}) \psi_w (\Psi_T \psi_w)^{-1}`
            TL_theta = backend.einsum("npq, zqr->znpr", self.TL_tensor, theta)

            # trace[TL_theta] is the force_bias
            vbias = backend.einsum("znpp->zn", TL_theta)
            if dump_result:
                logger.debug(
                    self, "norm of vbias:   %15.8f", backend.linalg.norm(vbias)
                )

            # compute local energy for each walker
            local_energy = self.local_energy(TL_theta, h1e, eri, vbias, gf)
            energy = backend.sum(
                [
                    self.walkers.weights[i] * local_energy[i]
                    for i in range(len(local_energy))
                ]
            )
            energy = energy / backend.sum(self.walkers.weights)

            # imaginary time propagation
            xbar = -backend.sqrt(self.dt) * (1j * 2 * vbias - self.mf_shift)
            # TODO: may apply bias bounding

            cfb, cmf = self.propagation(self.walkers, xbar, ltensor)

            #  phaseless approximation
            self.update_weight(overlap, cfb, cmf, local_energy, time)

            # periodic re-orthogonalization
            if int(time / self.dt) == self.renorm_freq:
                self.renormalization()

            # print energy and time
            if dump_result:
                time_list.append(time)
                energy_list.append(energy)
                logger.info(self, f" Time: {time:9.3f}    Energy: {energy:15.8f}")

            time += self.dt

        return time_list, energy_list
