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
import time
import warnings

from openms.qmc.trial import TrialHF
from . import generic_walkers as gwalker
from pyscf.lib import logger
from openms.lib.logger import task_title
from openms.lib.boson import Boson
from openms.qmc.estimators import local_eng_elec_chol
from openms.qmc.propagators import Phaseless, PhaselessElecBoson


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

        self.trial = kwargs.get("trial", None)
        self.mf = mf

        # walker parameters
        # TODO: move these variables into walker object
        self.__dict__.update(kwargs)
        self.taylor_order = taylor_order
        self.num_walkers = num_walkers
        self.renorm_freq = renorm_freq
        self.random_seed = random_seed

        # walker_tensors/coeff are moved into walkers class (phiw, weights) respectively
        # self.walker_coeff = None
        # self.walker_tensors = None

        #self.mf_shift = None
        self.print_freq = 10

        self.hybrid_energy = None

        self.batched = batched

        self.build()  # setup calculations

    def build(self):
        r"""
        Build up the afqmc calculations
        """
        # set up trial wavefunction
        logger.info(self, task_title("Initialize Trial WF and Walker"))
        self.spin_fac = 1.0
        if self.trial is None:
            if self.mf is not None:
                logger.info(self, f"Mean-field reference is {self.mf.__class__}")
                self.trial = TrialHF(self.mol, mf=self.mf)
            else:
                logger.info(self, f"Mean-field reference is None, we will build from RHF")
                self.trial = TrialHF(self.mol)

        # set up walkers
        # moved this into walker class
        # temp = self.trial.wf.copy()
        # self.walker_tensors = backend.array([temp] * self.num_walkers, dtype=backend.complex128)
        # self.walker_coeff = backend.array([1.] * self.num_walkers)
        logger.info(self, task_title("Set up walkers"))
        self.walkers = gwalker.Walkers_so(nwalkers=self.num_walkers, trial=self.trial)
        logger.info(self, "Done!")

        logger.info(self, task_title("Get integrals"))
        self.h1e, self.eri, self.ltensor = self.get_integrals()
        logger.info(self, "Done!")

        # prepare the propagator
        logger.info(self, task_title("preparing  propagator"))
        if isinstance(self.system, Boson):
            logger.info(
                self,
                "\nsystem is a electron-boson coupled system!"
                + "\nPhaselessElecBoson propagator is to be used!\n",
            )
            self.propagator = PhaselessElecBoson(
                dt=self.dt,
                taylor_order=self.taylor_order,
                energy_scheme=self.energy_scheme,
                quantization="first",
            )
        else:
            logger.info(
                self,
                "\nsystem is a bare electronic system!"
                + "\nPhaseless propagator is to be used!\n",
            )
            self.propagator = Phaseless(
                dt=self.dt,
                taylor_order=self.taylor_order,
                energy_scheme=self.energy_scheme,
            )
        logger.info(self, "Done!")

        self.propagator.dump_flags()


    def dump_flags(self):
        r"""dump flags (TBA)
        """
        pass

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

        warnings.warn(
            "The 'walker_trial_overlap' function is deprecated and will be removed in a future version. "
            "Please use the 'trial.ovlp_with_walkers' function instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        return backend.einsum("pr, zpq->zrq", self.trial.psi.conj(), self.walkers.phiw)

    def orthogonalization(self):
        r"""
        Renormalizaiton and orthogonaization of walkers
        """

        ortho_walkers = backend.zeros_like(self.walkers.phiw)
        for idx in range(self.walkers.phiw.shape[0]):
            ortho_walkers[idx] = backend.linalg.qr(self.walkers.phiw[idx])[0]
        self.walkers.phiw = ortho_walkers

    # renormalization is to be deprecated
    renormalization = orthogonalization

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

        e1 = 2.0 * backend.einsum("zSpq,pq->z", G1p, h1e) * self.spin_fac

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

        warnings.warn(
            "The qmc.local_energy function is deprecated and will be removed in a future version. "
            "Please use 'estimators.local_eng_elec or local_eng_elec_chol' function instead.",
            DeprecationWarning,
            stacklevel=2,
        )
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

        e1 = 2.0 * backend.einsum("zpq, pq->z", gf, h1e)
        energy = e1 + e2 + self.nuc_energy
        return energy

    def kernel(self, trial_wf=None):
        r"""main function for QMC time-stepping

        trial_wf: trial wavefunction
        walkers: walker function

        """

        backend.random.seed(self.random_seed)


        h1e = self.h1e
        eri = self.eri
        ltensor = self.ltensor
        propagator = self.propagator

        trial = self.trial if trial_wf is None else trial_wf
        walkers = self.walkers

        # setup propagator
        # self.build_propagator(h1e, eri, ltensor)
        propagator.build(h1e, eri, ltensor, self.trial)

        # start the propagation
        tt = 0.0
        energy_list = []
        time_list = []
        wall_t0 = time.time()
        while tt <= self.total_time:
            dump_result = int(tt / self.dt) % self.print_freq == 0

            # step 1): get force bias (note: TL_tensor and mf_shift moved into propagator.atrributes)
            gf, TL_theta = trial.force_bias(
                walkers, propagator.TL_tensor, verbose=dump_result
            )
            # trace[TL_theta] is the force_bias
            vbias = backend.einsum("znpp->zn", TL_theta)

            # step 2): property calculations
            # compute local energy for each walker
            # local_energy = self.local_energy(TL_theta, h1e, eri, vbias, gf)

            walkers.eloc = local_eng_elec_chol(TL_theta, h1e, eri, vbias, gf)
            walkers.eloc += self.nuc_energy

            energy = backend.dot(walkers.weights, walkers.eloc)
            energy = energy / backend.sum(walkers.weights)

            # imaginary time propagation
            # TODO: may apply bias bounding
            xbar = -backend.sqrt(self.dt) * (1j * 2 * vbias - propagator.mf_shift)

            # step 3): propagate walkers and update weights
            # self.propagation(walkers, xbar, ltensor)
            propagator.propagate_walkers(trial, walkers, vbias, ltensor)

            # moved phaseless approximation to propagation
            # since it is associated with propagation type
            # self.update_weight(overlap, cfb, cmf)
            if dump_result:
                logger.debug(self, f"local_energy:   {walkers.eloc}")

            # step 4): periodic re-orthogonalization
            if int(tt / self.dt) == self.renorm_freq:
                self.orthogonalization()

            # print energy and time
            if dump_result:
                time_list.append(tt)
                energy_list.append(energy)
                logger.info(self, f" Time: {tt:9.3f}    Energy: {energy:15.8f}")

            tt += self.dt

        return time_list, energy_list
