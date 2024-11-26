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
QMC method
----------

DF or Cholesky decomposition:

.. math::

    I_{pqrs} \simeq L_{\lambda, pq} L_{\lambda, rs}

Tensor hypercontraction:

.. math::

    I_{pqrs} \simeq X_{\mu p} X_{\mu q} Z_{\mu\nu} X_{\nu r} X_{\nu s}

it means that the :math:`L` is effectively decomposed into :math:`X`

.. math::

    L_{\lambda pq} = X_{\lambda p} X_{\lambda q}

"""

import sys
from pyscf import tools, lo, scf, fci
from pyscf.gto import mole
import numpy as backend
import scipy
import itertools
import logging
import h5py
import time
import warnings

from openms import runtime_refs, _citations
from . import generic_walkers as gwalker
from pyscf.lib import logger
from openms.lib.logger import task_title
from openms.lib.boson import Boson
from openms.qmc.trial import make_trial
from openms.qmc.estimators import local_eng_elec_chol
from openms.qmc.estimators import local_eng_elec_chol_new

from openms.qmc.propagators import Phaseless, PhaselessElecBoson


def kernel(mc, propagator, trial=None):
    r"""An universal kernel for qmc propagations
    """

    # prepare propagation
    backend.random.seed(mc.random_seed)

    # integrals
    h1e = mc.h1e
    ltensor = mc.ltensor
    propagator = mc.propagator

    trial = mc.trial if trial_wf is None else trial_wf
    walkers = mc.walkers

    # generalize the propagator build for any typs
    # (fermions, bosons, or fermion-boson mixture).
    propagator.build(h1e, ltensor, mc.trial, mc.geb)

    # start the propagation
    tt = 0.0
    energy_list = []
    time_list = []
    wall_t0 = time.time()
    while tt <= mc.total_time:
        dump_result = int(tt / mc.dt) % mc.print_freq == 0

        # step 1: compute bias and GFs

        # step 2: compute energies and other properties

        # step 3: propagate walkers

        # step 4: update weights and re-orthogonalize and re-normalize walkers

        # step 5: store intermediate variables (if needed)
        if dump_result:
            time_list.append(tt)
            energy_list.append(energy)
            # and other large data into .h5 file

        tt += self.dt

    # finalize the propagations
    mc.post_kernel()
    return time_list, energy_list


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
        propagator_options = None,
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
        # import openms
        # logger.info(self, openms.__logo__)

        if "pra2024" not in runtime_refs:
            runtime_refs.append("pra2024")

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

        self.print_freq = 10
        self.hybrid_energy = None
        self.batched = batched

        self.geb = None  # TODO: optimize the handling of geb
        self.chol_Xa = None  # TODO: optimize the handling of this term
        self.chol_bilinear = None  # TODO: optimize the handling of this term

        # update propagator options
        self.propagator_options = {
            "verbose": self.verbose,
            "stdout" : self.stdout,
            "quantizaiton": "first",
            "energy_scheme" : self.energy_scheme,
            "talor_order" : self.taylor_order,
        }

        if propagator_options is not None:
            self.propagator_options.update(propagator_options)

        self.build()  # setup calculations

    def build(self):
        r"""
        Build up the afqmc calculations
        """
        # set up trial wavefunction
        logger.note(self, task_title("Initialize Trial WF and Walker"))

        # make a trial_options for creating trail WF.
        trial_options = {
            "verbose": self.verbose,
            "stdout": self.stdout,
            "trail_type": "RHF",
            "numdets": 1,
        }

        self.spin_fac = 1.0
        if self.trial is None:
            self.trial = make_trial(self.mol, mf=self.mf, **trial_options)

        # set up walkers
        logger.note(self, task_title("Set up walkers"))
        self.walkers = gwalker.Walkers_so(nwalkers=self.num_walkers, trial=self.trial)
        logger.note(self, "Done!")

        logger.note(self, task_title("Get integrals"))
        self.h1e, self.eri, self.ltensor = self.get_integrals()
        logger.note(self, "Done!")

        # prepare the propagator
        logger.note(self, task_title("preparing  propagator"))
        if isinstance(self.system, Boson):
            logger.note(
                self,
                "\nsystem is a electron-boson coupled system!"
                + "\nPhaselessElecBoson propagator is to be used!\n",
            )
            self.propagator = PhaselessElecBoson(self.dt, **self.propagator_options)
            self.propagator.chol_bilinear = self.chol_bilinear
            self.propagator.chol_Xa  = self.chol_Xa
        else:
            logger.note(
                self,
                "\nsystem is a bare electronic system!"
                + "\nPhaseless propagator is to be used!\n",
            )
            self.propagator = Phaseless(dt=self.dt, **self.propagator_options)
        logger.info(self, "Done!")

        # FIXME: temporarily assign system to propgator as well
        # FIXME: need to decide how to handle access system from the propagation
        self.propagator.system = self.system

        # self.propagator.build(self.h1e, self.ltensor, self.geb)

    def dump_flags(self):
        r"""dump flags (TBA)

        """
        title = f"{self.__class__.__name__} simulation using OpenMS package"
        logger.note(self, task_title(title, level=0))

        logger.note(self, f" Time:              {self.total_time:9.3f}")
        logger.note(self, f" Time step :        {self.dt:6.3f}")
        logger.note(self, f" Energy scheme:     {self.energy_scheme}")
        logger.note(self, f" Number of walkers: {self.num_walkers:5d}")

        # flags of propagators (walkers)
        self.propagator.dump_flags()

        # flags of trial WF (TBA)
        logger.note(self, f" Trial state:       ")


    def get_integrals(self):
        r"""return oei, eri, and cholesky tensors in OAO

        .. note::

           this part will be replaced by the code in tools which provide
           either full of block decomposition of the eri.
        """

        # with h5py.File("input.h5") as fa:
        #    ao_coeff = fa["ao_coeff"][()]

        # Lowdin orthogonizaiton S^{-/2} -> X
        overlap = self.mol.intor("int1e_ovlp")
        Xmat = lo.orth.lowdin(overlap)
        norb = Xmat.shape[0]

        import tempfile

        # get h1e, and ori in OAO representation

        # ---------------------------------------
        # TODO: replace it with the block decomposition in tools
        # ---------------------------------------

        ftmp = tempfile.NamedTemporaryFile()
        tools.fcidump.from_mo(self.mol, ftmp.name, Xmat)
        h1e, eri, self.nuc_energy = read_fcidump(ftmp.name, norb)

        # Cholesky decomposition of eri
        # here eri uses chemist's notation
        # [il|kj]  a^\dag_i a^\dag_j a_k a_l  = <ij|kl> a^\dag_i a^\dag_j a_k a_l
        # [] -- chemist, <> physicist notations

        # Cholesky decomposition of eri (ij|kl) -> L_{\gamma,ij} L_{\gamma,kl}
        eri_2d = eri.reshape((norb**2, -1))
        u, s, v = scipy.linalg.svd(eri_2d)

        # mask = s > 1.e-15
        # ltensor = u[:, mask] * backend.sqrt(s[mask])

        ltensor = u * backend.sqrt(s)
        ltensor = ltensor.T
        ltensor = ltensor.reshape(ltensor.shape[0], norb, norb)

        # with h5py.File("input.h5", "r+") as fa:
        #    fa["h1e"] = h1e
        #    fa["nuc_energy"] = self.nuc_energy
        #    fa["cholesky"] = ltensor

        if isinstance(self.system, Boson):
            system = self.system
            # Add boson-mediated oei and eri:
            # shape [nm, nao, nao]

            chol_eb = system.gmat.copy()
            # transform into OAO
            chol_eb = backend.einsum(
                "ik, mkj, jl -> mil", Xmat.T, chol_eb, Xmat, optimize=True
            )

            # geb is the bilinear coupling term
            tmp = (system.omega * 0.5) ** 0.5
            self.geb = chol_eb * tmp[:, backend.newaxis, backend.newaxis]

            logger.debug(self, f"size of chol before adding DSE: {ltensor.shape[0]}")
            if backend.linalg.norm(chol_eb) > 1.0e-10:
                ltensor = backend.concatenate((ltensor, chol_eb), axis=0)
            logger.debug(self, f"size of chol after adding DSE:  {ltensor.shape[0]}")

            # add terms due to decoupling of bilinear term
            if self.propagator_options["decouple_bilinear"]:
                logger.debug(self, f"creating chol due to decomposition of bilinear term")

                # (1 + 1j) * \sqrt{\omega} (\lambda\cdot D)
                self.chol_bilinear = self.geb * (1.0 + 1j)
                self.chol_Xa = (system.omega) ** 0.5 * (1.0 + 1j)

        self.nfields = ltensor.shape[0]
        return h1e, eri, ltensor


    def propagate_walkers(self, walkers, xbar, ltensor):
        pass

    def measure_observables(self, operator):
        observables = None
        return observables

    def walker_trial_overlap(self):
        r"""
        Compute the overlap between trial and walkers:

        .. math::

            \langle \Psi_T \ket{\Psi_w} = \det[S]

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

        if self.walkers.phiw_b is not None:
            ortho_walkers = backend.zeros_like(self.walkers.phiw_b)
            #for idx in range(self.walkers.phiw_b.shape[0]):
            #    ortho_walkers[idx] = backend.linalg.qr(self.walkers.phiw_b[idx])[0]
            #self.walkers.phiw_b = ortho_walkers

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

        i.e. the Ecoul is :math:`\left[\frac{\bra{\Psi_T}L\ket{\Psi_w}}{\bra{\Psi_T}\Psi_w\rangle}\right]^2`,
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

        self.dump_flags()

        # Note of the rename
        # rename precomputed_ltensor -> TL_tensor
        # rename ltheta -> TL_theta
        # trace_ltheta -> trace_tltheta -> vbias

        # prepare propagation
        backend.random.seed(self.random_seed)

        # print("YZ: walkers WF      =", self.walkers.phiw)
        # print("YZ: walkers weights =", self.walkers.weights)

        h1e = self.h1e
        # eri = self.eri
        ltensor = self.ltensor
        propagator = self.propagator

        trial = self.trial if trial_wf is None else trial_wf
        walkers = self.walkers

        # setup propagator
        # self.build_propagator(h1e, eri, ltensor)
        propagator.build(h1e, ltensor, self.trial, self.geb)


        # start the propagation
        tt = 0.0
        energy_list = []
        time_list = []
        wall_t0 = time.time()
        while tt <= self.total_time:
            dump_result = int(tt / self.dt) % self.print_freq == 0

            # step 1): get force bias (note: TL_tensor and mf_shift moved into propagator.atrributes)
            # store Gf in walkers in order to recycle it in the propagators
            # gf, vbias = trial.get_vbias(walkers, ltensor, verbose=dump_result)
            gf, TL_theta = trial.force_bias(
                walkers, propagator.TL_tensor, verbose=dump_result
            )
            # trace[TL_theta] is the force_bias
            vbias = backend.einsum("znpp->zn", TL_theta)

            # step 2): property calculations
            # compute local energy for each walker
            # local_energy = self.local_energy(TL_theta, h1e, eri, vbias, gf)

            # walkers.eloc = local_eng_elec_chol_new(h1e, ltensor, gf)
            # walkers.eloc = local_eng_elec_chol(TL_theta, h1e, vbias, gf)
            walkers.eloc = propagator.compute_local_energies(TL_theta, h1e, vbias, gf)
            walkers.eloc += self.nuc_energy

            energy = backend.dot(walkers.weights, walkers.eloc)
            energy = energy / backend.sum(walkers.weights)

            # imaginary time propagation
            # TODO: may apply bias bounding
            xbar = -backend.sqrt(self.dt) * (1j * 2 * vbias - propagator.mf_shift)

            # step 3): propagate walkers and update weights
            # self.propagate_walkers(walkers, xbar, ltensor)
            propagator.propagate_walkers(trial, walkers, vbias, ltensor, verbose=int(dump_result))

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

        self.post_kernel()
        return time_list, energy_list

    def _finalize(self):
        '''Hook for dumping results and clearing up the object.'''
        pass

    def post_kernel(self):
        r"""Prints relevant citation information for calculation."""
        breakline = '='*80
        logger.note(self, f"\n{breakline}")
        logger.note(self, f"*  Hoollary, the job is done!\n")

        self._finalize()

        logger.note(self, f"Citations:")
        for i, key in enumerate(runtime_refs):
            logger.note(self, f"[{i+1}]. {_citations[key]}")
        logger.note(self, f"{breakline}\n")
