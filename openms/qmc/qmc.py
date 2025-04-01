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
from abc import abstractmethod
from pyscf import lo, scf, fci
from pyscf import tools as pyscftools
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
from . import tools

from pyscf.lib import logger
from openms.lib.logger import task_title
from openms.lib.boson import Boson
from openms.qmc.trial import make_trial, multiCI
from openms.qmc.estimators import local_eng_elec_chol
from openms.qmc.estimators import local_eng_elec_chol_new

from openms.qmc.propagators import Phaseless, PhaselessElecBoson


def qr_ortho(phiw):
    r"""
    phiw size is [nao, nalpha/nbeta]
    """
    Qmat, Rmat = backend.linalg.qr(phiw)
    Rdiag = backend.diag(Rmat)
    ## don't need to work on the sign
    # signs = backend.sign(Rdiag)
    # Qmat = backend.dot(Qmat, backend.diag(signs))
    log_det = backend.sum(backend.log(backend.abs(Rdiag)))

    return Qmat, log_det


def qr_ortho_batch(phiw):
    r"""
    phiw size is [nwalker, nao, nalpha/nbeta]
    """
    Qmat, Rmat = backend.linalg.qr(phiw)
    Rdiag = backend.einsum("wii->wi", Rmat)
    log_det = backend.einsum("wi->w", backend.log(abs(Rdiag)))
    return Qmat, log_det


def kernel(mc, propagator, trial=None):
    r"""An universal kernel for qmc propagations"""

    # prepare propagation
    backend.random.seed(mc.random_seed)
    logger.note(self, task_title(" Entering the main kernel of AFQMC"))

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


class QMCbase(object):
    r"""
    Basic QMC class
    """

    def __init__(
        self,
        system,  # or molecule
        mf=None,
        dt=0.005,
        # nsteps=25,
        total_time=5.0,
        num_walkers=100,
        renorm_freq=5,
        random_seed=1,
        taylor_order=6,
        energy_scheme=None,
        propagator_options=None,
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
           OAO:         use OAO representation or not
        """
        # import openms
        # logger.info(self, openms.__logo__)

        if "pra2024" not in runtime_refs:
            runtime_refs.append("pra2024")

        # io params
        self.stdout = sys.stdout
        self.verbose = kwargs.get("verbose", 1)
        self.print_freq = 10

        # system parameters
        self.system = self.mol = system
        self.uhf = kwargs.get("uhf", False)
        self.OAO = kwargs.get("OAO", True)
        #
        # TODO: when uhf is turned offf, check whehter the system is really a closed-shell system
        #
        self.use_so = kwargs.get("use_so", False)
        if "use_so" not in kwargs:
            self.use_so = kwargs.get("use_spinorbital", False)
        if self.uhf:  # set use_so true if uhf is used.
            self.use_so = True

        # TODO: when the eri size is larger than 50% of the available memory,
        # turn on the block_decompose_eri anyway!
        self.block_decompose_eri = kwargs.get("block_decompose_eri", False)
        self.chol_thresh = kwargs.get("chol_thresh", 1.0e-6)

        # propagator params
        self.dt = dt
        self.total_time = total_time
        # FIXME: in the future, we will use nsteps to control the propagation,
        # and deprecate the total_time
        self.nsteps = int(total_time / dt)  # nsteps
        # self.nsteps = nsteps  #
        self.propagator = None
        self.nblocks = 500  #
        self.eq_time = 2.0  # time of equilibration phase
        self.eq_steps = int(
            self.eq_time / self.dt
        )  # Number of time steps for the equilibration phase
        self.stablize_freq = 5  # Frequency of stablization(re-normalization) steps
        self.energy_scheme = energy_scheme

        # trial parameters
        self.trial = kwargs.get("trial", None)
        self.mf = mf

        # walker parameters
        # TODO: move these variables into walker object
        self.__dict__.update(kwargs)

        self.taylor_order = taylor_order
        # self.num_walkers = num_walkers
        self.random_seed = random_seed
        self.renorm_freq = renorm_freq
        self.stablize_freq = self.renorm_freq

        # walker_tensors/coeff are moved into walkers class (phiw, weights) respectively
        # self.walker_coeff = None
        # self.walker_tensors = None

        self.hybrid_energy = None
        self.batched = kwargs.get("batched", True)

        # parameters for walker weight control
        self.pop_control_freq = 5  # weight control frequency

        # other variables for Hamiltonian
        self.geb = None  # TODO: optimize the handling of geb
        self.chol_Xa = None  # TODO: optimize the handling of this term

        # update propagator options
        self.propagator_options = {
            "verbose": self.verbose,
            "stdout": self.stdout,
            "num_fake_fields": 0,
            "quantizaiton": "second",
            "energy_scheme": self.energy_scheme,
            "taylor_order": self.taylor_order,
        }

        self.walker_options = gwalker.default_walker_options
        tmp_walker_options = kwargs.get("walker_options", None)
        print("tmp_walker_options = ", tmp_walker_options)
        if tmp_walker_options is not None:
            self.walker_options.update(tmp_walker_options)
        if num_walkers is not None:
            self.walker_options["nwalkers"] = num_walkers

        if propagator_options is not None:
            self.propagator_options.update(propagator_options)

        # property calculations
        self.property_calc_freq = kwargs.get("property_calc_freq", 10)
        self.default_properties = ["energy"]  # TODO: more property calculators
        self.stacked_variables = [
            "weights",
            "unscaled_weights",
            "walker_hybrid_energies",
            "walker_local_energies",
        ]  # may add more variables
        self.property_buffer = backend.zeros(
            (len(self.stacked_variables),), dtype=backend.complex128
        )
        self.eshift = 0.0

        # set up calculations
        self.build()  # setup calculations

        # wall time analysis variables
        self.wt_propagator = 0.0
        self.wt_observables = 0.0
        self.wt_weight_control = 0.0
        self.wt_ortho = 0.0
        self.wt_io = 0.0
        self.wt_tot = 0.0


    def build(self):
        r"""
        Build up the afqmc calculations, including:

        1) setup trial WF
        2) setup integrals
        3) setup walkers
        4) setup propagators

        TODO: add the options to get integrals and trials from files
        """
        # 1) set up trial wavefunction
        logger.note(self, task_title("Initialize Trial WF and Walker"))

        # number of spin components
        self.ncomponents = 1
        if self.uhf or self.use_so:
            self.ncomponents = 2
        # if trial is not None, get ncomponents from trial

        self.spin_fac = 1.0 / self.ncomponents

        # FIXME: determine how to better handle trial options from qmc kwargs
        #        may directly make trial_options as a qmc class argument.
        # make a trial_options for creating trail WF.
        trial_options = {
            "verbose": self.verbose,
            "stdout": self.stdout,
            "trail_type": "RHF",
            "numdets": 1,
            "OAO": self.OAO,
            "uhf": self.uhf,
            "ncomponents": self.ncomponents,
        }

        # TODO: 1) simplify this the construction of trial
        #      2) handle different trial, RHF, UHF, GHF, ROHF, and MCSCF.
        if self.trial is None:
            self.trial = make_trial(self.mol, mf=self.mf, **trial_options)
            if self.mf is None:
                self.mf = self.trial.mf
            logger.debug(self, f"Debug: self.mf = {self.mf}")
            logger.debug(self, f"Debug: trail.mf = {self.trial.mf}")
        else:
            logger.info(self, "Trial WF is set from the input")

        # 2) make h1e in Spin orbital
        t0 = time.time()
        logger.note(self, task_title("Get integrals"))
        self.h1e, self.ltensor = self.get_integrals()

        # half-rotate integrals
        self.trial.half_rotate_integrals(self.h1e, self.ltensor)
        logger.note(
            self,
            task_title(f"Get integrals ... Done! Time used: {time.time()-t0: 7.3f} s"),
        )

        # 3) set up walkers
        t0 = time.time()
        logger.note(self, task_title("Set up walkers"))
        # self.walkers = gwalker.Walkers_so(self.trial, nwalkers=self.num_walkers)
        self.walkers = gwalker.make_walkers(self.trial, self.walker_options)

        # calculate green's function
        ovlp = self.trial.ovlp_with_walkers_gf(self.walkers)
        logger.debug(self, f"Debug: initial trial_walker overlap is\n {ovlp}")
        logger.note(
            self, f"Setup walkers ... Done! Time used: {time.time()-t0: 7.3f} s"
        )

        # 4) prepare the propagator
        # TODO: may use a dict to switch different propagator
        t0 = time.time()
        logger.note(self, task_title("Prepare propagator"))
        if isinstance(self.system, Boson):
            logger.note(
                self,
                "\nsystem is a electron-boson coupled system!"
                + "\nPhaselessElecBoson propagator is to be used!\n",
            )
            self.propagator = PhaselessElecBoson(self.dt, **self.propagator_options)
            self.propagator.chol_Xa = self.chol_Xa
        else:
            logger.note(
                self,
                "\nsystem is a bare electronic system!"
                + "\nPhaseless propagator is to be used!\n",
            )
            self.propagator = Phaseless(dt=self.dt, **self.propagator_options)
        # FIXME: temporarily assign system to propgator as well
        # FIXME: need to decide how to handle access system from the propagation
        self.propagator.system = self.system

        self.propagator.build(self.h1e, self.ltensor, self.trial, self.geb)

        logger.note(
            self,
            task_title(
                f"Prepare propagator ... Done! Time used: {time.time()-t0: 7.3f} s"
            ),
        )

    @abstractmethod
    def cast2backend(self):
        r"""cast the tensors to backend for following simulation"""
        raise NotImplementedError


    def dump_flags(self):
        r"""dump flags"""

        title = f"{self.__class__.__name__} simulation using OpenMS package"
        logger.note(self, task_title(title, level=0))

        representation = "OAO" if self.OAO else "MO"
        logger.note(self, f" {representation} representation is used")
        logger.note(self, f" Time step              : {self.dt:7.3f}")
        logger.note(self, f" Total time             : {self.total_time:7.3f}")
        logger.note(self, f" Number of steps        : {self.nsteps:7d}")
        logger.note(self, f" Energy scheme          : {self.energy_scheme}")
        logger.note(self, f" Number of chols        : {self.ltensor.shape[0]:5d}")
        logger.note(self, f" Threshold of chols     : {self.chol_thresh:7.3e}")
        logger.note(self, f" Use Spin orbital?      : {self.use_so}")
        logger.note(self, f" Unrestricted spin?     : {self.uhf}")
        logger.note(self, f" No. of spin components : {self.ncomponents:5d}")

        # flags of propagators (walkers)
        logger.note(
            self, f" Propagator is        : {self.propagator.__class__.__name__}"
        )
        self.propagator.dump_flags()

        # flags of trial WF and walkers
        self.trial.dump_flags()
        self.walkers.dump_flags()


    def measurements(self, walkers, step):
        r"""masurement of physical observables, e.g., energies"""

        pass


    def get_integrals(self):
        r"""return oei, eri, and cholesky tensors in OAO or MO

        .. note::

           this part will be replaced by the code in tools which provide
           either full of block decomposition of the eri.
        """

        # with h5py.File("input.h5") as fa:
        #    ao_coeff = fa["ao_coeff"][()]

        hcore = scf.hf.get_hcore(self.mol)
        logger.debug(self, f"Debug: norm of hcore = {backend.linalg.norm(hcore)}")
        if self.OAO:
            # Lowdin orthogonalization S^{-/2} -> X
            overlap = self.mol.intor("int1e_ovlp")
            Xmat = lo.orth.lowdin(overlap)
        else:
            Xmat = self.mf.mo_coeff
        logger.debug(self, f"Debug: basis_transform_matrix = \n{Xmat}")
        norb = Xmat.shape[0]

        #
        # old code, will remove this part, as we don't need the eri
        #

        # get h1e, and ori in OAO representation
        mol = self.mol._mol if isinstance(self.mol, Boson) else self.mol
        if not self.block_decompose_eri:
            import tempfile
            ftmp = tempfile.NamedTemporaryFile()

            pyscftools.fcidump.from_mo(mol, ftmp.name, Xmat)
            hcore, eri, self.nuc_energy = tools.read_fcidump(ftmp.name, norb)

            # Cholesky decomposition of eri
            # here eri uses chemist's notation
            # [il|kj]  a^\dag_i a^\dag_j a_k a_l  = <ij|kl> a^\dag_i a^\dag_j a_k a_l
            # [] -- chemist, <> physicist notations

            # Cholesky decomposition of eri (ij|kl) -> L_{\gamma,ij} L_{\gamma,kl}
            eri_2d = eri.reshape((norb**2, -1))
            u, s, v = scipy.linalg.svd(eri_2d)
            mask = s > self.chol_thresh
            # ltensor = u * backend.sqrt(s)
            ltensor = u[:, mask] * backend.sqrt(s[mask])
            ltensor = ltensor.T
            ltensor = ltensor.reshape(ltensor.shape[0], norb, norb)
        else:
            # ---------------------------------------
            #  use block decomposition of eri in tools
            # ---------------------------------------
            # get h1e, eri, ltensors in OAO/MO representation
            hcore, ltensor, self.nuc_energy = tools.get_h1e_chols(
                mol, Xmat=Xmat, thresh=self.chol_thresh
            )

        # shape of h1e [nspin, nao, nao]
        h1e = backend.array([hcore for _ in range(self.ncomponents)])
        logger.debug(self, f"\nDebug: h1e.shape = {h1e.shape}")
        logger.debug(
            self, f"Debug: norm of hcore in OAO/MO = {backend.linalg.norm(h1e[0])}"
        )
        logger.debug(
            self, f"Debug: norm of ltensor in OAO/MO = {backend.linalg.norm(ltensor)}"
        )

        # with h5py.File("input.h5", "r+") as fa:
        #    fa["h1e"] = h1e
        #    fa["nuc_energy"] = self.nuc_energy
        #    fa["cholesky"] = ltensor

        if isinstance(self.system, Boson):
            system = self.system
            # Add boson-mediated oei and eri:
            # shape [nm, nao, nao]

            chol_eb = system.gmat.copy()
            nmodes = system.gmat.shape[0]

            # transform into OAO
            chol_eb = backend.einsum(
                "ik, mkj, jl -> mil", Xmat.T, chol_eb, Xmat, optimize=True
            )

            # geb is the bilinear coupling term
            tmp = (system.omega * 0.5) ** 0.5
            self.geb = chol_eb * tmp[:, backend.newaxis, backend.newaxis]

            logger.debug(self, f"size of chol before adding DSE: {ltensor.shape[0]}")
            # add terms due to decoupling of bilinear term
            if self.propagator_options["decouple_bilinear"]:
                logger.debug(
                    self, f"creating chol due to decomposition of bilinear term"
                )

                # TODO: set A_\alpha and O_\alpha operator as input variables
                decoup_Afactor = backend.ones(nmodes)
                decoup_Ofactor =  (system.omega * 0.5) ** 0.5
                # \sqrt{1-A_\alpha} + \sqrt{A_\alpha}
                tmp = backend.sqrt(backend.ones(nmodes) - decoup_Afactor) + backend.sqrt(decoup_Afactor)
                Lga = chol_eb * tmp[:, backend.newaxis, backend.newaxis]
                for imode in range(nmodes):
                    if backend.linalg.norm(Lga[imode]) > 1.0e-10:
                        ltensor = backend.concatenate((ltensor, Lga), axis=0)

                # (1 + 1j) * \sqrt{\omega} (\lambda\cdot D)
                self.chol_Xa = backend.sqrt(decoup_Afactor) * decoup_Ofactor  * (1.0 + 1j)
            else:
                if backend.linalg.norm(chol_eb) > 1.0e-10:
                    ltensor = backend.concatenate((ltensor, chol_eb), axis=0)

        self.nfields = ltensor.shape[0]
        return h1e, ltensor


    def propagate_walkers(self, walkers, xbar, ltensor):
        pass


    def measure_observables(self, operator):
        r"""Placeholder for measure_observables.
        According to the operator, we measure the expectation values

        TODO: may not used this one!
        """
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

        .. note::

            Since the orthogonaization depends on the type of walkers, we are moving
            this function into walkers. Hence, the function here is to be deprecated!
        """

        # 1) old code
        # ortho_walkers = backend.zeros_like(self.walkers.phiwa)
        detR = backend.zeros(self.walkers.nwalkers, dtype=backend.complex128)
        for iw in range(self.walkers.phiwa.shape[0]):
            self.walkers.phiwa[iw], log_det = qr_ortho(self.walkers.phiwa[iw])

            if self.walkers.ncomponents > 1:
                self.walkers.phiwb[iw], log_det_b = qr_ortho(self.walkers.phiwb[iw])
                log_det += log_det_b

            detR[iw] = backend.exp(log_det - self.walkers.detR_shift[iw])
            self.walkers.log_detR[iw] += backend.log(detR[iw])
            self.walkers.detR[iw] = detR[iw]
            self.walkers.ovlp[iw] = self.walkers.ovlp[iw] / detR[iw]

        """
        # 2) batched code
        self.walkers.phiwa, log_det = qr_ortho_batch(self.walkers.phiwa)
        if self.walkers.ncomponents > 1:
            self.walkers.phiwb, log_det_b = qr_ortho_batch(self.walkers.phiwb)
            log_det += log_det_b

        self.walkers.detR = backend.exp(log_det - self.walkers.detR_shift)
        self.walkers.ovlp = self.walkers.ovlp / self.walkers.detR
        """


        if self.walkers.boson_phiw is not None:
            ortho_walkers = backend.zeros_like(self.walkers.boson_phiw)
            # for iw in range(self.walkers.boson_phiw.shape[0]):
            #    ortho_walkers[iw] = backend.linalg.qr(self.walkers.boson_phiw[iw])[0]
            # self.walkers.boson_phiw = ortho_walkers

    # renormalization is to be deprecated
    orthonormalization = orthogonalization

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
        r"""Compute local energy from oei, eri and GF.

        Warning: this function is Deprecaeted and moved to propagator to handle
        the cases of different trial and walkers.

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

    # we use the following function to deal with property calculations
    def property_stack(self, walkers, step):
        r"""
        The function handles the accumulation and periodic reduction of properties.

        Parameters:
            walkers: object
                Contains the walker states and associated properties.
            step: int
                The current simulation step.

        """
        # property_list = self.default_properties
        if step < 0:
            # Initial setup for the property buffer
            logger.debug(
                self, f"Debug: initial buffer shape (in accumulator) is {self.property_buffer.shape}"
            )
            # TODO: compute the initial values of the properties
            return

        # Dictionary linking property names to their computed values
        _data_dict = {
            "weights": backend.sum(walkers.weights),
            "unscaled_weights": backend.sum(walkers.weights_org),
            "walker_hybrid_energies": backend.sum(walkers.ehybrid * walkers.weights),
            "walker_local_energies": backend.sum(walkers.eloc * walkers.weights),
        }

        # Accumulate values for the specified properties
        tmp = [
            _data_dict.get(key, 0.0 + 0.0j) for key in self.stacked_variables
        ]
        self.property_buffer += backend.array(tmp)

        # logger.debug(self, f"Debug: updated buffer shape is {self.property_buffer.shape}")
        logger.debug(self, f"Debug: updated buffer is {self.property_buffer}")

        # Perform periodic property reduction and normalization
        if (step + 1) % self.property_calc_freq == 0:

            # Normalize energies by weights
            weights_idx = self.stacked_variables.index("weights")
            norm = self.property_buffer[weights_idx]

            for idx, name in enumerate(self.stacked_variables):
                if "energies" in name:
                    self.property_buffer[idx] /= norm

            # Note: dont' combine the following loop with above one, for energies,
            # we only need to normalized it against weights
            # Normalize weights over the calculation frequency
            for idx, name in enumerate(self.stacked_variables):
                if "weights" in name:
                    self.property_buffer[idx] /= self.property_calc_freq

            # Update the energy shift using normalized hybrid energies
            idx = self.stacked_variables.index("walker_hybrid_energies")
            self.eshift = self.property_buffer[idx]
            logger.debug(self, f"Debug: update eshift = {self.eshift}")

            # Reset the property buffer for the next accumulation cycle
            self.property_buffer.fill(0.0 + 0.0j)


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
        logger.info(self, f"\n Random seed is {self.random_seed}\n")
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

        logger.debug(self, f"Debug: the initial orthogonalise in walker")
        self.orthogonalization()

        # initialize property calculations
        self.property_stack(walkers, -1)

        # start the propagation
        tt = 0.0
        energy_list = []
        time_list = []
        wall_t0 = time.time()
        logger.info( self,
            f"{'Step':^8}{'Etot':^17}{'Norm':^17}{'Raw_Etot':^17}{'E1':^17}{'E2':^17}  Wall_time"
        )

        # while tt <= self.total_time:
        for step in range(self.nsteps):
            t0 = time.time()
            tt = self.dt * step
            dump_result = step % self.print_freq == 0
            logger.debug(
                self, f"\nDebug: -------------- qmc step {step} -----------------"
            )
            # step 3): periodic re-orthogonalization
            # (FIXME: whether put this at the begining or end, in principle, should not matter)
            if (step + 1) % self.renorm_freq == 0:
                wall_t1 = time.time()
                self.orthogonalization()
                logger.debug(self, f"Debug: orthogonalise at step {step}")
                self.wt_ortho += time.time() - wall_t1

            vbias = None
            # step 1): get force bias (note: TL_tensor and mf_shift moved into propagator.atrributes)
            # store Gf in walkers in order to recycle it in the propagators
            # gf, vbias = trial.get_vbias(walkers, ltensor, verbose=dump_result)
            """
            gf, TL_theta = trial.force_bias(
                walkers, propagator.TL_tensor, verbose=dump_result
            )

            # trace[TL_theta] is the force_bias
            vbias = backend.einsum("znpp->zn", TL_theta)

            # imaginary time propagation
            # TODO: may apply bias bounding
            xbar = -backend.sqrt(self.dt) * (1j * 2 * vbias - propagator.mf_shift)
            """

            wall_t1 = time.time()
            # step 3): propagate walkers and update weights
            # self.propagate_walkers(walkers, xbar, ltensor)
            propagator.propagate_walkers(
                trial, walkers, vbias, ltensor, eshift=self.eshift, verbose=int(dump_result)
            )
            self.wt_propagator += time.time() - wall_t1

            # step 2) weight control
            wall_t1 = time.time()
            self.walkers.weight_control(step)
            self.wt_weight_control += time.time() - wall_t1

            # moved phaseless approximation to propagation
            # since it is associated with propagation type
            # self.update_weight(overlap, cfb, cmf)
            if dump_result:
                logger.debug(self, f"local_energy:   {walkers.eloc}")

            # step 4): estimate energies and other properties if needed
            # We store weights, energies, and other properties of each estimator in local
            # buffer_variables and compute the properties at every print_freq
            wall_t1 = time.time()
            self.property_stack(walkers, step)

            # self.measurements(walkers, step)
            if (step + 1) % self.property_calc_freq == 0:
                # Compute energies and other observables
                energies = propagator.local_energy(h1e, ltensor, walkers, trial, enuc=self.nuc_energy)
                energy = energies[0] / energies[1]

                # Append time and energy to respective lists
                time_list.append(tt)
                energy_list.append(energy)

                # Log the computed energy and other properties
                logger.info(
                    self,
                    f"{step:6d}  {energy:15.7e}  "
                    f"{energies[0]:15.7e}  {energies[1]:15.7e}  "
                    f"{energies[2]:15.7e}  {energies[3]:15.7e}  "
                    f"{time.time() - t0:10.4f}s"
                )
            self.wt_observables += time.time() - wall_t1

            # step 5): TODO: code of checkpoint
            wall_t1 = time.time()
            # if dump_result:
            #     self.save_checkpoint()
            self.wt_io += time.time() - wall_t1

        # TODO: code of analysis, post processing, etc.
        self.wt_tot = time.time() - wall_t0

        self.post_kernel()
        return time_list, energy_list

    def _finalize(self):
        """Hook for dumping results and clearing up the object."""



        logger.note(self, task_title("Wall time analysis"))
        logger.note(self, f" Total             : {self.wt_tot: 9.3f}")
        logger.note(self, f" IO                : {self.wt_io: 9.3f}")
        logger.note(self, f" Measuremnets      : {self.wt_observables: 9.3f}")
        logger.note(self, f" Weight control    : {self.wt_weight_control: 9.3f}")
        logger.note(self, f" Orthogonalization : {self.wt_ortho: 9.3f}")
        logger.note(self, f" Propagator        : {self.wt_propagator: 9.3f}")
        logger.note(self, f"\n Breakdown of propagator:")
        logger.note(self, f"   Overlap & GF    : {self.propagator.wt_ovlp: 9.3f}")
        logger.note(self, f"   Updte weights   : {self.propagator.wt_weight: 9.3f}")
        logger.note(self, f"   Onebody term    : {self.propagator.wt_onebody: 9.3f}")
        logger.note(self, f"   Twobody term    : {self.propagator.wt_twobody: 9.3f}")
        logger.note(self, f"   Bilinear term   : {self.propagator.wt_bilinear: 9.3f}")
        logger.note(self, f"   Bosonic term    : {self.propagator.wt_boson: 9.3f}")
        logger.note(self, f"   Breakdown of twobody:")
        logger.note(self, f"     Force bias    : {self.propagator.wt_fbias: 9.3f}")
        logger.note(self, f"     HS of twobody : {self.propagator.wt_hs: 9.3f}")
        # more wall times TBA.
        logger.note(self, f"")


    def post_kernel(self):
        r"""Prints relevant citation information for calculation."""
        breakline = "=" * 86
        logger.note(self, f"\n{breakline}")
        logger.note(self, f"*  Hoollary, the job is done!\n")

        self._finalize()

        logger.note(self, task_title("Citations"))
        for i, key in enumerate(runtime_refs):
            logger.note(self, f"[{i+1}]. {_citations[key]}")
        logger.note(self, f"{breakline}\n")
