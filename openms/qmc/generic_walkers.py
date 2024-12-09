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
Walkers functions (TBA)
-----------------------

This class contains the walkers WF and weights
The walkers are adjusted according to the type of trial WF

"""

import sys, time
from abc import abstractmethod
import numpy as backend
from pyscf.lib import logger
from openms.lib.logger import task_title


def initialize_walkers(trial):
    r"""get initial walkers

    # TODO: 1) get init_waker for various trials
    # for SD, init_walker = trial.psi.copy()
    """

    # N copies of trial
    # shape: Nw x N_NO x N_AO for SD
    init_walker = trial.psi.copy()

    # 2) TODO: multiSD

    return init_walker


class BaseWalkers(object):
    def __init__(self, **kwargs):
        r"""Base walker class

        Parameters
        ----------

        trail: object
            Trial wavefunction
        nwalkers: int
            Number of walkers
        """

        self.nwalkers = kwargs.get("nwalkers", 100)
        self.logshift = backend.zeros(self.nwalkers)

        trial = kwargs.get("trial", None)
        if trial is None:
            raise Exception("Trial WF must be specified in walker")
        self.verbose = trial.verbose
        self.stdout = trial.stdout
        self.nalpha = trial.nalpha
        self.nbeta = trial.nbeta
        self.ncomponents = trial.ncomponents

        # variables for weights and weights control
        self.weights = backend.ones(self.nwalkers)
        self.weight_min = 0.1
        self.weight_max = 10.0
        #
        self.total_weight0 = self.nwalkers
        self.total_weight = self.nwalkers

        # variables for overlap, energies, ...
        self.ovlp = backend.ones(self.nwalkers, dtype=backend.complex128)
        self.sgn_ovlp = backend.ones(self.nwalkers)
        self.log_ovlp = backend.zeros(self.nwalkers)

        self.detR = backend.ones(self.nwalkers, dtype=backend.complex128)
        self.detR_shift = backend.zeros(self.nwalkers)
        self.log_detR = backend.zeros(self.nwalkers, dtype=backend.complex128)
        self.log_shift = backend.zeros(self.nwalkers)
        self.log_detR_shift = backend.zeros(self.nwalkers)

        self.eloc = backend.zeros(self.nwalkers)
        self.ehybrid = backend.zeros(self.nwalkers)  # or None?
        self.phiw = self.phiwa = self.phiwb = None
        self.boson_phiw = None  # bosonic walker WF

        self.spin_restricted = False
        # print("nwalkers in walkerfunciton is", self.nwalkers)

    def dump_flags(self):
        r"""
        dump flags
        """
        logger.note(self, task_title("Flags of walkers"))
        logger.note(self, f" Number of walkers        : {self.nwalkers:5d}")

    @abstractmethod
    def orthogonalization(self):
        r"""Renormalizaiton and orthogonaization of walkers"""
        raise NotImplementedError(
            "The 'orthogonalization' method must be implemented in a subclass "
            "to handle renormalization and orthogonalization of walkers."
        )

    @abstractmethod
    def weight_control(self, step, freq=5):
        r"""Handles the control of walker weights
        This method must be implemented in any subclass.
        """

        # raise NotImplementedError(
        #    "The 'weight_control' method must be implemented in a subclass "
        #    "to manage the walker weights effectively."
        # )
        t0 = time.time()
        weight_bound = 0.1 * self.total_weight
        if step > 0:
            backend.clip(
                self.weights, a_min=-weight_bound, a_max=weight_bound, out=self.weights
            )

        if (step + 1) % freq != 0:
            return

        logger.debug(self, f"Debug: pop control at step {step}")
        weights = backend.abs(self.weights)
        total_weight = backend.sum(weights)
        ratio = total_weight / self.total_weight0
        logger.debug(self, f"Debug: weights control is triggered! ratio is {ratio}")
        if total_weight < 1.0e-6:
            logger.warn(
                self, "The total weight is problematic, check the propagation!!!!"
            )

        self.total_weight = total_weight
        self.weights /= ratio


# walker in so orbital (akin ghf walker)
class Walkers_so(BaseWalkers):
    r"""walker in the SO

    TODO: make the walker can be either fermionic, bosonic
    or fermion-boson mixture walkers

    Walker shape: [nwalker, n_AO, n_electron]
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        trial = kwargs.get("trial", None)
        if trial is None:
            raise Exception("Trial WF must be specified in walker")

        self.nao = trial.psi.shape[-2]
        logger.debug(self, f"Debug: nao in walkers = {self.nao}")

        # 1) create walkers
        init_walker = initialize_walkers(trial)
        self.phiw = backend.array(
            [init_walker] * self.nwalkers, dtype=backend.complex128
        )
        # TODO: optimize the handling of phiw, phia/b with different ncomponents
        self.phiwa = self.phiw[:, :, : self.nalpha]
        if trial.ncomponents > 1:
            self.phiwb = self.phiw[:, :, self.nalpha :]

        # make bosonic walkers
        if trial.boson_psi is not None:
            self.boson_phiw = backend.array(
                [trial.boson_psi] * self.nwalkers, dtype=backend.complex128
            )

        # build Gfs?
        self.Ga = backend.zeros(
            (self.nwalkers, self.nao, self.nao), dtype=backend.complex128
        )
        self.Gb = backend.zeros(
            (self.nwalkers, self.nao, self.nao), dtype=backend.complex128
        )
        self.Ghalfa = backend.zeros(
            (self.nwalkers, self.nalpha, self.nao), dtype=backend.complex128
        )
        self.Ghalfb = backend.zeros(
            (self.nwalkers, self.nbeta, self.nao), dtype=backend.complex128
        )

