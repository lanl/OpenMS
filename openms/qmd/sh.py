"""
basic MQC module
"""

from copy import copy
import datetime
import os
import shutil
import textwrap
from typing import List, Union

import numpy as np

import openms
from openms.lib.misc import Molecule, fs2au, au2A, call_name, typewriter
from openms.qmd.es_driver import QuantumDriver
from openms.qmd.propagator import rk4
from .mqc import MQC


class SH(MQC):
    """Class for nuclear/electronic propagator used in surface hopping dynamics

    Attributes:
       :param object molecule: Molecule object

    """

    def __init__(
        self,
        molecule: List[Molecule],
        init_states: np.array,
        init_coef: np.array,
        qm: QuantumDriver,
        thermostat=None,
        **kwargs,
    ):
        """Surface hopping

        :param molecule: list of molecular objects
        :type molecule: List[Molecule]
        :param init_states: initial states of each molecule
        :type init_states: np.array
        :param init_coef: initial wavefunction coefficients of each molecule
        :type init_coef: np.array
        :param qm: electronic structure driver
        :type qm: QuantumDriver
        :param thermostat: thermostat, defaults to None
        :type thermostat: object, optional
        """
        # Initialize input values
        super().__init__(molecule, init_states, init_coef, qm, thermostat, **kwargs)
        self.__dict__.update(kwargs)
        self.md_type = self.__class__.__name__
        self.curr_ham = np.empty((len(self.mol), self.nstates, self.nstates))
        self.curr_coords = np.array([_.coords for _ in self.mol])
        self.curr_veloc = np.array([_.veloc for _ in self.mol])

    def initialize(self, *args):
        r"""Prepare the initial conditions for both quantum and classical EOM
        It should be implemented in derived class!
        """
        # call the BaseMD.initialize for the nuclear part.
        base_dir, md_dir, qm_log_dir = super().initialize(*args)

        # initialization for electronic part (TBD)
        return base_dir, md_dir, qm_log_dir
        # return NotImplementedError("Method not implemented!")

    def electronic_propagator(self):
        coef = copy(self.coef)
        for _ in range(self.nesteps):
            coef = self.quantum_step(self.current_time, coef)
        self.coef = coef
        # assign updated coefficients to each molecule
        for i, m in enumerate(self.mol):
            m.coef = coef[i]

    def quantum_step(self, t: float, coef: np.array):
        """Propagate one quantum step

        :param t: current time t
        :type t: float
        :param coef: coefficients at current time
        :type coef: np.array
        :return: coefficient at time t + quantum_step
        :rtype: np.array
        """
        return rk4(self.get_coef_dot, t, coef, self.edt)

    def get_coef_dot(self, t: float, coef: np.array):
        """Calculate the acceleration for the coefficient at given Hamiltonian

        :param t: current time
        :type t: float
        :param coef: coefficients at current time
        :type coef: np.array
        :return: time derivative of the coefficients at current time
        :rtype: np.array
        """
        # if saved time is different from given current time
        # the Hamiltonian needs to be recalculated
        if t != self.current_time:
            self.curr_coords += self.curr_veloc * self.edt / 2
            energies = self.qm.get_energies(self.curr_coords)
            nact = self.qm.get_nact(self.curr_coords)
            self.curr_ham = self.get_H(energies, nact)
            self.current_time = t
        return -1j * np.einsum("ijk, ik -> ij", self.curr_ham, coef)

    def dump_step(self):
        r"""Output coodinates, velocity, energies, electronic populations, etc.
        Universal properties will be dumped here (velocity, coordinate, energeis)
        Other will be dumped in derived class!
        """

        return NotImplementedError("Method not implemented!")

    def get_H(self, energies: np.array, nact: np.array):
        """Function to assemble the Hamiltonian at the current quantum step.
        side.

        :param energies: potential energies of all excited states
        :type E: np.array
        :param nact: non-adiabatic coupling term
        :type nact: np.array
        :return: Hamiltonian of the system in matrix representation
        :rtype: np.array
        """
        ham = np.empty((len(self.mol), self.nstates, self.nstates))
        for i in range(len(self.mol)):
            counter = 0
            for j in range(self.nstates):
                ham[i, j, j] = energies[i, j]
                for k in range(j, self.nstates):
                    ham[i, j, k] = nact[i, counter]
                    ham[i, k, j] = -nact[i, counter]
                    counter += 1
        return ham

    def get_densities(self):
        return np.einsum("ij, ik -> ijk", np.conj(self.coef), self.coef)

    def hopping_probability(self, density: np.array):
        # p = 2 * self.edt * density * self.curr_ham
        # for i in range(len(self.mol)):
        #     for j in range(self.states):
        #         for k in range(self.states):
        #             if j == k:
        #                 p[i, j, k] = 0
        #             else:
        #                 p[i, j, k] = -np.real(p[i, j, k]) / density[i, j, j]
        p = np.empty((len(self.mol), self.nstates))
        for i in range(len(self.mol)):
            current_state = self.states[i]
            for j in range(self.states):
                if j == current_state:
                    p[i, j] = 0
                else:
                    tmp = -np.real(
                        density[i, current_state, j]
                        * self.curr_ham[i, current_state, j]
                    )
                    p[i, j] = 2 * self.edt * tmp / density[i, j, j]
        return p

    def check_hops(self, probability: np.array, energies: np.array, nacr: np.array):
        tmp = self.prng.random(probability.shape)
        sort_idx = np.argsort(probability)
        sorted_probability = np.take_along_axis(probability, sort_idx, axis=1)
        possible_hops = sorted_probability > tmp
        for i in range(len(self.mol)):
            for j in range(self.nstates):
                if possible_hops[i, j]:
                    final_state = sort_idx[i, j]
                    (
                        self.states[i],
                        self.curr_veloc[i],
                        hop_type,
                    ) = self.velocity_rescaling(
                        self.states[i], final_state, self.curr_veloc[i], energies, nacr
                    )
                    continue

    def velocity_rescaling(
        self,
        state_i: int,
        state_f: int,
        veloc: np.array,
        energies: np.array,
        nacr: np.array,
    ):
        nuclear_force = (energies[state_i] - energies[state_f]) * nacr[state_i, state_f]
        tmp = veloc - nuclear_force * self.dt
        # frustrated hops
        if np.any((tmp * veloc) < 0):
            return state_i, veloc, 2
        else:
            return state_f, tmp, 1

    def instantaneous_decoherence(self, state):
        coef = np.zeros(self.nstates)
        coef[state] = 1.0
        return coef
