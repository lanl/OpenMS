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

r"""Variational Monte Carlo (VMC) for quantum many-body ground states and dynamics

Theoretical background
----------------------

The VMC leverages the variational principle to optimize the ground state energies.

The Variational Monte Carlo can be used to:

  - Compute expectation value of given operators with wavefunction, such as energy, correlation function, etc.
  - Optimize a parameterized trial wavefunction and energy (by leveraging variational principles)


For a given trial wavefunction :math:`\ket{\Psi(\theta)}`, the energy can be computed as

.. math::

   E = \frac{\bra{\Psi(\theta)}\hat{H}\ket{\Psi(\theta)}}{\bra{\Psi(\theta)}\Psi(\theta)\rangle}.


VMC compute the energy by sampling over configurations :math:`c_i` with probability functions,

.. math::

    E = & \frac{\sum_i\bra{\Psi(\theta, c_i)}\hat{H}\ket{\Psi(\theta, c_i)} }{\sum_i \bra{\Psi(\theta, c_i)}\Psi(\theta, c_i)\rangle}\\
      = & \frac{\sum_i\bra{\Psi(\theta, c_i)} \ket{\Psi(\theta, c_i)} \frac{\hat{H}\ket{\Psi(\theta, c_i)}} {\ket{\Psi(\theta, c_i)}}}
          {\sum_i \bra{\Psi(\theta, c_i)}\Psi(\theta, c_i)\rangle} \\
      = & \frac{\sum_i P_i  \frac{\hat{H}\ket{\Psi(\theta, c_i)}} {\ket{\Psi(\theta, c_i)}}}
         {\sum_i \bra{\Psi(\theta, c_i)}\Psi(\theta, c_i)\rangle}


More details TBA.


The energy within the VMC method is computed as

.. math::
    E = \sum^{N_s}_{i} \frac{E_{loc}(\Gamma_i)}{N_s}

where :math:`\Gamma_i` index the sample. And the local energy is:

.. math::
    E_{loc}(\sigma_i) & = \frac{\bra{\sigma}\hat{H}\ket{\psi}}{\bra{\sigma}\psi\rangle}
    =\sum_\eta \bra{\sigma}\hat{H}\ket{\eta}\frac{\psi(\eta)}{\psi(\sigma)} \\
    & = \sum_\eta \bra{\sigma}\hat{H}\ket{\eta}[\log(\psi(\eta)) - \log(\psi(\sigma))]


The sum in principle is over the exponentially large configuration space.



"""

import abc
from typing import Any
from pyscf.lib import logger
import numpy as np




class VMCBase(object):
    def __init__(
        self,
        ansatz, #: VariationalAnsatz,
        hamiltonian,
        sampler : Any,
        optimizer: Any,
        observables=None,
    ):

        """
        Universal Variational Monte Carlo (VMC) engine.

        Parameters:
        - ansatz: object with methods:
            - log_psi(config): log of ansatz amplitude
            - gradient(config): gradient of log_psi w.r.t. variational parameters
            - amplitude(config): ansatz amplitude (optional, can be derived)
        - hamiltonian: object with method:
            - local_energy(config, ansatz): computes local energy
        - sampler: object with method:
            - sample(ansatz, n_samples): returns list of configurations
        - optimizer: object with method:
            - step(gradients, overlap_matrix, energy_grad): updates parameters
        - observables: optional dictionary of callables taking config as input
        """
        self._ansatz = ansatz
        self._sampler = sampler
        self._optimizer = optimizer
        self.ham = hamiltonian
        self.obs = observables if observables else {}
        self._step = 0


    def __repr__(self):
        # TODO: add the optimizer and ansatz name in the __repr__
        return f"{self.__class__.__name__} class with xx optimizer and xx ansatz"


    @property
    def ansatz(self):
        return self._ansatz

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def step_count(self):
        r"""Return steps used to converge VMC"""
        return self._step

    @property
    def preconditioner(self):
        return self._precond

    @property
    def energy(self):
        r"""Return energy"""
        return 0.0 # TBA.


    # setters TBA

    # ********** end of property and setters **************

    def dump_flags(self):
        r"""Dump flags of VMC"""

        title = f"{self.__class__.__name__} simulation using OpenMS package"
        logger.note(self, task_title(title, level=0))

        logger.note(self, f" Sampler is             : xx")
        logger.note(self, f" Optimizer is           : xx")
        logger.note(self, f" Number of sites        : xx")
        logger.note(self, f" Size of hilbert space  : xx")


    def kernel(self, n_samples=1000, n_iter=100, verbose=True):
        r"""
        Run VMC optimization
        """

        for it in range(n_iter):
            configs = self.sampler.sample(self.psi, n_samples)
            E_locals = []
            grad_Oks = []
            log_psi_vals = []
            obs_acc = {key: [] for key in self.obs}

            for C in configs:
                log_psi = self.psi.log_psi(C)
                E_loc = self.ham.local_energy(C, self.psi)
                grad_Ok = self.psi.gradient(C)

                log_psi_vals.append(log_psi)
                E_locals.append(E_loc)
                grad_Oks.append(grad_Ok)

                for key, obs_func in self.obs.items():
                    obs_acc[key].append(obs_func(C))

            E_locals = np.array(E_locals)
            grad_Oks = np.array(grad_Oks)
            log_psi_vals = np.array(log_psi_vals)

            E_mean = np.mean(E_locals)
            Ok_mean = np.mean(grad_Oks, axis=0)

            # Stochastic Reconfiguration components
            S = np.cov(grad_Oks.T, bias=True)
            g = np.mean((E_locals[:, None] * grad_Oks), axis=0) - E_mean * Ok_mean

            # Update parameters
            self.opt.step(S, g)

            if verbose:
                print(f"Iteration {it + 1}: Energy = {E_mean:.6f}")
                for key in self.obs:
                    print(f"    ⟨{key}⟩ = {np.mean(obs_acc[key]):.6f}")

        return
