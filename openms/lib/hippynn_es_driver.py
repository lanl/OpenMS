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
# Author: Xinyang Li <lix@lanl.gov>
#

import os
from typing import List, Union

try:
    from hippynn.experiment.serialization import load_model_from_cwd
    from hippynn.graphs import Predictor

    HIPPYNN_AVAILABLE = True
except ImportError:
    HIPPYNN_AVAILABLE = False

try:
    import torch

    TORCH_AVAILABLE = True
    torch._C._set_grad_enabled(True)
except ImportError:
    TORCH_AVAILABLE = False

import numpy as np
from openms.lib.misc import Molecule, periodictable
from openms.qmd.es_driver import QuantumDriver

r"""
HIPNN-based electronic structure driver
"""

if HIPPYNN_AVAILABLE and TORCH_AVAILABLE:
    #
    class NNDriver(QuantumDriver):
        def __init__(
            self,
            molecule: Union[Molecule, List[Molecule]],
            nstates: int,
            model_path: str,
            coords_unit="Angstrom",
            energy_conversion=0.036749405469679,
            model_device="cpu",
            multi_targets=True,
        ):
            r"""Initialize the driver with a HIPNN model.

            :param molecule: object of a molecule or list of molecules
            :type molecule: Union[Molecule, List[Molecule]]
            :param nstates: number of states needed in the simulation.
            :type nstates: int
            :param model_path: the directory where the saved model is located.
            :type model_path: str
            :param model_device: the device where the model will be loaded.
                For better compatibility, defaults to "cpu".
            :type model_device: str, optional
            :param multi_targets: whether multi-targets nodes are used in
                hippynn, defaults to True.
            :type multi_targets: bool, optional
            :raises RuntimeError: error exit when PyTorch or hippynn is missing.
            """
            if not HIPPYNN_AVAILABLE or not TORCH_AVAILABLE:
                raise RuntimeError(
                    "PyTorch and hippynn must be installed to use the neural network"
                    " electronic structure driver."
                )
            super().__init__()
            self.mol = molecule
            self.nstates = nstates
            self.state_pairs = torch.triu_indices(nstates, nstates, 1).T
            current_dir = os.getcwd()
            # TODO: hippynn doesn't have the ability to load model other than the current
            # TODO: working directory. Might be useful to add such a function on the hippynn
            # TODO: side
            os.chdir(model_path)
            dtype = torch.get_default_dtype()
            self.model = load_model_from_cwd(model_device=model_device).to(dtype)
            os.chdir(current_dir)
            self.predictor = Predictor.from_graph(
                self.model, model_device=model_device, requires_grad=True
            )
            self.multi_targets = multi_targets
            self.coords_unit = coords_unit
            # eV to au
            self.energy_conversion = energy_conversion
            if coords_unit == "Angstrom":
                # Angstrom to au
                self.coords_conversion = 1.8897259886
            else:
                self.coords_conversion = 1

        def get_Z(self):
            r"""Convert the element symbols to a tensor of atomic numbers (Z). This only
            need to be run once. when the class is initialized."""
            if isinstance(self.mol, Molecule):
                Z = [[periodictable[a]["z"] for a in self.mol.elements]]
            else:
                Z = [[periodictable[a]["z"] for a in mol.elements] for mol in self.mol]
            # lower case tensor to keep the int type
            self.Z = torch.tensor(Z)
            del Z

        def get_R(self):
            r"""Convert the molecular positions to a tensor of coordinates (R). This
            conversion needs to be done every time when the coordinates are updated,
            for example, in MD simulations.
            """
            if isinstance(self.mol, Molecule):
                # R = [self.mol.atom_coords(unit=self.coords_unit)]
                R = [self.mol.coords]
            else:
                R = [_.coords for _ in self.mol]
                # R = [_.atom_coords(unit=self.coords_unit) for _ in self.mol]
            # converting list of np.nparray to tensor is extremely slow
            # could use some optimizations here
            # convert unit from a.u. to Angstrom
            self.R = torch.Tensor(np.array(R)) / self.coords_conversion
            del R

        # At this moment the model is assumed to
        #   1. have the same number of states as the simulation
        #   2. use multi targets, i.e., one node to predict all states
        #   3. have the node name I normally use
        # TODO: node name and no of states should be determined at least semi-automatically
        def make_predictions(self):
            r"""Make predictions for electronic structure properties based on the geometry of
            the input molecule or list of molecules. The predicted results are saved
            within the class, as NNDriver.pred.

            """
            self.get_R()
            self.pred = self.predictor(Z=self.Z, R=self.R)
            self.pred["E"] *= self.energy_conversion

        def nuc_grad(self):
            e = self.pred["E"]
            force = []
            for i in range(self.nstates + 1):
                force.append(
                    torch.autograd.grad(e[:, i].sum(), self.R, retain_graph=True)[0]
                )
            return -torch.stack(force, dim=1) / self.coords_conversion

        def _assign_forces(self, molecule: Molecule, force: torch.Tensor):
            for i in range(self.nstates):
                molecule.states[i].forces = force[i]

        def calculate_force(self):
            self.make_predictions()
            forces = self.nuc_grad().numpy()
            if isinstance(self.mol, Molecule):
                self._assign_forces(self.mol, forces[0])
            else:
                for i, mol in enumerate(self.mol):
                    self._assign_forces(mol, forces[i])

        def get_nact(self, nacr: torch.Tensor):
            r"""Return the non-adiabatic coupling terms (NACT) between all pairs of excited
               states. Calculated from NACR and nuclear velocities.

            .. math::
               :nowrap:

               \begin{align*}
                  NACT(t_i) = NACR(t_i) \cdot v(t_i)
               \begin{align*}

            :param nacr: NACR. Shape (n_molecules, n_state_pairs, natoms, 3).
            :type nacr: torch.Tensor
            :return: NACT. Shape (n_molecules, n_state_pairs).
            :rtype: torch.Tensor
            """
            if isinstance(self.mol, Molecule):
                v = [self.mol.veloc]
            else:
                v = [_.veloc for _ in self.mol]
            v = torch.Tensor(np.array(v))
            n_molecules, n_atoms, n_dims = v.shape
            # reshape velocities for batched matrix multiplications
            v = v.reshape(n_molecules, 1, 1, n_atoms * n_dims)
            nacr = nacr.reshape(n_molecules, len(self.state_pairs), n_atoms * n_dims, 1)
            # resulting a tensor with a shape of (n_molecules, n_pairs, 1, 1)
            # use squeeze to remove 1's
            return torch.matmul(v, nacr).squeeze(dim=(2, 3))

        def get_nacr(self):
            r"""Return the non-adiabatic coupling vectors (NACR) between all pairs of excited
                states.

            :return: NACR. Shape (n_molecules, n_state_pairs, natoms, 3).
            :rtype: torch.Tensor
            """
            # only take excited state energies
            e = self.get_energies()[:, 1:]
            # direct hippynn output is NACR * dE
            # in the shape of (n_molecules, npairs, natoms * ndim)
            nacr_de = self.pred["ScaledNACR"]
            de = []
            for i, j in self.state_pairs:
                # energy difference between two states
                de.append(e[:, j] - e[:, i])
            de = torch.stack(de, dim=1)
            nacr = nacr_de / de.unsqueeze(2)
            # rehape into (n_molecules, npairs, natoms, ndim)
            return nacr.reshape(*nacr.shape[:2], -1, 3)

        # current model is in eV
        # consider retrain the model
        def get_energies(self):
            r"""Return the molecular energies for all molecules and all states (the ground
                state and *nstates* excited states).

            :return: molecular energies. Shape (n_molecules, nstates + 1).
            :rtype: torch.Tensor
            """
            return self.pred["E"]

        def _assign_energies(self, molecule: Molecule, energy: torch.Tensor):
            for i in range(self.nstates):
                molecule.states[i].energy = energy[i].detach().numpy()

        def update_potential(self):
            e = self.pred["E"]
            if isinstance(self.mol, Molecule):
                self._assign_energies(self.mol, e[0])
            else:
                for i, mol in enumerate(self.mol):
                    self._assign_energies(mol, e[i])

        def get_dipoles(self):
            r"""Return the transition dipoles of all states.

            :return: transition dipoles. Shape (n_molecules, n_states, 3).
            :rtype: torch.Tensor
            """
            return self.pred["D"]

        def get_dipole_grad(self):
            r"""Return the gradients of transition dipoles

            :return: the gradients of the transition dipoles.
                Shape (n_molecules, n_states, natoms, 3).
            :rtype: torch.Tensor
            """
            d = self.pred["D"]
            d_grad = []
            for i in range(self.nstates):
                d_grad.append(
                    torch.autograd.grad(d[:, i].sum(), self.R, retain_graph=True)[0]
                )
            return torch.stack(d_grad, dim=1)
