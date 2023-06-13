import os
from typing import List, Union

import hippynn
import torch
from hippynn.experiment.serialization import load_model_from_cwd
from hippynn.graphs import Predictor

from ..lib.misc import Molecule
from ..qmd.es_driver import QuantumDriver


class NNDriver(QuantumDriver):
    def __init__(
        self, nstates: int, model_path: str, model_device="cpu", multi_targets=True
    ):
        """Initialize the driver with a HIPNN model

        Args:
            model_path (str): the directory where the saved model is located
            model_device (str, optional): the device where the model will be loaded.
                For better compatibility, defaults to "cpu".
            multi_targets (bool, optional): whether multi-targets nodes are used in
                hippynn. Defaults to True.
        """
        super().__init__()
        self.nstates = nstates
        self.state_pairs = torch.triu_indices(nstates, nstates, 1).T
        current_dir = os.getcwd()
        # TODO: hippynn doesn't have the ability to load model other than the current
        # TODO: working directory. Might be useful to add such a function on the hippynn
        # TODO: side
        os.chdir(model_path)
        self.model = load_model_from_cwd(model_device=model_device)
        os.chdir(current_dir)
        self.predictor = Predictor.from_graph(self.model, model_device=model_device)
        self.multi_targets = multi_targets

    def load_geom(self, molecule: Molecule):
        return super().load_geom(molecule)

    # At this moment the model is assumed to
    #   1. have the same number of states as the simulation
    #   2. use multi targets, i.e., one node to predict all states
    #   3. have the node name I normally use
    # TODO: node name and no of states should be determined at least semi-automatically
    def make_predictions(self, molecule: Union[Molecule, List[Molecule]]):
        if isinstance(molecule, Molecule):
            Z = [molecule.elements]
            R = [molecule.coord]
        else:
            Z = [_.elements for _ in molecule]
            R = [_.coord for _ in molecule]
        Z = torch.tensor(Z)
        R = torch.Tensor(R)
        self.pred = self.predictor(Z=Z, R=R)

    def nuc_grad(self):
        return self.pred["E"].grad

    def get_nact(self, molecule: Union[Molecule, List[Molecule]], nacr: torch.Tensor):
        if isinstance(molecule, Molecule):
            v = [molecule.veloc]
        else:
            v = [_.veloc for _ in molecule]
        v = torch.Tensor(v)
        n_molecules, n_atoms, n_dims = v.shape
        # reshape velocities for batched matrix multiplications
        v = v.reshape(n_molecules, 1, 1, n_atoms * n_dims)
        nacr = nacr.reshape(n_molecules, -1, n_atoms * n_dims, 1)
        # resulting a tensor with a shape of (n_molecules, n_pairs, 1, 1)
        # use squeeze to remove 1's
        return torch.matmul(v, nacr).squeeze()

    def get_nacr(self):
        # only take excited state energies
        e = self.get_energies()[:, 1:]
        # direct hippynn output is NACR * dE
        # in the shape of (n_molecules, natoms * ndim, npairs)
        nacr_de = self.pred["ScaledNACR"].permute(0, 2, 1)
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
        """Return the molecular energies for all molecules and all states (the ground
            state and *nstates* excited states)

        Returns:
            torch.Tensor: molecular energies. Shape (n molecules, nstates + 1)
        """
        return self.pred["E"]

    def get_dipoles(self):
        return self.pred["D"].permute(0, 2, 1)

    def get_dipole_grad(self, dipoles: torch.Tensor):
        return dipoles.grad
