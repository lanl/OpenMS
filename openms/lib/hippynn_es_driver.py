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


class NNDriver(QuantumDriver):
    def __init__(
        self, nstates: int, model_path: str, model_device="cpu", multi_targets=True
    ):
        """Initialize the driver with a HIPNN model.

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

    def get_Z_R(self, molecule: Union[Molecule, List[Molecule]]):
        """Convert the element symbols to a tensor of atomic numbers (Z) and molecular
            positions to a tensor of coordinates (R). No changes of species or number of
            molecules are expected, so the Z and R tensors are saved within this class.

        :param molecule: molecule or molecules.
        :type molecule: Union[Molecule, List[Molecule]]
        """
        if isinstance(molecule, Molecule):
            Z = [[periodictable[a]["z"] for a in molecule.elements]]
            R = [molecule.coords]
        else:
            Z = [[periodictable[a]["z"] for a in mol.elements] for mol in molecule]
            R = [_.coords for _ in molecule]
        # lower case tensor to keep the int type
        self.Z = torch.tensor(Z)
        # converting list of np.nparray to tensor is extremely slow
        # could use some optimizations here
        self.R = torch.Tensor(np.array(R))
        del Z, R

    # At this moment the model is assumed to
    #   1. have the same number of states as the simulation
    #   2. use multi targets, i.e., one node to predict all states
    #   3. have the node name I normally use
    # TODO: node name and no of states should be determined at least semi-automatically
    def make_predictions(self):
        """Make predictions for electronic structure properties based on the geometry of
        the input molecule or list of molecules. The predicted results are saved
        within the class, as NNDriver.pred.

        """
        self.pred = self.predictor(Z=self.Z, R=self.R)

    def nuc_grad(self):
        e = self.pred["E"]
        force = []
        for i in range(self.nstates + 1):
            force.append(
                torch.autograd.grad(e[:, i].sum(), self.R, retain_graph=True)[0]
            )
        return torch.stack(force, dim=1)

    get_forces = nuc_grad

    def get_nact(
        self, molecule: Union[Molecule, List[Molecule]], nacr: torch.Tensor
    ) -> torch.Tensor:
        """Return the non-adiabatic coupling terms (NACT) between all pairs of excited
           states. Calculated from NACR and nuclear velocities.

        .. math::
           :nowrap:

           \begin{align*}
              NACT(t_i) = NACR(t_i) \dcot v(t_i)
           \begin{align*}

        :param molecule: molecule or molecules.
        :type molecule: Union[Molecule, List[Molecule]]
        :param nacr: NACR. Shape (n_molecules, n_state_pairs, natoms, 3).
        :type nacr: torch.Tensor
        :return: NACT. Shape (n_molecules, n_state_pairs).
        :rtype: torch.Tensor
        """
        if isinstance(molecule, Molecule):
            v = [molecule.veloc]
        else:
            v = [_.veloc for _ in molecule]
        v = torch.Tensor(np.array(v))
        n_molecules, n_atoms, n_dims = v.shape
        # reshape velocities for batched matrix multiplications
        v = v.reshape(n_molecules, 1, 1, n_atoms * n_dims)
        nacr = nacr.reshape(n_molecules, len(self.state_pairs), n_atoms * n_dims, 1)
        # resulting a tensor with a shape of (n_molecules, n_pairs, 1, 1)
        # use squeeze to remove 1's
        return torch.matmul(v, nacr).squeeze(dim=(2, 3))

    def get_nacr(self) -> torch.Tensor:
        """Return the non-adiabatic coupling vectors (NACR) between all pairs of excited
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
    def get_energies(self) -> torch.Tensor:
        """Return the molecular energies for all molecules and all states (the ground
            state and *nstates* excited states).

        :return: molecular energies. Shape (n_molecules, nstates + 1).
        :rtype: torch.Tensor
        """
        return self.pred["E"]

    def get_dipoles(self) -> torch.Tensor:
        """Return the transition dipoles of all states.

        :return: transition dipoles. Shape (n_molecules, n_states, 3).
        :rtype: torch.Tensor
        """
        return self.pred["D"]

    def get_dipole_grad(self) -> torch.Tensor:
        """Return the gradients of transition dipoles

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
