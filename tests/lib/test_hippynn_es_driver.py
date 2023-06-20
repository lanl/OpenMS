import os
import unittest
from pathlib import Path
from typing import List, Union

import numpy as np

from openms.lib.hippynn_es_driver import NNDriver
from openms.lib.misc import Molecule

CWD = os.getcwd()
NULL = open(os.devnull, "w")
TESTS_ROOT_DIR = Path(__file__).parent.parent.absolute()

eth_xyz = [
    """
    H         -4.43923        0.71227       -0.88261
    H         -4.28070        0.14822        0.88261
    C         -3.82097        0.58174        0.00000
    C         -2.53501        0.94317        0.00000
    H         -2.07527        1.37669       -0.88261
    H         -1.91674        0.81264        0.88261
    """,
    """
    H         -5.94860        1.80664        0.07342
    H         -5.13955        3.47492       -0.07342
    C         -5.04031        2.39647       -0.00000
    C         -3.83840        1.81360       -0.00000
    H         -3.73915        0.73515        0.07342
    H         -2.93010        2.40343       -0.07342
    """,
]
n_atoms = 6
n_dims = 3


def build_molecules(xyz: Union[str, List[str]]):
    if isinstance(xyz, str):
        test_molecules = Molecule(atom=xyz)
        # generate random velocities
        test_molecules.veloc = np.random.rand(*test_molecules.veloc.shape)
    else:
        test_molecules = []
        for _ in xyz:
            mol = Molecule(atom=_)
            mol.veloc = np.random.rand(*mol.veloc.shape)
            test_molecules.append(mol)
    return test_molecules


class TestNNDriver(unittest.TestCase):
    def setUp(self):
        self.model_path = f"{TESTS_ROOT_DIR}/assets/models/3_states"
        self.nstates = 3

        self.test_molecules = []
        for _ in eth_xyz:
            self.test_molecules.append(Molecule(atom=_))

    def _check_init(self):
        return NNDriver(self.nstates, self.model_path)

    def _check_atomic_numbers(
        self, driver: NNDriver, molecules: Union[Molecule, List[Molecule]], n_mols: int
    ):
        try:
            driver.get_Z_R(molecules)
        except Exception as e:
            self.fail(f"get_Z_R() raised {e} unexpectedly!")
        self.assertEqual(driver.Z.shape, (n_mols, n_atoms))
        self.assertEqual(driver.R.shape, (n_mols, n_atoms, n_dims))

    def _check_predictions(self, driver: NNDriver, n_mols: int):
        try:
            driver.make_predictions()
        except Exception as e:
            self.fail(f"make_predictions() raised {e} unexpectedly!")
        self.assertIn("E", driver.pred)
        self.assertIn("D", driver.pred)
        self.assertIn("ScaledNACR", driver.pred)

    def _check_energies(self, driver: NNDriver, n_mols: int):
        e = driver.get_energies()
        n_states = driver.nstates + 1
        self.assertEqual(e.shape, (n_mols, n_states))

    def _check_dipoles(self, driver: NNDriver, n_mols: int):
        d = driver.get_dipoles()
        self.assertEqual(d.shape, (n_mols, driver.nstates, n_dims))
        d = driver.get_dipole_grad()
        self.assertEqual(d.shape, (n_mols, driver.nstates, n_atoms, n_dims))

    def _check_forces(self, driver: NNDriver, n_mols: int):
        f = driver.get_forces()
        n_states = driver.nstates + 1
        self.assertEqual(f.shape, (n_mols, n_states, n_atoms, n_dims))

    def _check_nacr(self, driver: NNDriver, n_mols: int):
        self.nacr = driver.get_nacr()
        n_pairs = self.nstates * (self.nstates - 1) // 2
        self.assertEqual(self.nacr.shape, (n_mols, n_pairs, n_atoms, n_dims))

    def _check_nact(
        self, driver: NNDriver, molecules: Union[Molecule, List[Molecule]], n_mols: int
    ):
        nact = driver.get_nact(molecules, self.nacr)
        n_pairs = self.nstates * (self.nstates - 1) // 2
        self.assertEqual(nact.shape, (n_mols, n_pairs))

    def _run_test(self, molecules: Union[Molecule, List[Molecule]]):
        if isinstance(molecules, Molecule):
            n_mols = 1
        else:
            n_mols = len(molecules)
        driver = self._check_init()
        self._check_atomic_numbers(driver, molecules, n_mols)
        self._check_predictions(driver, n_mols)
        self._check_energies(driver, n_mols)
        self._check_forces(driver, n_mols)
        self._check_dipoles(driver, n_mols)
        self._check_nacr(driver, n_mols)
        self._check_nact(driver, molecules, n_mols)

    def test1(self):
        molecules = build_molecules(eth_xyz[0])
        self._run_test(molecules)

    def test2(self):
        molecules = build_molecules(eth_xyz)
        self._run_test(molecules)
