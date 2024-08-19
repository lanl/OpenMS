
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
Su-Schrieffer-Heeger (SSH) model

.. math::

    H(R) = \sum_i
"""

from openms.qmd import QuantumDriver


class SSH(QuantumDriver):
    r"""
    Class for 1D SSH model

    object molecule: molecule object
    integer ne: the number of electrons
    integer length: the number of sites
    double onsite: onsite energy
    double hopping: hopping integral
    double U: U parameter (correlation)
    double mass: nuclei mass
    """

    def __init__(self, molecule, **kwargs):
        super().__init__()

        self.mol = molecule
        self.nelectrons = 1
        self.length = 1
        self.onsite = 0.0
        self.hopping = 1.0
        self.U = 0.0
        self.K = 1.0 # spring constant
        self.mass = 1837.0 

        self.__dict__.update(kwargs)

    def get_mo(self):
        r"""
        Construct molecular orbital of SSH model
        """
        pass

    #other func for energies, forces, etc (todo)
