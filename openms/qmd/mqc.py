"""
basic MQC module
"""

import numpy
import openms



from openms.lib.misc import fs2au, au2A, call_name, typewriter
import textwrap, datetime
import numpy as np
import os, shutil
from .bomd import BaseMD


class MQC(BaseMD):
    """Class for nuclear/electronic propagator used in MQC dynamics

    Attributes:
       :param object molecule: Molecule object

    """

    def __init__(self, molecule, thermostat=None, **kwargs):
        # Initialize input values
        self.init_state = 0
        self.init_coef = None
        self.nesteps = 20
        self.propagator = "rk4"
        self.elec_object = "density"
        self.l_print_dm = True
        super().__init__(molecule, thermostat, **kwargs)
        self.__dict__.update(kwargs)

        self.md_type = self.__class__.__name__

        # None for BOMD case
        if self.elec_object != None:
            self.elec_object = self.elec_object.lower()

        if not (self.elec_object in [None, "coefficient", "density"]):
            error_message = "Invalid electronic representation!"
            error_vars = f"elec_object = {self.elec_object}"
            raise ValueError(
                f"( {self.md_type}.{call_name()} ) {error_message} ( {error_vars} )"
            )

        if self.propagator != None:
            self.propagator = self.propagator.lower()

        if not (self.propagator in [None, "rk4"]):
            error_message = "Invalid electronic propagator!"
            error_vars = f"propagator = {self.propagator}"
            raise ValueError(
                f"( {self.md_type}.{call_name()} ) {error_message} ( {error_vars} )"
            )

        # Initialize coefficients and densities
        self.mol.get_coefficient(self.init_coef, self.init_state)

    def electronic_propagator(self):
        r"""Propagator for electronic EOM.
        It should be implemented in derived class!
        rk4 is used by default!
        """
        return NotImplementedError("Method not implemented!")

    def initialize(self, *args):
        r"""Prepare the initial conditions for both quantuma and classical EOM
        It should be implemented in derived class!
        """
        # call the BaseMD.initialize for the nuclear part.
        base_dir, md_dir, qm_log_dir = super().initialize(*args)

        # initialization for electroic part (TBD)
        return base_dir, md_dir, qm_log_dir
        # return NotImplementedError("Method not implemented!")

    def dump_step(self):
        r"""Output coodinates, velocity, energies, electronic populations, etc.
        Universal properties will be dumped here (velocity, coordinate, energeis)
        Other will be dumped in derived class!
        """

        return NotImplementedError("Method not implemented!")
