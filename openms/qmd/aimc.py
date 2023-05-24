
from openms.qmd.mqc import MQC

class CTQMC(MQC):
    r"""Basic Class for coupld-trajectory MQC (CTMQC) dynamics

    :param object,list molecules: List of molecular objects
    :param object thermostat: Thermostat object
    """

    def __init__(self, molecules, thermostat=None, **kwargs):
        # Initialize input values

        # there it may be better to define each trajectory (molecule) as an MQC object.
        # instead of using a single derived MQC object
        self.md_type = self.__class__.__name__
        self.mols = molecules

        super().__init__(molecules[0], thermostat, **kwargs)


    def kernel(self):
        raise NotImplementedError('Method Not Implemented')
