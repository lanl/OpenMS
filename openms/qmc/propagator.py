

from abc import abstractmethod, ABC

class PropagatorBase(ABC):

    r"""
    Base propagator class
    """
    def __build__(self, dt, verbose=1):
        self.dt = dt
        self.verbose = verbose
        self.time = 0.0

    @abstractmethod
    def build(self, walkers=None, mpi=None):
        pass

    @abstractmethod
    def propagate_walkers(self, hamiltonians, trial, walkers, eshift):
        pass

    @abstractmethod
    def propagate_walkers_onebody(self, hamiltonians, trial, walkers, eshift):
        pass

    @abstractmethod
    def propagate_walkers_twobody(self, hamiltonians, trial, walkers, eshift):
        pass


class Phaseless(PropagatorBase):
    r"""
    HS-transformation based AFQMC propagators
    """

    def __init__(self, dt, verbose=1):
        super().__init__(dt, verbose=verbose)

    def build(self, hamiltonian, trial=None, walkers=None, mpi=None, verbose=1):

        pass



