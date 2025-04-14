

from dataclasses import dataclass
from typing import TypeVar


class FakeComm:
    """Fake MPI communicator class to reduce logic."""

    def __init__(self):
        self.rank = 0
        self.size = 1
        self.buffer = {}

    def Get_size(self):
        return 1

    def Get_rank(self):
        return 0

    def Barrier(self):
        pass

    def barrier(self):
        pass

    def Get_rank(self):
        return 0

    def Get_size(self):
        return 1

    def Gather(self, sendbuf, recvbuf, root=0):
        recvbuf[:] = sendbuf

    def gather(self, sendbuf, root=0):
        return [sendbuf]

    def Allgather(self, sendbuf, recvbuf, root=0):
        recvbuf[:] = sendbuf

    def Bcast(self, sendbuf, root=0):
        return sendbuf

    def bcast(self, sendbuf, root=0):
        return sendbuf

    def isend(self, sendbuf, dest=None, tag=None):
        return FakeReq()

    def Isend(self, sendbuf, dest=None, tag=None):
        self.buffer[tag] = sendbuf
        return FakeReq()

    def recv(self, source=None, root=0):
        pass

    def Recv(self, recvbuff, source=None, root=0, tag=0):
        if self.buffer.get(tag) is not None:
            recvbuff[:] = self.buffer[tag].copy()

    def Allreduce(self, sendbuf, recvbuf, root=0):
        recvbuf[:] = sendbuf

    def allreduce(self, sendbuf, op=None, root=0):
        return sendbuf.copy()

    def Split(self, color: int = 0, key: int = 0):
        return self

    def Reduce(self, sendbuf, recvbuf, op=None, root=0):
        recvbuf[:] = sendbuf

    def Scatter(self, sendbuf, recvbuf, root=0):
        recvbuf[:] = sendbuf

    def Scatterv(self, sendbuf, recvbuf, root=0):
        recvbuf[:] = sendbuf

    def scatter(self, sendbuf, root=0):
        assert sendbuf.shape[0] == 1, "Incorrect array shape in FakeComm.scatter"
        return sendbuf[0]


class FakeReq:
    def __init__(self):
        pass

    def wait(self):
        pass


@dataclass
class FakeMPI:
    COMM_WORLD = FakeComm()
    SUM = None
    COMM_SPLIT_TYPE_SHARED = None
    COMM_TYPE_SHARED = None
    DOUBLE = None
    INT64_T = None
    Win = None
    IntraComm = TypeVar("IntraComm")


import builtins

# Store original print
original_print = builtins.print

# Overwrite print if MPI is available
def configure_print_for_mpi(MPI):
    global original_print  # make sure it's visible to other files
    if hasattr(builtins.print, '__wrapped__'):
        return  # already overridden

    try:
        original_print = print
        rank = MPI.COMM_WORLD.Get_rank()
        if rank != 0:
            def mpi_silent_print(*args, **kwargs): pass
            mpi_silent_print.__wrapped__ = True
            #builtins.print = lambda *args, **kwargs: None
            builtins.print = mpi_silent_print
    except:
        pass  # Use default print behavior if anything fails
        original_print = print

#
def load_mpi():
    try:
        from mpi4py import MPI

        comm_type = MPI.Intracomm
        configure_print_for_mpi(MPI)  # overwrite print function

        return MPI, comm_type
    except (ModuleNotFoundError, ImportError):
        # from openms.__mpi__ import FakeComm, FakeMPI

        comm_type = FakeComm
        MPI = FakeMPI

        configure_print_for_mpi(MPI)
        return MPI, comm_type


MPI, CommType = load_mpi()

# MPI wrapper for handling MPI calculations (if MPI is available)
class MPIWrapper(object):

    def __init__(self, verbose=1):
        self.verbose = verbose
        self.MPI = MPI
        comm = MPI.COMM_WORLD
        self.comm = comm
        self.size = comm.Get_size()
        self.rank = comm.Get_rank()

