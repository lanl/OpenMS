from .spin import Spin

class SystemSpins(Spin):
    r"""Sysytem class
    """
    def __init__(self, *args, **kwargs):

        self.zfs = None
        self.gyro = None

