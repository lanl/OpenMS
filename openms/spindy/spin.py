import numpy as np

#jmat(j, #s): Higher-order spin operators. s = "x", "y", "z", "+", or "-"
#from qutip import qeye, jmat
#form qutip import sigmax, sigmay, sigmaz, sigmap, sigmam

ELEC_GYRO = -17.608597050  # rad * MHz / G

def check_gyro(gyro):
    """
    Check if gyro is matrix or scalar.

    Args:
        gyro (ndarray or float): Gyromagnetic ratio matrix or float.

    Returns:
        tuple: tuple containing:

            * **ndarray or float**: Gyromagnetic ratio.
            * **bool**: True if gyro is float, False otherwise.
    """
    try:
        gyro = float(gyro)
        check = True
    except TypeError:
        check = False

    if not check:
        gyro = np.asarray(gyro)
        if gyro.ndim == 1:  # Assume array
            check = True
        elif not gyro.shape or gyro.shape[0] == 1:
            check = True
            gyro = gyro.reshape(1)[0]
        else:
            test_gyros = gyro.copy()
            indexes = np.arange(gyro.shape[-1])
            test_gyros[..., indexes, indexes] = 0

            diag_check = np.isclose(test_gyros, 0).all()
            same_check = ((gyro[..., 0, 0] == gyro[..., 1, 1]) & (gyro[..., 1, 1] == gyro[..., 2, 2])).all()
            check = diag_check & same_check
            if check:
                gyro = gyro[..., 0, 0][()]

    return gyro, check

def zfs_tensor(D, E=0):
    """
    Generate (3, 3) ZFS tensor from observable parameters D and E.

    Args:
        D (float or ndarray with shape (3, 3)): Longitudinal splitting (D) in ZFS **OR** total ZFS tensor.
        E (float): Transverse splitting (E) in ZFS.

    Returns:
        ndarray with shape (3, 3): Total ZFS tensor.
    """

    D = np.asarray(D)

    if D.size == 1:
        tensor = np.zeros((3, 3), dtype=np.float64)
        tensor[2, 2] = 2 / 3 * D
        tensor[1, 1] = -D / 3 - E
        tensor[0, 0] = -D / 3 + E
    else:
        tensor = D

    return tensor


class Spin(object):
    r"""
    Base Class for spin object, which contains the properties of spin, including:
    
    :param float spin: Total spin. Default: 0.5
    :param 1darray coord: Cartesian coordinates of spin. Default: (0, 0, 0)
    :param float D:  
    """

    def __init__(self, 
            coord=np.zeros(3),
            D = 0.0, E=0.0,
            **kwargs):

        self.coord = coord
        self.spin = 0.5
        self.D = 0.0
        self.E = 0.0
        self.alpha = None
        self.beta = None
        self._zfs = None
        self.gfactor = ELEC_GYRO

        self.__dict__.update(kwargs)

        if self._zfs is None:
            self.get_zfs(D, E)
        self.set_gfactor(self.gfactor)

        self.ndim = int(self.spin * 2 + 1 + 1e-8)
        #print("dimension of the spin is", self.ndim)


    def get_zfs(self, D, E):
        self._zfs = zfs_tensor(D, E)

    def set_gfactor(self, gyro):
        """
        Set gyromagnetic ratio of the central spin.

        Args:
            gyro (float or ndarray with shape (3, 3)): Gyromagnetic ratio of central spin in rad / ms / G.

                **OR**

                Tensor describing central spin interactions with the magnetic field.

        """
        check = not np.asarray(gyro).shape == (3, 3)
        if check:
            gyro, check = check_gyro(gyro)
            if check:
                gyro = np.eye(3) * gyro
        self._gyro = gyro



    def __str__(self):
        r"""
        Print function for the spin object
        """
        log_info=""

        return log_info


if __name__ == '__main__':
    center = Spin(spin=0.5)
    print(center.spin)


