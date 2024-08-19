#
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

import numpy

#jmat(j, #s): Higher-order spin operators. s = "x", "y", "z", "+", or "-"
#from qutip import qeye, jmat
#form qutip import sigmax, sigmay, sigmaz, sigmap, sigmam

#electronic Gyromagnetic ratio \gamma_e
ELEC_GYRO = -1.76085963023e5  # rad * MHz / T
#\gamma_e/2PI =
ELEC_GFAC = -2.00231930426256 # unit less

# print will be moved to logger system (todo)

def check_gfactor(gfactor):
    r"""
    Check if gfactor is matrix or scalar.
        
    :param float gfactor: (ndarray or float) g-factor 

    Returns::

        tuple: tuple containing:
            ndarray or float: g-factor tensor.
            bool: True if gfactor is float, False otherwise.
    """
    try:
        gfactor = float(gfactor)
        check = True
    except TypeError:
        check = False

    if not check:
        gfactor = numpy.asarray(gfactor)
        if gfactor.ndim == 1:  # Assume array
            check = True
        elif not gfactor.shape or gfactor.shape[0] == 1:
            check = True
            gfactor = gfactor.reshape(1)[0]
        else:
            test_gfactors = gfactor.copy()
            indexes = numpy.arange(gfactor.shape[-1])
            test_gfactors[..., indexes, indexes] = 0

            diag_check = numpy.isclose(test_gfactors, 0).all()
            same_check = ((gfactor[..., 0, 0] == gfactor[..., 1, 1]) & (gfactor[..., 1, 1] == gfactor[..., 2, 2])).all()
            check = diag_check & same_check
            if check:
                gfactor = gfactor[..., 0, 0][()]

    return gfactor, check

def zfs_tensor(D, E=0):
    r"""
    Generate (3, 3) ZFS tensor from observable parameters D and E.

       :param float or (3,3) tensor D:  Longitudinal splitting (D) in ZFS **OR** the total ZFS tensor.
       :param float E: Transverse splitting (E) in ZFS.

    Returns:
        tensor with shape (3, 3): Total ZFS tensor.

    .. math::

       \mathbf{D}= \begin{pmatrix}
        -\frac{1}{3}D+E & 0 & 0  \\
         & -\frac{1}{3}D-E & 0  \\
        0 & 0 & \frac{2}{3}D
        \end{pmatrix}
    """

    D = numpy.asarray(D)

    if D.size == 1:
        tensor = numpy.zeros((3, 3), dtype=numpy.float64)
        tensor[0, 0] = -D / 3.0 + E
        tensor[1, 1] = -D / 3.0 - E
        tensor[2, 2] =  2.0 / 3.0 * D
    else:
        tensor = D

    return tensor


class Spin(object):
    r"""
    Spin object base class.

    Base Class for spin object, which contains the properties of spin, including:

    :param float spin: Total spin. Default: 0.5
    :param 1darray coord: Cartesian coordinates of spin. Default: (0, 0, 0)
    :param float D: (longitudional splitting) parameter of central spin in ZFS tensor. Default 0.0, unit: MHz.
    :param float E: (transverse splitting) parameter of central spin in ZFS tensor. Default 0.0, unit: MHz.
    :param float/tensor gfactor: gfactor of central spin. Default: -2.00231930426256. unit: unitless.
    :param float spin: taotal spin
    :param float alpha: alpha state :math:`\ket{0}`
    :param float beta: beta state :math:`\ket{1}`
    :param float detuning: detuning from the Zeeman splitting. Default 0, unit: MHz.

    Other notes:
    
    Central spin is: 

    .. math::

        \hat{H} =& \mathbf{S}\cdot\mathbf{D}\cdot\mathbf{S} + \mathbf{B}\cdot\gamma\cdot\mathbf{S} \\
                =& \mathbf{S}\cdot\mathbf{D}\cdot\mathbf{S} + \frac{e}{2m} \mathbf{B} g \mathbf{S}.

    Gyromagnetic ratio (:math:`\gamma`) is related to gfactor (g) via :math:`\gamma=g\frac{e}{2m}`, i.e.,
    gyromagnetic ratio is equal to the g-factor times the fundamental charge-to-mass ratio.

    Unit of :math:`\gamma=\frac{e}{2m}g` is: rad * MHz/T
    Default unit is MHz
    """

    def __init__(self, 
            coord = numpy.zeros(3),
            D = 0.0, 
            E = 0.0,
            **kwargs):

        self.coord = coord
        self.spin = 0.5
        self.D = 0.0
        self.E = 0.0
        self.alpha = None
        self.beta = None
        self._zfs = None
        self.gfactor = ELEC_GFAC 
        self.detuning = 0.0

        self.__dict__.update(kwargs)

        if self._zfs is None:
            self.get_zfs(D, E)
        self.set_gfactor(self.gfactor)

        self.ndim = int(self.spin * 2 + 1 + 1e-8)
        #print("dimension of the spin is", self.ndim)

        """ Hamiltonian of central spin """
        self.Hamiltonian = None
        """ eigen energeis and states """
        self.eig = None
        self.evecs = None
        self.sigma = None

    def get_zfs(self, D, E):
        self._zfs = zfs_tensor(D, E)

    def set_gfactor(self, gfactor):
        r"""
        Set gfactormagnetic ratio of the central spin.

        :param float or (3,3) tensor gfactor: : g-factor of central spin **OR**
            Tensor describing central spin interactions with the magnetic field.
        """

        check = not numpy.asarray(gfactor).shape == (3, 3)
        if check:
            gfactor, check = check_gfactor(gfactor)
            if check:
                gfactor = numpy.eye(3) * gfactor
        self._gfactor = gfactor

    def get_sigma(self):
        r"""Template method to get sigma."""
        self.sigma = None # todo

    def sigma(self):
        r"""Template method to return sigma."""
        if self.sigma is None: self.get_sigma()
        return self.sigma

    def __repr__(self):
        r"""Return string of Python code that recreate the MyClass object."""
        log = ""
        return log

    def __str__(self):
        r"""Print function for the spin object."""
        log_info=""
        return log_info

class SpinSystem(Spin):
    r"""
    Class for containing all the central spins in the system.
    """
    def __init__(self, 
            coords = None,
            spin = None,
            coord = None,
            **kwargs):

        self.spin = [0.5]
        self.__dict__.update(kwargs)
        self.spin = numpy.asarray(self.spin)
        print("debug-zy:spin=", self.spin)
        self.size = self.spin.size
        print("debug-zy:size=", self.size)

        if coord is None:
            coord = numpy.asarray([[0, 0, 0]] * self.size)
        if spin is None:
            spin = 0


if __name__ == '__main__':
    center = Spin(spin=0.5)
    print(center.spin)
