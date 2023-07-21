"""
Basic class of electronic structure driver.
Derived class can be either quantum model system or ab initio solvers.
"""


# basic class of model
class QuantumDriver(object):
    def __init__(self):
        r"""Save name of QM calculator and its method"""
        self.qm_prog = str(self.__class__).split(".")[1]
        self.qm_method = self.__class__.__name__

    # or using get_energies_n_forces?
    def get_tdh(self, force_only=False):
        r"""Time-dependent Hamiltonain in adiabatic representation for
        propagating electronic EOM. This function will compute the adiabatic
        energies (call get_energies) and NACT among different states (via get_nact).
        """

        raise NotImplementedError("Method Not Implemented")

    def load_geom(self, molecule):
        r"""Load geometry"""

        raise NotImplementedError("Method Not Implemented")

    def nuc_grad(self):
        r"""Function for computing the nuclear gradients of energies
        Must be impelemented in derived classed

        Return array containing the gradients [Nat][ndim]
        """
        raise NotImplementedError("Method Not Implemented")

    get_forces = nuc_grad

    def get_nact(self, that):
        r"""Function for computing the NACT using wavefunction overlap method,
        i.e., overlap between self and that (a quantum object of previous time step):

        .. math::
           :nowrap:

           \begin{align*}
              NACT(t_i) =& \bra{\phi(t_i)}\partial_t\ket{\phi(t_i)}  & \\
                        =&-\bra{\phi(t_i)}\phi(t_i-1)\rangle  &
           \begin{align*}

        This function is implemented in each drived classed.
        """
        raise NotImplementedError("Method Not Implemented")

    def get_nacr(self):
        r"""Function for computing derivative couplings (analytically)

        .. math::

           NACR = \bra{\phi}\nabla_R\ket{\phi}
        """

        raise NotImplementedError("Method Not Implemented")

    def get_energies(self):
        r"""
        Compute the adiabatic states and corresponding energies
        """
        raise NotImplementedError("Method Not Implemented")
