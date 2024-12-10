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
# Authors:   Yu Zhang    <zhy@lanl.gov>
#          Ilia Mazin <imazin@lanl.gov>
#

from typing import Union, List
import warnings
import numpy
import scipy
from pyscf import lib
from pyscf import gto
from pyscf.lib import logger
from openms.lib.ov_blocks import one_e_blocks, block_diag
from openms.lib.ov_blocks import two_e_blocks, two_e_blocks_full
from openms.mqed import qedhf, scqedhf, vtqedhf


def get_bosonic_Ham(nmodes, nboson_states, omega, za, Fa):
    r"""Construct Bosonic Hamiltonian in different representation
    after integrating out the electronic DOF.

    Bosonic Hamiltonain after integrating out the electronic DOF:

    .. math::

        H_b = \omega_\alpha b^\dagger_\alpha b_\alpha +
              \sqrt{\frac{\omega_\alpha}{2}}\langle\lambda_\alpha\cdot\boldsymbol{D}
              \rangle (b^\dagger_\alpha + b_\alpha)

    It becomes diagonal in the CS representation:

    .. math::

        H_{cs} = \omega_\alpha b^\dagger_\alpha b_\alpha

    In the SC representation, it is

    .. math::

        H_{sc} = \omega_\alpha b^\dagger_\alpha b_\alpha + TBA.

    In the VSQ + VLF representation:

    .. math::

       H_{VSQ} = &\omega (\cosh(2r)\omega_\alpha b^\dagger_\alpha b_\alpha + \sinh^2(r)
                 -\frac{1}{2}\sinh(2r)[b^2_\alpha + b^{\dagger 2}_\alpha]
                 \\
                 & + e^{-r} \sqrt{\frac{\omega_\alpha}{2}}\langle\Delta\lambda_\alpha\cdot\boldsymbol{D}
                  \rangle (b^\dagger_\alpha + b_\alpha)
                 + TBA.
    """

    boson_size = sum(nboson_states)
    Hb = numpy.zeros((boson_size, boson_size))
    idx = 0
    for imode in range(nmodes):
        cosh2r = numpy.cosh(2.0 * Fa[imode])
        sinhr = numpy.sinh(Fa[imode])
        mdim = nboson_states[imode]
        # diaognal term, didn't include ZPE
        H0 = numpy.diag(numpy.arange(mdim) * cosh2r + sinhr * sinhr) * omega[imode]

        # off-diaognal term
        h_od = numpy.diag(numpy.sqrt(numpy.arange(1, mdim)), k = 1) \
            + numpy.diag(numpy.sqrt(numpy.arange(1, mdim)), k = -1)
        H0 += h_od * za[imode]

        Hb[idx:idx+mdim, idx:idx+mdim] = H0
        idx += mdim

    print (Hb)
    return Hb


def get_dipole_ao(mol, add_nuc_dipole=True, origin_shift=None):
    r"""Return dipole moment matrix in atomic orbital (AO) basis.

    The first dimension specifies a Cartesian coordinate:
    (**X**, **Y**, **Z**).

    Parameters
    ----------
    mol : :class:`MoleBase <pyscf.gto.mole.MoleBase>`
        PySCF molecule object.

    Keyword Arguments
    -----------------
    add_nuc_dipole : bool
        Nuclear component in dipole moment matrix,
        **optional** ``default = True``.
    origin_shift : :class:`~numpy.ndarray`
        If not None, dipole computed as if
        molecule has been shifted away from
        origin by some distance,
        **optional** ``default = True``.

    Return
    ------
    dipole_ao : :class:`~numpy.ndarray`
        Dipole moment matrix.
    """

    origin = numpy.zeros(3, dtype=float)

    # Center of nuclear charge
    if add_nuc_dipole:
        charge_center = lib.einsum("i,ix->x",
                                   mol.atom_charges(), mol.atom_coords())
        charge_center /= numpy.sum(mol.atom_charges())

        origin += charge_center

    # Shift away from origin
    if origin_shift is not None:
        origin -= origin_shift

    with mol.with_common_orig(origin):
        dipole_ao = -mol.intor_symmetric("int1e_r", comp=3)
        return dipole_ao


def get_quadrupole_ao(mol, add_nuc_dipole=True, origin_shift=None):
    r"""Return quadrupole moment matrix in AO basis.

    .. math::

        Q_{uv} = & \bra{u} (\mu_{tot} e)^2 \ket{v} \\
             = & \bra{u} (e^T  Q  e) \ket{v} + 2 \bra{u} \mu  e \ket{v} (\mu_{nuc} e) + (\mu_{nuc} e)^2

    The first dimension specifies the vector direction:

        ========= ========= =========
         [0]: XX   [3]: XY   [6]: XZ
         [1]: YX   [4]: YY   [7]: YZ
         [2]: ZX   [5]: ZY   [8]: ZZ
        ========= ========= =========

    Diagonal elements correspond to:

        quadrupole_ao[0] = :math:`\bm{Q}_{XX}`

        quadrupole_ao[4] = :math:`\bm{Q}_{YY}`

        quadrupole_ao[8] = :math:`\bm{Q}_{ZZ}`

    The lower- and upper-triangles are related as:

        quadrupole_ao[1] = :math:`\bm{Q}_{YX}`

        quadrupole_ao[2] = :math:`\bm{Q}_{ZX}`

        quadrupole_ao[3] = :math:`(\bm{Q}_{YX})^T`

        quadrupole_ao[5] = :math:`\bm{Q}_{ZY}`

        quadrupole_ao[6] = :math:`(\bm{Q}_{ZX})^T`

        quadrupole_ao[7] = :math:`(\bm{Q}_{ZY})^T`

    Parameters
    ----------
    mol : :class:`MoleBase <pyscf.gto.mole.MoleBase>`
        PySCF molecule object.

    Keyword Arguments
    -----------------
    add_nuc_dipole : bool
        Nuclear component in quadrupole moment matrix,
        **optional** ``default = True``.
    origin_shift : :class:`~numpy.ndarray`
        If not None, dipole computed as if
        molecule has been shifted away from
        origin by some distance,
        **optional** ``default = True``.

    Return
    ------
    quadrupole_ao : :class:`~numpy.ndarray`
        Quadrupole moment matrix.
    """

    origin = numpy.zeros(3, dtype=float)

    # Center of nuclear charge
    if add_nuc_dipole == True:
        charge_center = lib.einsum("i,ix->x",
                                   mol.atom_charges(), mol.atom_coords())
        charge_center /= numpy.sum(mol.atom_charges())

        origin += charge_center

    # Shift away from origin
    if origin_shift is not None:
        origin -= origin_shift

    with mol.with_common_orig(origin):
        quadrupole_ao = -mol.intor("int1e_rr")
        return quadrupole_ao


class Boson(object):
    r"""Boson base class.

    At minimum, :meth:`~Boson.__init__` requires:

        1. PySCF molecule object for electronic quantities,
        2. frequency value and vector for each ``mode`` you wish to
           create in constructing :class:`Boson` object.

    Number of provided frequencies and vectors determines value of
    :attr:`nmodes`. ``nboson_states`` keyword argument defines particle number
    value for each ``mode``, ``default = 1`` for all modes.

    Note
    ----
    If coupling constants :attr:`gfac` are not provided,
    normalization constants for :attr:`vec` of each ``mode``
    are used.

    Parameters
    ----------
    mol : :class:`MoleBase <pyscf.gto.mole.MoleBase>`
        PySCF molecule object.
    omega : float, :class:`list[floats] <list>`, :class:`~numpy.ndarray`
        Bosonic frequencies, :math:`\om_\al`.
    vec : list, :class:`~numpy.ndarray`
        Bosonic mode vectors, :math:`\bm{e}_\al`.

    Keyword Arguments
    -----------------
    gfac : float, :class:`list[floats] <list>`, :class:`~numpy.ndarray`
        Coupling constants, :math:`\la_\al`, **optional**.
    nboson_states : int, :class:`list[ints] <list>`, :class:`~numpy.ndarray`
        Number states for each mode, **optional**
        ``default = [1] * nmodes``.

    Attributes
    ----------
    boson_type : str
        Type of boson object.
    cavity_type : str
        Type of cavity environment.
    omega : :class:`~numpy.ndarray`
        Bosonic frequencies, :math:`\om_\al`.
    vec : :class:`~numpy.ndarray`
        Bosonic mode vectors, :math:`\bm{e}_\alpha`.
    gfac : :class:`~numpy.ndarray`
        Coupling constants, :math:`\la_\al`.
    nmodes : int
        Number of boson modes, :code:`len(omega)`.
    nboson_states : :class:`~numpy.ndarray`
        Number states of each ``mode``.
    use_cs : bool
        Coherent-state basis if ``True``,
        otherwise use Fock basis.
    z_alpha : :class:`~numpy.ndarray`
        :math:`z_\alpha` of each ``mode``,
        non-zero in coherent-state.
    boson_coeff : :class:`list[ndarray] <list>`
        Bosonic eigenvectors of each ``mode``.
    e_boson : float
        Bosonic energy.
    verbose : int
        Level of output message detail.
    stdout :
        Where/how to output messages.

    _mol : :class:`MoleBase <pyscf.gto.mole.MoleBase>`
        PySCF molecule object.
    nao : int
        Size of electronic basis set.
    nelectron : int
        Number of electrons.

    _mf : :class:`RHF <mqed.qedhf.RHF>`
        Instance of OpenMS mean-field class.
    add_nuc_dipole : bool
        Include nuclear component in dipole/quadrupole,
        **optional** ``default = True``.
    origin_shift : :class:`~numpy.ndarray`
        XYZ coordinates by which to shift
        molecule away from origin,
        **optional** ``default = None``.
    complete_basis : bool
        Controls how OEI component is computed,
        **optional** ``default = True``.
    couplings : :class:`~numpy.ndarray`
        Coupling terms, :math:`\la_\al`,
        copy of :attr:`gfac`.
    couplings_bilinear : :class:`~numpy.ndarray`
        Bilinear coupling terms,
        :math:`\sqrt{\frac{\om_\al}{2}} \la_\al`.
    couplings_self : :class:`~numpy.ndarray`
        Dipole self-energy coupling terms,
        :math:`\frac{1}{2} \la_\al^2`.
    couplings_var : :class:`~numpy.ndarray`
        Variational parameters, :math:`\bm{f}`.

        - ``Default = 0`` for each mode.
        - For SC-QED-HF, ``= 1`` for each ``mode``.
        - For VT-QED-HF, ``= 0.5`` for each ``mode``
          (if no values provided in arguments).
    couplings_res : :class:`~numpy.ndarray`
        Difference between :math:`1.0` and :attr:`couplings_var`.
        ``Default = 1`` for each ``mode``.
    optimize_varf : bool
        Optimize variational parameters, :math:`\bm{f}`.
        ``Default = False``.

    Raises
    ------
    ValueError
        Whenever provided parameter is not expected type or shape.
    NotImplementedError
        Most methods in :class:`Boson` are templates that
        are meant to be overwritten by its subclasses.
    """

    def __init__(
        self, mol, omega=None, vec=None, gfac=None,
        mf=None,
        nboson_states: Union[int, List[int]] = 1,
        add_nuc_dipole=True,
        **kwargs
    ):

        # PySCF molecule object
        self._mol = None
        if not isinstance(mol, gto.mole.MoleBase):
            err_msg = f"Parameter 'mol' is not an instance " + \
                      f"of MoleBase class, part of the PySCF " + \
                      f"quantum chemistry software package."
            logger.error(self, err_msg)
            raise ValueError(err_msg)

        else:
            self._mol = mol
            self.verbose = mol.verbose
            self.stdout = mol.stdout
            self.nelectron = mol.nelectron

        # Default: Include nuclear component in dipole/quadrupole
        self.add_nuc_dipole = add_nuc_dipole

        # Additional shift of molecule from dipole/quadrupole origin
        self.origin_shift = None
        if "origin_shift" in kwargs and kwargs["origin_shift"] is not None:

            o_shift = kwargs["origin_shift"].copy()
            if isinstance(o_shift, list):
                o_shift = numpy.asarray(o_shift, dtype=float)

            if (not isinstance(o_shift, numpy.ndarray)
                or o_shift.size != 3):
                err_msg = f"Parameter 'omega_shift' does " + \
                          f"not have dimension: len(3)."
                logger.error(self, err_msg)
                raise ValueError(err_msg)

            else:
                self.origin_shift = o_shift.copy()
                del (o_shift)


        # Determines if QED-OEI contribution constructed from
        # quadrupole moment matrix or product of dipole moment matrices
        self.complete_basis = kwargs.get("complete_basis", True)

        # Cavity modes and frequencies
        self.omega = self.nmodes = None

        if isinstance(omega, list):
            omega = numpy.asarray(omega, dtype=float)

        if not isinstance(omega, (float, numpy.ndarray)):
            err_msg = f"Parameter 'omega' is not float, " + \
                      f"list of floats, or ndarray of floats."
            logger.error(self, err_msg)
            raise ValueError(err_msg)

        else:
            if isinstance(omega, float):
                self.omega = numpy.asarray(omega, dtype=float)
            else:
                self.omega = omega.copy()
            self.nmodes = omega.size

        # Molecular coupling strengths
        self.gfac = None

        if gfac is not None:
            if isinstance(gfac, list):
                gfac = numpy.asarray(gfac, dtype=float)

            # Check if gfac is an array and has the same size as omega
            if (not isinstance(gfac, numpy.ndarray)
                or gfac.size != self.nmodes):
                err_msg = f"Parameter 'gfac' does not " + \
                          f"have dimension: len(omega)."
                logger.error(self, err_msg)
                raise ValueError(err_msg)

            else:
                self.gfac = gfac.copy()

        else:
            debug_msg = f"Parameter 'gfac' not provided, will use " + \
                        f"normalization constant of 'vec' parameter."
            logger.debug(self, debug_msg)

        # Cavity mode polarization vectors
        self.vec = None

        if isinstance(vec, list):
            vec = numpy.asarray(vec, dtype=float)

        if (not isinstance(vec, numpy.ndarray)
            or vec.shape != (self.nmodes, 3)):
            err_msg = f"Parameter 'vec' does not have " + \
                      f"dimensions: (len(omega), 3)."
            logger.error(self, err_msg)
            raise ValueError(err_msg)

        # Normalize vectors first, use normalization constants
        # as the gfac values, if coupling strengths were not provided
        else:
            m_cnsts = numpy.zeros(self.nmodes)
            for a in range(self.nmodes):
                cnst = m_cnsts[a] = numpy.sqrt(numpy.dot(vec[a], vec[a]))
                if cnst > 1e-15: # Prevent division by zero
                    vec[a] = vec[a] / cnst

            if self.gfac is None:
                self.gfac = m_cnsts.copy()
            del m_cnsts

            self.vec = vec.copy()

        # Cavity mode photon occupation numbers
        self.nboson_states = numpy.ones(self.nmodes, dtype=int)

        # Photon occupation specified by integer or list of integers
        if isinstance(nboson_states, int): # same occupation for all modes
            nboson_states = numpy.repeat(nboson_states, self.nmodes)
        elif isinstance(nboson_states, list): # each mode has own occupation
            nboson_states = numpy.asarray(nboson_states, dtype=int)

        if (isinstance(nboson_states, numpy.ndarray)
            and (nboson_states.size == self.nmodes)):

            if not all(i > 0 for i in nboson_states):
                err_msg = f"Elements of 'nboson_states' are " + \
                          f"not all integers > 0."
                logger.error(self, err_msg)
                raise ValueError(err_msg)

            else:
                self.nboson_states = nboson_states.copy()

        else:
            err_msg = f"Parameter 'nboson_states' is not int, " + \
                      f"list of ints, or ndarray of ints."
            logger.error(self, err_msg)
            raise ValueError(err_msg)

        # Photon wavefunction (default : coherent-state (CS))
        # Fock state representation if CS flag set to False
        self.use_cs = kwargs.get('use_cs', True)
        self.z_alpha = numpy.zeros(self.nmodes, dtype=float)

        # Providing CS "z_alpha" values also sets CS flag to False
        if "z_alpha" in kwargs and kwargs["z_alpha"] is not None:

            z_a = kwargs["z_alpha"].copy()
            if isinstance(z_a, list):
                z_a = numpy.asarray(z_a, dtype=float)

            if (not isinstance(z_a, numpy.ndarray) or z_a.size != self.nmodes):
                err_msg = f"Parameter 'z_alpha' does " + \
                          f"not have dimension: len(omega)."
                logger.error(self, err_msg)
                raise ValueError(err_msg)

            else:
                self.use_cs = False
                self.z_alpha = z_a.copy()

        if self.use_cs:
            logger.info(self, "CS basis for photon is used")
        else:
            logger.info(self, "Fock basis for photon is used")

        if "couplings_var" in kwargs:
            self.couplings_var = kwargs["couplings_var"]
            self.optimize_varf = False
        else:
            self.optimize_varf = True
            self.couplings_var = 0.5 * numpy.ones(self.nmodes)

        self.squeezed_cs = kwargs.get("squeezed_cs", False)
        self.optimize_vsq = kwargs.get("optimize_vsq", False)
        self.squeezed_var = numpy.zeros(self.nmodes)
        if "squeezed_var" in kwargs:
            self.optimize_vsq = False
            self.squeezed_var = kwargs["squeezed_var"]

        # Pre-factors of PF Hamiltonian components from coupling strengths
        self.couplings = numpy.asarray(self.gfac, dtype=float)
        self.couplings_bilinear = numpy.zeros(self.nmodes, dtype=float)
        self.couplings_self = numpy.zeros(self.nmodes, dtype=float)
        self.e_boson_grad_r = numpy.zeros(self.nmodes, dtype=float)
        self.couplings_res = numpy.zeros(self.nmodes, dtype=float)

        for a in range(self.nmodes):
            self.couplings_bilinear[a] = (
                self.couplings[a] * numpy.sqrt(0.5 * self.omega[a])
            )
            self.couplings_res[a] = 1.0 - self.couplings_var[a]
            self.couplings_self[a] = 0.5 * self.couplings[a] ** 2

        # Variational attributes (default : QED-HF)
        self.optimize_varf = False

        # Boson info
        self.boson_type = self.__class__.__name__
        self.cavity_type = None
        self.e_boson = 0.0

        boson_size = sum(self.nboson_states)
        self.boson_coeff = numpy.zeros((boson_size, boson_size))
        for imode in range(self.nmodes):
            mdim = self.nboson_states[imode]
            idx = sum(self.nboson_states[:imode])
            self.boson_coeff[idx : idx + mdim, idx : idx + mdim] = numpy.eye(mdim)

        # Mean-field info
        self._mf = mf
        self.nao = self._mol.nao_nr()
        self.dipole_ao = None
        self.quadrupole_ao = None

        # Values used by 'post-hf integrals' methods
        # whether to shfit the h1 in post-hf integral with DSE_oei
        self.shift = kwargs.get("shift", False)
        # self.shift = shift

        # ------------------------------------
        self.spin = mol.spin
        self.intor = mol.intor
        self.symmetry = mol.symmetry
        self.intor_symmetric = mol.intor_symmetric
        self._pseudo = mol._pseudo
        self._ecpbas = mol._ecpbas
        self._add_suffix = mol._add_suffix
        self.nao_nr = mol.nao_nr
        self.ao_loc_nr = mol.ao_loc_nr

        self._atm = mol._atm
        self._bas = mol._bas
        self._env = mol._env
        self.energy_nuc = mol.energy_nuc
        self.nelec = mol.nelec
        # ------------------------------------

    # ---------------------
    # General boson methods
    # ---------------------

    def get_boson_dm(self, mode=None):
        r"""Return photon ``mode`` density matrix.

        Parameters
        ----------
        mode : int
            Index for ``mode`` stored in the object.

        Returns
        -------
        :class:`~numpy.ndarray`
            photon mode density matrix
        """

        if mode is None:
            bc = self.boson_coeff[:, 0].copy()
            return numpy.outer(numpy.conj(bc), bc)

        mdim = self.nboson_states[mode]
        idx = 0
        if mode > 0: # non-vacuum state density, not tested
            idx = sum(self.nboson_states[:mode]) + mode

        bc = self.boson_coeff[idx:idx+mdim, 0].copy()
        return numpy.outer(numpy.conj(bc), bc)


    def update_boson_coeff(self, dm):
        r"""Update eigenvectors for all modes in :class:`Boson`.

        Construct and diagonalize photonic component of the
        Pauli-Fierz Hamiltonian for ``mode`` of :class:`Boson`.
        Store eigenvectors in :attr:`boson_coeff`, update
        photonic energy, :attr:`e_boson`.

        .. math::
            \hat{H}_\tt{photon}
            &= \sum_\al \hat{H}^\al_\tt{photon} \\
            \hat{H}^\al_\tt{photon}
            &= \om_\al \cb{\al} \ab{\al}
             + \sqrt{\frac{\om_\al}{2}} \bm{e}_\al
               \cdot \sum_{\mu\nu} \rho_{\mu\nu}
               \cdot \mel*{\mu}{\la_\al \cdot \hat{D}}{\nu} (\cb{\al} + \ab{\al})
        where :math:`\bm{\rho}` is the provided electronic density matrix, ``dm``.

        --------------------------------------------------------------------

        The bosonic creation and annihilation operators act on photon number
        states as:

        .. math::
            \cb{} \ab{} \ket{m} &= m \ket{m} \\
                  \cb{} \ket{m} &= \sqrt{m+1} \ket{m+1} \\
                  \ab{} \ket{m} &= \sqrt{m} \ket{m-1}
        In the Fock/particle number basis, the off-diagonal bilinear
        interaction terms are:

        .. math::
            \mel*{n}{\cb{} + \ab{}}{m}
            &= \mel*{n}{\cb{}}{m} + \mel*{n}{\ab{}}{m} \\
            &= \mel*{n}{\sqrt{m+1}}{m+1} + \mel*{n}{\sqrt{m}}{m-1} \\
            &= \d{m+1}{n} \sqrt{m+1} + \d{m-1}{n} \sqrt{m}

        .. note::
            In coherent-state representation, photonic Hamiltonian
            has only diagonal terms.
            :math:`\hat{H}_{\tt{photon}}=\sum_{\al}\om_{\al}\cb{\al}\ab{\al}`


        In the VSQ representation: (TBA)

        Parameters
        ----------
        dm : :class:`~numpy.ndarray`
            Density matrix.

        Returns
        -------
        :class:`~numpy.ndarray`, :class:`~numpy.ndarray`
            Eigenvalues and eigenvectors of the photonic Hamiltonian.
        """

        # use the new get_bosonic_Ham to get the matrix

        # # Total photon energy
        # ph_e_tot = 0.0

        # for a in range(self.nmodes):
        #     mdim = self.nboson_states[a]

        #     # Diagonal terms
        #     hmat = numpy.diag(self.omega[a] * numpy.arange(mdim))

        #     # Off diagonal terms, scaled by bilinear-coupling term and residual
        #     # coupling values, if the cavity mode has non-zero photon occupation
        #     # Also, photonic Hamiltonian is diagonal in CS representation
        #     if mdim > 1 and self.use_cs == False:

        #         h_od = numpy.diag(numpy.sqrt(numpy.arange(1, mdim)), k = 1) \
        #                 + numpy.diag(numpy.sqrt(numpy.arange(1, mdim)), k = -1)

        #         za = lib.einsum("pq, qp-> ", self.get_geb_ao(a), dm)
        #         za *= self.couplings_res[a]

        #         hmat -= (h_od * za)

        #     # Photon cavity mode eigenvectors
        #     self.boson_coeff[a] = h_evec = self._mf.eig(hmat, numpy.eye(mdim))[1]

        #     # Photon mode energy
        #     mode_e_tot = 0.0
        #     for n in range(mdim):
        #         coeff_term = numpy.conj(h_evec[n, 0]) * h_evec[n, 0]
        #         mode_e_tot += self.omega[a] * n * coeff_term
        #     ph_e_tot += mode_e_tot # add to total photon energy
        # self.e_boson = ph_e_tot

        # we assume noninteracting bosons at this moment
        za = lib.einsum("pq, Xpq ->X", dm, self.gmat) - self.z_alpha
        za *= self.couplings_res * numpy.sqrt(self.omega/2.0) # only consider the residual part

        Fa = self.squeezed_var
        hmat = get_bosonic_Ham(self.nmodes, self.nboson_states, self.omega, za, Fa)

        # Photon cavity mode eigenvectors
        # h_evec = self._mf.eig(hmat, numpy.eye(sum(nboson_states)))[1]
        e, c = scipy.linalg.eigh(hmat)
        idx = numpy.argmax(abs(c.real), axis=0)
        e = e[idx]
        c[:,c[idx,numpy.arange(len(e))].real<0] *= -1

        # Photon ground state energy
        Etot = 0.0
        # we did't include ZPE here
        idx = 0
        for imode in range(self.nmodes):
            mdim = self.nboson_states[imode]
            sinh2r = numpy.sinh(2.0 * Fa[imode])
            cosh2r = numpy.cosh(2.0 * Fa[imode])

            p = c[idx:idx+mdim,0].conj() * c[idx:idx+mdim,0]
            nw = (numpy.arange(mdim) + 0.5) * cosh2r - 0.5
            nw_grad = (numpy.arange(mdim) + 0.5) * sinh2r * 2.0
            Etot += self.omega[imode] * numpy.dot(p, nw)
            self.e_boson_grad_r[imode] = self.omega[imode] * numpy.dot(p, nw_grad)
            idx += mdim

        self.e_boson = Etot
        self.boson_coeff = c
        return self


    # def update_mean_field(self, mf, **kwargs):
    #     r"""
    #     Update with attributes from mean-field object, parameter ``mf``.

    #     Called by OpenMS mean-field constructors, save in :attr:`._mf`.
    #     Overwrite :attr:`verbose` and :attr:`stdout` values. For each ``mode``,
    #     call :func:`init_boson_coeff_guess` to save initial boson coefficient
    #     guess in :attr:`boson_coeff`.

    #     If ``couplings_var`` keyword argument is provided to function:
    #         1. store in :attr:`couplings_var`,
    #         2. update :attr:`optimize_varf` flag to ``True``,
    #         3. call :func:`update_couplings` to update
    #            :attr:`couplings_res`.

    #     :attr:`optimize_varf` flag update **(in step 2, above)** can
    #     be overwritten/undone if additional ``optimize_varf=False``
    #     keyword argument is provided.

    #     Parameters
    #     ----------
    #     mf : :class:`RHF <mqed.qedhf.RHF>`
    #         Instance of OpenMS mean-field class.

    #     Keyword Arguments
    #     -----------------
    #     couplings_var : float, :class:`list[floats] <list>`, :class:`~numpy.ndarray`
    #         Variational parameter values, :math:`f_\al`.
    #         **Optional**
    #     optimize_varf : bool
    #         Whether to optimize variational parameters.
    #         **Optional,** ``default = False``

    #     Raises
    #     ------
    #     ValueError
    #         Whenever provided parameter is not expected type or shape.
    #     """

    #     self._mf = mf
    #     self.verbose = mf.verbose
    #     self.stdout = mf.stdout
    #     self.nao = mf.nao

    #     # Variational parameters and optimization flag (if provided)
    #     if "couplings_var" in kwargs and kwargs["couplings_var"] is not None:

    #         # QED-HF
    #         if type(self._mf) == qedhf.RHF:
    #             warn_msg = f"QED-HF does not require variational parameters."
    #             logger.warn(self, warn_msg)

    #         # SC/VT-QED-HF
    #         elif type(self._mf) in (scqedhf.RHF, vtqedhf.RHF):

    #             if isinstance(kwargs["couplings_var"], list):
    #                 kwargs["couplings_var"] = numpy.asarray(kwargs["couplings_var"],
    #                                                         dtype=float)

    #             if (not isinstance(kwargs["couplings_var"], numpy.ndarray) or
    #                 kwargs["couplings_var"].size != self.nmodes):
    #                 err_msg = f"Parameter 'couplings_var' does " + \
    #                           f"not have dimension: len(omega)."
    #                 logger.error(self, err_msg)
    #                 raise ValueError(err_msg)

    #             # SC-QED-HF
    #             if type(self._mf) == scqedhf.RHF:

    #                 if not all(i == 1.0 for i in kwargs["couplings_var"]):
    #                     err_msg = f"Value of 'f' parameter must be 1.0 for " + \
    #                               f"each mode when mean-field is SC-QED-HF."
    #                     logger.error(self, err_msg)
    #                     raise ValueError(err_msg)

    #                 if ("optimize_varf" in kwargs and kwargs["optimize_varf"] == True):
    #                     warn_msg = f"Cannot optimize 'f' in SC-QED-HF. " + \
    #                                f"Valus fixed to 1.0 for each mode."
    #                     logger.warn(self, warn_msg)

    #                 self.couplings_var = kwargs["couplings_var"].copy()
    #                 self.update_couplings()
    #                 self.optimize_varf = False

    #             # VT-QEDHF
    #             if type(self._mf) == vtqedhf.RHF:

    #                 self.couplings_var = kwargs["couplings_var"].copy()
    #                 self.update_couplings()
    #                 self.optimize_varf = True

    #                 if ("optimize_varf" in kwargs and kwargs["optimize_varf"] == False):
    #                     self.optimize_varf = False

    #         else:
    #             err_msg = f"Mean-field object must be " + \
    #                       f"instance of OpenMS QED object."
    #             logger.error(self, err_msg)
    #             raise ValueError(err_msg)

    #     # Ensure SC/VT-QEDHF coupling is correct
    #     else:
    #         if type(self._mf) == scqedhf.RHF:
    #             self.couplings_var = numpy.ones(self.nmodes,
    #                                             dtype=float)
    #             self.update_couplings()
    #             self.optimize_varf = False

    #         elif type(self._mf) == vtqedhf.RHF:
    #             self.couplings_var = 0.5 * numpy.ones(self.nmodes,
    #                                                   dtype=float)
    #             self.update_couplings()
    #             self.optimize_varf = True

    #     # Modified dipole moment matrix
    #     self.dipole_ao = self.get_dipole_ao()
    #     self.gmat = self.get_gmat_ao()

    #     # Modified quadrupole moment matrix, only for "complete_basis"
    #     if self.complete_basis:
    #         self.quadrupole_ao = self.get_quadrupole_ao()
    #         self.q_lambda_ao = self.get_q_lambda_ao()

    #     return self


    def update_couplings(self):
        """Update :attr:`couplings_res` by ``1.0 - couplings_var``."""
        self.couplings_res = numpy.ones(self.nmodes) - self.couplings_var


    def dump_flags(self):
        if self.verbose < logger.INFO:
            return self

        logger.info(self, '\n\n******** %s ********', self.__class__)
        logger.info(self, 'boson_type = %s', self.__class__.__name__)
        logger.info(self, 'complete_basis = %s', self.complete_basis)
        logger.info(self, 'add_nuc_dipole = %s', self.add_nuc_dipole)
        if self.origin_shift is not None:
            logger.info(self, 'origin_shift = %.3f   %.3f   %.3f', *(self.origin_shift))
        logger.info(self, 'optimize_varf = %s', self.optimize_varf)
        logger.info(self, 'use_cs = %s', self.use_cs)
        logger.info(self, 'nmodes = %d\n', self.nmodes)
        for a in range(self.nmodes):
            logger.info(self, '------ cavity mode #%s ------', (a + 1))
            logger.info(self, 'nboson_states[%s] = %d', *(a, self.nboson_states[a]))
            logger.info(self, 'omega[%s] = %.1f', *(a, self.omega[a]))
            logger.info(self, 'vec[%s] = %.3f   %.3f   %.3f', *(a, *self.vec[a]))
            if self.use_cs:
                logger.info(self, 'coherent_state[%s] = %.10f', *(a, self.z_alpha[a]))
            logger.info(self, '== %s ==', ("PF Hamiltonian"))
            logger.info(self, 'couplings[%s] = %.10f', *(a, self.couplings[a]))
            logger.info(self, 'couplings_bilinear[%s] = %.10f', *(a, self.couplings_bilinear[a]))
            logger.info(self, 'couplings_dse[%s] = %.10f', *(a, self.couplings_self[a]))
            logger.info(self, '== %s ==', ("SC/VT-QED-HF"))
            logger.info(self, 'couplings_var[%s] = %.10f', *(a, self.couplings_var[a]))
            logger.info(self, 'couplings_res[%s] = %.10f', *(a, self.couplings_res[a]))
        return self

    # ----------------------
    # Template boson methods
    # ----------------------

    def update_cs(self):
        r"""Template method to update :math:`z_\al` values in CS representation."""
        raise NotImplementedError("Subclasses must implement this method.")

    def get_gmat_so(self):
        """Template method to get coupling matrix in SO."""
        raise NotImplementedError("Subclasses must implement this method.")

    def get_dse_hcore(self):
        r"""Template method to construct DSE-mediated OEI in AO basis."""
        raise NotImplementedError("Subclasses must implement this method.")

    def get_dse_jk(self):
        r"""Template method to construct DSE-mediated J/K matrices in AO basis."""
        raise NotImplementedError("Subclasses must implement this method.")

    def get_quadrupole_ao(self):
        r"""Template method to construct quadrupole moment matrix in AO basis."""
        raise NotImplementedError("Subclasses must implement this method.")

    def get_polarized_quadrupole_ao(self):
        r"""Template method to construct polarized quadrupole moment matrix in AO basis."""
        raise NotImplementedError("Subclasses must implement this method.")

    def get_q_lambda_ao(self):
        r"""Template method to construct polarized dipole self-energy matrix in AO basis."""
        raise NotImplementedError("Subclasses must implement this method.")

    def get_gmat_ao(self):
        r"""Template method to construct polarized interaction matrix in AO basis."""
        raise NotImplementedError("Subclasses must implement this method.")

    def get_geb_ao(self):
        r"""Template method to construct bilinear interaction matrix in AO basis."""
        raise NotImplementedError("Subclasses must implement this method.")

    def get_dipole_ao(self):
        r"""Template method to construct dipole moment matrix in AO basis."""
        raise NotImplementedError("Subclasses must implement this method.")


    def construct_g_dse_JK(self):
        r"""
        DSE-mediated JK matrix
        """
        raise NotImplementedError


class Photon(Boson):
    r"""Photon subclass."""
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.gmat = None
        self.q_lambda_ao = None

        # Values used by 'post-hf integrals' methods
        self.ptot = self.pa = self.pb = None
        self.ca = self.cb = None
        self.na = self.nb = None
        self.const = None

    # --------------
    # QED-HF methods
    # --------------

    def update_cs(self, dm):
        r"""Update :math:`z_\alpha` values.

        If :attr:`use_cs` ``= True``, update values
        stored in :attr:`z_alpha`.

        .. math::

            z_\alpha = \langle \lambda\cdot \boldsymbol{D}\rangle


        Parameters
        ----------
        dm : :class:`~numpy.ndarray`
            Density matrix.

        Returns
        -------
        :class:`~numpy.ndarray`
            Updatec :attr:`z_alpha`, stored in :attr:`z_alpha`.

        .. warning::
            Relate to the working equations, found in the theory,
            the values of :math:`z_\alpha` computed here differ by
            a factor of :math:`-\frac{1}{\sqrt{2\om_\al}}`.
        """

        if self.use_cs:
            self.z_alpha = lib.einsum("pq, Xpq ->X", dm, self.gmat)


    def add_oei_ao(self, dm, s1e=None, residue=False):
        r"""Compute QED-RHF boson-mediated 1e- integrals.

        return DSE-mediated oei.. This is universal for bare HF or QED-HF.
        DSE-mediated oei:

        (to be updated for the many-photon case)

        .. math::

            & -\langle \lambda\cdot D\rangle g^\alpha_{uv} - 0.5 q^\alpha_{uv} + z^2 S/N_e \\
            & = -\text{Tr}[\rho g^\alpha] g^\alpha_{uv} - 0.5 q^\alpha_{uv} + z^2 S/N_e \\
            & = -zg - 0.5 * q (\text{or }g^2) + z^2 S/N_e

        since :math:`z^2 = z^2 * \text{Tr}[S D] /N_e = \text{Tr}[(z^2/N_eS)D]`
        i.e., we can add z^2/Ne*S into oei, where S is the overlap, Ne is total energy
        and Ne = Tr[SD].

        Regardless of Fock basis or coherent-state (CS) representation,
        return quadrupole-modified OEI contribution, which arises
        from the DSE term of the PF Hamiltonian:

        .. math::
            \bm{h}_{\tt{DSE}} &= \bm{h}_{\tt{bare}} - \frac{1}{2}
                                 \sum_\al \sum_{\mu\nu} \rho_{\mu\nu}
                                 \cdot \bm{\tilde{q}}^\al_{\mu\nu} \\
            \bm{\tilde{q}}^\al_{\mu\nu} &= \la_{\al}^2
                                           \cdot \bm{Q}^\al_{\mu\nu}
        where :math:`\bm{Q}^\al` is the polarized quadrupole moment matrix
        in the AO basis.

        In CS representation, when :attr:`use_cs` ``= True``, additional
        DSE-mediated contribution to the OEI is included:

        .. math::
            \bm{h}_{\tt{CS}} &= \bm{h}_{\tt{DSE}} - \bm{\tilde{d}}^\al
                                \sum_\al \sum_{\mu\nu} \rho_{\mu\nu}
                                \cdot \mel*{\mu}{\hat{D}}{\nu} \\
            \bm{\tilde{d}}^\al_{\mu\nu} &= \la_\al
                                           \cdot \bm{D}^\al_{\mu\nu}
        where :math:`\bm{D}^\al` is the polarized dipole moment matrix in
        the AO basis.

        Also in CS representation, :math:`E_{\tt{QED-HF}}` includes DSE
        term:

        .. math::
            E_{\tt{QED-HF}} = E_{\tt{HF}}
                            + \frac{1}{2} \sum_\al
                              \langle
                              [ \la_\al \cdot
                                (\hat{D} - \ev*{\hat{D}}_{\mu\nu})
                              ]^2
                              \rangle
        that can also be included via the OEI, since:

        .. math::
            \langle
            [ \la_\al
            \cdot (\hat{D} - \ev*{\hat{D}}_{\mu\nu})]^2
            \rangle
            &= \langle [ \la_\al
               \cdot (\hat{D} - \ev*{\hat{D}}_{\mu\nu})]^2
               \rangle
               \cdot \frac{\Tr[\bm{S} \cdot \bm{\rho}]}
               {N_{\tt{elec}}} \\
            &= \tt{Tr}
               \left[
               \frac{[ \la_\al \cdot
               (\hat{D} - \ev*{\hat{D}}_{\mu\nu})]^2}
               {N_{\tt{elec}}} \bm{S} \cdot \bm{\rho}
               \right]
        where :math:`\bm{\rho}` is the provided electronic density matrix,
        ``dm``.

        Parameters
        ----------
        dm : :class:`~numpy.ndarray`
            Density matrix.
        s1e : :class:`~numpy.ndarray`
            Overlap matrix, **optional**
            (computed if not provided).
        residue : bool
            Multiply by pre-factor if ``True``.

              - used by VT-QED-HF solver
              - **optional,** ``default = False``

        Return
        ------
        dse_oei : :class:`~numpy.ndarray`
            DSE-mediated OEI for all photon modes.

        """

        self.get_q_dot_lambda()
        self.get_gmatao()
        if s1e is None:
            s1e = self._mf.get_ovlp(self._mol)

        if self.use_cs:
            self.update_cs(dm)

        gvar2 = numpy.ones(self.nmodes)
        if residue:
            gvar2 = self.couplings_res**2 # element-wise

        oei = - lib.einsum("Xpq, X->pq", self.gmat, gvar2 * self.z_alpha)
        if self.complete_basis:
            oei -= 0.5 * lib.einsum("Xpq, X->pq", self.q_lambda_ao, gvar2)
        else:
            s_eval, s_evec = scipy.linalg.eigh(s1e)
            idx = s_eval > 1e-15
            s_inv = numpy.dot(s_evec[:, idx] / s_eval[idx], s_evec[:, idx].conj().T)

            tmp_term = lib.einsum("Xpr, rs, Xsq-> Xpq", self.gmat, s_inv, self.gmat)
            oei += 0.5 * lib.einsum("X, Xpq-> pq", gvar2, tmp_term)
            del (tmp_term)

        # DSE + boson energy (beyond |0> state)
        z2s = (0.5 * numpy.sum(self.z_alpha**2 * gvar2) + self.e_boson)* s1e/self._mol.nelectron
        # z2s = (0.5 * numpy.sum(self.z_alpha**2 * gvar2))* s1e/self._mol.nelectron
        oei += z2s

        # bilinear term
        # off-diaognal (photonic) block
        idx = 0
        for imode in range(self.nmodes):
            shift = numpy.eye(self.gmat.shape[1]) * self.z_alpha[imode] #/ mol.nelectron
            shift = s1e * self.z_alpha[imode]
            gtmp = (self.gmat[imode] - shift) * self.couplings_res[imode]
            gtmp *= numpy.sqrt(0.5*self.omega[imode])

            ci = self.boson_coeff[idx : idx + self.nboson_states[imode], 0]
            pdm = numpy.outer(numpy.conj(ci), ci)
            mdim = self.nboson_states[imode]
            h_od = numpy.diag(numpy.sqrt(numpy.arange(1, mdim)), k = 1) \
               + numpy.diag(numpy.sqrt(numpy.arange(1, mdim)), k = -1)
            ph_exp_val = numpy.sum(h_od * pdm)
            oei += gtmp * ph_exp_val

            idx += self.nboson_states[imode]
        return oei


    def get_dse_hcore(self, dm=None, s1e=None, residue=False):
        r"""Compute QED-RHF boson-mediated 1e- integrals.

        Deprecation warning:
        this function will be deprecated as it same as add_oei_ao and
        this term include bilinear as well. the function name is misleading

        Regardless of Fock basis or coherent-state (CS) representation,
        return quadrupole-modified OEI contribution, which arises
        from the DSE term of the PF Hamiltonian:

        .. math::
            \bm{h}_{\tt{DSE}} &= \bm{h}_{\tt{bare}} - \frac{1}{2}
                                 \sum_\al \sum_{\mu\nu} \rho_{\mu\nu}
                                 \cdot \bm{\tilde{q}}^\al_{\mu\nu} \\
            \bm{\tilde{q}}^\al_{\mu\nu} &= \la_{\al}^2
                                           \cdot \bm{Q}^\al_{\mu\nu}
        where :math:`\bm{Q}^\al` is the polarized quadrupole moment matrix
        in the AO basis.

        In CS representation, when :attr:`use_cs` ``= True``, additional
        DSE-mediated contribution to the OEI is included:

        .. math::
            \bm{h}_{\tt{CS}} &= \bm{h}_{\tt{DSE}} - \bm{\tilde{d}}^\al
                                \sum_\al \sum_{\mu\nu} \rho_{\mu\nu}
                                \cdot \mel*{\mu}{\hat{D}}{\nu} \\
            \bm{\tilde{d}}^\al_{\mu\nu} &= \la_\al
                                           \cdot \bm{D}^\al_{\mu\nu}
        where :math:`\bm{D}^\al` is the polarized dipole moment matrix in
        the AO basis.

        Also in CS representation, :math:`E_{\tt{QED-HF}}` includes DSE
        term:

        .. math::
            E_{\tt{QED-HF}} = E_{\tt{HF}}
                            + \frac{1}{2} \sum_\al
                              \langle
                              [ \la_\al \cdot
                                (\hat{D} - \ev*{\hat{D}}_{\mu\nu})
                              ]^2
                              \rangle
        that can also be included via the OEI, since:

        .. math::
            \langle
            [ \la_\al
            \cdot (\hat{D} - \ev*{\hat{D}}_{\mu\nu})]^2
            \rangle
            &= \langle [ \la_\al
               \cdot (\hat{D} - \ev*{\hat{D}}_{\mu\nu})]^2
               \rangle
               \cdot \frac{\Tr[\bm{S} \cdot \bm{\rho}]}
               {N_{\tt{elec}}} \\
            &= \tt{Tr}
               \left[
               \frac{[ \la_\al \cdot
               (\hat{D} - \ev*{\hat{D}}_{\mu\nu})]^2}
               {N_{\tt{elec}}} \bm{S} \cdot \bm{\rho}
               \right]
        where :math:`\bm{\rho}` is the provided electronic density matrix,
        ``dm``.

        Parameters
        ----------
        dm : :class:`~numpy.ndarray`
            Density matrix.
        s1e : :class:`~numpy.ndarray`
            Overlap matrix, **optional**
            (computed if not provided).
        residue : bool
            Multiply by pre-factor if ``True``.

              - used by VT-QED-HF solver
              - **optional,** ``default = False``

        Return
        ------
        dse_oei : :class:`~numpy.ndarray`
            DSE-mediated OEI for all photon modes.
        """

        warnings.warn(
            "The 'get_dse_hcore' function is deprecated, please use add_oei_ao" +
            "instead because since boson-mediated oei part includes both DSE and bilinear term",
            DeprecationWarning,
            stacklevel=2,
        )

        if s1e is None:
            s1e = self._mf.get_ovlp(self._mol)

        # Always shift OEI with photon ground state energy
        dse_oei = self.e_boson * s1e / self.nelectron

        # DSE contribution
        g_dse = numpy.ones(self.nmodes) if not residue else \
                (self.couplings_res ** 2)

        if self.complete_basis:
            dse_oei -= 0.5 * lib.einsum("X, Xpq-> pq", g_dse, self.q_lambda_ao)
        else:
            s_eval, s_evec = scipy.linalg.eigh(s1e)
            idx = s_eval > 1e-15
            s_inv = numpy.dot(s_evec[:, idx] / s_eval[idx], s_evec[:, idx].conj().T)

            tmp_term = lib.einsum("Xpr, rs, Xsq-> Xpq", self.gmat, s_inv, self.gmat)
            dse_oei += 0.5 * lib.einsum("X, Xpq-> pq", g_dse, tmp_term)
            del (tmp_term)

        # DSE contributions to OEI and total energy from coherent-state representation
        if self.use_cs == True:
            dse_oei += 0.5 * numpy.sum(self.z_alpha**2 * g_dse) * s1e / self.nelectron
            dse_oei -= lib.einsum("X, Xpq-> pq", g_dse * self.z_alpha, self.gmat)

        # E-P bilinear coupling contribution
        g_ep = numpy.ones(self.nmodes) if not residue else \
               self.couplings_res

        ep_term = numpy.zeros((self.nao, self.nao))
        for a in range(self.nmodes):

            shift = s1e * self.z_alpha[a]
            gtmp = (self.gmat[a] - shift) * numpy.sqrt(0.5 * self.omega[a])
            gtmp *= g_ep[a]

            ph_exp_val = self.get_bdag_minus_b_expval(a)
            ep_term += (gtmp * ph_exp_val)

        dse_oei -= ep_term
        del (g_ep, shift, gtmp, ph_exp_val, ep_term)

        return dse_oei


    def get_dse_jk(
        self, dm, residue=False):
        r"""
        Return DSE-mediated :math:`J` and :math:`K` matrices.

        Uses parts of :external:func:`hf.get_jk <pyscf.scf.hf.get_jk>`
        function from PySCF to generalize construction of DSE-mediated
        :math:`J/K` matrices for an input of multiple density matrices.

        .. warning::
            This term exists in both Fock/particle-number basis and
            within the coherent-state (CS) representation. Therefore,
            stored :attr:`z_alpha` values are not used, as these are
            set to an array of zeros when not in CS representation.

        .. math::
            J^\tt{DSE}_{\mu\nu} &= \sum_{\la\si} \rho_{\la\si}
                                   \sum_\al (\mu\nu|\la\si)^\al
                                 = \sum_{\la\si} \rho_{\la\si}
                                   \sum_\al g^\al_{\mu\nu} g^\al_{\la\si} \\
            K^\tt{DSE}_{\mu\nu} &= \sum_{\la\si} \rho_{\la\si}
                                   \sum_\al (\mu\si|\la\nu)^\al
                                 = \sum_{\la\si} \rho_{\la\si}
                                   \sum_\al g^\al_{\mu\si} g^\al_{\la\nu}

        Parameters
        ----------
        dm : :class:`~numpy.ndarray`
            Density matrix.
        residue : bool
            Multiply by pre-factor if ``True``.

              - used by VT-QED-HF solver
              - **optional,** ``default = False``

        Returns
        -------
        :class:`~numpy.ndarray`, :class:`~numpy.ndarray`
            DSE contributions to Coulomb :math:`(J)` and Exchange
            :math:`(K)` matrices of all photon modes, for each
            density matrix provided.
        """

        # Support input 'dm' being multiple density matrices
        dm = numpy.asarray(dm, order='C')
        dm_shape = dm.shape
        dm = dm.reshape(-1, self.nao, self.nao)
        n_dm = dm.shape[0]
        if n_dm > 1:
            log_msg = f"J/K matices constructed for {n_dm} density matrices."
            logger.debug(self, log_msg)

        # Scale by residual in VT-QED-HF
        g_dse = numpy.ones(self.nmodes) if not residue else \
                self.couplings_res * self.couplings_res

        j_dse = numpy.zeros((n_dm, self.nao, self.nao))
        k_dse = numpy.zeros((n_dm, self.nao, self.nao))

        # J/K matrices for all input density matrices
        for i in range(n_dm):

            # Coulomb (J)
            tmp_term = lib.einsum("Xpq, qp-> X", self.gmat, dm[i])
            j_dse[i] += lib.einsum("X, Xpq-> pq", g_dse * tmp_term, self.gmat)
            del (tmp_term)

            # Exchange (K)
            tmp_term = lib.einsum("Xpr, Xsq, sr-> Xpq", self.gmat, self.gmat, dm[i])
            k_dse[i] += lib.einsum("X, Xpq-> pq", g_dse, tmp_term)
            del (tmp_term)

        return j_dse.reshape(dm_shape), k_dse.reshape(dm_shape)


    def get_quadrupole_ao(self):
        r"""Return quadrupole moment matrix in AO basis."""
        return get_quadrupole_ao(self._mol,
                                 add_nuc_dipole=self.add_nuc_dipole,
                                 origin_shift=self.origin_shift)


    def get_polarized_quadrupole_ao(self, mode):
        r"""Return product of ``mode`` vector and quadrupole moment matrix.

        Returns product of the transversal polarization vector of
        photon ``mode`` and quadrupole moment matrix :attr:`quadrupole_ao`.

        .. math::
            \bm{Q}_\al = \bm{e}_\al
                         \cdot \mel*{\mu}{\hat{Q}}{\nu}
                         \cdot \bm{e}_\al

        Parameters
        ----------
        mode : int
            Index for ``mode`` stored in the object.

        Returns
        -------
        :class:`~numpy.ndarray`
            Quadrupole moment matrix polarized by photon mode.
        """

        outer_prod = numpy.outer(self.vec[mode], self.vec[mode]).reshape(-1)
        return lib.einsum("X, Xuv-> uv", outer_prod, self.quadrupole_ao)


    def get_q_lambda_ao(self):
        r"""Compute dipole self-energy matrix for all modes in AO basis.

        .. math::
            \bm{\tilde{q}} = \sum_\al \cdot
                             \mel*{\mu}{(\bm{\la}_\al\cdot\hat{D})^2}{\nu}
        Function only runs if :attr:`q_lambda_ao` is not ``None``.

        Returns
        -------
        :class:`~numpy.ndarray`
            Modified quadrupole moment matrix in AO basis.
        """

        q_lambda_ao = numpy.zeros((self.nmodes, self.nao, self.nao))

        for a in range(self.nmodes):

            q_lambda_ao[a] += self.get_polarized_quadrupole_ao(a)

            debug_msg = f"{self.boson_type} mode #{a + 1}: " + \
                        f"Norm of polarized quadrupole moment 'Q_ao' = " + \
                        f"{numpy.linalg.norm(q_lambda_ao[a])}"
            logger.debug(self, debug_msg)

        return lib.einsum("X, Xuv-> Xuv", (self.couplings ** 2), q_lambda_ao)


    def get_q_dot_lambda(self):
        r"""same as get_q_lambda_ao, one of them will be deprecated!!!"""
        # Tensor:  <u|r_i * r_y> * v_x * v_y
        if self.q_lambda_ao is None:
            nao = self._mol.nao_nr()
            self.q_lambda_ao = numpy.empty((self.nmodes, nao, nao))
            if self.quadrupole_ao is None:
                self.quadrupole_ao = self.get_quadrupole_ao()
            for mode in range(self.nmodes):
                x_out_y = numpy.outer(self.vec[mode], self.vec[mode]).reshape(-1)
                x_out_y *= self.couplings[mode] ** 2
                self.q_lambda_ao[mode] = numpy.einsum("J,Juv->uv", x_out_y, self.quadrupole_ao)

        logger.debug(self, f" Norm of Q_ao {numpy.linalg.norm(self.q_lambda_ao)}")


    def get_dipole_ao(self):
        r"""Return dipole moment matrix in AO basis."""
        return get_dipole_ao(self._mol,
                             add_nuc_dipole=self.add_nuc_dipole,
                             origin_shift=self.origin_shift)


    def get_polarized_dipole_ao(self, mode):
        """
        Gets the product between the photon transversal polarization
        and the dipole moments.
        """

        if self.dipole_ao is None:
            self.dipole_ao = self.get_dipole_ao()
        x_dot_mu_ao = numpy.einsum("x,xuv->uv", self.vec[mode], self.dipole_ao)
        return x_dot_mu_ao


    def get_gmat_ao(self):
        r"""Compute interaction matrix for all modes in AO basis.

        .. math::
            \bm{\tilde{d}} = \sum_\al \cdot
                             \mel*{\mu}{\bm{\la}_\al\cdot\hat{D}}{\nu}
        Function only runs if :attr:`gmat` is not ``None``.


        Returns
        -------
        :class:`~numpy.ndarray`
            Interaction matrix, stored in :attr:`gmat`.
        """

        gmat = lib.einsum("Xc, cuv-> Xuv", self.vec, self.dipole_ao)

        for a in range(self.nmodes):
            debug_msg = f"{self.boson_type} mode #{a + 1}: " + \
                        f"Norm of 'gmat_ao' w/o lambda: " + \
                        f"{numpy.linalg.norm(gmat[a])}"
            logger.debug(self, debug_msg)

        return lib.einsum("X, Xuv-> Xuv", self.couplings, gmat)


    def get_gmatao(self):
        r"""Compute interaction matrix for all modes in AO basis.

        Only difference from get_geb_ao is the sqrt(w/2) factor in get_geb_ao

        TODO: get_gmat_ao or this one will be deprecated

        .. math::
            \bm{\tilde{d}} = \sum_\al \cdot
                             \mel*{\mu}{\bm{\la}_\al\cdot\hat{D}}{\nu}
        Function only runs if :attr:`gmat` is not ``None``.


        Returns
        -------
        :class:`~numpy.ndarray`
            Interaction matrix, stored in :attr:`gmat`.
        """

        if self.gmat is None:
            nao = self._mol.nao_nr()
            gmat = numpy.empty((self.nmodes, nao, nao))
            for mode in range(self.nmodes):
                gmat[mode] = self.get_polarized_dipole_ao(mode) #* self.couplings[mode]
                logger.debug(self, f" Norm of gao without w {numpy.linalg.norm(gmat[mode])}")
                gmat[mode] *= self.couplings[mode]
                #gmat = numpy.einsum("Jx,J,xuv->Juv", self.vec, self.gfac, self.dipole_ao)
            self.gmat = gmat


    def get_geb_ao(self, mode):
        r"""Return bilinear interaction term of ``mode`` in :class:`Boson`.

        The bilinear interaction term of ``mode`` is :attr:`gmat[mode] <gmat>`,
        scaled by a frequency-dependent term:

        .. math::
            \sqrt{\frac{\om_\al}{2}} \bm{\tilde{d}}^\al
            = \sqrt{\frac{\om_\al}{2}}
              \cdot \mel*{\mu}{\bm{\la}_\al \cdot \hat{D}}{\nu}

        Parameters
        ----------
        mode : int
            Index for ``mode`` stored in the object.

        Returns
        -------
        :class:`~numpy.ndarray`
            Interaction matrix of mode scaled by frequency term.
        """
        return (numpy.sqrt(0.5 * self.omega[mode]) * self.gmat[mode])


    def get_bdag_minus_b_expval(self, mode):

        mdim = self.nboson_states[mode]

        h_od = numpy.diag(numpy.sqrt(numpy.arange(1, mdim)), k = 1) \
               + numpy.diag(numpy.sqrt(numpy.arange(1, mdim)), k = -1)
        pdm = self.get_boson_dm(mode)
        return numpy.sum(h_od * pdm)


    # -----------------------------------
    # Post-HF integrals (coupled-cluster)
    # -----------------------------------

    def get_omega(self):
        return self.omega


    def get_mos(self):
        r"""
        get MO coefficients.
        """

        mf = self._mf
        self.nmo = 2 * self.nao
        if mf.mo_coeff is None:
            mf.kernel()
        if mf.mo_occ.ndim == 1:
            ca = cb = mf.mo_coeff
            na = nb = int(mf.mo_occ.sum() // 2)
        else:
            ca, cb = mf.mo_coeff
            na, nb = int(mf.mo_occ[0].sum()), int(mf.mo_occ[1].sum())
        self.ca, self.cb = ca, cb
        self.na, self.nb = na, nb
        self.pa = numpy.einsum("ai,bi->ab", ca[:, :na], ca[:, :na])
        self.pb = numpy.einsum("ai,bi->ab", cb[:, :nb], cb[:, :nb])
        self.ptot = block_diag(self.pa, self.pb)


    def tmat(self):
        """
        Returns T-matrix in spin orbital (SO) basis.
        """
        t = self._mf.get_hcore()
        return block_diag(t, t)


    def fock(self):
        from pyscf import scf
        from pyscf.scf import ghf

        if self.pa is None or self.pb is None:
            raise Exception("Cannot build Fock without density ")

        h1 = self._mf.get_hcore()

        # add DSE-oei contribution
        if not self.shift:
            h1 += self.add_oei_ao(self.pa+self.pb)


        ptot = block_diag(self.pa, self.pb)
        h1 = block_diag(h1, h1)

        # this only works for bare HF
        #myhf = scf.GHF(self._mol)
        #fock = h1 + myhf.get_veff(self._mol, dm=ptot)

        # we use jk_buld from mf object instead
        jkbuild = self._mf.get_jk
        vj, vk = ghf.get_jk(self._mol, dm=ptot, hermi=1, jkbuild=jkbuild)
        #vj, vk = self._mf.get_jk(self._mol, dm=self.pa+self.pb, hermi=1) # in ao
        fock = h1 + vj - vk

        return fock


    def hf_energy(self):
        # this only works with bare HF
        F = self.fock()
        T = self.tmat()
        ptot = block_diag(self.pa, self.pb)

        Ehf = numpy.einsum("ij,ji->", ptot, F)
        Ehf += numpy.einsum("ij,ji->", ptot, T)
        # print(f"Electronic energy in hf_energy()= {Ehf}")
        if self.shift:
            return 0.5 * Ehf + self._mf.energy_nuc() + self.const
        else:
            return 0.5 * Ehf + self._mf.energy_nuc()


    def g_fock(self):
        if self.ca is None: self.get_mos()

        na, nb = self.na, self.nb
        va, vb = self.nmo // 2 - na, self.nmo // 2 - nb
        Co = block_diag(self.ca[:, :na], self.cb[:, :nb])
        Cv = block_diag(self.ca[:, na:], self.cb[:, nb:])

        F = self.fock()
        logger.debug(self, f" -YZ: F.shape = {F.shape}")
        logger.debug(self, f" -YZ: Norm of F with DSE oei {numpy.linalg.norm(F)}")

        if self.shift:
            Foo = numpy.einsum("pi,pq,qj->ij", Co, F, Co) - 2 * numpy.einsum(
                "I,pi,Ipq,qj->ij", self.xi, Co, self.gmatso, Co
            )
            Fov = numpy.einsum("pi,pq,qa->ia", Co, F, Cv) - 2 * numpy.einsum(
                "I,pi,Ipq,qa->ia", self.xi, Co, self.gmatso, Cv
            )
            Fvo = numpy.einsum("pa,pq,qi->ai", Cv, F, Co) - 2 * numpy.einsum(
                "I,pa,Ipq,qi->ai", self.xi, Cv, self.gmatso, Co
            )
            Fvv = numpy.einsum("pa,pq,qb->ab", Cv, F, Cv) - 2 * numpy.einsum(
                "I,pa,Ipq,qb->ab", self.xi, Cv, self.gmatso, Cv
            )
        else:
            Foo = numpy.einsum("pi,pq,qj->ij", Co, F, Co)
            Fov = numpy.einsum("pi,pq,qa->ia", Co, F, Cv)
            Fvo = numpy.einsum("pa,pq,qi->ai", Cv, F, Co)
            Fvv = numpy.einsum("pa,pq,qb->ab", Cv, F, Cv)
        return one_e_blocks(Foo, Fov, Fvo, Fvv)


    def get_I(self, full=False):
        from pyscf import ao2mo
        if self.ca is None: self.get_mos()

        na, nb = self.na, self.nb
        va, vb = self.nmo // 2 - na, self.nmo // 2 - nb
        nao = self.nmo // 2
        C = numpy.hstack((self.ca, self.cb))

        if False:
            # don't add DSE-mediated eri
            eri = ao2mo.general(self._mol, [C,]*4, compact=False).reshape([self.nmo,]*4)
        else:
            # add the DSE-mediated eri
            bare_eri =  self._mol.intor("int2e", aosym="s1")
            for mode in range(self.nmodes):
                bare_eri += numpy.einsum("pq,rs->pqrs", self.gmat[mode], self.gmat[mode])
            eri = ao2mo.general(bare_eri, [C,]*4, compact=False).reshape([self.nmo,]*4)

        eri[:nao, nao:] = eri[nao:, :nao] = eri[:, :, :nao, nao:] = eri[:, :, nao:, :nao] = 0

        Ua_mo = eri.transpose(0, 2, 1, 3) - eri.transpose(0, 2, 3, 1)
        logger.debug(self, f" -YZ: Norm of I with DSE eri {numpy.linalg.norm(Ua_mo)}")
        if full: return Ua_mo

        temp = [i for i in range(self.nmo)]
        oidx = temp[:na] + temp[self.nmo // 2 : self.nmo // 2 + nb]
        vidx = temp[na : self.nmo // 2] + temp[self.nmo // 2 + nb :]
        # print("\nnorm(I_mo)=", numpy.linalg.norm(Ua_mo))

        vvvv = Ua_mo[numpy.ix_(vidx, vidx, vidx, vidx)]
        vvvo = Ua_mo[numpy.ix_(vidx, vidx, vidx, oidx)]
        vovv = Ua_mo[numpy.ix_(vidx, oidx, vidx, vidx)]
        vvoo = Ua_mo[numpy.ix_(vidx, vidx, oidx, oidx)]
        oovv = Ua_mo[numpy.ix_(oidx, oidx, vidx, vidx)]
        vovo = Ua_mo[numpy.ix_(vidx, oidx, vidx, oidx)]
        vooo = Ua_mo[numpy.ix_(vidx, oidx, oidx, oidx)]
        ooov = Ua_mo[numpy.ix_(oidx, oidx, oidx, vidx)]
        oooo = Ua_mo[numpy.ix_(oidx, oidx, oidx, oidx)]

        #vvov = Ua_mo[numpy.ix_(vidx, vidx, oidx, vidx)]
        #ovvv = Ua_mo[numpy.ix_(oidx, vidx, vidx, vidx)]
        #voov = Ua_mo[numpy.ix_(vidx, oidx, oidx, vidx)]
        #ovvo = Ua_mo[numpy.ix_(oidx, vidx, vidx, oidx)]
        #ovov = Ua_mo[numpy.ix_(oidx, vidx, oidx, vidx)]
        #oovo = Ua_mo[numpy.ix_(oidx, oidx, vidx, oidx)]
        #ovoo = Ua_mo[numpy.ix_(oidx, vidx, oidx, oidx)]

        return two_e_blocks(vvvv=vvvv,
               vvvo=vvvo, vovv=vovv,
               vvoo=vvoo, oovv=oovv,
               vovo=vovo, vooo=vooo,
               ooov=ooov, oooo=oooo)


    g_aint = get_I


    def mfG(self):
        if self.pa is None: self.get_mos()
        ptot = block_diag(self.pa, self.pb)
        g = self.gmatso
        if self.shift:
            mfG = numpy.zeros(self.nmodes)
        else:
            mfG = numpy.einsum("Ipq,qp->I", g, ptot)

        return (mfG, mfG)


    def gint(self):
        if self.ca is None: self.get_mos()
        g = self.gmatso.copy()
        na = self.na
        nb = self.nb
        Co = block_diag(self.ca[:, :na], self.cb[:, :nb])
        Cv = block_diag(self.ca[:, na:], self.cb[:, nb:])

        oo = numpy.einsum("Ipq,pi,qj->Iij", g, Co, Co)
        ov = numpy.einsum("Ipq,pi,qa->Iia", g, Co, Cv)
        vo = numpy.einsum("Ipq,pa,qi->Iai", g, Cv, Co)
        vv = numpy.einsum("Ipq,pa,qb->Iab", g, Cv, Cv)
        g = one_e_blocks(oo, ov, vo, vv)

        return (g, g)


    def get_gmat_so(self):
        r"""
        e-photon coupling matrix
        """

        if self.dipole_ao is None:
            self.get_dipole_ao()
        if self.quadrupole_ao is None:
            self.get_quadrupole_ao()
        if self.pa is None: self.get_mos()

        self.get_gmat_ao()

        # gmatso
        #gmatso = [
        #    block_diag(self.gmat[i], self.gmat[i]) for i in range(len(self.gmat))
        #]
        # add factor of sqrt(w/2) into the coupling
        gmatso = [
           block_diag(
               self.gmat[i] * numpy.sqrt(self.omega[i] / 2),
               self.gmat[i] * numpy.sqrt(self.omega[i] / 2),
           )
           for i in range(len(self.gmat))
        ]
        self.gmatso = numpy.asarray(gmatso)

        if self.shift:
            self.xi = numpy.einsum("Iab,ab->I", self.gmatso, self.ptot) / self.omega
            self.const = -numpy.einsum("I,I->", self.omega, self.xi**2)


    kernel = get_gmat_so


# class phonon which will compute the phonon modes and e-ph coupling strength
class Phonon(Boson):
    r"""Phonon subclass."""
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)


    def relax(self):
        raise NotImplementedError("Method not implemented!")

if __name__ == "__main__":
    import numpy
    from pyscf import gto
    from openms.mqed import qedhf

    mol = gto.M()
    mol.atom = """H 0 0 0; F 0 0 1.75202"""
    mol.unit = "Angstrom"
    mol.basis = "sto3g"
    mol.verbose = 3
    mol.build()

    nmodes = 1
    omega = numpy.zeros(nmodes)
    omega[0] = 0.5
    vec = numpy.zeros((nmodes, 3))
    vec[0, :] = 0.05 * numpy.asarray([0.0, 0.0, 1.0])

    qed = Photon(mol, omega=omega, vec=vec)
    hf = qedhf.RHF(mol, qed)
    hf.max_cycle = 200
    hf.conv_tol = 1.0e-8
    hf.diis_space = 10
    mf = hf.run()

    for i in range(nmodes):
        g_eb = hf.qed.get_geb_ao(i)
        print("e-ph coupling matrix of mode ", i, "is \n", g_eb)

    # get I
    I = hf.qed.get_I()
    F = hf.qed.g_fock()
