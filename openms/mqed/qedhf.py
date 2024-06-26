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

import sys
import copy
import numpy
from openms import __config__
from pyscf import lib
from pyscf.scf import hf
from pyscf.dft import rks
# from mqed.lib      import logger

from pyscf.lib import logger
#from pyscf.scf import diis
from pyscf.scf import _vhf
from pyscf.scf import chkfile
from pyscf import __config__


WITH_META_LOWDIN = getattr(__config__, 'scf_analyze_with_meta_lowdin', True)
PRE_ORTH_METHOD = getattr(__config__, 'scf_analyze_pre_orth_method', 'ANO')
MO_BASE = getattr(__config__, 'MO_BASE', 1)
TIGHT_GRAD_CONV_TOL = getattr(__config__, 'scf_hf_kernel_tight_grad_conv_tol', True)
MUTE_CHKFILE = getattr(__config__, 'scf_hf_SCF_mute_chkfile', False)


# For code compatibility in python-2 and python-3
if sys.version_info >= (3,):
    unicode = str


r"""
Theoretical background of QEDHF
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Within the Coherent State (CS) representation (for photonic DOF), the
QEDHF wavefunction ansatz is

.. math::

   \ket{\Psi} = & \prod_\alpha e^{z_\alpha b^\dagger_\alpha - z^*_\alpha b_\alpha } \ket{HF}\otimes{0_p} \\
              = & U(\mathbf{z}) \ket{HF}\otimes{0_p}.

where :math:`z_\alpha=-\frac{\lambda_\alpha\cdot\langle\boldsymbol{D}\rangle}{\sqrt{2\omega_\alpha}}` denotes
the photon displacement due to the coupling with electrons.

Consequently, we can use :math:`U(\mathbf{z})` to transform the original PF Hamiltonian
into CS representation

.. math::

    H_{CS} = & U^\dagger(\mathbf{z}) H U(\mathbf{z}) \\
           = & H_e+\sum_\alpha\Big\{\omega_\alpha b^\dagger_\alpha b_\alpha
               +\frac{1}{2}[\lambda_\alpha\cdot(\boldsymbol{D}-\langle\boldsymbol{D}\rangle)]^2  \\
             & -\sqrt{\frac{\omega_\alpha}{2}}[\lambda_\alpha\cdot(\boldsymbol{D} -
             \langle\boldsymbol{D}\rangle)](b^\dagger_\alpha+ b_\alpha)\Big\}.


With the ansatz, the QEDHF energy is

.. math::

  E_{QEDHF}= E_{HF} + \frac{1}{2}\langle \boldsymbol{lambda}\cdot [\boldsymbol{D}-\langle \boldsymbol{D}\rangle)]^2\rangle,

"""

def kernel(mf, conv_tol=1e-10, conv_tol_grad=None,
           dump_chk=True, dm0=None, callback=None, conv_check=True, **kwargs):
    r"""
    kernel: the QEDHF SCF driver.

    The only difference against bare hf kernel is that we include get_hcore within
    the scf cycle because qedhf has DSE-mediated hcore which depends on DM

    Args:
        mf : an instance of QEDSCF class
            mf object holds all parameters to control SCF.  One can modify its
            member functions to change the behavior of SCF.  The member
            functions which are called in kernel are

            | mf.get_init_guess
            | mf.get_hcore
            | mf.get_ovlp
            | mf.get_veff
            | mf.get_fock
            | mf.get_grad
            | mf.eig
            | mf.get_occ
            | mf.make_rdm1
            | mf.energy_tot
            | mf.dump_chk

    Kwargs:
        conv_tol : float
            converge threshold.
        conv_tol_grad : float
            gradients converge threshold.
        dump_chk : bool
            Whether to save SCF intermediate results in the checkpoint file
        dm0 : ndarray
            Initial guess density matrix.  If not given (the default), the kernel
            takes the density matrix generated by ``mf.get_init_guess``.
        callback : function(envs_dict) => None
            callback function takes one dict as the argument which is
            generated by the builtin function :func:`locals`, so that the
            callback function can access all local variables in the current
            environment.

    Returns:
        A list : scf_conv, e_tot, mo_energy, mo_coeff, mo_occ

        scf_conv : bool
            True means SCF converged
        e_tot : float
            Hartree-Fock energy of last iteration
        mo_energy : 1D float array
            Orbital energies.  Depending the eig function provided by mf
            object, the orbital energies may NOT be sorted.
        mo_coeff : 2D array
            Orbital coefficients.
        mo_occ : 1D array
            Orbital occupancies.  The occupancies may NOT be sorted from large
            to small.

    Examples:

    >>> from pyscf import gto, scf
    >>> from openms.mqed import qedhf
    >>> mol = gto.M(atom='Li 0 0 0; H 0 0 1.1', basis='cc-pvdz')
    >>> qedmf = qedhf.RHF(mol, cavity_freq=0.1, cavity_mode=numpy.asarray([0, 0, 0.1]))
    >>> qedmf.kernel()
    >>> conv, e, mo_e, mo, mo_occ = scf.hf.kernel(scf.hf.SCF(mol), dm0=numpy.eye(mol.nao_nr()))
    >>> print('conv = %s, E(QEDHF) = %.12f' % (self.conv, self.e_tot))
    conv = True, E(QEDHF) = -xxx
    """
    if 'init_dm' in kwargs:
        raise RuntimeError('''
You see this error message because of the API updates in pyscf v0.11.
Keyword argument "init_dm" is replaced by "dm0"''')
    cput0 = (logger.process_clock(), logger.perf_counter())
    if conv_tol_grad is None:
        conv_tol_grad = numpy.sqrt(conv_tol)
        logger.info(mf, 'Set gradient conv threshold to %g', conv_tol_grad)

    mol = mf.mol
    s1e = mf.get_ovlp(mol)
    if dm0 is None:
        dm = mf.get_init_guess(mol, mf.init_guess)
    else:
        dm = dm0

    if mf.qed.use_cs:
        mf.qed.update_cs(dm)

    mf.initialize_var_param(dm)

    # construct h1e, gmat in DO representation (used in SC/VT-QEDHF class)
    mf.get_h1e_DO(mol, dm=dm)

    h1e = mf.get_hcore(mol, dm)
    vhf = mf.get_veff(mol, dm)
    e_tot = mf.energy_tot(dm, h1e, vhf)
    logger.info(mf, 'init E= %.15g', e_tot)

    scf_conv = False
    mo_energy = mo_coeff = mo_occ = None

    s1e = mf.get_ovlp(mol)
    cond = lib.cond(s1e)
    logger.debug(mf, 'cond(S) = %s', cond)
    if numpy.max(cond)*1e-17 > conv_tol:
        logger.warn(mf, 'Singularity detected in overlap matrix (condition number = %4.3g). '
                    'SCF may be inaccurate and hard to converge.', numpy.max(cond))

    # Skip SCF iterations. Compute only the total energy of the initial density
    if mf.max_cycle <= 0:
        fock = mf.get_fock(h1e, s1e, vhf, dm)  # = h1e + vhf, no DIIS
        mo_energy, mo_coeff = mf.eig(fock, s1e)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        return scf_conv, e_tot, mo_energy, mo_coeff, mo_occ

    if isinstance(mf.diis, lib.diis.DIIS):
        mf_diis = mf.diis
    elif mf.diis:
        assert issubclass(mf.DIIS, lib.diis.DIIS)
        mf_diis = mf.DIIS(mf, mf.diis_file)
        mf_diis.space = mf.diis_space
        mf_diis.rollback = mf.diis_space_rollback

        # We get the used orthonormalized AO basis from any old eigendecomposition.
        # Since the ingredients for the Fock matrix has already been built, we can
        # just go ahead and use it to determine the orthonormal basis vectors.
        fock = mf.get_fock(h1e, s1e, vhf, dm)
        _, mf_diis.Corth = mf.eig(fock, s1e)
    else:
        mf_diis = None

    if dump_chk and mf.chkfile:
        # Explicit overwrite the mol object in chkfile
        # Note in pbc.scf, mf.mol == mf.cell, cell is saved under key "mol"
        chkfile.save_mol(mol, mf.chkfile)

    # A preprocessing hook before the SCF iteration
    mf.pre_kernel(locals())

    from openms import mqed

    cput1 = logger.timer(mf, 'initialize scf', *cput0)
    for cycle in range(mf.max_cycle):
        dm_last = dm
        last_hf_e = e_tot

        mf.get_h1e_DO(mol, dm=dm)
        mf.get_var_gradient(dm)

        #if isinstance(mf, mqed.qedhf.RHF):
        h1e = mf.get_hcore(mol, dm)

        fock = mf.get_fock(h1e, s1e, vhf, dm, cycle, mf_diis)
        mo_energy, mo_coeff = mf.eig(fock, s1e)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        dm = mf.make_rdm1(mo_coeff, mo_occ)
        vhf = mf.get_veff(mol, dm, dm_last, vhf)
        #if isinstance(mf, mqed.qedhf.RHF):
        h1e = mf.get_hcore(mol, dm)
        e_tot = mf.energy_tot(dm, h1e, vhf)

        # Here Fock matrix is h1e + vhf, without DIIS.  Calling get_fock
        # instead of the statement "fock = h1e + vhf" because Fock matrix may
        # be modified in some methods.
        fock = mf.get_fock(h1e, s1e, vhf, dm)  # = h1e + vhf, no DIIS
        norm_gorb = numpy.linalg.norm(mf.get_grad(mo_coeff, mo_occ, fock))
        if not TIGHT_GRAD_CONV_TOL:
            norm_gorb = norm_gorb / numpy.sqrt(norm_gorb.size)
        norm_eta = mf.get_var_norm()
        norm_gorb += norm_eta
        norm_ddm = numpy.linalg.norm(dm-dm_last)
        logger.info(mf, 'cycle= %d E= %.15g  delta_E= %4.3g  |g|= %4.3g  |ddm|= %4.3g',
                    cycle+1, e_tot, e_tot-last_hf_e, norm_gorb, norm_ddm)

        if callable(mf.check_convergence):
            scf_conv = mf.check_convergence(locals())
        elif abs(e_tot-last_hf_e) < conv_tol and norm_gorb < conv_tol_grad:
            scf_conv = True

        if dump_chk:
            mf.dump_chk(locals())

        if callable(callback):
            callback(locals())

        cput1 = logger.timer(mf, 'cycle= %d'%(cycle+1), *cput1)

        if scf_conv:
            break

    if scf_conv and conv_check:
        # An extra diagonalization, to remove level shift
        #fock = mf.get_fock(h1e, s1e, vhf, dm)  # = h1e + vhf
        mo_energy, mo_coeff = mf.eig(fock, s1e)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        dm, dm_last = mf.make_rdm1(mo_coeff, mo_occ), dm
        vhf = mf.get_veff(mol, dm, dm_last, vhf)
        #if isinstance(mf, mqed.qedhf.RHF):
        h1e = mf.get_hcore(mol, dm)
        e_tot, last_hf_e = mf.energy_tot(dm, h1e, vhf), e_tot

        fock = mf.get_fock(h1e, s1e, vhf, dm)
        norm_gorb = numpy.linalg.norm(mf.get_grad(mo_coeff, mo_occ, fock))
        if not TIGHT_GRAD_CONV_TOL:
            norm_gorb = norm_gorb / numpy.sqrt(norm_gorb.size)
        norm_ddm = numpy.linalg.norm(dm-dm_last)

        conv_tol = conv_tol * 10
        conv_tol_grad = conv_tol_grad * 3
        if callable(mf.check_convergence):
            scf_conv = mf.check_convergence(locals())
        elif abs(e_tot-last_hf_e) < conv_tol or norm_gorb < conv_tol_grad:
            scf_conv = True
        logger.info(mf, 'Extra cycle  E= %.15g  delta_E= %4.3g  |g|= %4.3g  |ddm|= %4.3g',
                    e_tot, e_tot-last_hf_e, norm_gorb, norm_ddm)
        if dump_chk:
            mf.dump_chk(locals())

    logger.timer(mf, 'scf_cycle', *cput0)
    # A post-processing hook before return
    mf.post_kernel(locals())
    return scf_conv, e_tot, mo_energy, mo_coeff, mo_occ

# difference from hf.get_fock:
# 1) generalize diis to include other variational  parameters;
# 2) Due to 1), we now use pre_update_prams and set_params pre/post processing
#    for DIIS update
# 3) h1e depends on dm
def get_fock(
    mf,
    h1e=None,
    s1e=None,
    vhf=None,
    dm=None,
    cycle=-1,
    diis=None,
    diis_start_cycle=None,
    level_shift_factor=None,
    damp_factor=None,
):
    """F = h^{core} + V^{HF}

    Special treatment (damping, DIIS, or level shift) will be applied to the
    Fock matrix if diis and cycle is specified (The two parameters are passed
    to get_fock function during the SCF iteration)

    Kwargs:
        h1e : 2D ndarray
            Core hamiltonian
        s1e : 2D ndarray
            Overlap matrix, for DIIS
        vhf : 2D ndarray
            HF potential matrix
        dm : 2D ndarray
            Density matrix, for DIIS
        cycle : int
            Then present SCF iteration step, for DIIS
        diis : an object of :attr:`SCF.DIIS` class
            DIIS object to hold intermediate Fock and error vectors
        diis_start_cycle : int
            The step to start DIIS.  Default is 0.
        level_shift_factor : float or int
            Level shift (in AU) for virtual space.  Default is 0.
    """
    # copied from hf get_fock, the only difference is that we update h1 in eacy iteration

    mf.initialize_var_param(dm)

    h1e = mf.get_hcore(dm=dm)
    if vhf is None:
        vhf = mf.get_veff(mf.mol, dm)

    f = h1e + vhf
    if cycle > -1:
        mf.update_variational_params()

    if cycle < 0 and diis is None:  # Not inside the SCF iteration
        return f

    if diis_start_cycle is None:
        diis_start_cycle = mf.diis_start_cycle
    if level_shift_factor is None:
        level_shift_factor = mf.level_shift
    if damp_factor is None:
        damp_factor = mf.damp
    if s1e is None:
        s1e = mf.get_ovlp()
    if dm is None:
        dm = mf.make_rdm1()

    if 0 <= cycle < diis_start_cycle - 1 and abs(damp_factor) > 1e-4:
        f = damping(s1e, dm * 0.5, f, damp_factor)
    if diis is not None and cycle >= diis_start_cycle:
        #params = diis.update(s1e, dm, f, mf, h1e, vhf)
        variables, gradients = mf.pre_update_params()
        params = diis.update(s1e, dm, f, mf, h1e, vhf, var=variables, var_grad=gradients)
        f = mf.set_params(params, fock_shape=f.shape)
    if abs(level_shift_factor) > 1e-4:
        f = level_shift(s1e, dm * 0.5, f, level_shift_factor)
    return f


TIGHT_GRAD_CONV_TOL = getattr(__config__, "scf_hf_kernel_tight_grad_conv_tol", True)

# in the future, replace it with our own object?
class TDMixin(lib.StreamObject):
    conv_tol = getattr(__config__, "tdscf_rhf_TDA_conv_tol", 1e-9)
    nstates = getattr(__config__, "tdscf_rhf_TDA_nstates", 3)
    singlet = getattr(__config__, "tdscf_rhf_TDA_singlet", True)
    lindep = getattr(__config__, "tdscf_rhf_TDA_lindep", 1e-12)
    level_shift = getattr(__config__, "tdscf_rhf_TDA_level_shift", 0)
    max_space = getattr(__config__, "tdscf_rhf_TDA_max_space", 50)
    max_cycle = getattr(__config__, "tdscf_rhf_TDA_max_cycle", 100)


from openms.lib.boson import Photon
import openms

class RHF(hf.RHF):
    # class HF(lib.StreamObject):
    r"""
    QEDSCF base class.   non-relativistic RHF.

    """

    def __init__(self, mol, xc=None, **kwargs):
        # print headers
        logger.info(self, openms.__logo__)
        if openms._citations["pccp2023"] not in openms.runtime_refs:
            openms.runtime_refs.append(openms._citations["pccp2023"])

        hf.RHF.__init__(self, mol)
        # if xc is not None:
        #    rks.KohnShamDFT.__init__(self, xc)

        cavity = None
        qed = None
        add_nuc_dipole = True # False

        if "cavity" in kwargs:
            cavity = kwargs["cavity"]
        if "add_nuc_dipole" in kwargs:
            add_nuc_dipole = kwargs["add_nuc_dipole"]
        if "qed" in kwargs:
            qed = kwargs["qed"]
        else:
            z_lambda = None
            if "z_lambda" in kwargs:
                z_lambda = kwargs["z_lambda"]
            if "cavity_mode" in kwargs:
                cavity_mode = kwargs["cavity_mode"]
            else:
                raise ValueError("The required keyword argument 'cavity_mode' is missing")

            if "cavity_freq" in kwargs:
                cavity_freq = kwargs["cavity_freq"]
            else:
                raise ValueError("The required keyword argument 'cavity_freq' is missing")
            logger.debug(self, f"cavity_freq = {cavity_freq}")
            logger.debug(self, f"cavity_mode = {cavity_mode}")

            nmode = len(cavity_freq)
            gfac = numpy.zeros(nmode)
            for i in range(nmode):
                gfac[i] = numpy.sqrt(numpy.dot(cavity_mode[i], cavity_mode[i]))
                if gfac[i] != 0:  # Prevent division by zero
                    cavity_mode[i] /= gfac[i]
            qed = Photon(mol, mf=self, omega=cavity_freq, vec=cavity_mode, gfac=gfac,
                         add_nuc_dipole=add_nuc_dipole,
                         z_lambda = z_lambda,
                         shift=False)

        # end of define qed object

        self.qed = qed

        # make dipole matrix in AO
        #self.make_dipolematrix() # replaced by qed functions
        self.qed.get_gmatao()
        self.qed.get_q_dot_lambda()

        self.gmat = self.qed.gmat
        self.qd2 = self.qed.q_dot_lambda

        self.bare_h1e = None # bare oei
        self.oei = None

    def get_h1e_DO(self, mol=None, dm=None):
        pass

    def get_hcore(self, mol=None, dm=None):
        #
        # DSE-mediated oei: -<\lambda\cdot D> g^\alpha_{uv} - 0.5 q^\alpha_{uv}
        #                = -Tr[\rho g^\alpha] g^\alpha_{uv} - 0.5 q^\alpha_{uv}
        #

        if mol is None: mol = self.mol
        #if self.bare_h1e is None: # cause problems in constructing QEDCCSD eris (return different h1e)
        #    self.bare_h1e = hf.get_hcore(mol)
        self.bare_h1e = hf.get_hcore(mol)

        if dm is not None:
            self.oei = self.qed.add_oei_ao(dm)
            return self.bare_h1e + self.oei
        return self.bare_h1e

    def get_g_dse_JK(self, dm, residue=False):
        r"""DSE-mediated JK terms
        Note: this term exists even if we don't use CS representation
        don't simply replace this term with z since z can be zero without CS.
        effective DSE-mediated eri is:

             I' = lib.einsum("Xpq, Xrs->pqrs", gmat, gmat)
        """

        dm_shape = dm.shape
        nao = dm_shape[-1]
        dm = dm.reshape(-1,nao,nao)
        n_dm = dm.shape[0]
        logger.debug(self, "No. of dm is %d", n_dm)

        vj_dse = numpy.zeros((n_dm,nao,nao))
        vk_dse = numpy.zeros((n_dm,nao,nao))
        for i in range(n_dm):
            # DSE-medaited J
            gtmp = self.gmat
            if residue: gtmp = self.gmat * self.qed.couplings_res
            scaled_mu = lib.einsum("pq, Xpq ->X", dm[i], gtmp)# <\lambada * D>
            vj_dse[i] += lib.einsum("Xpq, X->pq", gtmp, scaled_mu)

            # DSE-mediated K
            vk_dse[i] += lib.einsum("Xpr, Xqs, rs -> pq", gtmp, gtmp, dm[i])
            #gdm = lib.einsum("Xqs, rs -> Xqr", gtmp, dm)
            #vk += lib.einsum("Xpr, Xqr -> pq", gtmp, gdm)

        vj = vj_dse.reshape(dm_shape)
        vk = vk_dse.reshape(dm_shape)
        return vj, vk

    get_fock = get_fock

    def get_jk(self, mol=None, dm=None, hermi=1, with_j=True, with_k=True, omega=None):
        # Note the incore version, which initializes an _eri array in memory.
        if mol is None:
            mol = self.mol
        if dm is None:
            dm = self.make_rdm1()
        if not omega and (
            self._eri is not None or mol.incore_anyway or self._is_mem_enough()
        ):
            if self._eri is None:
                self._eri = mol.intor("int2e", aosym="s8")
            vj, vk = hf.dot_eri_dm(self._eri, dm, hermi, with_j, with_k)
        else:
            vj, vk = RHF.get_jk(self, mol, dm, hermi, with_j, with_k, omega)

        vj_dse, vk_dse = self.get_g_dse_JK(dm)
        vj += vj_dse
        vk += vk_dse

        return vj, vk

    # get_veff = get_veff
    def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        r"""QED Hartree-Fock potential matrix for the given density matrix

        .. math::

            V_{eff} = J - K/2 + \bra{i}\lambda\cdot\mu\ket{j}

        """
        if mol is None:
            mol = self.mol
        if dm is None:
            dm = self.make_rdm1()

        if self.qed.use_cs:
            self.qed.update_cs(dm)

        if self._eri is not None or not self.direct_scf:
            vj, vk = self.get_jk(mol, dm, hermi)
            vhf = vj - vk * 0.5
        else:
            ddm = numpy.asarray(dm) - numpy.asarray(dm_last)
            vj, vk = self.get_jk(mol, ddm, hermi)
            vhf = vj - vk * 0.5
            vhf += numpy.asarray(vhf_last)


        return vhf

    def dump_flags(self, verbose=None):
        return hf.RHF.dump_flags(self, verbose)

    def get_var_gradient(self, dm=None):
        pass

    def get_var_norm(self):
        r"""QEDHF does not have additional variables
        """
        return 0.0

    def initialize_var_param(self, dm = None):
        r"""
        initialize additional variational parameters
        """
        pass

    def update_variational_params(self):
        pass

    def pre_update_params(self):
        return None, None

    def set_params(self, params, fock_shape=None):
        fsize = numpy.prod(fock_shape)
        f = params[:fsize].reshape(fock_shape)
        return f

    def dse(self, dm, residue=False):
        r"""
        compute dipole self-energy due to CS basis.
        we can add this part into oei as well, then don't need to compute dse separtely:
        since z^2 = z^2 * Tr[S D] /Ne = Tr[(z^2/Ne*S)*D]
        i.e., we can add z^2/Ne*S into oei, where S is the overlap, Ne is total energy
        and Ne = Tr[SD].

        Feb 10: deprecated, we move this term into oei
        """
        # dip = self.dip_moment(dm=dm)
        # print("dipole_moment=", dip)
        e_dse = 0.0
        z_lambda = self.qed.z_lambda
        if residue: z_lambda *= self.qed.couplings_res
        e_dse += 0.5 * numpy.dot(z_lambda, z_lambda)
        #print("dipole self-energy=", e_dse)
        return e_dse

    def energy_elec(self, dm=None, h1e=None, vhf=None):
        r"""
        Our get_hcore depends on dm, so overwrite the default hf energy_elec
        """

        if dm is None: dm = self.make_rdm1()
        if h1e is None: h1e = self.get_hcore(self.mol, dm)
        if vhf is None: vhf = self.get_veff(self.mol, dm)
        e1 = numpy.einsum('ij,ji->', h1e, dm).real
        e_coul = numpy.einsum('ij,ji->', vhf, dm).real * .5
        self.scf_summary['e1'] = e1
        self.scf_summary['e2'] = e_coul
        logger.debug(self, 'E1 = %s  E_coul = %s', e1, e_coul)
        return e1+e_coul, e_coul

    def post_kernel(self, envs):
        r"""
        Use the post kernel to print citation informations
        """
        breakline = '='*80
        print(f"\n{breakline}")
        print(f"*  Hoollary, the job is done!\n")
        print(f"Citations:")
        for i, citation in enumerate(openms.runtime_refs):
            print(f"[{i+1}]. {citation}")
        print(f"{breakline}\n")

    def scf(self, dm0=None, **kwargs):

        cput0 = (logger.process_clock(), logger.perf_counter())

        self.dump_flags()
        self.build(self.mol)

        if self.max_cycle > 0 or self.mo_coeff is None:
            self.converged, self.e_tot, \
                    self.mo_energy, self.mo_coeff, self.mo_occ = \
                    kernel(self, self.conv_tol, self.conv_tol_grad,
                           dm0=dm0, callback=self.callback,
                           conv_check=self.conv_check, **kwargs)
        else:
            # Avoid to update SCF orbitals in the non-SCF initialization
            # (issue #495).  But run regular SCF for initial guess if SCF was
            # not initialized.
            self.e_tot = kernel(self, self.conv_tol, self.conv_tol_grad,
                                dm0=dm0, callback=self.callback,
                                conv_check=self.conv_check, **kwargs)[1]

        logger.timer(self, 'SCF', *cput0)
        self._finalize()
        return self.e_tot
    kernel = lib.alias(scf, alias_name='kernel')

class RKS(rks.KohnShamDFT, RHF):
    def __init__(self, mol, xc="LDA,VWN", **kwargs):
        RHF.__init__(self, mol, **kwargs)
        rks.KohnShamDFT.__init__(self, xc)

    def dump_flags(self, verbose=None):
        RHF.dump_flags(self, verbose)
        return rks.KohnShamDFT.dump_flags(self, verbose)

    get_veff = rks.get_veff
    get_vsap = rks.get_vsap
    energy_elec = rks.energy_elec


# depreciated standalone qedhf function
def qedrhf(model, options):
    # restricted qed hf
    # make a copy for qedhf
    mf = copy.copy(model.mf)
    conv_tol = 1.0e-10
    conv_tol_grad = None
    dump_chk = False
    callback = None
    conv_check = False
    noscf = False
    if "noscf" in options:
        noscf = options["noscf"]

    # converged bare HF coefficients
    na = int(mf.mo_occ.sum() // 2)
    ca = mf.mo_coeff
    dm = 2.0 * numpy.einsum("ai,bi->ab", ca[:, :na], ca[:, :na])
    mu_ao = model.dmat

    if conv_tol_grad is None:
        conv_tol_grad = numpy.sqrt(conv_tol)
    mol = mf.mol

    # initial guess
    h1e = mf.get_hcore(mol)
    vhf = mf.get_veff(mol, dm)

    e_tot = mf.energy_tot(dm, h1e, vhf)
    nuc = mf.energy_nuc()
    scf_conv = False
    mo_energy = mo_coeff = mo_occ = None
    s1e = mf.get_ovlp(mol)
    cond = lib.cond(s1e)

    # Skip SCF iterations. Compute only the total energy of the initial density
    if mf.max_cycle <= 0:
        fock = mf.get_fock(h1e, s1e, vhf, dm)  # = h1e + vhf, no DIIS
        mo_energy, mo_coeff = mf.eig(fock, s1e)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        return scf_conv, e_tot, mo_energy, mo_coeff, mo_occ

    if isinstance(mf.diis, lib.diis.DIIS):
        mf_diis = mf.diis
    elif mf.diis:
        assert issubclass(mf.DIIS, lib.diis.DIIS)
        mf_diis = mf.DIIS(mf, mf.diis_file)
        mf_diis.space = mf.diis_space
        mf_diis.rollback = mf.diis_space_rollback

        # We get the used orthonormalized AO basis from any old eigendecomposition.
        # Since the ingredients for the Fock matrix has already been built, we can
        # just go ahead and use it to determine the orthonormal basis vectors.
        fock = mf.get_fock(h1e, s1e, vhf, dm)
        _, mf_diis.Corth = mf.eig(fock, s1e)
    else:
        mf_diis = None

    if dump_chk and mf.chkfile:
        # Explicit overwrite the mol object in chkfile
        # Note in pbc.scf, mf.mol == mf.cell, cell is saved under key "mol"
        chkfile.save_mol(mol, mf.chkfile)

    # A preprocessing hook before the SCF iteration
    mf.pre_kernel(locals())

    print("converged Tr[D]=", numpy.trace(dm) / 2.0)
    nmode = model.vec.shape[0]

    for cycle in range(mf.max_cycle):
        dm_last = dm
        last_hf_e = e_tot

        # fock = mf.get_fock(h1e, s1e, vhf, dm)
        fock = h1e + vhf

        """
        fock = mf.get_fock(h1e, s1e, vhf, dm, cycle, mf_diis)
        """

        mu_mo = lib.einsum("pq, Xpq ->X", dm, mu_ao)

        scaled_mu = 0.0
        z_lambda = 0.0
        for imode in range(nmode):
            z_lambda -= numpy.dot(mu_mo, model.vec[imode])
            scaled_mu += numpy.einsum("pq, pq ->", dm, model.gmat[imode])

        dse = 0.5 * z_lambda * z_lambda

        # oei = numpy.zeros((h1e.shape[0], h1e.shape[1]))
        oei = model.gmat * z_lambda
        oei -= model.qd2
        oei = numpy.sum(oei, axis=0)
        fock += oei

        #  <>
        for imode in range(nmode):
            fock += scaled_mu * model.gmat[imode]
        fock -= 0.5 * numpy.einsum("Xpr, Xqs, rs -> pq", model.gmat, model.gmat, dm)

        # e_tot = mf.energy_tot(dm, h1e, fock - h1e + oei) + dse
        e_tot = 0.5 * numpy.einsum("pq,pq->", (oei + h1e + fock), dm) + nuc + dse

        mo_energy, mo_coeff = mf.eig(fock, s1e)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)

        # factor of 2 is applied (via mo_occ)
        dm = mf.make_rdm1(mo_coeff, mo_occ)
        # attach mo_coeff and mo_occ to dm to improve DFT get_veff efficiency
        dm = lib.tag_array(dm, mo_coeff=mo_coeff, mo_occ=mo_occ)
        vhf = mf.get_veff(mol, dm, dm_last, vhf)

        """
      # Here Fock matrix is h1e + vhf, without DIIS.  Calling get_fock
      # instead of the statement "fock = h1e + vhf" because Fock matrix may
      # be modified in some methods.

      fock = mf.get_fock(h1e, s1e, vhf, dm)  # = h1e + vhf, no DIIS
      oei = model.gmat * model.gmat
      oei -= model.qd2
      oei = numpy.sum(oei, axis=0)

      fock += oei

      mu_mo = lib.einsum('pq, Xpq ->X', 2 * dm, mu_ao)
      z_lambda = 0.0
      scaled_mu = 0.0
      for imode in range(nmode):
          z_lambda += -numpy.dot(mu_mo, model.vec[imode])
          scaled_mu += numpy.einsum('pq, pq ->', dm, model.gmat[imode])
      dse = 0.5 * z_lambda * z_lambda

      for imode in range(nmode):
          fock += 2 * scaled_mu * model.gmat[imode]
      fock -= numpy.einsum('Xpr, Xqs, rs -> pq', model.gmat, model.gmat, dm)
      """

        norm_gorb = numpy.linalg.norm(mf.get_grad(mo_coeff, mo_occ, fock))
        if not TIGHT_GRAD_CONV_TOL:
            norm_gorb = norm_gorb / numpy.sqrt(norm_gorb.size)
        norm_ddm = numpy.linalg.norm(dm - dm_last)
        print(
            "cycle= %3d E= %.12g  delta_E= %4.3g |g|= %4.3g |ddm|= %4.3g |d*u|= %4.3g dse= %4.3g"
            % (cycle + 1, e_tot, e_tot - last_hf_e, norm_gorb, norm_ddm, z_lambda, dse)
        )
        if noscf:
            break

        if callable(mf.check_convergence):
            scf_conv = mf.check_convergence(locals())
        elif abs(e_tot - last_hf_e) < conv_tol and norm_gorb < conv_tol_grad:
            scf_conv = True

        if dump_chk:
            mf.dump_chk(locals())

        if callable(callback):
            callback(locals())

        if scf_conv:
            break

    # to be updated
    if scf_conv and conv_check:
        # An extra diagonalization, to remove level shift
        # fock = mf.get_fock(h1e, s1e, vhf, dm)  # = h1e + vhf
        mo_energy, mo_coeff = mf.eig(fock, s1e)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        dm, dm_last = mf.make_rdm1(mo_coeff, mo_occ), dm
        dm = lib.tag_array(dm, mo_coeff=mo_coeff, mo_occ=mo_occ)
        vhf = mf.get_veff(mol, dm, dm_last, vhf)
        # e_tot, last_hf_e = mf.energy_tot(dm, h1e, vhf), e_tot
        e_tot, last_hf_e = mf.energy_tot(dm, h1e, fock - h1e + oei) + dse, e_tot

        fock = mf.get_fock(h1e, s1e, vhf, dm)
        norm_gorb = numpy.linalg.norm(mf.get_grad(mo_coeff, mo_occ, fock))
        if not TIGHT_GRAD_CONV_TOL:
            norm_gorb = norm_gorb / numpy.sqrt(norm_gorb.size)
        norm_ddm = numpy.linalg.norm(dm - dm_last)

        conv_tol = conv_tol * 10
        conv_tol_grad = conv_tol_grad * 3
        if callable(mf.check_convergence):
            scf_conv = mf.check_convergence(locals())
        elif abs(e_tot - last_hf_e) < conv_tol or norm_gorb < conv_tol_grad:
            scf_conv = True
        print(
            "Extra cycle  E= %.15g  delta_E= %4.3g  |g|= %4.3g  |ddm|= %4.3g",
            e_tot,
            e_tot - last_hf_e,
            norm_gorb,
            norm_ddm,
        )
        if dump_chk:
            mf.dump_chk(locals())

    # A post-processing hook before return
    mf.post_kernel(locals())
    print("HOMO-LUMO gap=", mo_energy[na] - mo_energy[na - 1])
    print("QEDHF energy=", e_tot)

    return scf_conv, e_tot, dse, mo_energy, mo_coeff, mo_occ


if __name__ == "__main__":
    # will add a loop
    import numpy
    from pyscf import gto, scf

    itest = 1
    zshift = itest * 2.0

    atom = f"C   0.00000000   0.00000000    {zshift};\
             O   0.00000000   1.23456800    {zshift};\
             H   0.97075033  -0.54577032    {zshift};\
             C  -1.21509881  -0.80991169    {zshift};\
             H  -1.15288176  -1.89931439    {zshift};\
             C  -2.43440063  -0.19144555    {zshift};\
             H  -3.37262777  -0.75937214    {zshift};\
             O  -2.62194056   1.12501165    {zshift};\
             H  -1.71446384   1.51627790    {zshift}"

    mol = gto.M(
        atom = atom,
        basis="sto3g",
        #basis="cc-pvdz",
        unit="Angstrom",
        symmetry=True,
        verbose=3,
    )
    print("mol coordinates=\n", mol.atom_coords())

    hf = scf.HF(mol)
    hf.max_cycle = 200
    hf.conv_tol = 1.0e-8
    hf.diis_space = 10
    hf.polariton = True
    mf = hf.run(verbose=4)

    print("electronic energies=", mf.energy_elec())
    print("nuclear energy=     ", mf.energy_nuc())
    dm = mf.make_rdm1()

    print("\n=========== QED-HF calculation  ======================\n")

    from openms.mqed import qedhf

    nmode = 2 # create a zero (second) mode to test the code works for multiple modes
    cavity_freq = numpy.zeros(nmode)
    cavity_mode = numpy.zeros((nmode, 3))
    cavity_freq[0] = 0.5
    cavity_mode[0, :] = 0.1 * numpy.asarray([0, 0, 1])

    qedmf = qedhf.RHF(mol, xc=None, cavity_mode=cavity_mode, cavity_freq=cavity_freq)
    qedmf.max_cycle = 500
    qedmf.kernel(dm0=dm)
    print(f"Total energy:     {qedmf.e_tot:.10f}")

    qed = qedmf.qed

    # get I
    qed.kernel()
    I = qed.get_I()
    F = qed.g_fock()
