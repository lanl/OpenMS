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

r"""

.. Gutzwiller Approximation (GA) based scf method
.. ==============================================

Renormalization factors:

There are two steps:

  1) build the effective GA Hamiltonian, and solve it using HF-like equations
  2) optimize m_{i\gamma}

Ref:
     https://online.kitp.ucsb.edu/online/excitcm09/ho/pdf/Ho_ExcitationsCM_KITP.pdf
     https://www.tandfonline.com/doi/full/10.1080/00268976.2020.1734243


The onsite Gutzwiller operator is expressed in the local (fermionic) Fock
states :math:`\ket{\Gamma_I}`,

..  math::
    \mathcal{P}_I = \sum_{\Gamma_I\Gamma'_I} \Phi_{\Gamma_I\Gamma'_I} \sqrt{P^0_{\Gamma_I}}
    \ket{\Gamma_I}\bra{\Gamma'_I}.

where the :math:`P^0_{\Gamma_I}` is the occupation propabaility of the uncorrelated
state:

.. math::
    P^0_{\Gamma_I} \equiv |\bra{\Psi_0}\Gamma_I\rangle|^2.

And the expansion coefficients :math:`\Phi_{\Gamma_I\Gamma'_I}` are the variational
parameters, which are grouped into a Hermitian matrix :math:`\Phi_I`.

For single band, :math:`\Phi_I` is a diagonal matrix,

.. math::
    \Phi_I =
    \begin{bmatrix}
    \phi_{I0} & 0 & 0 & 0 \\
    0 & \phi_{I\uparrow} & 0 & 0 \\
    0 & 0 & \phi_{I\downarrow} & 0 \\
    0 & 0 & 0 & \phi_{I2} \\
    \end{bmatrix}

which are computed from the embedding Hamilonian.

Quasiparticle Hamiltonian
-------------------------

.. math::
    H^{qp} = \sum_{IJ}\sum_{\alpha\beta} T^{IJ}_{\alpha\beta}
             \mathcal{R}_{I\alpha}\mathcal{R}_{J\beta} c^\dagger_{I\alpha} c_{J\beta}
             + \sum_I \mu_i \hat{n}_{I\alpha}.

where the renormalization factors are

.. math::
    \mathcal{R}_{I\sigma} = \frac{\text{Tr}\left[\Phi^\dagger_I M^\dagger_{I\sigma} \Phi_i M_{I\sigma}\right]}
    {\sqrt{n_{I\sigma}(1-n_{I\sigma})}}

Where :math:`M_I` is the matrix representation of the electronic annihilation operator
:math:`c_{I\sigma}` in the local basis.


Embedding Hamiltonian
---------------------

The embedding Hamiltonian is:

.. math::
    H^{eb}_{I} = \frac{\Delta_{I}M_I + \Delta^*_{I}M^\dagger_I}{\sqrt{n^0_{I}(1-n^0_{I})}} +
               \sum_\sigma \mu_{i\sigma} N_{\sigma} + U\mathcal{D}.

where :math:`\Delta_I` is

.. math::
  \Delta_{I\sigma} = \sum_J t_{IJ} \mathcal{R}_{J\sigma} \rho_{IJ,\sigma}

SCF steps:
----------

SCF loop afer initializing density matrix, variational parameters, and lambda:

  #. update Renormalization factors :math:`\mathcal{R}_I`
  #. construct quasiparticle Hamiltonian
  #. diagonalize qp Hamiltonian and get RDM
  #. update :math:`\Delta_{I\sigma}`
  #. update EB Hamiltonian and diagonalize it to update variational parameters, and lambda
  #. check convergence.

Program references
------------------
"""

import sys
import tempfile
import scipy
import openms

import numpy as backend
import numpy

# from mqed.lib      import logger
from pyscf.scf import hf
from pyscf.scf import chkfile
from pyscf.lib import logger
from pyscf import lib
from pyscf import __config__


WITH_META_LOWDIN = getattr(__config__, "scf_analyze_with_meta_lowdin", True)
PRE_ORTH_METHOD = getattr(__config__, "scf_analyze_pre_orth_method", "ANO")
MO_BASE = getattr(__config__, "MO_BASE", 1)
TIGHT_GRAD_CONV_TOL = getattr(__config__, "scf_hf_kernel_tight_grad_conv_tol", True)
MUTE_CHKFILE = getattr(__config__, "scf_hf_SCF_mute_chkfile", False)

global EPS
EPS = 1.e-7


def eig(h, s):
    '''Solver for generalized eigenvalue problem

    .. math:: HC = SCE
    '''
    e, c = scipy.linalg.eigh(h, s)
    idx = numpy.argmax(abs(c.real), axis=0)
    c[:,c[idx,numpy.arange(len(e))].real<0] *= -1
    return e, c

def get_occ(mf, mo_energy=None, mo_coeff=None):
    '''Label the occupancies for each orbital

    Kwargs:
        mo_energy : 1D ndarray
            Obital energies

        mo_coeff : 2D ndarray
            Obital coefficients

    '''
    if mo_energy is None: mo_energy = mf.mo_energy
    e_idx = numpy.argsort(mo_energy)
    e_sort = mo_energy[e_idx]
    nmo = mo_energy.size
    mo_occ = numpy.zeros_like(mo_energy)
    nocc = mf.mol.nelectron // 2
    mo_occ[e_idx[:nocc]] = 2
    if mf.verbose >= logger.INFO and nocc < nmo:
        if e_sort[nocc-1]+1e-3 > e_sort[nocc]:
            logger.warn(mf, 'HOMO %.15g == LUMO %.15g',
                        e_sort[nocc-1], e_sort[nocc])
        else:
            logger.info(mf, '  HOMO = %.15g  LUMO = %.15g',
                        e_sort[nocc-1], e_sort[nocc])

    if mf.verbose >= logger.DEBUG:
        numpy.set_printoptions(threshold=nmo)
        logger.debug(mf, '  mo_energy =\n%s', mo_energy)
        numpy.set_printoptions(threshold=1000)
    return mo_occ



def kernel(
    ga,
    conv_tol=1e-10,
    conv_tol_grad=None,
    dump_chk=True,
    dm0=None,
    callback=None,
    conv_check=True,
    **kwargs,
):
    r"""kernel: the GA-SCF driver.

    Parameters
    ----------
    ga : an instance of SCF class. ga object holds all parameters to control GA-SCF.
         One can modify its member functions to change the behavior of GA-SCF.
         The member functions which are called in kernel are

    conv_tol (float, optional): Convergence tolerance. Defaults to 1e-10.
    conv_tol_grad (float, optional): Convergence tolerance for gradients. Defaults to None.
    dm0 (numpy.ndarray, optional): Initial guess density matrix. Defaults to None.

    kwargs: (TODO)

    Returns: (TODO)

    """

    logger.info(ga, f"\n{'*' * 70}\n {' ' * 20}Gutzwiller calculation \n{'*' * 70}")

    if "init_dm" in kwargs:
        raise RuntimeError(
            '''You see this error message because of the API updates in pyscf v0.11.''' +
            '''Keyword argument "init_dm" is replaced by "dm0"'''
        )
    cput0 = (logger.process_clock(), logger.perf_counter())
    if conv_tol_grad is None:
        conv_tol_grad = numpy.sqrt(conv_tol)
        logger.info(ga, "Set gradient conv threshold to %g", conv_tol_grad)

    mol = ga.mol
    s1e = ga.get_ovlp(mol)

    # initialize r and lambda
    R = kwargs.get("R", None)
    lamda = kwargs.get("lamda", None)
    ga.init_gwf_r_lambda(R, lamda)

    # initial fock matrix
    init_fock = ga.update_qp_ham()
    print('initial fock matrix is', init_fock)

    # initialize dm0
    if dm0 is None:
        #dm = ga.get_init_guess(mol, ga.init_guess, s1e=s1e, **kwargs)
        mo_energy, mo_coeff = eig(init_fock, s1e)
        mo_occ = ga.get_occ(mo_energy, mo_coeff)
        dm = ga.make_rdm1(mo_coeff, mo_occ)
    else:
        dm = dm0

    print("\n dm is \n", numpy.trace(dm))

    # initialize variational parameters
    ga.init_var_params(dm)

    print(" ga variaitonal params are \n", ga.f)

    # TODO: use DIIS or Newton solver for the SCF.

    # scf params
    scf_conv = False
    e_tot = 0.0
    ga.cycles = 0
    fold = ga.f
    for cycle in range(ga.max_cycle):
        # update embedding Hamiltonian and xx
        dm_last = dm
        last_e = e_tot

        ga.update_renormalizations(dm)

        # get fock matrix (qp) matrix
        fock = ga.update_qp_ham()

        # Diagonalize Fock matrix and update density matrix
        mo_energy, mo_coeff = eig(fock, s1e)
        mo_occ = ga.get_occ(mo_energy, mo_coeff)
        dm = ga.make_rdm1(mo_coeff, mo_occ)

        # update deltas
        ga.get_deltas(dm)

        # update var_params (f, or phi) and lambda
        ga.update_var_params(dm)
        err = ga.update_lambda(dm)
        delta_f = fold - ga.f

        norm_ddm = numpy.linalg.norm(dm-dm_last)
        gnorm = numpy.linalg.norm(delta_f)

        if gnorm < conv_tol or err < conv_tol_grad:
            scf_conv = True

        logger.info(ga, '\ncycle= %d E= %.15g  delta_E= %4.3g  |g|= %4.3g  |g_var|= %4.3g  |ddm|= %4.3g',
                    cycle+1, e_tot, e_tot-last_e, err, gnorm, norm_ddm)
        # logger.debug(ga, "cycle= %d times: h1e_do = %.6g eta_grad = %.6g hcore = %.6g veff = %.6g  fock = %.6g scf = %.6g",
        fold = ga.f

    ga.cycles = cycle + 1

    logger.timer(ga, "scf_cycle", *cput0)
    # A post-processing hook before return
    ga.post_kernel(locals())
    return scf_conv, e_tot #, mo_energy, mo_coeff, mo_occ


def gwf_rl_step():
    r"""
    one step of GWF update for updating R and lambda (la)
    """

    # update effective hamiltonian
    # hm_expand_all_general(r, r_coeff)

    # hm_expand_all_herm(la, la_coeff)

    # modify_r_lambda_frozen()

    # map_wh_bnd_matrix(r, r)

    # map_wh_bnd_matrix(la, la)

    # calc_band_all()

    # gutz_fermi()

    # bnd_modify_frozen()

    # calc_mup_dn()

    # calc_nks()

    # calc_nks_pp()

    # eval_sl_vec_all(1)

    # map_wh_bnd_matrix(nks, nks)

    # calc_da0()

    # calc_isimix()

    # calc_da()

    # calc_vdc_list()

    # calc_lambdac()

    # solve_hembed_list

    # calc_r01_pp

    # calc_ncvar_pp

    # calc_ncphy_pp

    # eval_sl_vec_all(2)

    # calc_total_energy

    return None


def gwf_fvec():
    r"""
    gradients of GWF (fvec) used to update the variational parameters.
    """

    pass


class GASCF(lib.StreamObject): # (hf.RHF):
    # class HF(lib.StreamObject):
    r"""
    QEDSCF base class.   non-relativistic RHF.

    The GWF method assume the following structure in the Hamiltonain

    .. math::

       \hat{H} = \sum_{IJ}\sum_{pq} h^{IJ}_{pq}  +
                 \sum_I \sum_{pqrs} U^I_{pqrs} c^\dagger_p c^\dagger_q c_r c_s
               \equiv  \sum_{IJ} \hat{T}_{IJ} + \sum_{I} \hat{H}^{loc}_I

    i.e., we assume a local correlation (or the eri is block diagonal).

    Important references :cite:`Lanata:2012td`.

    """

    conv_tol = getattr(__config__, 'scf_hf_SCF_conv_tol', 1e-9)
    conv_tol_grad = getattr(__config__, 'scf_hf_SCF_conv_tol_grad', None)
    conv_check = getattr(__config__, 'scf_hf_SCF_conv_check', True)
    max_cycle = getattr(__config__, 'scf_hf_SCF_max_cycle', 50)
    init_guess = getattr(__config__, 'scf_hf_SCF_init_guess', 'minao')

    callback = None

    # only works for one-band model, i.e, 3 independent params for s band
    num_GAparam_per_site = 3
    Mpp = numpy.zeros((num_GAparam_per_site, num_GAparam_per_site))
    Npp = numpy.zeros((num_GAparam_per_site, num_GAparam_per_site))
    Upp = numpy.zeros((num_GAparam_per_site, num_GAparam_per_site))
    init_guess = getattr(__config__, 'scf_hf_SCF_init_guess', 'minao')

    def __init__(self, mol, xc=None, **kwargs):
        # hf.RHF.__init__(self, mol)
        # if xc is not None:
        #    rks.KohnShamDFT.__init__(self, xc)

        self.mol = mol
        self.verbose = mol.verbose
        self.max_memory = mol.max_memory
        self.stdout = mol.stdout

        if MUTE_CHKFILE:
            self.chkfile = None
        else:
            # the chkfile will be removed automatically, to save the chkfile, assign a
            # filename to self.chkfile
            self._chkfile = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
            self.chkfile = self._chkfile.name

        # print headers
        logger.info(self, openms.__logo__)
        # if openms._citations["pccp2023"] not in openms.runtime_refs:
        #     openms.runtime_refs.append(openms._citations["pccp2023"])


        self.e_tot = 0
        self.converged = False
        self.cycles = 0
        self.scf_summary = {}

        self.nao = nao = self.mol.nao_nr()

        # GA variational parameters
        self.f = backend.ones((3, nao))
        self.R = backend.ones(self.nao)
        self.lamda = backend.zeros(self.nao)


        logger.debug(self, f"GWF initialization")
        self.Mpp[0, 1] = self.Mpp[1, 0] = self.Mpp[1, 2] = self.Mpp[2, 1] = numpy.sqrt(2.0)
        self.Npp[1, 1] = 0.5
        self.Npp[2, 2] = 1.0
        self.Upp[2, 2] = 1.0

    # ------------------------------------------------------
    # new functions added in Oct 2024
    # ------------------------------------------------------

    def norbs_per_site(self):
        r"""return number of orbitals per site (atom)"""
        return 1

    def get_init_guess(self, mol, init_guess, s1e):

        pass

    def check_sanity(self):
        s1e = self.get_ovlp()
        cond = lib.cond(s1e)
        logger.debug(self, 'cond(S) = %s', cond)
        if numpy.max(cond)*1e-17 > self.conv_tol:
            logger.warn(self, 'Singularity detected in overlap matrix (condition number = %4.3g). '
                        'SCF may be inaccurate and hard to converge.', numpy.max(cond))
        return super().check_sanity()

    def build(self, mol=None):
        if mol is None: mol = self.mol
        if self.verbose >= logger.WARN:
            self.check_sanity()
        return self

    # one atom is a site
    # need to get atom-wise density matrix rho[i,j] from the density matrix

    def init_gwf_r_lambda(self, R=None, lamda=None):
        r"""TODO: check consistency in size"""
        if R is None:
            self.R = backend.ones(self.nao)
        else:
            self.R = R

        if lamda is None:
            self.lamda = backend.zeros(self.nao)
        else:
            self.lamda = lamda

    def init_var_params(self, dm):
        r"""initialization of gwf parameters"""
        # TODO: generalize it for multi-band

        n0 = backend.diagonal(dm)
        self.f[0] = 1.0 - n0
        self.f[1] = backend.sqrt(n0 * (1.0 - n0))
        self.f[2] = n0;

    def get_bare_hcore(self, mol=None, dm=None):
        r"""One-body integral of hopping term
        """
        if mol is None: mol = self.mol
        return hf.get_hcore(mol)

    def update_renormalizations(self, dm):
        r"""
        Update renormalization factor R

        .. math::

             R_{I,\alpha\beta} = \frac{\text{Tr}[\phi^\dagger_I M^\dagger_{I\alpha}
                              \phi_I M_{I\beta}]}{\sqrt{n^0_{I\beta}(1-n^0_{I\beta})}}

        where :math:`M^\dagger_{I\alpha}` is the matrix representation of the electron
        annihilation operator :math:`c_{I\alpha}`.

        Note that

        .. math::
               m_{I\Gamma} & = \bra{\Psi_G}\hat{m}_{I\Gamma}\ket{\Psi_G} \\
               \hat{m}_{I\Gamma} &= \ket{I,\Gamma}\bra{I,\Gamma}

        So that

        .. math::
             \phi_{I\Gamma} = \sqrt{\bra{\Psi_G} \ket{\Gamma_I}\bra{\Gamma_I}\ket{\Psi_0}}
                            = \sqrt{\bra{\Psi_0} \mathcal{P}^\dagger\ket{\Gamma_I}
                              \bra{\Gamma_I}\mathcal{P}\ket{\Psi_0}}
                            \equiv \sqrt{m_{I\Gamma}}.
        """
        n0 = backend.diagonal(dm)

        tmp = self.f[2].conj() * self.f[1] + self.f[1].conj() * self.f[0]
        self.R = tmp / numpy.sqrt(abs(n0 * (1.0 - n0)) + EPS)
        self.R[self.R> 1.0] = 1.0 # prevent the R larger than 1.0
        logger.debug(self, f"test: new renormalizaiton factors are {self.R}")


    def get_deltas(self, dm):
        r"""Compute the :math:`\Delta_{I\sigma}` (for embedding Hamiltonain)

        .. math::
            \Delta_{I\sigma} = \sum_J t_{IJ} \mathcal{R}_{J\sigma} \rho_{IJ,\sigma}

        """
        # go through the neighbors of site i to update the delta

        h1e = self.get_bare_hcore()
        self.Delta = numpy.sum(h1e * dm.T,  axis=1)  * self.R.conj()

    def compute_entropy_term(self, dm):
        pass


    def update_eb_ham(self, dm, i, f, R, Delta):
        r"""Update embedding Hamiltonian

        """
        # 1) get number of configurations per site
        n0 = rho[i, i]
        nf = f[1].conj() * f[1] + f[2].conj() * f[2]
        lmd = -4.0 * abs(R * Delta) * (nf - 0.5) / (abs(nf * (1.0 - nf)) + EPS)

        # 2) construct the embedding Hamiltonian
        Hpp = (Delta[i] / numpy.sqrt(abs(n0 * (1. - n0))) + EPS) * self.Mpp
        Hpp += self._eri[i, i, i, i] * self.Upp + (lmd - 2.0 * self.lamda[i]) * self.Npp


    def update_var_params(self, rho, mix=0.7, T_e=300.0):
        r"""
        get local embedding Hamiltonian H Emb and diagonalize
        it to update the variational parameters.

        Onsite U is used here to update the local embedding Hamiltonian
        and the f factors where are then used in update_lambda()

        The embedding Hamiltonian is

        .. math::
            H^{eb}_{I} = \frac{\Delta_{I}M_I + \Delta^*_{I}M^\dagger_I}{\sqrt{n^0_{I}(1-n^0_{I})}} +
                       \sum_\sigma \mu_{i\sigma} N_{\sigma} + U\mathcal{D}.

        where :math:`\mathcal{D}` is the matrix representaiton of local configurations.
        """

        newf = backend.ones((3, self.nao))
        ftmp = self.f.copy()
        # vectize the following code
        for i in range(self.nao):
            # construction of local embedding Hamiltonian and compute the eigenstates
            # TODO: move the contruction of local Hamiltonian into update_eb_ham

            #Hpp = self.update_eb_ham(dm, i, self.f[i], self.R[i], self.Delta[i])

            n0 = rho[i, i]
            nf = self.f[1, i].conj() * self.f[1, i] + self.f[2,i].conj() * self.f[2,i]
            lmd = -4.0 * abs(self.R[i] * self.Delta[i]) * (nf - 0.5) / (abs(nf * (1.0 - nf)) + EPS)
            Hpp = (self.Delta[i] / numpy.sqrt(abs(n0 * (1. - n0))) + EPS) * self.Mpp
            Hpp += self._eri[i, i, i, i] * self.Upp + (lmd - 2.0 * self.lamda[i]) * self.Npp

            # diagonalzie Hpp
            e, c = scipy.linalg.eigh(Hpp)
            # FIXME: sort and get smallest energy
            cmin = c[0]

            newf[:,i] = cmin
            newf[1,i] /= numpy.sqrt(2.0)
            print('newf is', newf[:,i])

        # update f params
        mixed_f = mix * (ftmp.conj() * ftmp) + (1.0 - mix) * (newf.conj() * newf)
        self.f = numpy.sqrt(mixed_f)


    def update_lambda(self, rho, R=None):
        r"""update the lambda parameters"""
        # lambda (of each site)
        if R is None: R = self.R
        n0 = numpy.diagonal(rho) #

        npp = self.f[1,:].conj() * self.f[1,:]  + self.f[2,:].conj() * self.f[2,:]
        self.lamda += (n0 - npp) * R

        delta_sum = sum(abs(n0 - npp))
        delta_sum /= self.nao # divided by number of sites
        return delta_sum


    def make_rdm1(self, mo_coeff, mo_occ):
        mocc = mo_coeff[:,mo_occ>0]
        dm = (mocc*mo_occ[mo_occ>0]).dot(mocc.conj().T)
        return lib.tag_array(dm, mo_coeff=mo_coeff, mo_occ=mo_occ)

    def get_fock(self, h1e=None, veff=None):
        return h1e + veff


    def update_qp_ham(self, R=None, lamda=None):
        r"""
        Update quasiparticle (QP) Hamiltonian with renormalizaiton
        factor (R, for hopping) and lamda (for onsite term)

        .. math::
            H^{qp} = \sum_{IJ}\sum_{\alpha\beta} T^{IJ}_{\alpha\beta}
                     \mathcal{R}_{I\alpha}\mathcal{R}_{J\beta} c^\dagger_{I\alpha} c_{J\beta}
                     + \sum_I \mu_i \hat{n}_{I\alpha}.
        """

        if R is None: R = self.R
        if lamda is None: lamda = self.lamda

        ham = numpy.zeros((self.nao, self.nao))
        numpy.fill_diagonal(ham, lamda)

        h1e = self.get_bare_hcore()
        ham += numpy.outer(R, R) * h1e
        return ham


    def gradients(self):
        r"""return the gradients for use in optimizaiton code like Newton method

        """

        # TODO: gradient of E w.r.t. f.  i.e., \partial E / \partial f
        # TODO: gradient of E w.r.t. DM. i.e., \partial E / \partial rho


        pass

    def Jacobian(self):
        r"""return Jacobina

        J = d
        """
        pass

    def scf(self, dm0=None, **kwargs):

        cput0 = (logger.process_clock(), logger.perf_counter())

        self.dump_flags()
        self.build(self.mol)

        if self.max_cycle > 0: # or self.mo_coeff is None:
            self.converged, self.e_tot = \
                kernel(self, self.conv_tol, self.conv_tol_grad,
                       dm0=dm0, callback=self.callback,
                       conv_check=self.conv_check, **kwargs)
        else:
            self.e_tot = kernel(self, self.conv_tol, self.conv_tol_grad,
                                dm0=dm0, callback=self.callback,
                                conv_check=self.conv_check, **kwargs)[1]
        logger.timer(self, 'SCF', *cput0)
        self._finalize()

        return self.e_tot

    kernel = lib.alias(scf, alias_name='kernel')

    def _finalize(self):
        '''Hook for dumping results and clearing up the object.'''
        if self.converged:
            logger.note(self, 'converged SCF energy = %.15g', self.e_tot)
            logger.note(self, 'converged R = [%s]', '  '.join(f'{z:.12g}' for z in self.R))
            # logger.note(self, 'converged Z = [%s] (TBA)', ', '.join(f'{f:.15g}' for f in self.qed.couplings_var))
        else:
            logger.note(self, 'SCF not converged.')
            logger.note(self, 'SCF energy = %.15g', self.e_tot)
        return self


    def dump_flags(self, verbose=None):
        r"""dump flags for the class"""
        # more flags dumping TBA.
        log = logger.new_logger(self, verbose)
        if log.verbose < logger.INFO:
            return self

        log.info('\n')
        log.info('******** %s ********', self.__class__)
        log.info('method = %s', self.__class__.__name__)

        log.info('SCF conv_tol = %g', self.conv_tol)
        log.info('SCF conv_tol_grad = %s', self.conv_tol_grad)
        log.info('SCF max_cycles = %d', self.max_cycle)
        if self.chkfile:
            log.info('chkfile to save SCF result = %s', self.chkfile)
        log.info('max_memory %d MB (current use %d MB)',
                 self.max_memory, lib.current_memory()[0])
        return self

    def post_kernel(self, envs):
        r"""Prints relevant citation information for calculation."""
        breakline = '='*80
        logger.note(self, f"\n{breakline}")
        logger.note(self, f"*  Hooray, the GA-SCF job is done!\n")
        logger.note(self, f"Citations:\n")
        for i, key in enumerate(openms.runtime_refs):
            logger.note(self, f"[{i+1}]. {openms._citations[key]}")
        logger.note(self, f"{breakline}\n")
        return self


    get_occ = get_occ

    # ------------------------------------------------------
    # old HF-like functions
    # ------------------------------------------------------

    def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        r"""
        GA effective Hamiltonain

        .. math::
            \hat{H}^{\text{eff}}_{ij} = H^{\text{core}}_{ij} + g_i * \sum_{kl} [P_{kl} * g_k * (2*J_{klij} - K_{klij})] * g_j
        """
        # Be carefule with the effects of :attr:`SCF.direct_scf` on this function
        if mol is None:
            mol = self.mol
        if dm is None:
            dm = self.make_rdm1()
        if self.direct_scf:
            ddm = backend.asarray(dm) - dm_last
            logger.debug(self, f"ddm.shape = {ddm.shape}")
            # ddm *= self.f[:, backend.newaxis]
            vj, vk = self.get_jk(mol, ddm, hermi=hermi)
            veff = vj - vk * 0.5

            # veff = backend.einsum("i,ij,j->ij", self.f, vj - vk * .5, self.f)
            return vhf_last + veff
        else:
            # dm = dm * self.f[:, backend.newaxis]
            vj, vk = self.get_jk(mol, dm, hermi=hermi)
            veff = (
                vj - vk * 0.5
            )  # veff = backend.einsum("i,ij,j->ij", self.f, vj - vk * .5, self.f)
            return veff

    def get_hcore(self, mol=None, dm=None):
        r"""
        GA kinetic term
        """
        if mol is None:
            mol = self.mol

        h = mol.intor_symmetric("int1e_kin")

        if mol._pseudo:
            # Although mol._pseudo for GTH PP is only available in Cell, GTH PP
            # may exist if mol is converted from cell object.
            from pyscf.gto import pp_int

            h += pp_int.get_gth_pp(mol)
        else:
            h += mol.intor_symmetric("int1e_nuc")

        if len(mol._ecpbas) > 0:
            h += mol.intor_symmetric("ECPscalar")

        # TODO: renormalization factor

        return h

    def update_params(self):
        pass

    def calc_co_lambda():
        r"""Compute the residues of lambda paramters
        TODO:
        """
        res = None
        # hs_l(:, :, i)
        # p f / p d_n

        return res


    # def energy_tot(self, dm=None, h1e=None, vhf=None):
    #    r"""
    #    GA total energy"
    #    """
    #    pass


def _hubbard_hamilts_pbc(L, U):
    h1e = backend.zeros((L, L))
    g2e = backend.zeros((L,) * 4)
    for i in range(L):
        h1e[i, (i + 1) % L] = h1e[(i + 1) % L, i] = -1
        g2e[i, i, i, i] = U
    return h1e, g2e


if __name__ == "__main__":
    import numpy
    from pyscf import gto, scf, ao2mo

    # one-dimensional Hubbard-U model
    L = 10
    U = 4

    mol = gto.M()
    mol.nelectron = L
    mol.nao = L
    mol.spin = 0
    mol.incore_anyway = True
    mol.build()

    # set hamiltonian
    h1e, eri = _hubbard_hamilts_pbc(L, U)
    mf = scf.UHF(mol)
    mf.get_hcore = lambda *args: h1e
    mf._eri = ao2mo.restore(1, eri, L)
    mf.get_ovlp = lambda *args: numpy.eye(L)

    print(mf._eri.shape)
    mf.kernel()
    dm = mf.make_rdm1()

    gamf = GASCF(mol)
    gamf.get_hcore = lambda *args: h1e
    gamf._eri = ao2mo.restore(1, eri, L)
    gamf.get_ovlp = lambda *args: numpy.eye(L)
    gamf.max_cycle = 500
    gamf.verbose = 4
    gamf.kernel()

    """
    atom = f"C   0.00000000   0.00000000  0.0;\
             O   0.00000000   1.23456800  0.0;\
             H   0.97075033  -0.54577032  0.0;\
             C  -1.21509881  -0.80991169  0.0;\
             H  -1.15288176  -1.89931439  0.0;\
             C  -2.43440063  -0.19144555  0.0;\
             H  -3.37262777  -0.75937214  0.0;\
             O  -2.62194056   1.12501165  0.0;\
             H  -1.71446384   1.51627790  0.0"

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
    #print("nuclear energy=     ", mf.energy_nuc())
    #dm = mf.make_rdm1()

    print("\n=========== GA-HF calculation  ======================\n")
    from openms.gwf import gahf

    mol.verbose = 5
    gamf = gahf.GASCF(mol)
    gamf.max_cycle = 500
    gamf.kernel()

    print("\nHF energy is=  ", mf.e_tot)
    print("\nGASCF energy is=", gamf.e_tot)
    """
