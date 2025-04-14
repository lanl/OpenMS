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

Quasiparticle Hamiltonian
-------------------------

.. math::
    H^{qp} = \sum_{IJ}\sum_{\alpha\beta} T^{IJ}_{\alpha\beta}
             \mathcal{R}_{I\alpha}\mathcal{R}_{J\beta} c^\dagger_{I\alpha} c_{J\beta}
             + \sum_I \mu_i \hat{n}_{I\alpha}.


Embedding Hamiltonian
---------------------

The embedding Hamiltonian is:

.. math::
    H^{eb}_{I} =


Program reference
-----------------
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

    # initialize variational parameters
    ga.init_var_params(dm)

    #print(" ga variaitonal params are \n", ga.f)

    # Skip SCF iterations. Compute only the total energy of the initial density
    if ga.max_cycle <= 0:
        fock = ga.get_fock(h1e, s1e, vhf, dm)  # = h1e + vhf, no DIIS
        mo_energy, mo_coeff = ga.eig(fock, s1e)
        mo_occ = ga.get_occ(mo_energy, mo_coeff)
        return scf_conv, e_tot, mo_energy, mo_coeff, mo_occ

    if isinstance(ga.diis, lib.diis.DIIS):
        ga_diis = ga.diis
    elif ga.diis:
        assert issubclass(ga.DIIS, lib.diis.DIIS)
        ga_diis = ga.DIIS(ga, ga.diis_file)
        ga_diis.space = ga.diis_space
        ga_diis.rollback = ga.diis_space_rollback
        ga_diis.damp = ga.diis_damp

        # We get the used orthonormalized AO basis from any old eigendecomposition.
        # Since the ingredients for the Fock matrix has already been built, we can
        # just go ahead and use it to determine the orthonormal basis vectors.
        fock = ga.get_fock(h1e, s1e, vhf, dm)
        _, ga_diis.Corth = ga.eig(fock, s1e)
    else:
        ga_diis = None

    if dump_chk and ga.chkfile:
        # Explicit overwrite the mol object in chkfile
        # Note in pbc.scf, ga.mol == ga.cell, cell is saved under key "mol"
        chkfile.save_mol(mol, ga.chkfile)

    # A preprocessing hook before the SCF iteration
    ga.pre_kernel(locals())

    # scf params
    cput1 = logger.timer(ga, "initialize scf", *cput0)
    fock_last = None
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

        if callable(ga.check_convergence):
            scf_conv = ga.check_convergence(locals())
        elif abs(e_tot - last_hf_e) < conv_tol and norm_gorb < conv_tol_grad:
            scf_conv = True

        if dump_chk:
            ga.dump_chk(locals())

        if callable(callback):
            callback(locals())

        cput1 = logger.timer(ga, "cycle= %d" % (cycle + 1), *cput1)

        if scf_conv:
            break

    if scf_conv and conv_check:
        # An extra diagonalization, to remove level shift
        # fock = ga.get_fock(h1e, s1e, vhf, dm)  # = h1e + vhf
        mo_energy, mo_coeff = ga.eig(fock, s1e)
        mo_occ = ga.get_occ(mo_energy, mo_coeff)
        dm, dm_last = ga.make_rdm1(mo_coeff, mo_occ), dm
        vhf = ga.get_veff(mol, dm, dm_last, vhf)
        e_tot, last_hf_e = ga.energy_tot(dm, h1e, vhf), e_tot

        fock = ga.get_fock(h1e, s1e, vhf, dm)
        norm_gorb = numpy.linalg.norm(ga.get_grad(mo_coeff, mo_occ, fock))
        if not TIGHT_GRAD_CONV_TOL:
            norm_gorb = norm_gorb / numpy.sqrt(norm_gorb.size)
        norm_ddm = numpy.linalg.norm(dm - dm_last)

        conv_tol = conv_tol * 10
        conv_tol_grad = conv_tol_grad * 3
        if callable(ga.check_convergence):
            scf_conv = ga.check_convergence(locals())
        elif abs(e_tot - last_hf_e) < conv_tol or norm_gorb < conv_tol_grad:
            scf_conv = True
        logger.info(
            ga,
            "Extra cycle  E= %.15g  delta_E= %4.3g  |g|= %4.3g  |ddm|= %4.3g",
            e_tot,
            e_tot - last_hf_e,
            norm_gorb,
            norm_ddm,
        )
        if dump_chk:
            ga.dump_chk(locals())

    logger.timer(ga, "scf_cycle", *cput0)
    # A post-processing hook before return
    ga.post_kernel(locals())
    return scf_conv, e_tot, mo_energy, mo_coeff, mo_occ


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

from openms.gwf import ga_sband

class GASCF(ga_sband.GASCF):
    # class HF(lib.StreamObject):
    r"""
    Multiband GA method with local correlation.

    This is a generalized GA method for multiple bands (per sidte) and local correlation.

    Ref:

       - https://online.kitp.ucsb.edu/online/excitcm09/ho/pdf/Ho_ExcitationsCM_KITP.pdf
       - https://www.tandfonline.com/doi/full/10.1080/00268976.2020.1734243

    Assume a (lattice) system with :math:`\mathcal{N}` fragments, each with mulitple orbitals.
    The total Hamiltonian is

    .. math::
        H = & \sum^{\mathcal{N}}_{IJ} \hat{T}_{IJ} + \sum_I \hat{H}^{loc}_I \\
        \hat{T}_{IJ} = & \sum^{N_I N_J}_{pq} T_{Ip, Jq} c^\dagger_{Ip} c_{Jq} \\
        H^{loc}_I = & \sum_{pqrs} U^I_{pqrs}  c^\dagger_p c^\dagger_q c_r c_s.

    Within the GA formalism, the quasiparticle (QP) and embedding (EB)
    Hamiltonains are

    .. math::
        H^{QP} = \sum^{\mathcal{N}}_{IJ}\sum_{ab} [\mathcal{R}^\dagger_I T_{IJ} \mathcal{R}_{J}]_{ab}
                  f^\dagger_{Ia} f_{Jb} + \sum_{I}\sum_{ab} [\Lambda_I]_{ab} f^\dagger_{Ia} f_{Ib}

    and

    .. math::

        H^{EB} = \sum_{I} H^{loc}_I[c_{I\alpha}, c^\dagger_{I\alpha}] + \sum_{a}\sum_{\alpha}
                 \left([\mathcal{D}_I]_{a\alpha} c^\dagger_{I\alpha} b_{Ia} + h.c.\right)
                 + \sum_{ab} [\Lambda^c_I]_{ab} b_{Ib} b^\dagger_{Ia},

    respectively.

    Operators:

      - :math:`c_{I\alpha}, c^\dagger_{I\alpha}`: fermionic operator in the original basis set
      - :math:`f_{Ia}, f^\dagger_{Ia}`:
      - :math:`b_{Ia}, b^\dagger_{Ia}`:

    """

    def __init__(self, mol, xc=None, **kwargs):
        # print headers
        logger.info(self, openms.__logo__)
        # if openms._citations["pccp2023"] not in openms.runtime_refs:
        #     openms.runtime_refs.append(openms._citations["pccp2023"])

        super().__init__(self, mol)


        # GA variational parameters
        self.ga_params = backend.ones((3, nao))
        self.R = backend.ones(self.nao)
        self.lamda = backend.zeros(self.nao)


        logger.debug(self, f"GWF initialization")

    # ------------------------------------------------------
    # new functions added in Oct 2024
    # ------------------------------------------------------

    # one atom is a site
    # need to get atom-wise density matrix rho[i,j] from the density matrix

    def init_gwf_params(self, dm):
        r"""initialization of gwf parameters"""
        n0 = backend.diaognal(dm)
        self.ga_params[0] = 1.0 - n0
        self.ga_params[1] = backend.sqrt(n0 * (1.0 - n0))
        self.ga_parmas[2] = n0;


    def get_renormalization(self, dm):
        r"""
        Update renormalization factor r
        """
        n0 = backend.diaognal(dm)

        tmp = self.ga_params[2].conj() * self.ga_params[1] + self.f[1].conj() * self.ga_params[0]
        self.R = tmp / sqrt(abs(n0 * (1.0 - n0)) + EPS)
        self.R[self.R> 1.0] = 1.0 # prevent the R larger than 1.0

    def get_deltas(self, dm):
        r"""Compute the Detal (for embedding Hamiltonain)
        """
        # go through the neighbors of site i to update the delta

        Delta = backend.zeros(self.nao)
        for n in neighbors:
            # j is the neighbor
            Delta += h1e_atom[i,j] * self.R[i].conj() * dm[j,i]

    def compute_entropy_term(self, dm):
        pass

    def update_var_params(self, rho, mix=1.0, T_e=300.0):
        r"""
        get local embedding Hamiltonian H Emb and diagonalize
        it to update the variational parameters..
        """
        for i in range(self.nsite):
            Hpp = backend.zeros((nloc[i], nloc[i]))
            n0 = rho[i, i]
            nf = self.f[1, i].conj() * self.f[1, i] + self.f[2,i].conjg() * self.f[2,i]
            lmd = -4.0 * abs(self.R[i] * self.Delta[i]) * (nf - 0.5) / (abs(nf * (1.0 - nf)) + EPS)
            Hpp = (self.Delta[i] / numpy.sqrt(abs(n0 * (1. - n0))) + EPS) * self.Mpp
            Hpp += onsiteU[i] * self.Upp + (lmd - 2.0 * self.lamda) * self.Npp

            # diagonalzie Hpp
            e, c = scipy.linalg.eigh(Hpp)
            # FIXME: sort and get smallest energy
            #
            emin = c[0]

            ftmp = self.f.copy()
            # update f params (TODO)

    def update_lambda(self, rho, r):
        r"""update the lambda parameters"""
        # lambda (of each site)
        n0 = numpy.diagonal(rho) #
        npp = self.f[1,:].conj() * self.f[1,:]  + self.f[2,:].conjg() * self.f[2,:]
        self.lamda += (n0 - npp) * r

        delta_sum = sum(abs(n0 - npp))
        delta_sum /= nsites # divided by number of sites
        return delta_sum


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



    # ------------------------------------------------------

    def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        r"""
        GA effective Hamiltonain:

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
            # ddm *= self.ga_params[:, backend.newaxis]
            vj, vk = self.get_jk(mol, ddm, hermi=hermi)
            veff = vj - vk * 0.5

            # veff = backend.einsum("i,ij,j->ij", self.ga_params, vj - vk * .5, self.ga_params)
            return vhf_last + veff
        else:
            # dm = dm * self.ga_params[:, backend.newaxis]
            vj, vk = self.get_jk(mol, dm, hermi=hermi)
            veff = (
                vj - vk * 0.5
            )  # veff = backend.einsum("i,ij,j->ij", self.ga_params, vj - vk * .5, self.ga_params)
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

    kernel = kernel

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
