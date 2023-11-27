r"""

"""
# import warnings
from typing import Union, List
import numpy
from pyscf import gto
from pyscf import lib
from pyscf.lib import logger
from cqcpy import utils


def get_dipole_ao(mol, add_nuc_dipole=True):
    r"""
    dipole integral
    """
    # get dip matrix
    dipole_ao = None
    charges = mol.atom_charges()
    coords = mol.atom_coords()
    charge_center = (0, 0, 0)  # numpy.einsum('i,ix->x', charges, coords)
    if add_nuc_dipole:
        charge_center = (
            numpy.einsum("i,ix->x", charges, coords) / mol.tot_electrons()
        )  # charges.sum()
    with mol.with_common_orig(charge_center):
        dipole_ao = mol.intor_symmetric("int1e_r", comp=3)
    return dipole_ao


def get_quadrupole_ao(mol, add_nuc_dipole=True):
    r"""
    quadrupole integral
    | xx, xy, xz |
    | yx, yy, yz |
    | zx, zy, zz |
    xx <-> rrmat[0], xy <-> rrmat[3], xz <-> rrmat[6]
                     yy <-> rrmat[4], yz <-> rrmat[7]
                                      zz <-> rrmat[8]
    Q_uv = <u| (mu_tot dot e)^2 |v>
         = <u| (e^T dot Q dot e) |v> + 2 <w| mu dot e | v> (mu_nuc dot e) + (mu_nuc dot e)^2
    """

    quadrupole_ao = None
    rrmat = None
    # r2mat = None
    charges = mol.atom_charges()
    coords = mol.atom_coords()
    charge_center = (0, 0, 0)  # numpy.einsum('i,ix->x', charges, coords)
    if add_nuc_dipole:
        charge_center = (
            numpy.einsum("i,ix->x", charges, coords) / mol.tot_electrons()
        )  # charges.sum()
    with mol.with_common_orig(charge_center):
        rrmat = mol.intor("int1e_rr")
        quadrupole_ao = -rrmat
    return quadrupole_ao


class Boson(object):
    """
    define boson object
    """

    def __init__(
        self,
        mol,
        mf,
        omega=None,
        vec=None,
        gfac=None,
        n_boson_states: Union[int, List[int]] = 1,
        shift=False,
        add_nuc_dipole = True,
        **kwargs
    ):
        r"""
        :param object mol: molecule object
        :param object mf: mean-field electronic structure solver
        :param array omega: boson frequencies
        :param array vec: boson mode vector (normalized)
        :param float gfac: coupling constant :math:`\lambda_\alpha`
        :param bool shift: shift with DSE?
        """

        self._mol = mol
        self._mf = mf
        self.nmodes = None
        self.verbose = self._mol.verbose
        self.stdout = self._mol.stdout
        self.shift = shift
        self.add_nuc_dipole = add_nuc_dipole

        if omega is None:
            logger.warn(
                self,
                "omega is not set. Defaulting to None may lead to unexpected behavior.",
            )
            # warnings.warn("omega is not set. Defaulting to None may lead to unexpected behavior.", UserWarning)
        else:
            self.omega = omega
            self.nmodes = len(self.omega)

            # Check if vec is a 2D array and has the same first dimension size as omega
            if vec is not None and (
                not isinstance(vec, numpy.ndarray)
                or vec.ndim != 2
                or len(vec) != len(omega)
            ):
                raise ValueError(
                    f"The size of the first axis of 'vec' {len(vec):d} must be the same as the length of 'omega'."
                )
            else:
                self.vec = vec

            if isinstance(n_boson_states, list):
                if len(n_boson_states) != len(omega):
                    raise ValueError(
                        "n_boson_state must be an integer or a list with the same as the length of 'omega'"
                    )
                self.n_boson_states = n_boson_states
            else:
                self.n_boson_states = [n_boson_states for i in range(self.nmodes)]

            # Check if gfac is ann array and has the same size as omega
            if gfac is not None and (
                not isinstance(gfac, numpy.ndarray)
                or gfac.ndim != 1
                or len(gfac) != len(omega)
            ):
                raise ValueError(
                    "The size of the first axis of 'vec' must be the same as the length of 'omega'."
                )
            else:
                self.gfac = gfac

        self.use_cs = kwargs['use_cs'] if 'use_cs' in kwargs else True
        if 'z_lambda' in kwargs:
            self.z_lambda = kwargs['z_lambda']
            self.use_cs = False
            if len(self.z_lambda) != self.nmodes:
                raise ValueError("z_lambda should has the same size cavity_freq!")
        else:
            self.z_lambda = numpy.zeros(self.nmodes)

        # If gfac is None and vec is not None,
        # normalize vec and set gfac as the normalization factor
        if gfac is None and vec is not None:
            self.gfac = numpy.zeros(len(vec))
            for i in range(len(vec)):
                self.gfac[i] = numpy.sqrt(numpy.dot(vec[i], vec[i]))
                if self.gfac[i] != 0:  # Prevent division by zero
                    vec[i] /= self.gfac[i]
            self.vec = vec
        else:
            self.vec = vec
            self.gfac = gfac

        nao = self._mol.nao_nr()
        self.n_electrons = None
        self.is_optimizing_bosons = True
        self.ao_oei = None
        self.boson_type = self.__class__.__name__

        # self.polarizations = numpy.zeros((3, self.nmodes), dtype=float) #replaced by vec
        self.cs_z = numpy.zeros(self.nmodes, dtype=float)
        self.couplings = self.gfac
        self.couplings_bilinear = numpy.zeros(self.nmodes, dtype=float)
        self.couplings_var = numpy.zeros(self.nmodes, dtype=float)
        self.couplings_res = numpy.zeros(self.nmodes, dtype=float)
        self.couplings_self = numpy.zeros(self.nmodes, dtype=float)

        for k in range(self.nmodes):
            self.couplings_bilinear[k] = (
                self.couplings[k] * (self.omega[k] * 0.5) ** 0.5
            )
            self.couplings_res[k] = 1.0 - self.couplings_var[k]
            self.couplings_self[k] = 0.5 * self.couplings[k] ** 2

        self.print_summary()

    def print_summary(self):
        r"""
        Summary of bosonic features
        """

        if self.nmodes > 0:
            logger.info(self, "\nnumber of %s modes = %d", self.boson_type, self.nmodes)
            for i in range(self.nmodes):
                logger.info(self, "Info of mode %d:", i)
                logger.info(self, "   mode frequencies is:  %f", self.omega[i])
                logger.info(
                    self,
                    "   mode vector is:       %s",
                    " ".join(f"{self.vec[i,j]:.5f}" for j in range(3)),
                )
                logger.info(self, "   number of states is:  %d", self.n_boson_states[i])
                logger.info(self, "   coupling strength is: %f", self.couplings[i])
                logger.info(
                    self, "   bilinear coupling is: %f", self.couplings_bilinear[i]
                )

    def get_gmat_so(self):
        """Template method to get coupling matrix in SO."""
        raise NotImplementedError("Subclasses must implement this method.")

    def get_h_dse_ao(self):
        r"""
        DSE-mediated effective potential
        """
        raise NotImplementedError

    def get_h_dse_ao_residue(self):
        r"""
        Residual DSE-mediated effective potential (for VT-HF)
        """
        raise NotImplementedError

    def construct_g_dse_JK(self):
        r"""
        DSE-mediated JK matrix
        """
        raise NotImplementedError

    def get_geb_ao(self, mode):
        r"""
        get electron-boson interaction matrix in AO
        """
        raise NotImplementedError

    def get_geb_ao_1der(self, mode):
        raise NotImplementedError

    def update_coherent_states(self, ao_density):
        """
        Coherent state z is given from the bilinear coupling g_b and frequency w,
        z = - <g_b> / w = - Tr[D g_b] / w

        Written by Yu Zhang, June 2023
        """

        print(" update coherent state!!!!")
        g_wx = numpy.zeros((self.nao, self.nao))

        for mode in range(self.nmodes):
            g_wx = self.get_geb_ao(mode)
            self.cs_z[mode] = -np.sum(ao_density * g_wx) / self.omega[mode]

    def update_settings(self):
        for k in range(self.nmodes):
            pass

# class phonon which will compute the phonon modes and e-ph coupling strength
class Phonon(Boson):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vmat = None  #

    def relax(self):
        raise NotImplementedError("Method not implemented!")


# -------------------
# class photon
# -------------------
class Photon(Boson):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # whether to add nuclear dipole contribution
        #self.add_nuc_dipole = add_nuc_dipole

        self.dipole_ao = None
        self.quadrupole_ao = None
        self.gmat = None
        self.q_dot_lambda = None
        self.const = None
        self.ca = None
        self.pa = None

    # -------------------------------------------
    # for mf scf

    def get_dipole_ao(self):
        r"""
        return dipole matrix
        """
        self.dipole_ao = get_dipole_ao(self._mol, add_nuc_dipole=self.add_nuc_dipole)

    def get_quadrupole_ao(self):
        self.quadrupole_ao = get_quadrupole_ao(self._mol, add_nuc_dipole=self.add_nuc_dipole)

    def get_polarized_dipole_ao(self, mode):
        """
        Gets the product between the photon transversal polarization
        and the dipole moments.
        Written by Yu Zhang, Sep 2023
        """

        if self.dipole_ao is None:
            self.get_dipole_ao()
        x_dot_mu_ao = numpy.einsum("x,xuv->uv", self.vec[mode], self.dipole_ao)
        return x_dot_mu_ao

    def get_geb_ao(self, mode):
        """
        Gets the bilinear interaction term in the AO basis,
        the g in g(b+b^+).
        Written by Yu Zhang, June 2023
        """

        logger.debug(self, " construct bilinear interation term in AO")
        g_wx = self.get_polarized_dipole_ao(mode)
        logger.debug(self, f" Norm of gao without w {numpy.linalg.norm(g_wx)}")
        g_wx *= self.couplings_bilinear[mode]

        return g_wx

    def get_gmatao(self):
        if self.gmat is None:
            nao = self._mol.nao_nr()
            gmat = numpy.empty((self.nmodes, nao, nao))
            for mode in range(self.nmodes):
                gmat[mode] = self.get_polarized_dipole_ao(mode) #* self.couplings[mode]
                logger.debug(self, f" Norm of gao without w {numpy.linalg.norm(gmat[mode])}")
                gmat[mode] *= self.couplings[mode]
                #gmat = numpy.einsum("Jx,J,xuv->Juv", self.vec, self.gfac, self.dipole_ao)
            self.gmat = gmat

    def get_q_dot_lambda(self):
        # Tensor:  <u|r_i * r_y> * v_x * v_y
        if self.q_dot_lambda is None:
            nao = self._mol.nao_nr()
            self.q_dot_lambda = numpy.empty((self.nmodes, nao, nao))
            if self.quadrupole_ao is None: self.get_quadrupole_ao()
            for mode in range(self.nmodes):
                x_out_y = 0.5 * numpy.outer(self.vec[mode], self.vec[mode]).reshape(-1)
                x_out_y *= self.couplings[mode] ** 2
                self.q_dot_lambda[mode] = numpy.einsum("J,Juv->uv", x_out_y, self.quadrupole_ao)

        logger.debug(self, f" Norm of Q_ao {numpy.linalg.norm(self.q_dot_lambda)}")

    # hf  utilties
    def update_cs(self, dm):
        r"""
        Update coherent state z_\alpha = \langle \lambda\cdot \boldsymbol{D}\rangle
        """
        #mu_mo = lib.einsum("pq, Xpq ->X", dm, self.dipole_ao)
        #self.z_lambda = 0.0
        #for imode in range(self.nmodes):
        #    self.z_lambda -= self.couplings[imode] * numpy.dot(mu_mo, self.vec[imode]) # e_\alpha \cdot <D>

        # CS z_\alpha = <\lambda\cdot D>
        self.z_lambda = lib.einsum("pq, Xpq ->X", dm, self.gmat)

    # -------------------------------------------
    # post-hf integrals
    # -------------------------------------------
    def add_oei_ao(self, dm):
        r"""
        return DSE-mediated oei.. This is universal for bare HF or QED-HF.
        DSE-mediated oei

        .. math::

            -<\lambda\cdot D> g^\alpha_{uv} - 0.5 q^\alpha_{uv}
            = -Tr[\rho g^\alpha] g^\alpha_{uv} - 0.5 q^\alpha_{uv}

        """
        self.get_q_dot_lambda()
        self.get_gmatao()
        if self.use_cs:
            self.update_cs(dm)
        oei = - lib.einsum("Xpq, X->pq", self.gmat, self.z_lambda)
        oei -= numpy.sum(self.q_dot_lambda, axis=0)
        return oei

    def get_mos(self):
        r"""
        get mo coefficients
        """

        mf = self._mf
        self.nmo = 2*self._mol.nao_nr()
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
        self.ptot = utils.block_diag(self.pa, self.pb)

        logger.debug(self, "dimension of mo_occ = %d", mf.mo_occ.ndim)

    def tmat(self):
        """Return T-matrix in the spin orbital basis."""
        t = self._mol.get_hcore()
        return utils.block_diag(t, t)

    def fock(self):
        from pyscf import scf
        from pyscf.scf import ghf

        if self.pa is None or self.pb is None:
            raise Exception("Cannot build Fock without density ")

        #h1 = self._mol.get_hcore()
        h1 = self._mf.get_hcore()

        # add DSE-oei contribution
        h1 += self.add_oei_ao(self.pa+self.pb)

        ptot = utils.block_diag(self.pa, self.pb)
        h1 = utils.block_diag(h1, h1)

        # this only works for bare HF
        #myhf = scf.GHF(self._mol)
        #fock = h1 + myhf.get_veff(self._mol, dm=ptot)

        # we use jk_buld from mf object instead
        jkbuild = self._mf.get_jk
        vj, vk = ghf.get_jk(self._mol, dm=ptot, hermi=1, jkbuild=jkbuild)
        #vj, vk = self._mf.get_jk(self._mol, dm=self.pa+self.pb, hermi=1) # in ao
        fock = h1 + vj - vk

        return fock

    # this only works with bare HF
    def hf_energy(self):
        F = self.fock()
        T = self.tmat()
        ptot = utils.block_diag(self.pa, self.pb)

        Ehf = numpy.einsum("ij,ji->", ptot, F)
        Ehf += numpy.einsum("ij,ji->", ptot, T)
        # print(f"Electronic energy in hf_energy()= {Ehf}")
        if self.shift:
            return 0.5 * Ehf + self.energy_nuc + self.const
        else:
            return 0.5 * Ehf + self.energy_nuc

    def g_fock(self):
        if self.ca is None: self.get_mos()

        na, nb = self.na, self.nb
        va, vb = self.nmo // 2 - na, self.nmo // 2 - nb
        Co = utils.block_diag(self.ca[:, :na], self.cb[:, :nb])
        Cv = utils.block_diag(self.ca[:, na:], self.cb[:, nb:])

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

    def get_I(self):
        from pyscf import ao2mo
        if self.ca is None: self.get_mos()

        na, nb = self.na, self.nb
        va, vb = self.nmo // 2 - na, self.nmo // 2 - nb
        nao = self.nmo // 2
        C = numpy.hstack((self.ca, self.cb))

        # add the DSE-mediated eri (todo)

        eri = ao2mo.general(
            self.mol,
            [
                C,
            ]
            * 4,
            compact=False,
        ).reshape(
            [
                self.nmo,
            ]
            * 4
        )
        eri[:nao, nao:] = eri[nao:, :nao] = eri[:, :, :nao, nao:] = eri[
            :, :, nao:, :nao
        ] = 0
        # print("\nnorm(I_ao)=", numpy.linalg.norm(eri))
        Ua_mo = eri.transpose(0, 2, 1, 3) - eri.transpose(0, 2, 3, 1)
        temp = [i for i in range(self.nmo)]
        oidx = temp[:na] + temp[self.nmo // 2 : self.nmo // 2 + nb]
        vidx = temp[na : self.nmo // 2] + temp[self.nmo // 2 + nb :]
        # print("\nnorm(I_mo)=", numpy.linalg.norm(Ua_mo))

        vvvv = Ua_mo[numpy.ix_(vidx, vidx, vidx, vidx)]
        vvvo = Ua_mo[numpy.ix_(vidx, vidx, vidx, oidx)]
        vvov = Ua_mo[numpy.ix_(vidx, vidx, oidx, vidx)]
        vovv = Ua_mo[numpy.ix_(vidx, oidx, vidx, vidx)]
        ovvv = Ua_mo[numpy.ix_(oidx, vidx, vidx, vidx)]
        vvoo = Ua_mo[numpy.ix_(vidx, vidx, oidx, oidx)]
        vovo = Ua_mo[numpy.ix_(vidx, oidx, vidx, oidx)]
        voov = Ua_mo[numpy.ix_(vidx, oidx, oidx, vidx)]
        ovvo = Ua_mo[numpy.ix_(oidx, vidx, vidx, oidx)]
        ovov = Ua_mo[numpy.ix_(oidx, vidx, oidx, vidx)]
        oovv = Ua_mo[numpy.ix_(oidx, oidx, vidx, vidx)]
        ooov = Ua_mo[numpy.ix_(oidx, oidx, oidx, vidx)]
        oovo = Ua_mo[numpy.ix_(oidx, oidx, vidx, oidx)]
        ovoo = Ua_mo[numpy.ix_(oidx, vidx, oidx, oidx)]
        vooo = Ua_mo[numpy.ix_(vidx, oidx, oidx, oidx)]
        oooo = Ua_mo[numpy.ix_(oidx, oidx, oidx, oidx)]

        return two_e_blocks_full(
            vvvv=vvvv,
            vvvo=vvvo,
            vovv=vovv,
            voov=voov,
            ovvv=ovvv,
            ovoo=ovoo,
            oovo=oovo,
            vvov=vvov,
            ovvo=ovvo,
            ovov=ovov,
            vvoo=vvoo,
            oovv=oovv,
            vovo=vovo,
            vooo=vooo,
            ooov=ooov,
            oooo=oooo,
        )

    def mfG(self):
        if self.pa is None: self.get_mos()
        ptot = utils.block_diag(self.pa, self.pb)
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
        Co = utils.block_diag(self.ca[:, :na], self.cb[:, :nb])
        Cv = utils.block_diag(self.ca[:, na:], self.cb[:, nb:])

        # g_mo = numpy.einsum('Ipq,pi,qj->Iij',g,Cab,Cab)
        # print("\n|gmat|=", numpy.linalg.norm(g_mo))

        oo = numpy.einsum("Ipq,pi,qj->Iij", g, Co, Co)
        ov = numpy.einsum("Ipq,pi,qa->Iia", g, Co, Cv)
        vo = numpy.einsum("Ipq,pa,qi->Iai", g, Cv, Co)
        vv = numpy.einsum("Ipq,pa,qb->Iab", g, Cv, Cv)
        # print('|g.oo|=', numpy.linalg.norm(oo))
        # print('|g.ov|=', numpy.linalg.norm(ov))
        # print('|g.vo|=', numpy.linalg.norm(vo))
        # print('|g.vv|=', numpy.linalg.norm(vv))
        g = one_e_blocks(oo, ov, vo, vv)
        return (g, g)

    def get_gmat_so(self):

        r"""
        e-photon coupling matrix
        """
        if self.omega is None:
            logger.warn(self, "warning, omega is not given")

        if self.dipole_ao is None:
            self.get_dipole_ao()
        if self.quadrupole_ao is None:
            self.get_quadrupole_ao()
        if self.pa is None: self.get_mos()

        nao = self._mol.nao_nr()
        self.get_gmatao()

        # gmatso
        gmatso = [
            utils.block_diag(self.gmat[i], self.gmat[i]) for i in range(len(self.gmat))
        ]
        self.gmatso = numpy.asarray(gmatso)

        if self.shift:
            self.xi = numpy.einsum("Iab,ab->I", self.gmatso, self.ptot) / self.omega
            self.const = -numpy.einsum("I,I->", self.omega, self.xi**2)

    kernel = get_gmat_so


if __name__ == "__main__":
    from pyscf import scf

    mol = gto.M()
    mol.atom = """H 0 0 0; F 0 0 1.75202"""
    mol.unit = "Angstrom"
    mol.basis = "sto3g"
    mol.verbose = 5
    mol.build()

    nmodes = 1
    omega = numpy.zeros(nmodes)
    gfac = numpy.zeros(nmodes)
    vec = numpy.zeros((nmodes, 3))

    gfac[0] = 0.05
    omega[0] = 0.5
    vec[0, :] = [0.0, 0.0, 1.0]  # [1.0, 1.0, 1.0]
    vec[0, :] = vec[0, :] / numpy.sqrt(numpy.dot(vec[0], vec[0]))

    hf = scf.HF(mol)
    hf.max_cycle = 200
    hf.conv_tol = 1.0e-8
    hf.diis_space = 10
    mf = hf.run()

    qed = Photon(mol, mf=mf, omega=omega, vec=vec, gfac=gfac)
    qed.kernel()

    #
    for i in range(nmodes):
        g_wx = qed.get_geb_ao(i)
        print("e-ph coupling matrix of mode ", i, "is \n", g_wx)
