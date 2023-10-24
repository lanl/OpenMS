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


class RHF(hf.RHF):
    # class HF(lib.StreamObject):
    r"""
    QEDSCF base class.   non-relativistic RHF.

    """

    def __init__(self, mol, xc=None, **kwargs):
        hf.RHF.__init__(self, mol)
        # if xc is not None:
        #    rks.KohnShamDFT.__init__(self, xc)

        cavity = None
        if "cavity" in kwargs:
            cavity = kwargs["cavity"]
        if "cavity_mode" in kwargs:
            cavity_mode = kwargs["cavity_mode"]
        else:
            raise ValueError("The required keyword argument 'cavity_mode' is missing")

        if "cavity_freq" in kwargs:
            cavity_freq = kwargs["cavity_freq"]
        else:
            raise ValueError("The required keyword argument 'cavity_freq' is missing")

        print("cavity_freq=", cavity_freq)
        print("cavity_mode=", cavity_mode)

        self.cavity_freq = cavity_freq
        self.cavity_mode = cavity_mode
        self.nmode = len(cavity_freq)
        nao = self.mol.nao_nr()

        # make dipole matrix in AO
        self.make_dipolematrix()
        self.gmat = numpy.empty((self.nmode, nao, nao))
        self.gmat = lib.einsum("Jx,xuv->Juv", self.cavity_mode, self.mu_mat_ao)
        x_out_y = 0.5 * numpy.outer(cavity_mode, cavity_mode).reshape(-1)
        self.qd2 = lib.einsum("J, Juv->uv", x_out_y, self.qmat)

        # print(f"{cavity} cavity mode is used!")
        # self.verbose    = mf.verbose
        # self.stdout     = mf.stdout
        # self.mol        = mf.mol
        # self.max_memory = mf.max_memory
        # self.chkfile    = mf.chkfile
        # self.wfnsym     = None
        self.dip_ao = mol.intor("int1e_r", comp=3)

    def make_dipolematrix(self):
        """
        return dipole and quadrupole matrix in AO

        Quarupole:
        # | xx, xy, xz |
        # | yx, yy, yz |
        # | zx, zy, zz |
        # xx <-> rrmat[0], xy <-> rrmat[3], xz <-> rrmat[6]
        #                  yy <-> rrmat[4], yz <-> rrmat[7]
        #                                   zz <-> rrmat[8]
        """

        self.mu_mo = None
        charges = self.mol.atom_charges()
        coords = self.mol.atom_coords()
        charge_center = (0, 0, 0)  # numpy.einsum('i,ix->x', charges, coords)
        with self.mol.with_common_orig(charge_center):
            self.mu_mat_ao = self.mol.intor_symmetric("int1e_r", comp=3)
            self.qmat = -self.mol.intor("int1e_rr")

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
        if self._eri is not None or not self.direct_scf:
            vj, vk = self.get_jk(mol, dm, hermi)
            vhf = vj - vk * 0.5
        else:
            ddm = numpy.asarray(dm) - numpy.asarray(dm_last)
            vj, vk = self.get_jk(mol, ddm, hermi)
            vhf = vj - vk * 0.5
            vhf += numpy.asarray(vhf_last)

        # add photon contribution
        if self.mu_mat_ao is None:
            self.make_dipolematrix()

        mu_mo = lib.einsum("pq, Xpq ->X", dm, self.mu_mat_ao)
        scaled_mu = 0.0
        self.mu_lambda = 0.0
        for imode in range(self.nmode):
            self.mu_lambda -= numpy.dot(mu_mo, self.cavity_mode[imode])
            scaled_mu += numpy.einsum("pq, pq ->", dm, self.gmat[imode])

        self.oei = self.gmat * self.mu_lambda
        self.oei -= self.qd2
        self.oei = numpy.sum(self.oei, axis=0)

        vhf += self.oei

        for imode in range(self.nmode):
            vhf += scaled_mu * self.gmat[imode]
        vhf -= 0.5 * numpy.einsum("Xpr, Xqs, rs -> pq", self.gmat, self.gmat, dm)

        return vhf

    def dump_flags(self, verbose=None):
        return hf.RHF.dump_flags(self, verbose)

    def dse(self, dm):
        r"""
        compute dipole self-energy
        """
        dip = self.dip_moment(dm=dm)
        # print("dipole_moment=", dip)
        e_dse = 0.0
        e_dse += 0.5 * self.mu_lambda * self.mu_lambda

        print("dipole self-energy=", e_dse)
        return e_dse

    def energy_tot(self, dm=None, h1e=None, vhf=None):
        r"""Total QED Hartree-Fock energy, electronic part plus nuclear repulstion
        See :func:`scf.hf.energy_elec` for the electron part

        Note this function has side effects which cause mf.scf_summary updated.
        """

        nuc = self.energy_nuc()
        e_tot = self.energy_elec(dm, h1e, vhf)[0] + nuc
        e_tot += 0.5 * numpy.einsum("pq,pq->", self.oei, dm)
        dse = self.dse(dm)  # dipole sefl-energy
        e_tot += dse
        self.scf_summary["nuc"] = nuc.real
        return e_tot

    """
    def get_jk(self, mol=None, dm=None, hermi=1, with_j=True, with_k=True,
               omega=None):
        # Note the incore version, which initializes an _eri array in memory.
        #print("debug-zy: qed get_jk")
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        if (not omega and
            (self._eri is not None or mol.incore_anyway or self._is_mem_enough())):
            if self._eri is None:
                self._eri = mol.intor('int2e', aosym='s8')
            vj, vk = hf.dot_eri_dm(self._eri, dm, hermi, with_j, with_k)
        else:
            vj, vk = SCF.get_jk(self, mol, dm, hermi, with_j, with_k, omega)

        # add photon contribution, not done yet!!!!!!! (todo)
        # update molecular-cavity couplings
        vp = numpy.zeros_like(vj)

        vj += vp
        vk += vp
        return vj, vk
    """


if __name__ == "__main__":
    # will add a loop
    import numpy
    from pyscf import gto, scf

    itest = 1
    zshift = itest * 2.0

    mol = gto.M(
        atom=f"H          0.86681        0.60144        {5.00000+zshift};\
        F         -0.86681        0.60144        {5.00000+zshift};\
        O          0.00000       -0.07579        {5.00000+zshift};\
        He         0.00000        0.00000        {7.50000+zshift}",
        basis="cc-pvdz",
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

    from openms.mqed import mqed_hf as qedhf

    nmode = 1
    cavity_freq = numpy.zeros(nmode)
    cavity_mode = numpy.zeros((nmode, 3))
    cavity_freq[0] = 0.5
    cavity_mode[0, :] = 0.05 * numpy.asarray([0, 0, 1])

    qedmf = qedhf.RHF(mol, xc=None, cavity_mode=cavity_mode, cavity_freq=cavity_freq)
    qedmf.max_cycle = 500
    qedmf.kernel(dm0=dm)
