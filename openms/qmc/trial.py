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

# files for trial WF and random walkers

r"""
Single and multi Slater determinant trial wavefunction
------------------------------------------------------

Both SD and MSD can be written as the generalized formalism

.. math::

    \ket{\Psi_T} = \sum^{N_d}_{\mu} c_\mu\ket{\psi_\mu} = \sum_\mu c_\mu \hat{\mu} \ket{\psi_0}

Where :math:`\ket{\psi_0}` is the mean-field reference (like HF) and
:math:`\hat{\mu}` is the excitation operator that generate the configuration
:math:`\ket{\psi_mu}` from the MF reference and :math:`N_d` is the number of
determinants.

Overlap
~~~~~~~

Within AFQMC, we need to compute the overlap between trial and walker:

.. math::

   \langle \Psi_T\ket{\psi_w} = \sum_\mu c_\mu \langle \psi_mu\ket{\psi_w}.

For SD, which involves the agebra in the matrix representation as

.. math::

    \langle \psi_0\ket{\psi_w} = det[C^\dagger_{\psi_0} C_{\psi_w}]
    = det[C^\dagger_{\psi_w} C^*_{\psi_0}]

i.e., in the einsum format:

>>> S = backend.einsum("zpi, pj->zij", phi, psi.conj())

For MSD, the overlap is

.. math::

    S = \langle \psi_0\ket{\psi_w}\left[1 + \sum_\mu c_\mu \frac{\bra{\psi_0} \hat{\mu}^\dagger \ket{\psi_w}}{\psi_0\ket{\psi_w}}  \right]

The summation over determinants in the second term can be performed via the aid of Wick's theorem.


"""

import sys
from abc import abstractmethod
from pyscf import tools, lo, scf, fci
from openms.mqed.qedhf import RHF as QEDRHF
from pyscf.lib import logger
import numpy as backend
import h5py



class TrialWFBase(object):
    r"""
    Base class for trial wavefunction
    """

    def __init__(
        self,
        mol,
        # ne: Tuple[int, int],
        # n_mo : int,
        mf=None,
        **kwargs,
    ):
        self.verbose = kwargs.get("verbose", mol.verbose)
        self.mol = mol
        self.stdout = mol.stdout

        if mf is None:
            mf = scf.RHF(self.mol)
            mf.kernel()
            logger.info(self, "MF energy is   {mf.e_tot}")

        self.mf = mf

        # TBA: set number of electrons, alpha/beta electrons
        # and number of SO # we assume using spin orbitals
        self.nso = 0  # number of spin orbitals
        self.nelectrons = mol.nelectron
        self.spin = mol.spin
        self.nalpha = (self.nelectrons + self.spin) // 2
        self.nbeta = self.nalpha - self.spin
        assert self.nalpha >= 0 and self.nbeta >= 0
        if self.nalpha + self.nbeta != self.nelectrons:
            raise RuntimeError(
                "Electron number %d and spin %d are not consistent\n"
                "Note mol.spin = 2S = Nalpha - Nbeta, not 2S+1"
                % (self.nelectrons, self.spin)
            )

        logger.debug(self, " nalpha/nbeta = %d  %d " % (self.nalpha, self.nbeta))

        # self.num_elec = num_elec # number of electrons
        # self.n_mo = n_mo
        # only works for spin-restricted reference at this moment
        # self.nalpha = self.nbeta = self.num_elec // 2

        self._numdets = kwargs.get("numdets", 1)
        self._numdets_props = kwargs.get("numdets_props", 1)
        self._numdets_chunks = kwargs.get("numdets_chunks", 1)

        self.build()

    @abstractmethod
    def build(self):
        r"""
        build initial trial wave function
        """
        pass

    @property
    def numdets(self) -> int:
        return self._numdets

    @numdets.setter
    def numdets(self, value: int):
        self._numdets = value

    @property
    def num_electrons(self) -> int:
        return self.nelectrons

    # functions
    @abstractmethod
    def force_bias(self, walkers):
        r"""Compute the force bias"""
        pass


# single determinant HF trial wavefunction
class TrialHF(TrialWFBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build(self):
        r"""
        initialize the Trial WF.

        The trial WF is the tensor product of electron
        and boson. Boson part is none by default.

        Representation of boson: 1) Fock, 2) CS, 3), VLF, 4) real space.
        """
        overlap = self.mol.intor("int1e_ovlp")
        Xmat = lo.orth.lowdin(overlap)

        xinv = backend.linalg.inv(Xmat)

        self.psi = self.mf.mo_coeff
        # Nao * Na
        self.psi = xinv.dot(self.mf.mo_coeff[:, : self.mol.nelec[0]])
        self.psi_boson = None
        # self.boson_basis = "Fock"

    def ovlp_with_walkers(self, walkers):
        r"""Compute the overlap between trial and walkers:

        .. math::

            \langle \Psi_T \ket{\Psi_w} = det[S]

        where

        .. math::

            S = C^*_{\psi_T} C_{\psi_w}

        and :math:`C_{\psi_T}` and :math:`C_{\psi_w}` are the coefficient matrices
        of Trial and Walkers, respectively.
        """
        return backend.einsum("pr, zpq->zrq", self.psi.conj(), walkers.phiw)

    def get_vbias(self, walkers, ltensor, verbose=False):
        overlap = self.ovlp_with_walkers(walkers)
        inv_overlap = backend.linalg.inv(overlap)

        if verbose:
            logger.debug(
                self,
                "\nnorm of walker overlap: %15.8f",
                backend.linalg.norm(overlap),
            )

        # theta is the Ghalf
        theta = backend.einsum("zqp, zpr->zqr", walkers.phiw, inv_overlap)
        Gf = backend.einsum("zqr, pr->zpq", theta, self.psi.conj())
        vbias = backend.einsum("npq,zpq->zn", ltensor, Gf)

        return Gf, vbias



    def force_bias(self, walkers, TL_tensor, verbose=False):
        r"""Update the force bias

        precisely, we compute vbias here.

        Green's function is:

        .. math::

            G_{pq} = [\psi_w (\Psi^\dagger_T \psi_w)^{-1} \Psi^\dagger_T]_{qp}

        TL_tensor (precomputed) is:

        .. math::

            (TL)_{\gamma,pq} = (\Psi^\dagger_T L_\gamma)_{pq}

        And :math:`\Theta_w` is:

        .. math::

            \Theta_w = \psi_w (\Psi^\dagger_T\psi_w)^{-1} = \psi_w S^{-1}_w

        where :math:`S_w` is the walker-trial overlap.

        Then :math:`(TL)\Theta_w` determines the force bias:

        .. math::

           F_\gamma = \sqrt{-\Delta\tau} \sum_\sigma [(TL)\Theta_w]

        TODO: may use L * G contraction to get the vbias, instead of using theta_w???
        """
        overlap = self.ovlp_with_walkers(walkers)
        inv_overlap = backend.linalg.inv(overlap)

        if verbose:
            logger.debug(
                self,
                "\nnorm of walker overlap: %15.8f",
                backend.linalg.norm(overlap),
            )

        # theta is the Ghalf
        theta = backend.einsum("zqp, zpr->zqr", walkers.phiw, inv_overlap)
        Gf = backend.einsum("zqr, pr->zpq", theta, self.psi.conj())
        # :math:`(\Psi_T L_{\gamma}) \psi_w (\Psi_T \psi_w)^{-1}`

        # since TL_tehta is too big, we will avoid constructing it in the propagation
        TL_theta = backend.einsum("npq, zqr->znpr", TL_tensor, theta)

        # trace[TL_theta] give the force_bias
        # vbias = backend.einsum("znpp->zn", TL_theta)
        return Gf, TL_theta


from openms.qmc import estimators


# single determinant unrestricted HF trial wavefunction
class TrialUHF(TrialWFBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build(self):
        overlap = self.mol.intor("int1e_ovlp")  # AO Overlap Matrix, S
        Xmat = lo.orth.lowdin(overlap)  # Eigenvectors of S**(1/2) = X
        xinv = backend.linalg.inv(Xmat)  # S**(-1/2)

        # TODO: name change MO_ALPHA/beta -> psia/b
        MO_ALPHA = self.mf.mo_coeff[
            0, :, : self.mol.nelec[0]
        ]  # Occupied ALPHA MO Coeffs
        MO_BETA = self.mf.mo_coeff[1, :, : self.mol.nelec[1]]  # Occupied BETA MO Coeffs
        self.psi = [
            backend.dot(xinv, MO_ALPHA)
        ]  # ALPHA ORBITALS AFTER LOWDIN ORTHOGONALIZATION
        self.psi.append(
            backend.dot(xinv, MO_BETA)
        )  # BETA ORBITALS AFTER LOWDIN ORTHOGONALIZATION
        self.psi = backend.array(
            self.psi
        )  # self.psi.shape = (spin, nocc mos per spin, nAOs)

        self.Gf, self.Gf_half = estimators.GF_so(
            self.pwsi, self.psi, self.nalpha, self.nbeta
        )

    def ovlp_with_walkers(self, walkers):
        r"""Compute the overlap with walkers"""
        pass

    def force_bias(self, walkers, TL_tensor, verbose=False):
        r"""Compute the force bias (Eqns.67-69 of Ref. :cite:`zhang2021jcp`)

        .. math::

            F_\gamma = & \sqrt{-\Delta\tau}\sum_{ij}\sum_\sigma (L^\sigma_\gamma)_{ij} G^\sigma_{ij} \\
                     = & \sqrt{-\Delta\tau}\sum_\sigma\text{Tr}\left[L^\sigma_{ij}\psi^\sigma_w(\Psi^{\sigma\dagger}_T
                          \psi^\sigma_w)^{-1}\Psi^{\sigma\dagger}_T\right] \\
                     \\
                     = & \sqrt{-\Delta\tau}\sum_\sigma\left[(\Psi^{\sigma\dagger}_TL^\sigma_\gamma)\Theta^\sigma_w\right]

        where :math:`\Theta^\sigma_w=\psi^\dagger_w(\Psi^{\sigma\dagger}_T\psi^\sigma_w)^{-1}`
        and :math:`(\Psi^{\sigma\dagger}_TL^\sigma_\gamma)` is independent of walkeers are pre-computed.
        :math:`\Theta^\sigma_w` is calculated by solving a linear equation with LU decomposition of
        :math:`\Psi^{\sigma\dagger}_T\psi^\sigma_w`.
        """

        overlap = self.ovlp_with_walkers(walkers)
        inv_overlap = backend.linalg.inv(overlap)

        # theta is also the Ghalf
        # Ghalfa = backend.einsum("zqp, zpr->zqr", walkers.phiw, inv_overlap)
        # Ghalfb = backend.einsum("zqp, zpr->zqr", walkers.phiw, inv_overlap)
        # Gfa = backend.einsum("zqr, pr->zpq", Ghalfa, self.psi.conj())
        # Gfb = backend.einsum("zqr, pr->zpq", Ghalfb, self.psi.conj())

        # :math:`\sum_\sigma(\Psi_T L_{\gamma}) \psi_w (\Psi_T \psi_w)^{-1}`
        # vbias = backend.einsum("npq, zqr->znpr", TL_tensor, Ghalfa)

def get_ci(mol, cas):
    from pyscf import mcscf, fci

    # TODO: use UHF if openshell system
    mf = scf.RHF(mol)
    e_hf = mf.kernel()

    # casscf
    M, N = cas
    mc = mcscf.CASSCF(mf, M, N)

    # get fcivec
    nocca = (N + mol.spin) // 2
    noccb = N - nocca
    e_tot, e_cas, fcivec, mo, mo_energy = mc.kernel()

    # TODO, get mo_occ from mc

    logger.debug(mol, f"\nHF     total energy: {e_hf}")
    logger.debug(mol, f"casscf total energy: {e_tot}")
    logger.debug(mol, f"correlation energy:  {e_tot - e_hf}")

    # logger.debug(mol, f"hf.mo_occ ?= cassc.mo_occ = {backend.allclose(mf.mo_occ, mc.mo_occ, rtol=1e-05)} \n")
    logger.debug(
        mol,
        f"hf.mo_coeff ?= cassc.mo_coeff = {backend.allclose(mf.mo_coeff, mc.mo_coeff, rtol=1e-05)} \n",
    )
    logger.debug(mol, f"cassc.mo_occ   = {mc.mo_occ}")
    logger.debug(mol, f"cassc.mo_coeff = {mc.mo_coeff}")

    coeff, occa, occb = zip(
        *fci.addons.large_ci(fcivec, M, (nocca, noccb), tol=1e-8, return_strs=False)
    )
    return mf, coeff, (occa, occb)


class multiCI(TrialWFBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ci_coeffs = kwargs.get("ci_coeffs", None)
        self.occs = kwargs.get("occupations", None)
        self.cas = kwargs.get("cas", None)
        if self.ci_coeffs is None:
            # build a mcscf calculations and get the values
            if self.cas is None:
                raise RuntimeError(
                    "Cas must be specified if ci_coefficients are not provided"
                )
            self.mf, self.ci_coeffs, self.occs = get_ci(self.mol, self.cas)
        logger.debug(self, f"mo_occ   = {self.mf.mo_occ}")
        logger.debug(self, f"occupation (a, b) = {self.occs}")

        # TODO: get X

        # update other parameters
        self._numdets = len(self.ci_coeffs)
        logger.debug(self, f"number of determinents = {self._numdets:8d}")

        self.build()

    # def build(self):
    #    # get chol
    #    pass

    def force_bias(self, walkers):
        r"""Compute the force bias with multiSD (Eqns.83-84 of Ref. :cite:`zhang2021jcp`)

        .. math::

            F = &\sqrt{-\Delta\tau} \frac{\bra{\Psi_T}L_\gamma\ket{\psi_w}}{\langle\Psi_T\ket{\psi_w}} \\
              = &\sqrt{-\Delta\tau}\sum_\sigma
        """

        # Fix this:
        # get Ga/b
        # Ga = xx
        # Gb = xx
        # vbias = backend.einsum("npq, zqp->zn", Ltensor, Ga)
        # vbias += backend.einsum("npq,zqp->zn ", Ltensor, Gb)

        raise NotImplementedError("Force bias with MSD is not implemented yet.")


# =====================================
# define walker class
# =====================================


if __name__ == "__main__":
    from pyscf import gto, scf, fci

    bond = 1.6
    natoms = 2
    atoms = [("H", i * bond, 0, 0) for i in range(natoms)]
    mol = gto.M(atom=atoms, basis="sto-6g", unit="Bohr", verbose=3)
    trial = TrialHF(mol)

    cas = (2, 2)
    trial = multiCI(mol, cas=cas)
