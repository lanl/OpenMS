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


**Overlap**:


Within AFQMC, we need to compute the overlap between trial and walker:

.. math::
   \langle \Psi_T\ket{\psi_w} = \sum_\mu c_\mu \langle \psi_\mu\ket{\psi_w}.

For SD, which involves the agebra in the matrix representation as

.. math::
    \langle \psi_0\ket{\psi_w} = det[C^\dagger_{\psi_0} C_{\psi_w}]
    = det[C^\dagger_{\psi_w} C^*_{\psi_0}]

i.e., in the einsum format:

>>> S = backend.einsum("zpi, pj->zij", phi, psi.conj())

**Green's Function***:

One-body GF is

.. math::
    G^\sigma_{ij} = & \frac{\bra{\Psi_T}a^\dagger_i a_j \ket{\psi_w}}{\langle\Psi_T\ket{\psi_w}} \\
                  = & \left[\psi^\sigma_w (\Psi^{\sigma\dagger}_T\psi^\sigma_w)^{-1} \Psi^{\sigma\dagger}_T \right]
                  \equiv & \left[\Theta^\sigma_w \Psi^{\sigma\dagger}_T \right].

Where :math:`\Theta^\sigma_w=\psi^\sigma_w (\Psi^{\sigma\dagger}_T\psi^\sigma_w)^{-1}`.

The **Force bias** can be calculated from the GFs via:

.. math::
     F_\gamma  = & \sqrt{-\Delta\tau} \sum_{ij,\sigma} [L^\sigma_\gamma]_{ij} G^\sigma_{ij} \\
               = & \sqrt{-\Delta\tau} \sum_\sigma \text{Tr}[L^\sigma_{\gamma}\psi^\sigma_w
                   (\Psi^{\sigma\dagger}_T\psi^\sigma_w)^{-1} \Psi^{\sigma\dagger}_T] \\
               \equiv & \sqrt{-\Delta\tau} \sum_\sigma \text{Tr}[(\Psi^{\sigma\dagger}_T L^\sigma_\gamma)\Theta^\sigma_w]

Since :math:`(\Psi^{\sigma\dagger}_T L^\sigma_\gamma)` is independent of walker, we pre-compute it and contract it with
:math:`\Theta^\sigma_w` to update the force bias.

**Energy**

The one-body temr is

.. math::

    E_1 = \text{Tr}[hG] = \sum_\sigma (\Psi^{\sigma\dagger}_T h_1) \Theta^\sigma_w.

So again, we pre-compute :math:`(\Psi^{\sigma\dagger}_T h_1)` and contract it with
:math:`\Theta^\sigma_w` without explicitly constructing the GF to get the one-body
energy.

Similarly, the two-body term is given by:

.. math::

    E_2 = & \frac{1}{2}\sum_{\gamma, ijkl, \sigma\sigma'} [L_\gamma]_{il}[L^*_\gamma]_{kj}
          \frac{\bra{\Psi_T}a^\dagger_{i\sigma} a^\dagger_{j\sigma'} a_{k\sigma'}
           a_{l\sigma}\ket{\psi_w}}{\langle\Psi_T\ket{\psi_w}} \\
        = & \frac{1}{2}\sum_{\gamma, ijkl, \sigma\sigma'} [L_\gamma]_{il}[L^*_\gamma]_{kj}
            \left[G^\sigma_{il} G^{\sigma'}_{jk} - \delta_{\sigma\sigma'}G^\sigma_{ik} G^{\sigma'}_{jl} \right]\\
        = & \frac{1}{2}\sum_\gamma (\text{Tr}[L_\gamma G])^2
          - \frac{1}{2} \sum_{\sigma}\text{Tr} \left[(\Psi^{\sigma\dagger}_T L^\sigma_\gamma)\Theta^\sigma_w
            (\Psi^{\sigma\dagger}_T L^\sigma_\gamma)\Theta^\sigma_w\right].

Once, again, we only need the contract the precomputed :math:`(\Psi^{\sigma\dagger}_T L^\sigma_\gamma)` with
:math:`\Theta^\sigma_w` for computing the energies.


Multi Slater determinant (MSD) trial wavefunction
-------------------------------------------------


The :math:`\ket{\phi_\mu}` is a Slater determinants with matrix representation :math:`{\psi_\mu^\sigma}`
for spin :math:`\sigma`.


**Overlap**:

For MSD, the overlap is

.. math::
    S = \langle \psi_0\ket{\psi_w}\left[1 + \sum_\mu c_\mu \frac{\bra{\psi_0} \hat{\mu}^\dagger \ket{\psi_w}}{\psi_0\ket{\psi_w}}  \right]

The summation over determinants in the second term can be performed via the aid of Wick's theorem.

More details TBA.

**Green's functions**:


**Force bias**:


**Efficient Implementation**:



"""

import sys
import time
from abc import abstractmethod
from pyscf import tools, lo, scf, fci
from pyscf.lib import logger
from openms.lib.boson import Boson
from openms.mqed.qedhf import RHF as QEDRHF
from openms.lib.logger import task_title
import numpy as backend
import h5py


def half_rotate_integrals(
    ncomponents,
    psia,
    psib,
    h1e,
    ltensor):
    r"""
    Perform half-rotation of integrals.

    1. Half-rotated h1e:

    .. math::
        \tilde{h}_{iq} = \sum_{p} C^*_{jp} h_{pq}


    2. Half-rotated chols:

    .. math::
       \tilde{L}_{\gamma, iq} = \sum_{p} C^*_{jp} L_{\gamma, pq}

    Parameters
    ----------
    psia : ndarray
        Wavefunction of alpha spin, shape `(nao, nmo)`.
    psib : ndarray
        Wavefunction of beta spin, shape `(nao, nmo)`.
    h1e : ndarray
        One-body integrals, shape `(ncomponents, nao, nao)`.
    ltensor : ndarray
        Cholesky tensor, shape `(nchol, nao, nao)`.

    Returns
    -------
    rh1e : tuple of numpy.ndarray
        Rotated one-body integrals, each component has shape `(ndets, nao, na/b)`
        for alpha and beta spins.
    rotated_ltensor : tuple of numpy.ndarray
        Rotated Cholesky tensors, each component has the shape `(ndets, nchol, nao, na/b)`
        for alpha and beta spins.
    """

    if len(psia.shape) != 3:
        raise ValueError(f"'psia' must have 3 dimensions, but its shape is {psia.shape}")
    if ncomponents > 1:
        if len(psib.shape) != 3:
            raise ValueError(f"'psib' must have 3 dimensions, but its shape is {psib.shape}")
    if len(h1e.shape) != 3:
        raise ValueError(f"'h1e' must have 3 dimensions, but its shape is {h1e.shape}")
    if len(ltensor.shape) != 3:
        raise ValueError(f"'ltensor' must have 3 dimensions, but its shape is {ltensor.shape}")

    # nao = h1e.shape[-1]
    # nchol = ltensor.shape[0]
    # nalpha = psia.shape[-1]
    # nbeta = psib.shape[-1]

    # Half-Rotating h1e
    rotated_h1a = backend.einsum("Jpi, pq->Jqi", psia.conj(), h1e[0])

    # half-rotating chols
    # TODO: distributed MPI of rotating ltensor
    # shape of rotated chol: (numdet, nchol, nmo, nao)
    rotated_ltensora = backend.einsum("Jpi, npq->Jnqi", psia.conj(), ltensor, optimize=True)

    rotated_h1b = None
    rotated_ltensorb = None
    if ncomponents > 1:
        rotated_h1b = backend.einsum("Jpi, pq->Jqi", psib.conj(), h1e[1])
        rotated_ltensorb = backend.einsum("Jpi, npq->Jnqi", psib.conj(), ltensor, optimize=True)

    # print(f"Debug: rotated_h1a.shape  = ", rotated_h1a.shape)
    # print(f"Debug: shape of original ltensor=  ", ltensor.shape)
    # print(f"Debug: rotated_h1a        =\n", rotated_h1a)
    # print(f"Debug: norm of rotated_h1a=  ", backend.linalg.norm(rotated_h1a))
    # print(f"Debug: norm of rotated_ltensora=  ", backend.linalg.norm(rotated_ltensora))
    #if ncomponents > 1:
    #    print(f"Debug: norm of rotated_ltensorb=  ", backend.linalg.norm(rotated_ltensorb))
    #    print(f"Debug: shape of rotated ltensorb=  ", rotated_ltensorb.shape)

    return (rotated_h1a, rotated_h1b), (rotated_ltensora, rotated_ltensorb)



#
# ****** functions for computing trial_walker overlap in single determinant formalism
#


def trial_walker_ovlp_base(phiw, psi):
    r"""
    An universal function for computing the trial_walker overlap

    Parameters
    ----------
    phiw: ndarray
        walker wavefunction
    psi: ndarray
       trial wavefunction
    """
    return backend.einsum("zpi, pj-> zij", phiw, psi.conj())


def trial_walker_ovlp_gf_base(phiw, psi):
    r"""
    An universal function for computing the trial_walker overlap
    and Green's function

    Parameters
    ----------
    phiw: ndarray
        walker wavefunction
    psi: ndarray
       trial wavefunction
    """
    ovlp = backend.einsum("zpi, pj-> zij", phiw, psi.conj())
    inv_ovlp = backend.linalg.inv(ovlp)
    Ghalf = backend.einsum("zij, zpj->zpi", inv_ovlp, phiw)

    return ovlp, Ghalf


def calc_walker_gf(walker, trial, ovlp):
    r"""
    Compute walker green's function with singlet trial and
    pre-computed trial_walker overlap (tuple of each spin, not total overlap)

    TODO: decide whether we need this function. The GF_so in the estimator
    does the same thing! (but not using half_rotated).

    The walker Ghalfa/b has the shape of [nwalker, na/b, nao]
    The walker (full) GF has the shape of [nwalker, nao, nao]

    Parameters
    ----------
    walker : object
        Single object.
    trial : object
        Trial wavefunction object
    ovlp: tuple of ndarray (length is ncomponents)
        trial_walker overlap
    """
    inv_ovlp = backend.linalg.inv(ovlp[0])
    walker.Ghalfa = backend.einsum("zij, zpj->zpi", inv_ovlp, walker.phiwa)
    # TODO: test the code without half_rotation
    if not trial.half_rotated:
        # print("Debug: trial is NOT half-rotated, construct the full Green funciton here!")
        walker.Ga = backend.einsum("pi, nqi->npq", trial.psia.conj(), walker.Ghalfa)

    if trial.ncomponents > 1:
        inv_ovlp = backend.linalg.inv(ovlp[1])
        walker.Ghalfb = backend.einsum("zij, zpj->zpi", inv_ovlp, walker.phiwb)
        if not trial.half_rotated:
            walker.Gb = backend.einsum("pi, nqi->npq", trial.psib.conj(), walker.Ghalfb)


def calc_trial_walker_ovlp(walker, trial):
    r"""Compute the trial walker overlap only

    Parameters
    ----------
    walker : object
        Single object.
    trial : object
        Trial wavefunction object.
    Returns
    -------
    ovlp : float64 / complex128
        Overlap between walker and trial
    """
    # slogdet: sign and (natural) logarithm of the determinant of an array.
    ovlp_a = trial_walker_ovlp_base(walker.phiwa, trial.psia)
    sign_a, logovlp_a = backend.linalg.slogdet(ovlp_a)
    if trial.ncomponents > 1:
        ovlp_b = trial_walker_ovlp_base(walker.phiwb, trial.psib)
        sign_b, logovlp_b = backend.linalg.slogdet(ovlp_b)

        ovlp = sign_a * sign_b * backend.exp(logovlp_a + logovlp_b - walker.logshift)
    else:
        ovlp = sign_a * backend.exp(logovlp_a - walker.logshift)
        ovlp *= ovlp
    return ovlp


def calc_trial_walker_ovlp_gf(walker, trial, return_signs=False):
    r"""Compute walker green's function with singlet trial

    This function also contains the code to compute the trial walker overlap

    Parameters
    ----------
    walker : object
        Single object.
    trial : object
        Trial wavefunction object.
    Returns
    -------
    ovlp : float64 / complex128
        Overlap between walker and trial
    """

    ovlp_a, walker.Ghalfa = trial_walker_ovlp_gf_base(walker.phiwa, trial.psia)
    sign_a, logovlp_a = backend.linalg.slogdet(ovlp_a)

    # TODO: test the code without half_rotation
    if not trial.half_rotated:
        # print("Debug: trial is NOT half-rotated, construct the full Green funciton here!")
        walker.Ga = backend.einsum("pi, nqi->npq", trial.psia.conj(), walker.Ghalfa)

    if trial.boson_psi is not None:
        # phi_{w, pi} Psi^T_{pj} --> zij, if i, j is only 1
        # then ovlp_b is phi_{z,F} \Psi^T_{F} -> z
        # ovlp_b_inv = 1/S
        # \phi_{zF} /S_{z,ij}  --> [boson_Ghalf]_{z,Fj}
        #  [boson_Ghalf]_{z,Fj} * Psi_{F'j} -> G_{FF'}
        walker.boson_ovlp = trial.boson_ovlp_with_walkers(walker)
        inv_ovlp = 1.0 / walker.boson_ovlp
        walker.boson_Ghalf = backend.einsum("zN, z->zN", walker.boson_phiw, inv_ovlp)
        walker.boson_Gf = backend.einsum("zM, N->zMN", walker.boson_Ghalf, trial.boson_psi.conj())

    sign_b = None
    if trial.ncomponents > 1:
        ovlp_b, walker.Ghalfb = trial_walker_ovlp_gf_base(walker.phiwb, trial.psib)
        sign_b, logovlp_b = backend.linalg.slogdet(ovlp_b)
        if not trial.half_rotated:
            walker.Gb = backend.einsum("pi, nqi->npq", trial.psib.conj(), walker.Ghalfb)

        ovlp = sign_a * sign_b * backend.exp(logovlp_a + logovlp_b - walker.logshift)
    else:
        ovlp = sign_a * backend.exp(logovlp_a - walker.logshift)
        ovlp *= ovlp

    if return_signs:
        return ovlp, sign_a, sign_b
    return ovlp


#
# ****** functions for computing trial_walker overlap in MSD formalism ******
#

def calc_ovlp_iexc(exc_order, Gf, cre_idx, anh_idx, occ_map, nao_frozen):
    r"""Order `exc_order` excitaiton's contribution to the trial_walker overlap

    An universal function for either alpha or beta component

    Not working yet!
    """

    # NOT DONE YET

    nwalkers = Gf.shape[0]
    numdets = len(cre_idx[exc_order])

    ovlp = None
    if numdets > 0:
        ovlp = backend.zeros((nwalkers, numdets), dtype=backend.complex128)
        ovlp_tmp = backend.zeros((exc_order, exc_order), dtype=backend.complex128)

        # loop over the CIs up to exc_order
        #
        for idet in range(numdets):
            for iexc in range(exc_order):
                p = occ_map[cre_idx[idet, iexc]] + nao_frozen
                q = anh_idx[idet, iexc] + nao_frozen
                ovlp_tmp[iex, iex] = Gf[:, p, q]
                for jexc in range(iexc + 1, exc_order):
                    r = mapping[cre_idx[idet, jexc]] + nao_frozen
                    s = anh_idx[idet, jexc] + nao_frozen
                    ovlp_tmp[iex, jex] = Gf[:, p, s]
                    ovlp_tmp[jex, iex] = Gf[:, r, q]
            ovlp[:, idet] = backend.linalg.det(det)
    return ovlp



def compute_MSD_ovlp(GFs, trial):
    r"""
    Compute overlap in MSD formalism
    """

    Ga, Gb = GFs
    nwalkers = Ga.shape[0]
    ndets = trial._numdets

    ovlp_a = backend.ones((nwalkers, ndets), dtype=backend.complex128)
    ovlp_b = backend.ones((nwalkers, ndets), dtype=backend.complex128)

    #for iexc in range(1, trial.max_exc):
    #    tmp_ovlps = calc_ovlp_iexc(iexc, Ga, trial.cre_ex_a, trial.anh_ex_a, trial.occ_map_a, trial.nao_frozen)
    #    #ovlp_a[trial.exc_map_a[iexc], :] = tmp_ovlps[0]
    #    #ovlp_b[trial.exc_map_b[iexc], :] = tmp_ovlps[1]

    return ovlp_a, ovlp_b


def calc_MSD_trial_walker_ovlp(walker, trial):
    r"""Compute the trial walker overlap only

    Parameters
    ----------
    walker : object
        MSD walker.
    trial : object
        MSD trial
    Returns
    -------
    ovlp : float64 / complex128
        Overlap between walker and trial
    """

    # 1) compute the overlap and GFs for the reference determinant
    ovlp0 = calc_trial_walker_ovlp_gf(walker, trial)
    logger.debug(trial, f"Debug: overlap of reference determinants is {ovlp0}")

    # 2) computer other determinants
    walker.CIa.fill(0.0 + 0.0j)
    walker.CIb.fill(0.0 + 0.0j)

    return ovlp0
    # raise NotImplementedError("Trail_Walker overlap with MSD is not implemented yet.")



def calc_MSD_trial_walker_ovlp_gf(walker, trial):
    r"""Compute walker green's function with singlet trial

    This function also contains the code to compute the trial walker overlap
    """
    # build the overlap and GFs for the reference part
    # which is the same as the SD case, so we recycle the SD code

    logger.debug(trial, "\nDebug: computing the overlap of reference determinants")
    ovlp0, sign_a, sign_b = calc_trial_walker_ovlp_gf(walker, trial, return_signs=True)
    # Note: the wallerks.Ga/b are for the reference determinants only
    # return ovlp0

    logger.debug(trial, f"Debug: overlap of reference determinants is {ovlp0}")
    return ovlp0

    # TODO: add the contribution from the other determinants
    ovlp_dets_a, ovlp_dets_b = compute_MSD_ovlp([walker.Ghalfa, walker.Ghalfb], trial)

    ovlp_dets_a *= trial.phase_a[None, :]
    ovlp_dets_b *= trial.phase_b[None, :]

    # phase_ovlp_dets_ab = backend.einsum("", xxx)
    # phase_ovlp_dets_ba = backend.einsum("", xxx)

    #
    # GF contributed by the singles
    #
    # walkers.Ga_dets += backend.einsum("w,wpq->wpq", ovlps, walkers.Ga)
    # walkers.Gb_dets += backend.einsum("w,wpq->wpq", ovlps, walkers.Gb)

    #
    # TODO: GF contribution from doulbes and above
    #

    return ovlp0
    # raise NotImplementedError("Greens function and Trail_Walker overlap with MSD is not implemented yet.")


def permutation_sign(anh_idx, cre_idx, ref_det, target_det):
    r"""
    Determine the sign of permutation from reference determinants to the target determinants

    """

    nmove = 0
    perm = 0
    for o in anh_idx:
        io = backend.where(target_det == o)[0]
        perm += io - nmove
        nmove += 1
    nmove = 0
    for o in cre_idx:
        io = backend.where(ref_det == o)[0]
        perm += io - nmove
        nmove += 1

    if perm % 2 == 1: return -1
    return 1



def initialize_boson_trial_with_z(zalpha, boson_states):
    from math import factorial

    nmodes = len(zalpha)
    coefficients = []
    for imode in range(nmodes):
        for n in range(boson_states[imode]):
            alpha = zalpha[imode]
            coeff = (backend.exp(-backend.abs(alpha)**2 / 2) * (alpha**n) / backend.sqrt(factorial(n)))
            coefficients.append(coeff)
    coefficients = backend.array(coefficients)
    norm_factor = backend.sqrt(backend.sum(backend.abs(coefficients)**2))
    return coefficients / norm_factor

class TrialWFBase(object):
    r"""
    Base class for trial wavefunction

    Names of the trial and walekr wavefunctions:

       - psia and psib: trial WF for alpha and beta spins
       - phiw: fermionic walker
       - boson_psi: bosonic trial WF
       - boson_phiw: bosonic walker WF

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
            logger.info(self, f"Mean-field reference is None, we will build from RHF")
            mf = scf.RHF(self.mol)
            mf.kernel()
            logger.info(self, "MF energy is   {mf.e_tot}")

        self.mf = mf
        logger.info(self, f"Mean-field reference is {mf.__class__}")

        # TBA: set number of electrons, alpha/beta electrons
        # and number of SO # we assume using spin orbitals
        self.ncomponents = kwargs.get("ncomponents", 1)  # number of spin components
        self.nelectrons = mol.nelectron
        self.nao = mol.nao_nr()
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

        # self.num_elec = num_elec # number of electrons
        # self.n_mo = n_mo
        # only works for spin-restricted reference at this moment
        # self.nalpha = self.nbeta = self.num_elec // 2

        self._numdets = kwargs.get("numdets", 1)
        self._numdets_props = kwargs.get("numdets_props", 1)
        self._numdets_chunks = kwargs.get("numdets_chunks", 1)
        self.OAO = kwargs.get("OAO", True)
        self.half_rotated = False

        self.boson_psi = None

        # self.build()

    def dump_flags(self):
        r"""dump flags"""
        logger.note(self, task_title("Flags of trial wavefunction"))
        logger.note(self, f" Trail WF is                  : {self.__class__.__name__}")
        logger.note(self, f" Number of determinants       : {self._numdets:5d}")
        logger.note(self, f" Number of spin components    : {self.ncomponents:5d}")
        logger.info(self, f" Number of total orbitals     : {self.nao:5d}")
        logger.note(self, f" Number of total electrons    : {self.nelectrons: 5d}")
        logger.note(self, f" Number of alpha electrons    : {self.nalpha: 5d}")
        logger.note(self, f" Number of beta electrons     : {self.nbeta: 5d}")
        if isinstance(self.mol, Boson):
            logger.note(self, f" Number of bosonic models     : {self.mol.nmodes}")
            logger.note(self, f" Number of bosonic states     : {self.mol.nboson_states}")
        logger.note(self, f"")


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


    @abstractmethod
    def half_rotate_integrals(self, h1e, ltensor):
        pass


    def initialize_boson_trial_with_z(self, zalpha, boson_states):
        self.boson_psi = initialize_boson_trial_with_z(zalpha, boson_states)
        logger.info(self, f"updated boson_psi is {self.boson_psi}")



# single determinant HF trial wavefunction
class TrialHF(TrialWFBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._numdets = 1
        self.header = "SD_RHF"
        if self.ncomponents > 1:
            self.header = "SD_UHF"

    def build(self):
        r"""
        initialize the Trial WF.

        The trial WF is the tensor product of electron
        and boson. Boson part is none by default.

        Representation of boson: 1) Fock, 2) CS, 3), VLF, 4) real space.
        """

        if self.OAO:
            overlap = self.mol.intor("int1e_ovlp")
            Xmat = lo.orth.lowdin(overlap)
            xinv = backend.linalg.inv(Xmat)
            # Nao * Na
            # self.psia = self.psib = xinv.dot(self.mf.mo_coeff[:, : self.mol.nelec[0]])
            self.psia = xinv.dot(self.mf.mo_coeff[:, : self.mol.nelec[0]])
            self.psib = xinv.dot(self.mf.mo_coeff[:, : self.mol.nelec[1]])
        else:
            nmo = self.mf.mo_coeff.shape[-1]
            tmp = backend.identity(nmo)[:, self.mf.mo_occ > 0]
            #self.psia = self.psib = tmp
            # psia and psib may have different size, when nalpha and nbeta are different
            self.psia = tmp[:, self.nalpha]
            self.psib = tmp[:, self.nbeta]

        # print("Debug: nelec  = ", self.mol.nelec)
        # print("Debug: mo_occ = ", self.mf.mo_occ)
        # print(f"Debug: shape of psia/b = {self.psia.shape} {self.psib.shape}")

        if self.ncomponents > 1:
            self.psi = backend.hstack([self.psia, self.psib])
            # TODO: build Fermionic Gf and Ghalf in SO
            # TODO: the new function trail_walker_gf can computer such green's functions
            # TODO: may longer need the GF_so function
            self.Gf, self.Gf_half = estimators.GF_so(self.psi, self.psi, self.nalpha, self.nbeta)
        else:
            self.psi = self.psia
            self.Gf, self.Gf_half = estimators.GF_so(self.psi, self.psi, self.nalpha, 0)

        # build bosonic trial
        # self.boson_basis = "Fock"

        # with h5py.File("input.h5", "w") as f:
        #    f["Xmat"] = Xmat
        #    f["xinv"] = xinv
        #    f["trial"] = self.psi

    def half_rotate_integrals(self, h1e, ltensor):
        r""" rotate h1e and ltensor by half

        Shape of rotated ltensor is [nchol, nao, ]
        """

        # ltensor.shape: (nchol, nao, nao)
        logger.info(self, task_title("half rotate integrals ..."))
        t0 = time.time()
        nao = self.psia.shape[0]
        nchol = ltensor.shape[0]
        psia = self.psia.reshape(self._numdets, nao, self.nalpha)
        psib = None if self.ncomponents == 1 else self.psib.reshape(self._numdets, nao, self.nbeta)

        # rh1e and rltensor are both tuple of alpha and beta components
        rotated_h1e, rltensor = half_rotate_integrals(self.ncomponents, psia, psib, h1e, ltensor)
        # logger.debug(self, f"Debug: shape of rotated_h1e {rotated_h1e[0].shape}")
        # logger.debug(self, f"Debug: shape of rotated ltensor {rltensor[0].shape}")

        # remove the determinent index as single-determinent trial only has one CI, no need to store the index
        self.rh1a = rotated_h1e[0][0]
        self.rltensora = rltensor[0][0]
        if self.ncomponents > 1:
            self.rh1b = rotated_h1e[1][0]
            self.rltensorb = rltensor[1][0]
        self.half_rotated = True

        logger.info(self, f"Time to rotate integrals is:    {time.time() - t0: 9.4f}")
        logger.info(self, task_title("half rotate integrals ... Done!"))


    def calc_gf(self, walkers):
        r""" TODO: calculate the green's function with walkers
        """
        #for w in range(walkers.nwalkers):
        #    ovlp = numpy.dot(...
        pass


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

        return calc_trial_walker_ovlp(walkers, self)


    def boson_ovlp_with_walkers(self, walkers):
        sb = None
        # for many-bosons, there should be a permenant
        if walkers.boson_phiw is not None:
            sb = backend.einsum("N, zN->z", self.boson_psi.conj(), walkers.boson_phiw)
        return sb


    def ovlp_with_walkers_gf(self, walkers):
        r"""compute trial_walker overlap and green's function
        """
        return calc_trial_walker_ovlp_gf(walkers, self)


    def get_vbias(self, walkers, ltensor, verbose=False):
        r"""compute the force bias without constructing the big TL_theta tensor

        we construct teh vbias from Ghalf with the rotated ltensors

        \Psi L -> rotated_L
        Since Gf = Psi * Ghalf, vbias = LG = rotated_L * Ghalf

        Vbias shape of [nwalker, nchols]
        """
        ## old code assuming rhf and without using rotated ltensor

        # overlap = self.ovlp_with_walkers(walkers)
        # inv_overlap = backend.linalg.inv(overlap)
        # if verbose:
        #     logger.debug(
        #         self,
        #         "\nnorm of walker overlap: %15.8f",
        #         backend.linalg.norm(overlap),
        #     )
        # Ghalf = backend.einsum("zqp, zpr->zqr", walkers.phiw, inv_overlap)
        # Gf = backend.einsum("zqr, pr->zpq", Ghalf, self.psi.conj())
        # vbias = backend.einsum("npq,zpq->zn", ltensor, Gf)

        # new code using rotated ltensor and Ghalf
        # shape of walkers.Ghalf   : [nwalker, nao, nalpha]
        # shape of rotated ltensor : [nchol, nao, nalpha]
        vbias = (2.0 / self.ncomponents) * backend.einsum("nqi, zqi->zn", self.rltensora, walkers.Ghalfa)
        if self.ncomponents > 1:
            vbias += backend.einsum("nqi, zqi->zn", self.rltensorb, walkers.Ghalfb)
        return vbias # Gf, vbias


    def get_boson_vbias(self, walkers, chols):
        r"""Compute the force bias for bosons"""
        # chols shape: (nfield, nfock, nfock)
        # boson_Gf shape: (nwalkers, nfock, nfock)

        vbias = backend.einsum("nMN, zMN->zn", chols, walkers.boson_Gf)
        return vbias


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
        TODO: eventually use get_vbias instead and avoid using the big TL_theta tensor
        """
        # works for ncomponents = 1
        if self.half_rotated:
            pass

        # here, we use the intermediate Ghalf to improve the efficiency
        overlap = self.ovlp_with_walkers(walkers)
        inv_overlap = backend.linalg.inv(overlap)

        logger.debug(self, "\nnorm of walker overlap: %15.8f", backend.linalg.norm(overlap))
        # theta is the Ghalf
        theta = backend.einsum("zqp, zpr->zqr", walkers.phiw, inv_overlap)

        Gf = backend.einsum("zqr, pr->zpq", theta, self.psi.conj())
        # :math:`(\Psi_T L_{\gamma}) \psi_w (\Psi_T \psi_w)^{-1}`

        # since TL_tehta is too big, we will avoid constructing it in the propagation
        TL_theta = backend.einsum("npq, zqr->znpr", TL_tensor, theta)
        # so TL_tensor * theta = L_tensor * Psi_T * Ghalf = L_tensor * G
        # vbias is L_tensor * G contraction

        # trace[TL_theta] give the force_bias
        # vbias = backend.einsum("znpp->zn", TL_theta)

        # bosonic part
        if self.boson_psi is not None:
            # phi_{w, pi} Psi^T_{pj} --> zij, if i, j is only 1
            # then ovlp_b is phi_{z,F} \Psi^T_{F} -> z
            # ovlp_b_inv = 1/S
            # \phi_{zF} /S_{z,ij}  --> [boson_Ghalf]_{z,Fj}
            #  [boson_Ghalf]_{z,Fj} * Psi_{F'j} -> G_{FF'}
            walkers.boson_ovlp = trial.boson_ovlp_with_walkers(walkers)
            inv_ovlp = 1.0 / walkers.boson_ovlp
            theta = backend.einsum("zN, z->zN", walkers.boson_phiw, inv_ovlp)
            boson_Gf = backend.einsum("zM, N->zMN", theta, self.boson_psi.conj())
            return [Gf, boson_Gf], TL_theta

        return Gf, TL_theta


from openms.qmc import estimators


# single determinant unrestricted HF trial wavefunction
class TrialUHF(TrialWFBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.header = "SD_UHF"
        self.ncomponents = 2

    def build(self):
        overlap = self.mol.intor("int1e_ovlp")  # AO Overlap Matrix, S
        Xmat = lo.orth.lowdin(overlap)  # Eigenvectors of S**(1/2) = X
        xinv = backend.linalg.inv(Xmat)  # S**(-1/2)

        # TODO: name change MO_ALPHA/beta -> psia/b
        MO_ALPHA = self.mf.mo_coeff[0, :, : self.mol.nelec[0]]  # Occupied ALPHA MO Coeffs
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

        # green's function in SO
        # TODO: need to update this using new function
        self.Gf, self.Gf_half = estimators.GF_so(
            self.psi, self.psi, self.nalpha, self.nbeta
        )


    def ovlp_with_walkers(self, walkers):
        r"""Compute the overlap with walkers"""
        super().ovlp_with_walkers(walkers)


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


def get_ci(mol, cas, ci_thresh=1.e-8):
    from pyscf import mcscf, fci

    # TODO: use UHF if openshell system
    mf = scf.RHF(mol)
    e_hf = mf.kernel()

    # casscf
    ncas, neleccas = cas
    mc = mcscf.CASSCF(mf, ncas, neleccas)

    # get fcivec
    nelecasa = (neleccas + mol.spin) // 2
    nelecasb = neleccas - nelecasa
    e_tot, e_cas, fcivec, mo, mo_energy = mc.kernel()

    # TODO, get mo_occ from mc

    logger.debug(mol, f"\nHF     total energy: {e_hf}")
    logger.debug(mol, f"casscf total energy: {e_tot}")
    logger.debug(mol, f"correlation energy:  {e_tot - e_hf}")

    # logger.debug(mol, f"hf.mo_occ ?= cassc.mo_occ = {backend.allclose(mf.mo_occ, mc.mo_occ, rtol=1e-05)} \n")
    # logger.debug(mol, f"cassc.mo_occ   = {mc.mo_occ}")
    # logger.debug(mol, f"cassc.mo_coeff = {mc.mo_coeff}")

    coeff, occa, occb = zip(
        *fci.addons.large_ci(fcivec, ncas, (nelecasa, nelecasb), tol=ci_thresh, return_strs=False)
    )
    # may need to return mc, instead of mf
    return mf, coeff, occa, occb


class multiCI(TrialWFBase):
    r"""Trial WF based on multi CI cofigurations.

    Here, we construct the Trial WF based on PySCF CASSCF calculations

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.header = "MSD_UHF"
        self.ncomponents = 2 # FIXME: decide whether to consider 1 component in this trial class

        self.cas = kwargs.get("cas", None)  # active space (norbital, nelectron)
        self.use_cas = kwargs.get("use_cas", True)
        ci_coeffs, occa, occb = kwargs.get("cas_wfn", (None, None, None))
        self.nchunks = kwargs.get("nchunks", 1)
        self._numdets = kwargs.get("numdets", -1)

        if ci_coeffs is None:
            # build a mcscf calculations and get the values
            if self.cas is None:
                raise RuntimeError(
                    "CAS (ncas, neleccas) must be specified if ci_coefficients are not provided"
                )
            self.mf, ci_coeffs, occa, occb = get_ci(self.mol, self.cas)
        assert len(occa) == len(occb)
        assert len(occa) == len(ci_coeffs)

        if self.nbeta > 0:
            max_orbital = max(backend.max(occa), backend.max(occb)) + 1
        else:
            max_orbital = backend.max(occa) + 1
        logger.debug(self, f"Debug: mo_occ = {self.mf.mo_occ}")
        logger.debug(self, f"Debug: max_orbital is {max_orbital}")

        #
        # update other parameters
        #
        self.max_numdets = len(ci_coeffs)
        if self._numdets < 0: self._numdets = self.max_numdets
        assert self._numdets <= self.max_numdets


        logger.info(self, f"Info: num_dets = {self._numdets:8d}")
        logger.info(self, f"Info: max_num_dets = {self.max_numdets:8d}")
        logger.debug(self, f"Debug: occa0 in cas = {occa}")
        logger.debug(self, f"Debug: occb0 in cas = {occb}")
        # occa/b are:



    def ovlp_with_walkers_gf(self, walkers):
        r"""compute trial_walker overlap and green's function
        """
        return calc_MSD_trial_walker_ovlp_gf(walkers, self)


    def ovlp_with_walkers(self, walkers):
        r"""Compute the overlap between trial and walkers (MDS)
        """

        return calc_MSD_trial_walker_ovlp(walkers, self)


    def get_vbias(self, walkers, ltensor, verbose=False):
        r"""compute the force bias without constructing the big TL_theta tensor

        we construct teh vbias from Ghalf with the rotated ltensors

        \Psi L -> rotated_L
        Since Gf = Psi * Ghalf, vbias = LG = rotated_L * Ghalf

        Vbias shape of [nwalker, nchols]
        """

        # may make multiCi as inheritance of TrialHF instead of TrialWFBase so that
        # it can inherits the TrialHF functions
        vbias = (2.0 / self.ncomponents) * backend.einsum("nqi, zqi->zn", self.rltensora, walkers.Ghalfa)
        if self.ncomponents > 1:
            vbias += backend.einsum("nqi, zqi->zn", self.rltensorb, walkers.Ghalfb)
        return vbias


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
# define joint fermion-boson trial
# =====================================
from openms.qmc.trial_boson import *


class trail_EPH(object):
    def __init__(self, trial_e, trial_b):
        self.trial_e = trial_e
        self.trial_b = trial_b

    def force_bias(self, walkers):
        r"""compute the force bias"""

        pass

    def ovlp_with_walkers(self, walkers):
        r"""compute the trial-walker overlap"""

        pass


def make_trial(mol, mf=None, **kwargs):
    r"""make trial WF according to the options

    in electron-boson case, the trial WF is

    .. math::

        \ket{\Psi_T} = \ket{\Psi^e_T} \otimes \ket{\Psi^b_T}

    instead makeing a big tensor (Nfock * Nspin * Nao * Nao), we use
    two small tensor (Nfock) for boson and (Nspin * Nao * Nao) for electrons
    """

    # TODO: 1) only pass mo_coff, mo_occ, and mol to trial to construc the
    # corresponding trial WF.
    # if MF is none: according to multiplicity and options to choose UHF/ROHF/RHF/MCSCF

    trial = TrialHF(mol, mf=mf, **kwargs)
    trial.build()
    logger.debug(trial, f"Debug: trial WF is {trial.psi}")

    if isinstance(mol, Boson):
        # here, we temporarily append trial WF for boson into the trial
        # set the initial condition according to Z (TBA)
        boson_size = sum(mol.nboson_states)
        trial.boson_psi = backend.zeros(boson_size)
        trial.boson_psi[0] = 1.0 # / backend.sqrt(boson_size)
        # trial.boson_psi[:] = 1.0 / backend.sqrt(boson_size)

        # return trial
    #else:
    #    return trial

    # else:
    #    raise ValueError(f"system type {mol.__class__} is supported!" +
    #                    " A trial function must be provided!")

    return trial


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
    trial.buld()

    cas = (2, 2)
    trial = multiCI(mol, cas=cas)

    # example of using bosonic trial

    # example of using joint fermionic-bosonic trial
