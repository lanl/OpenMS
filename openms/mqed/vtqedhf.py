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

r"""
Theoretical background of VT-QEDHF methods :cite:`Li:2023qedhf`

VT-QEDHF method
^^^^^^^^^^^^^^^

Transformation is:

.. math::

   U(f) = \exp\left[-\frac{f_\alpha\boldsymbol{\lambda}_\alpha\cdot\boldsymbol{D}}{\sqrt{2\omega}}(b-b^\dagger)\right].

where :math:`f_\alpha\in [0, 1]`. With :math:`U`, the transformed QED Hamiltonian
becomes

.. math::

    \mathcal{H} = & \mathcal{H}_e + (1-f_\alpha)\sqrt{\frac{\omega_\alpha}{2}}\boldsymbol{\lambda}_\alpha\cdot\boldsymbol{D} (b+b^\dagger) \\
                  & + \frac{1}{2} (1-f_\alpha)^2 \left(\boldsymbol{\lambda}_\alpha\cdot\boldsymbol{D}\right)^2.

Where the dressed electronic Hamiltonian is

.. math::

    \mathcal{H}_e =  \tilde{h}_{\mu\nu}\hat{E}_{\mu\nu} + \tilde{I}_{\mu\nu\lambda\sigma}\hat{e}_{\mu\nu\lambda\sigma}.

and

.. math::

    \tilde{h}_{\mu\nu}&=\sum_{\mu'\nu} h_{\mu'\nu'}X^\dagger_{\mu\mu'}X_{\nu\nu'} \\
    \tilde{I}_{\mu\nu\lambda\sigma}&= \sum_{\mu'\nu'\lambda'\sigma'}X^\dagger_{\mu\mu'}X^\dagger_{\nu\nu'}I_{\mu'\nu'\lambda'\sigma'} X_{\lambda\lambda'}X_{\sigma\sigma'}.

And

.. math::

    X_{\mu\nu} = \exp\left[-\sum_{\alpha}\frac{f_\alpha}{\sqrt{2\omega_\alpha}} \boldsymbol{\lambda}_\alpha\cdot\boldsymbol{D} (\hat{a}^\dagger_\alpha - \hat{a}_\alpha) \right]|_{\mu\nu}.

Then we derive the QEDHF functional and Fock matrix accordingly based on the HF ansatz.


VSQ-QEDHF method
^^^^^^^^^^^^^^^^

Details TBA.


"""

import numpy
from scipy import linalg
import warnings

from pyscf import lib
from pyscf.lib import logger

import openms
from openms.mqed import scqedhf
from openms import __config__

TIGHT_GRAD_CONV_TOL = getattr(__config__, "TIGHT_GRAD_CONV_TOL", True)
LINEAR_DEP_THRESHOLD = getattr(__config__, "LINEAR_DEP_THRESHOLD", 1e-8)
CHOLESKY_THRESHOLD = getattr(__config__, "CHOLESKY_THRESHOLD", 1e-10)
FORCE_PIVOTED_CHOLESKY = getattr(__config__, "FORCE_PIVOTED_CHOLESKY", False)
LINEAR_DEP_TRIGGER = getattr(__config__, "LINEAR_DEP_TRIGGER", 1e-10)


def newton_update(var, gradient, precond0):
    r"""TBA"""
    var -= precond0 * gradient
    return var


def entropy(rho):
    r"""Compute entropy for given density matrix

    .. math::

        S = - K_B Tr[\rho \text{ln}(\rho)]
    """
    e, _ = linalg.eigh(rho)
    e = e[e > 0]
    tmp = e * numpy.log(e)
    return -1.0 * numpy.sum(tmp)



class RHF(scqedhf.RHF):
    r"""Non-relativistic VT-QED-HF subclass."""

    def __init__(self, mol, **kwargs):

        super().__init__(mol, **kwargs)

        # print headers
        logger.info(self, openms.__logo__)

        if "vtqedhf" not in openms.runtime_refs:
            openms.runtime_refs.append("vtqedhf")

        self.vlf_grad = numpy.zeros(self.qed.nmodes)
        self.vsq_grad = numpy.zeros(self.qed.nmodes)

        if "couplings_var" in kwargs:
            self.qed.couplings_var = numpy.asarray(kwargs["couplings_var"])
            self.qed.optimize_varf = False
        else:
            self.qed.optimize_varf = True
            self.qed.couplings_var = 0.5 * numpy.ones(self.qed.nmodes)
        if "squeezed_cs" in kwargs:
            self.qed.squeezed_cs = kwargs["squeezed_cs"]

        self.qed.update_couplings()
        self.vhf_dse = None
        self.grad_vhf_dse = None


    def get_hcore(self, mol=None, dm=None, dress=True):

        h1e = super().get_hcore(mol, dm, dress)

        if dm is not None:
            # this is zero for scqedhf
            # self.oei = self.qed.get_dse_hcore(dm, residue=True) # this is zero for scqedhf
            self.oei, self.grad_oei = self.qed.add_oei_ao(dm, residue=True, compute_grad=True)

            return h1e + self.oei

        return h1e


    def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        r"""VTQED Hartree-Fock potential matrix for the given density matrix

        .. math::
            V_{eff} = J - K/2 + \bra{i}\lambda\cdot\mu\ket{j}

        """
        vhf = super().get_veff(mol, dm, dm_last, vhf_last, hermi)

        if self.qed.use_cs:
            self.qed.update_cs(dm)

        # two-electron part (residue, to be tested!)
        # vj_dse, vk_dse = self.qed.get_dse_jk(dm, residue=True)
        vj_dse, vk_dse, self.grad_vhf_dse = self.get_dse_jk(dm, residue=True, compute_grad=True)

        self.vhf_dse = vj_dse - 0.5 * vk_dse
        vhf += self.vhf_dse

        return vhf

    def gaussian_derivative_sq_vector(self, eta, imode, onebody=True):
        r"""
        Compute derivative of FC factor with respect to F (squeezing)

        .. math::

            \frac{d G}{\partial f_\alpha} =
        """
        nao = self.qed.gmat[imode].shape[0]
        if onebody:
            p, q = numpy.ogrid[:nao, :nao]
            diff_eta = eta[imode, q] - eta[imode, p]
        else:
            p, q, r, s = numpy.ogrid[:nao, :nao, :nao, :nao]
            diff_eta = eta[imode, q] - eta[imode, p] + eta[imode, s] - eta[imode, r]

        tau = numpy.exp(self.qed.squeezed_var[imode])
        tmp = tau / self.qed.omega[imode]

        derivative = -numpy.exp((-0.5 * (tmp * diff_eta) ** 2)) \
                     * ((tmp * diff_eta) ** 2)

        if onebody:
            return derivative.reshape(nao, nao)
        else:
            return derivative.reshape(nao, nao, nao, nao)


    def gaussian_derivative_f_vector(self, eta, imode, onebody=True):
        r"""
        Compute derivative of FC factor with respect to f

        .. math::

            \frac{d G}{\partial f_\alpha} =
        """

        nao = self.qed.gmat[imode].shape[0]
        # Number of boson states
        mdim = self.qed.nboson_states[imode]

        if onebody:
            p, q = numpy.ogrid[:self.nao, :self.nao]
            diff_eta = eta[imode, p] - eta[imode, q]
        else:
            p, q, r, s = numpy.ogrid[:self.nao, :self.nao, :self.nao, :self.nao]
            diff_eta = eta[imode, p] - eta[imode, q] + eta[imode, r] - eta[imode, s]

        tau = numpy.exp(self.qed.squeezed_var[imode])
        tmp = tau / self.qed.omega[imode]

        # Displacement operator derivative
        if mdim > 1:
            idx = sum(self.qed.nboson_states[:imode])
            ci = self.qed.boson_coeff[idx : idx + mdim, idx]
            pdm = numpy.outer(numpy.conj(ci), ci)

            derivative = self.qed.displacement_deriv_vt(imode, tmp * diff_eta, pdm)

        # Apply vacuum derivative formula
        else:
            derivative = -numpy.exp((-0.5 * (tmp * diff_eta) ** 2)) \
                          * ((tmp * diff_eta) ** 2)

        # in principle, the couplings_var should be > 0.0
        if self.qed.couplings_var[imode] < -0.05 or self.qed.couplings_var[imode] > 1.05:
            logger.warn(self, f"Warning: Couplings_var should be in [0,1], which is {self.qed.couplings_var[imode]}")

        derivative /= self.qed.couplings_var[imode]

        if onebody:
            return derivative.reshape(self.nao, self.nao)
        else:
            return derivative.reshape(self.nao, self.nao, self.nao, self.nao)


    def get_vsq_gradient(self, dm_do, g_DO, dm=None):
        r"""
        Compute dE/dF where F is the variational squeezed parameters
        See equations in Ref. :cite:`Mazin:2024lm`.

        """

        # gradient w.r.t f_\alpha
        nao = self.mol.nao_nr()
        nmodes = self.qed.nmodes

        onebody_dvsq = numpy.zeros(nmodes)
        twobody_dvsq = numpy.zeros(nmodes)

        for imode in range(nmodes):
            tau = numpy.exp(self.qed.squeezed_var[imode])
            if abs(self.qed.squeezed_var[0]) > 1.0e-5:
                g2_dot_D = 2.0 * numpy.einsum("pp, p->", dm_do[imode], g_DO[imode]**2)

        # will be replaced with c++ code
        for imode in range(nmodes):
            tau = numpy.exp(self.qed.squeezed_var[imode])
            # one-electron part
            derivative = self.gaussian_derivative_sq_vector(self.eta, imode)
            h_dot_g = self.h1e_DO * derivative # element_wise
            oei_derivative = numpy.einsum("pq, pq->", h_dot_g, dm_do[imode])

            # one-electron part, diaognal part
            tmp = numpy.einsum("pp, p->p", dm_do[imode], g_DO[imode])
            tmp = 2.0 * numpy.einsum("p,q->", tmp, tmp)
            tmp -= numpy.einsum("pq, pq, p, q->", dm_do[imode], dm_do[imode], g_DO[imode], g_DO[imode])
            # oei_derivative += tmp / self.qed.omega[imode]

            onebody_dvsq[imode] += oei_derivative

            # two-electron part
            derivative = self.gaussian_derivative_sq_vector(self.eta, imode, onebody=False)
            derivative *= (2.0 * self.eri_DO - self.eri_DO.transpose(0, 3, 2, 1))
            tmp = lib.einsum('pqrs, rs-> pq', derivative, dm_do[imode], optimize=True)
            tmp = lib.einsum('pq, pq->', tmp, dm_do[imode], optimize=True)
            twobody_dvsq[imode] = tmp / 4.0

        self.vsq_grad = onebody_dvsq + twobody_dvsq

        # photon energy part:
        self.vsq_grad += self.qed.e_boson_grad_r


    def grad_var_params(self, dm_do, g_DO, dm=None):
        r"""Compute dE/df where f is the variational transformation parameters.

        :math:`\eta` here is the eigenvalue of :math:`\lambda \sqrt{\omega_\alpha/2} (\boldsymbol{d}\cdot \boldsymbol{e}_\alpha)`,
        not just the eigenvalue of :math:`\boldsymbol{d}\cdot \boldsymbol{e}_\alpha`.

        Define :math:`\eta_p` as eigenvalue of :math:`\boldsymbol{d}\cdot \boldsymbol{e}_\alpha'
        and :math:`\tilde{\eta}_p` as eigenvalue of :math:`\lambda \sqrt{\omega_\alpha/2} (\boldsymbol{d}\cdot \boldsymbol{e}_\alpha)`,
        then:

        .. math::

            \tilde{\eta}_p = \eta_p  \sqrt{\omega_\alpha/2}.

        And the Gaussian factor is:

        .. math::

            \exp(-\lambda^2(\eta_p - \eta_q)^2/4\omega) =
            \exp[ -1/(2\omega^2_\alpha) (\tilde{\eta}_p - \tilde{\eta}_q)^2 ].

        """

        # gradient w.r.t eta
        self.get_eta_gradient(dm_do, g_DO, dm)

        if self.qed.optimize_vsq:
            self.get_vsq_gradient(dm_do, g_DO, dm)

        if not self.qed.optimize_varf:
            return

        # gradient w.r.t f_\alpha
        nmodes = self.qed.nmodes
        onebody_dvlf = numpy.zeros(nmodes)
        twobody_dvlf = numpy.zeros(nmodes)

        for a in range(nmodes):

            tau = numpy.exp(self.qed.squeezed_var[a])
            if abs(self.qed.couplings_var[a]) > 1.e-5:
                g2_dot_D = 2.0 * numpy.einsum("pp, p->", dm_do[a], g_DO[a]**2)
                onebody_dvlf[a] += g2_dot_D / self.qed.omega[a] / self.qed.couplings_var[a]

            # one-electron part
            derivative = self.gaussian_derivative_f_vector(self.eta, a)

            h_dot_g = self.h1e_DO * derivative  # element_wise
            oei_derivative = numpy.einsum("pq, pq->", h_dot_g, dm_do[a])
            tmp = numpy.einsum("pp, p->p", dm_do[a], g_DO[a])
            tmp = 2.0 * numpy.einsum("p,q->", tmp, tmp)
            tmp -= numpy.einsum(
                "pq, pq, p, q->", dm_do[a], dm_do[a], g_DO[a], g_DO[a]
            )
            oei_derivative += tmp / self.qed.omega[a] / self.qed.couplings_var[a]

            onebody_dvlf[a] += oei_derivative

            # two-electron part
            derivative = self.gaussian_derivative_f_vector(self.eta, a, onebody=False)
            derivative *= (2.0 * self.eri_DO - self.eri_DO.transpose(0, 3, 2, 1))

            tmp = lib.einsum("pqrs, rs-> pq", derivative, dm_do[a], optimize=True)
            tmp = lib.einsum("pq, pq->", tmp, dm_do[a], optimize=True)
            twobody_dvlf[a] = tmp / 4.0

            self.vlf_grad[a] = onebody_dvlf[a] + twobody_dvlf[a]

        if abs(1.0 - self.qed.couplings_var[0]) > 1.0e-4 and self.vhf_dse is not None:
            # only works for nmode == 1
            ## E_DSE = (1-f)^2 * original E_DSE,
            ## so the gradient =  -2(1-f) * original E_DSE =  - E_DSE*2/(1-f)
            # old code
            # self.dse_tot = self.energy_elec(dm, self.oei, self.vhf_dse)[0]
            # self.vlf_grad[0] -= self.dse_tot * 2.0 / (1.0 - self.qed.couplings_var[0])

            #  Now, we compute the gradient from add_oei and get_dse_jk
            self.vlf_grad[0] += self.energy_elec(dm, self.grad_oei, self.grad_vhf_dse)[0]
        return self


    def norm_var_params(self):
        var_norm = linalg.norm(self.eta_grad) / numpy.sqrt(self.eta.size)
        var_norm += linalg.norm(self.vlf_grad) / numpy.sqrt(self.vlf_grad.size)
        var_norm += linalg.norm(self.vsq_grad) / numpy.sqrt(self.vsq_grad.size)
        return var_norm


    def update_var_params(self):

        self.eta -= self.precond * self.eta_grad
        if self.qed.optimize_varf:
            #TODO: give user warning to use smaller precond if it diverges
            self.qed.couplings_var -= self.precond * self.vlf_grad
            self.qed.update_couplings()
        if self.qed.optimize_vsq:
            self.qed.squeezed_var -= self.precond * self.vsq_grad


    def pre_update_var_params(self):
        variables = self.eta
        gradients = self.eta_grad

        if self.qed.optimize_varf:
            variables = numpy.hstack([variables.ravel(), self.qed.couplings_var])
            gradients = numpy.hstack([gradients.ravel(), self.vlf_grad])
        if self.qed.optimize_vsq:
            variables = numpy.hstack([variables.ravel(), self.qed.squeezed_var])
            gradients = numpy.hstack([gradients.ravel(), self.vsq_grad])

        return variables, gradients


    def set_var_params(self, params):
        r"""set the additional variaitonal params"""

        params = numpy.hstack([p.ravel() for p in params])

        fsize = 0
        etasize = self.eta.size
        if params.size > fsize:
            self.eta = params[fsize:fsize+etasize].reshape(self.eta_grad.shape)
        fsize += etasize

        varsize = self.qed.couplings_var.size
        if params.size > fsize and self.qed.optimize_varf:
            self.qed.couplings_var = params[fsize : fsize + varsize].reshape(
                self.vlf_grad.shape
            )
            self.qed.update_couplings()

        fsize += varsize
        if params.size > fsize and self.qed.optimize_vsq:
            self.qed.squeezed_var = params[fsize :].reshape(
                self.vsq_grad.shape
            )


    def set_params(self, params, fock_shape=None):
        r""" get size of the variational parameters
        """

        fsize = numpy.prod(fock_shape)
        f = params[:fsize].reshape(fock_shape)
        etasize = self.eta.size
        varsize = self.qed.couplings_var.size
        if params.size > fsize:
            self.eta = params[fsize : fsize + etasize].reshape(self.eta_grad.shape)

        fsize += etasize
        if self.qed.optimize_varf and params.size > fsize:
            self.qed.couplings_var = params[fsize : fsize + varsize].reshape(
                self.vlf_grad.shape
            )
            self.qed.update_couplings()

        fsize += varsize
        if self.qed.optimize_vsq and params.size > fsize:
            self.qed.squeezed_var = params[fsize :].reshape(
                self.vsq_grad.shape
            )
        return f

    def make_boson_nocc_org(self, nfock=10):
        r"""boson occ in original fock representation"""
        import math

        _, _, rho_b = self.make_rdm1_org(self.mo_coeff, self.mo_occ, nfock=nfock)
        nocc = numpy.dot(numpy.diagonal(rho_b), numpy.array(range(nfock)))
        print("nocc =", nocc)
        return nocc


    def make_rdm1_org(self, mo_coeff, mo_occ, nfock=2, **kwargs):
        r"""One-particle density matrix in original AO-Fock representation

        .. math::

            \ket{\Phi} = & \hat{U}(\hat{f}, F)\ket{HF}\otimes \ket{0}
                 = \exp[-\frac{f_\alpha \lambda\cdot D}{\sqrt{2\omega}}]
                 \sum_\mu c_\mu \ket{\mu, 0} \\
                 = & \sum_\mu c_\mu \exp[-g(b-b^\dagger)] \ket{\mu} \otimes \ket{0} \\
                 = & \sum_\mu c_\mu U^\dagger\exp[-\tilde{g}(b-b^\dagger)]U \ket{\mu}\otimes\ket{0}

        where :math:`\tilde{g}` is the diagonal matrix.
        It is obvious that the VT-QEDHF WF is no longer the single produc state

        """
        import math

        mocc = mo_coeff[:,mo_occ>0]
        rho = (mocc*mo_occ[mo_occ>0]).dot(mocc.conj().T)

        nao = rho.shape[0]
        imode = 0

        U = self.ao2dipole[imode]
        Uinv = linalg.inv(U)
        # transform into Dipole
        rho_DO = scqedhf.unitary_transform(U, rho)

        tau = numpy.exp(self.qed.squeezed_var[imode])
        rho_tot = numpy.zeros((nfock, nao, nfock, nao))
        for m in range(nfock):
            for n in range(nfock):
                # <m | D(z_alpha) |0>
                zalpha = tau * self.qed.couplings_var[imode] * self.eta[imode]
                zalpha /= self.qed.omega[imode]

                z0 = numpy.exp(-0.5 * zalpha ** 2)
                zm = z0 * zalpha ** m * numpy.sqrt(math.factorial(m))
                # zn = z0 * (-zalpha) ** n * numpy.sqrt(math.factorial(n))
                zn = z0 * (zalpha) ** n * numpy.sqrt(math.factorial(n))

                rho_tmp = rho_DO * numpy.outer(zm, zn)
                # back to AO
                rho_tmp = scqedhf.unitary_transform(Uinv, rho_tmp)
                rho_tot[m, :, n, :] = rho_tmp

        rho_e = numpy.einsum("mpmq->pq", rho_tot)
        rho_b = numpy.einsum("mpnp->mn", rho_tot) / numpy.trace(rho_e)

        return rho_tot, rho_e, rho_b


if __name__ == "__main__":
    import numpy
    from pyscf import gto

    atom = """
        H          0.86681        0.60144       0.00000
        F         -0.86681        0.60144       0.00000
        O          0.00000       -0.07579       0.00000
        He         0.00000        0.00000       2.50000
        """

    mol = gto.M(atom = atom,
                basis = "sto-3g",
                unit = "Angstrom",
                symmetry = True,
                verbose = 3)

    nmode = 1
    cavity_freq = numpy.zeros(nmode)
    cavity_mode = numpy.zeros((nmode, 3))
    cavity_freq[0] = 0.5
    cavity_mode[0, :] = 0.1 * numpy.asarray([0, 0, 1])

    qedmf = RHF(mol, omega = cavity_freq, vec = cavity_mode)
    qedmf.max_cycle = 500
    qedmf.kernel()
