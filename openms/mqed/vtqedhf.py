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

import numpy
from scipy import linalg

from pyscf import dft
from pyscf import lib

import openms
#from openms.lib import logger
from openms.mqed import scqedhf
from openms import __config__

TIGHT_GRAD_CONV_TOL = getattr(__config__, "TIGHT_GRAD_CONV_TOL", True)
LINEAR_DEP_THRESHOLD = getattr(__config__, "LINEAR_DEP_THRESHOLD", 1e-8)
CHOLESKY_THRESHOLD = getattr(__config__, "CHOLESKY_THRESHOLD", 1e-10)
FORCE_PIVOTED_CHOLESKY = getattr(__config__, "FORCE_PIVOTED_CHOLESKY", False)
LINEAR_DEP_TRIGGER = getattr(__config__, "LINEAR_DEP_TRIGGER", 1e-10)

# class VTRHF(scqedhf.SCRHF):
#     r"""Non-relativistic VT-QED-HF subclass."""
#     def __init__(
#         self, mol, qed=None, xc=None, **kwargs):

#         super().__init__(mol, qed, xc, **kwargs)

#         if "vtqedhf" not in openms.runtime_refs:
#             openms.runtime_refs.append("vtqedhf")

#         self.var_grad = numpy.zeros(self.qed.nmodes)
#         self.vhf_dse = None


#     def get_hcore(self, mol=None, dm=None):
#         h1e = super().get_hcore(mol, dm)

#         if dm is not None:
#             self.oei = self.qed.get_dse_hcore(dm, residue=True) # this is zero for scqedhf
#             return h1e + self.oei
#         return h1e


#     def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
#         r"""VTQED Hartree-Fock potential matrix for the given density matrix

#         .. math::
#             V_{eff} = J - K/2 + \bra{i}\lambda\cdot\mu\ket{j}

#         """
#         vhf = super().get_veff(mol, dm, dm_last, vhf_last, hermi)

#         self.qed.update_cs(dm)

#         # two-electron part (residue, to be tested!)
#         vj_dse, vk_dse = self.qed.get_dse_jk(dm, residue=True)
#         self.vhf_dse = vj_dse - 0.5 * vk_dse
#         vhf += self.vhf_dse

#         return vhf


#     def gaussian_derivative_f(self, eta, imode, p, q, r=None, s=None):
#         r"""
#         Compute derivative of FC factor with respect to f

#         .. math::

#             \frac{d G}{\partial f_\alpha} =
#         """

#         diff_eta = (eta[imode, q] - eta[imode, p])
#         if r is not None:
#             diff_eta += (eta[imode, s] - eta[imode, r])
#         tmp = 1.0 #self.qed.couplings_var[imode]
#         if False: # depending on wether eta has sqrt(w/2) factors:
#             tmp = tmp / numpy.sqrt(2.0 * self.qed.omega[imode])
#         else:
#             tmp = tmp / self.qed.omega[imode]
#         derivative = - numpy.exp(-0.5*(tmp*diff_eta)**2) * (tmp * diff_eta) ** 2

#         # in principle, the couplings_var should be > 0.0
#         if self.qed.couplings_var[imode] < -0.05 or self.qed.couplings_var[imode] > 1.1:
#             warnings.warn(f"Warning: Couplings_var should be in [0,1], which is {self.qed.couplings_var[imode]}")
#             #raise ValueError(f"Couplings_var should be in [0,1], which is {self.qed.couplings_var[imode]}")

#         derivative /= self.qed.couplings_var[imode]
#         return derivative


#     def gaussian_derivative_f_vector(self, eta, imode, onebody=True):
#         r"""
#         Compute derivative of FC factor with respect to f

#         .. math::

#             \frac{d G}{\partial f_\alpha} =
#         """
#         nao = self.qed.gmat[imode].shape[0]
#         if onebody:
#             p, q = numpy.ogrid[:nao, :nao]
#             diff_eta = eta[imode, q] - eta[imode, p]
#         else:
#             p, q, r, s = numpy.ogrid[:nao, :nao, :nao, :nao]
#             diff_eta = eta[imode, q] - eta[imode, p] +  eta[imode, s] - eta[imode, r]

#         tmp = 1.0 #self.qed.couplings_var[imode]
#         if False: # depending on wether eta has sqrt(w/2) factors:
#             tmp = tmp / numpy.sqrt(2.0 * self.qed.omega[imode])
#         else:
#             tmp = tmp / self.qed.omega[imode]
#         derivative = - numpy.exp(-0.5*(tmp * diff_eta)**2) * (tmp * diff_eta) ** 2

#         # in principle, the couplings_var should be > 0.0
#         if self.qed.couplings_var[imode] < -0.05 or self.qed.couplings_var[imode] > 1.05:
#             self.logger.warn(f"Warning: Couplings_var should be in [0,1], which is {self.qed.couplings_var[imode]}")
#             #raise ValueError(f"Couplings_var should be in [0,1], which is {self.qed.couplings_var[imode]}")
#         derivative /= self.qed.couplings_var[imode]

#         if onebody:
#             return derivative.reshape(nao, nao)
#         else:
#             return derivative.reshape(nao, nao, nao, nao)


#     def grad_var_params(self, dm_do, g_DO, dm=None):
#         r"""
#         Compute dE/df where f is the variational transformation parameters
#         """
#         # gradient w.r.t eta
#         self.get_eta_gradient(dm_do, g_DO, dm)

#         if not self.qed.optimize_varf: return
#         # gradient w.r.t f_\alpha
#         nao = self.mol.nao_nr()
#         nmodes = self.qed.nmodes

#         onebody_dvar = numpy.zeros(nmodes)
#         twobody_dvar = numpy.zeros(nmodes)

#         for imode in range(nmodes):
#             if abs(self.qed.couplings_var[0]) > 1.e-5:
#                 g2_dot_D = 2.0 * numpy.einsum("pp, p->", dm_do[imode], g_DO[imode]**2)
#                 onebody_dvar[imode] += g2_dot_D / self.qed.omega[imode] / self.qed.couplings_var[imode]

#         # will be replaced with c++ code
#         for imode in range(nmodes):
#             # one-electron part
#             derivative = self.gaussian_derivative_f_vector(self.eta, imode)
#             h_dot_g = self.h1e_DO * derivative # element_wise
#             oei_derivative = numpy.einsum("pq, pq->", h_dot_g, dm_do[imode])
#             tmp = numpy.einsum("pp, p->p", dm_do[imode], g_DO[imode])
#             tmp = 2.0 * numpy.einsum("p,q->", tmp, tmp)
#             tmp -= numpy.einsum("pq, pq, p, q->", dm_do[imode], dm_do[imode], g_DO[imode], g_DO[imode])
#             oei_derivative += tmp / self.qed.omega[imode] / self.qed.couplings_var[imode]

#             onebody_dvar[imode] += oei_derivative

#             # two-electron part
#             derivative = self.gaussian_derivative_f_vector(self.eta, imode, onebody=False)
#             derivative *= (2.0 * self.eri_DO - self.eri_DO.transpose(0, 3, 2, 1))
#             tmp = lib.einsum('pqrs, rs-> pq', derivative, dm_do[imode], optimize=True)
#             tmp = lib.einsum('pq, pq->', tmp, dm_do[imode], optimize=True)
#             twobody_dvar[imode] = tmp/4.0

#         self.var_grad = onebody_dvar + twobody_dvar

#         if abs(1.0 - self.qed.couplings_var[0]) > 1.e-4 and self.vhf_dse is not None:
#             # only works for nmode == 1
#             # E_DSE = (1-f)^2 * original E_DSE,
#             # so the gradient =  -2(1-f) * original E_DSE =  - E_DSE*2/(1-f)
#             self.dse_tot = self.energy_elec(dm, self.oei, self.vhf_dse)[0]
#             #self.dse_tot += self.dse(dm, residue=True) #we moved this into fock
#             self.var_grad[0] -= self.dse_tot * 2.0 / (1.0 - self.qed.couplings_var[0])


#     def norm_var_params(self):
#         var_norm = linalg.norm(self.eta_grad)/numpy.sqrt(self.eta.size)
#         var_norm += linalg.norm(self.var_grad)/numpy.sqrt(self.var_grad.size)
#         return var_norm


#     def update_var_params(self):
#         self.eta -= self.precond * self.eta_grad
#         if self.qed.optimize_varf:
#             #TODO: give user warning to use smaller precond if it diverges
#             self.qed.couplings_var -= self.precond * self.var_grad
#             self.qed.update_couplings()


#     def pre_update_var_params(self):
#         variables = self.eta
#         gradients = self.eta_grad
#         if self.qed.optimize_varf:
#             variables = numpy.hstack([variables.ravel(), self.qed.couplings_var])
#             gradients = numpy.hstack([gradients.ravel(), self.var_grad])
#         return variables, gradients


#     def set_params(self, params, fock_shape=None):

#         fsize = numpy.prod(fock_shape)
#         f = params[:fsize].reshape(fock_shape)
#         etasize = self.eta.size
#         varsize = self.qed.couplings_var.size
#         if params.size > fsize:
#             self.eta = params[fsize:fsize+etasize].reshape(self.eta_grad.shape)
#         if params.size > fsize + etasize:
#             self.qed.couplings_var = params[fsize+etasize:].reshape(self.var_grad.shape)
#             self.qed.update_couplings()
#         return f


# class VTRKS(dft.rks.KohnShamDFT, VTRHF):
#     r"""Template class for QED-VT-RKS. WIP."""
#     def __init__(
#         self, mol, xc="LDA,VWN", **kwargs):
#         raise NotImplementedError("""RKS object is currently unsupported. WIP.""")
#         #VTRHF.__init__(self, mol, **kwargs)
#         #dft.rks.KohnShamDFT.__init__(self, xc)

#     #get_veff = dft.rks.get_veff
#     #get_vsap = dft.rks.get_vsap
#     #energy_elec = dft.rks.energy_elec


# if __name__ == "__main__":
#     import numpy
#     from pyscf import gto

#     itest = 0
#     zshift = itest * 2.0

#     atom = f"C    0.00000000    0.00000000    {zshift};\
#              O    0.00000000    1.23456800    {zshift};\
#              H    0.97075033   -0.54577032    {zshift};\
#              C   -1.21509881   -0.80991169    {zshift};\
#              H   -1.15288176   -1.89931439    {zshift};\
#              C   -2.43440063   -0.19144555    {zshift};\
#              H   -3.37262777   -0.75937214    {zshift};\
#              O   -2.62194056    1.12501165    {zshift};\
#              H   -1.71446384    1.51627790    {zshift}"

#     mol = gto.M(
#         atom = atom,
#         basis="sto3g",
#         #basis="cc-pvdz",
#         unit="Angstrom",
#         symmetry=True,
#         verbose=1)

#     nmode = 1
#     cavity_freq = numpy.zeros(nmode)
#     cavity_freq[0] = 0.5
#     cavity_mode = numpy.zeros((nmode, 3))
#     cavity_mode[0, :] = 0.1 * numpy.asarray([0, 0, 1])

#     qedmf = VTRHF(mol, omega=cavity_freq, vec=cavity_mode)
#     qedmf.max_cycle = 500
#     qedmf.kernel()
#     print(f"Electronic energy = {qedmf.e_tot}")


class VTRHF(scqedhf.SCRHF):
    r"""Non-relativistic VT-QED-HF subclass."""
    def __init__(
        self, mol, qed=None, xc=None, **kwargs):

        super().__init__(mol, qed, xc, **kwargs)

        if "vtqedhf" not in openms.runtime_refs:
            openms.runtime_refs.append("vtqedhf")

        self.var_grad = numpy.zeros(self.qed.nmodes)
        self.vhf_dse = None


    def get_hcore(self, mol=None, dm=None, dress=False):
        h1e = super().get_hcore(mol, dm, dress)

        if dm is not None:
            self.oei = self.qed.get_dse_hcore(dm, residue=True) # this is zero for scqedhf
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
        vj_dse, vk_dse = self.qed.get_dse_jk(dm, residue=True)
        self.vhf_dse = vj_dse - 0.5 * vk_dse
        vhf += self.vhf_dse

        return vhf

    def gaussian_derivative_f(self, eta, imode, p, q, r=None, s=None):
        r"""
        Compute derivative of FC factor with respect to f

        .. math::

            \frac{d G}{\partial f_\alpha} =
        """

        diff_eta = (eta[imode, q] - eta[imode, p])
        if r is not None:
            diff_eta += (eta[imode, s] - eta[imode, r])
        tmp = 1.0 #self.qed.couplings_var[imode]
        if False: # depending on wether eta has sqrt(w/2) factors:
            tmp = tmp / numpy.sqrt(2.0 * self.qed.omega[imode])
        else:
            tmp = tmp / self.qed.omega[imode]
        derivative = - numpy.exp(-0.5*(tmp*diff_eta)**2) * (tmp * diff_eta) ** 2

        # in principle, the couplings_var should be > 0.0
        if self.qed.couplings_var[imode] < -0.05 or self.qed.couplings_var[imode] > 1.1:
            warnings.warn(f"Warning: Couplings_var should be in [0,1], which is {self.qed.couplings_var[imode]}")
            #raise ValueError(f"Couplings_var should be in [0,1], which is {self.qed.couplings_var[imode]}")

        derivative /= self.qed.couplings_var[imode]
        return derivative

    def gaussian_derivative_f_vector(self, eta, imode, onebody=True):
        r"""
        Compute derivative of FC factor with respect to f

        .. math::

            \frac{d G}{\partial f_\alpha} =
        """
        nao = self.qed.gmat[imode].shape[0]
        if onebody:
            p, q = numpy.ogrid[:nao, :nao]
            diff_eta = eta[imode, q] - eta[imode, p]
        else:
            p, q, r, s = numpy.ogrid[:nao, :nao, :nao, :nao]
            diff_eta = eta[imode, q] - eta[imode, p] +  eta[imode, s] - eta[imode, r]

        tmp = 1.0 #self.qed.couplings_var[imode]
        if False: # depending on wether eta has sqrt(w/2) factors:
            tmp = tmp / numpy.sqrt(2.0 * self.qed.omega[imode])
        else:
            tmp = tmp / self.qed.omega[imode]
        derivative = - numpy.exp(-0.5*(tmp * diff_eta)**2) * (tmp * diff_eta) ** 2

        # in principle, the couplings_var should be > 0.0
        if self.qed.couplings_var[imode] < -0.05 or self.qed.couplings_var[imode] > 1.05:
            warnings.warn(f"Warning: Couplings_var should be in [0,1], which is {self.qed.couplings_var[imode]}")
            #raise ValueError(f"Couplings_var should be in [0,1], which is {self.qed.couplings_var[imode]}")
        derivative /= self.qed.couplings_var[imode]

        if onebody:
            return derivative.reshape(nao, nao)
        else:
            return  derivative.reshape(nao, nao, nao, nao)

    def get_var_gradient(self, dm_do, g_DO, dm=None):
        r"""
        Compute dE/df where f is the variational transformation parameters
        """
        # gradient w.r.t eta
        self.get_eta_gradient(dm_do, g_DO, dm)

        if not self.qed.optimize_varf: return
        # gradient w.r.t f_\alpha
        nao = self.mol.nao_nr()
        nmodes = self.qed.nmodes

        onebody_dvar = numpy.zeros(nmodes)
        twobody_dvar = numpy.zeros(nmodes)

        for imode in range(nmodes):
            if abs(self.qed.couplings_var[0]) > 1.e-5:
                g2_dot_D = 2.0 * numpy.einsum("pp, p->", dm_do[imode], g_DO[imode]**2)
                onebody_dvar[imode] += g2_dot_D / self.qed.omega[imode] / self.qed.couplings_var[imode]

        # will be replaced with c++ code
        for imode in range(nmodes):
            # one-electron part
            derivative = self.gaussian_derivative_f_vector(self.eta, imode)
            h_dot_g = self.h1e_DO * derivative # element_wise
            oei_derivative = numpy.einsum("pq, pq->", h_dot_g, dm_do[imode])
            tmp = numpy.einsum("pp, p->p", dm_do[imode], g_DO[imode])
            tmp = 2.0 * numpy.einsum("p,q->", tmp, tmp)
            tmp -= numpy.einsum("pq, pq, p, q->", dm_do[imode], dm_do[imode], g_DO[imode], g_DO[imode])
            oei_derivative += tmp / self.qed.omega[imode] / self.qed.couplings_var[imode]

            onebody_dvar[imode] += oei_derivative

            # two-electron part
            derivative = self.gaussian_derivative_f_vector(self.eta, imode, onebody=False)
            derivative *= (2.0 * self.eri_DO - self.eri_DO.transpose(0, 3, 2, 1))
            tmp = lib.einsum('pqrs, rs-> pq', derivative, dm_do[imode], optimize=True)
            tmp = lib.einsum('pq, pq->', tmp, dm_do[imode], optimize=True)
            twobody_dvar[imode] = tmp/4.0

        self.var_grad = onebody_dvar + twobody_dvar

        if abs(1.0 - self.qed.couplings_var[0]) > 1.e-4 and self.vhf_dse is not None:
            # only works for nmode == 1
            # E_DSE = (1-f)^2 * original E_DSE,
            # so the gradient =  -2(1-f) * original E_DSE =  - E_DSE*2/(1-f)
            self.dse_tot = self.energy_elec(dm, self.oei, self.vhf_dse)[0]
            #self.dse_tot += self.dse(dm, residue=True) #we moved this into fock
            self.var_grad[0] -= self.dse_tot * 2.0 / (1.0 - self.qed.couplings_var[0])

    def get_var_norm(self):
        var_norm = linalg.norm(self.eta_grad)/numpy.sqrt(self.eta.size)
        var_norm += linalg.norm(self.var_grad)/numpy.sqrt(self.var_grad.size)
        return var_norm

    def update_variational_params(self):
        self.eta -= self.precond * self.eta_grad
        if self.qed.optimize_varf:
            #TODO: give user warning to use smaller precond if it diverges
            self.qed.couplings_var -= self.precond * self.var_grad
            self.qed.update_couplings()

    def pre_update_params(self):
        variables = self.eta
        gradients = self.eta_grad
        if self.qed.optimize_varf:
            variables = numpy.hstack([variables.ravel(), self.qed.couplings_var])
            gradients = numpy.hstack([gradients.ravel(), self.var_grad])
        return variables, gradients

    def set_params(self, params, fock_shape=None):

        fsize = numpy.prod(fock_shape)
        f = params[:fsize].reshape(fock_shape)
        etasize = self.eta.size
        varsize = self.qed.couplings_var.size
        if params.size > fsize:
            self.eta = params[fsize:fsize+etasize].reshape(self.eta_grad.shape)
        if params.size > fsize + etasize:
            self.qed.couplings_var = params[fsize+etasize:].reshape(self.var_grad.shape)
            self.qed.update_couplings()
        return f



if __name__ == "__main__":
    import numpy
    from pyscf import gto, scf

    itest = -2
    zshift = itest * 2.5
    print(f"zshift={zshift}")

    atom = """H 0 0 0; F 0 0 1.75202"""
    atom = f"H          0.86681        0.60144        {5.00000+zshift};\
        F         -0.86681        0.60144        {5.00000+zshift};\
        O          0.00000       -0.07579        {5.00000+zshift};\
        He         0.00000        0.00000        {7.50000+zshift}"

    mol = gto.M(
        atom=atom,
        basis="sto3g",
        #basis="cc-pvdz",
        unit="Angstrom",
        symmetry=True,
        verbose=3,
    )
    print("mol coordinates=\n", mol.atom_coords())

    """
    hf = scf.HF(mol)
    hf.max_cycle = 200
    hf.conv_tol = 1.0e-8
    hf.diis_space = 10
    hf.polariton = True
    mf = hf.run(verbose=4)

    print("electronic energies=", mf.energy_elec())
    print("nuclear energy=     ", mf.energy_nuc())
    dm = mf.make_rdm1()
    """

    print(
        "\n=========== self-consistent SC-QED-HF calculation  ======================\n"
    )

    from openms.mqed import vtqedhf as qedhf

    nmode = 1
    cavity_freq = numpy.zeros(nmode)
    cavity_mode = numpy.zeros((nmode, 3))
    cavity_freq[0] = 0.5
    cavity_mode[0, :] = 1.e-1 * numpy.asarray([0, 0, 1])

    mol.verbose = 4

    qedmf = qedhf.RHF(mol, xc=None, cavity_mode=cavity_mode, cavity_freq=cavity_freq, add_nuc_dipole=True)
    qedmf.max_cycle = 500
    qedmf.verbose = 5
    qedmf.init_guess ="hcore"
    qedmf.kernel()
