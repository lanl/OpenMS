from openms.lib.boson import Photon
from openms.qmc import qmc


class BAFQMC(qmc.QMCBase):
    r"""Bosonic AFQMC class"""
    def __init__(self, system, *args, **kwargs):
        super().__init__(system, *args, **kwargs)
        self.exp_h1b = None

    def dump_flags(self):
        r"""
        Dump flags
        """
        print(f"\n========  Bosonic AFQMC simulation using OpenMS package ========\n")

    def build_propagator(self, h1b, h2b, ltensor):
        r"""Pre-compute the propagators"""
        warnings.warn(
            "\nThis 'build_propagator' function in afqmc is deprecated"
            + "\nand will be removed in a future version. "
            "Please use the 'propagators.build()' function instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        # shifted h1b
        shifted_h1b = backend.zeros(h1b.shape)
        rho_mf = self.trial.psi.dot(self.trial.psi.T.conj())
        self.mf_shift = 1j * backend.einsum("npq,pq->n", ltensor, rho_mf)

        traceV = backend.einsum('npr,nrq->pq', ltensor.conj(), ltensor)
        shifted_h1b = h1b - 0.5 * traceV
        shifted_h1b -= backend.einsum("n, npq->pq", self.mf_shift, 1j * ltensor)

        self.TL_tensor = backend.einsum("pr, npq->nrq", self.trial.psi.conj(), ltensor)
        self.exp_h1b = scipy.linalg.expm(-self.dt / 2.0 * shifted_h1b)
        logger.debug(
            self, "norm of shifted_h1b: %15.8f", backend.linalg.norm(shifted_h1b)
        )
        logger.debug(
            self, "norm of TL_tensor:   %15.8f", backend.linalg.norm(self.TL_tensor)
        )


    def propagation_onebody(self, phi_w):
        r"""Propgate one-body term"""
        return backend.einsum("pq, zqr->zpr", self.exp_h1b, phi_w)

    def propagation_twobody(self, vbias, phi_w):
        r"""Propgate two-body term"""
        pass

    def propagate_walkers(self, walkers, xbar, ltensor):
        r"""propagate the walkers"""
        # step 1) half-step one-body term
        # self.psi_w = self.propagation_onebody(psi_w)

        # step 2) two-body term
        # self.psi_w = self.propagation_twobody(psi_w)

        # step 3) half-step one-body term
        # self.psi_w = self.propagation_onebody(psi_w)
        pass

if __name__ == "__main__":
    num_walkers = 500
