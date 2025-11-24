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

AFQMC back-propagation (BP) estimators
======================================

For a given length back propagation :math:`L_{BP}`, BP reconstructs the left state by
back‑propagating the trial determinant :math:`\ket{T}` with the adjoint propagators,

.. math::
    \ket{L} = B^\dagger_{n} B^\dagger_{n-1} \cdots B^\dagger_{n-L_{BP}+1} \ket{T},

with periodic re‑orthonormalization. Then the BP Green’s function is

.. math::
    G = R(L^\dagger R)^{-1} L^\dagger

This function includes:
 - A back-propagation buffer (stores last :math:`L_{BP}` auxiliary fields)
 - A routine to build the left Slater matrix :math:`\ket{L}`
 - A routine to compute the BP Green's function :math:`G`

"""

import numpy as np
import numpy as backend

from dataclasses import dataclass, field
from typing import Deque, List, Optional
from collections import deque
import math

Array = np.ndarray

def thin_qr(mat: Array) -> Array:
    """Return Q from a thin QR of mat (Nbasis x Ne)."""
    Q, _ = np.linalg.qr(mat, mode='reduced')
    return Q


def left_apply_onebody(L, exp_h1e):
    """
    L: (nwalkers, nocc, nao)
    exp_h1e: (nao, nao)
    returns: L @ exp_h1e
    """
    # einsum: w q n , n p  -> w q p
    return backend.einsum("wqn,np->wqp", L, exp_h1e)

def left_apply_onebody_uhf(La, Lb, exp_h1e_a, exp_h1e_b):
    La = left_apply_onebody(La, exp_h1e_a)
    Lb = left_apply_onebody(Lb, exp_h1e_b)
    return La, Lb

def left_apply_HS_taylor(L, eri_op, taylor_order):
    """
    Right-apply exp(eri_op) to the bra:
    L <- L @ exp(eri_op)  using the same Taylor series as forward.
    L: (nwalkers, nocc, nao)
    eri_op: (nwalkers, nao, nao)
    """
    out = L.copy()
    term = L.copy()  # term = L @ A^n / n!
    for n in range(1, taylor_order+1):
        # term <- term @ eri_op
        term = backend.einsum("wqn,wnp->wqp", term, eri_op)
        out = out + term * (1.0 / math.factorial(n))
    return out


def make_eri_op_from_xshift(dt, ltensor, xshift):
    """
    dt: scalar
    ltensor: (nchol, nao, nao) — same as used in forward path
    xshift: (nwalkers, nfields=nchol)
    returns eri_op: (nwalkers, nao, nao)
    """
    sqrtdt = 1j * backend.sqrt(dt)
    nchol, nao = ltensor.shape[:-1]
    return sqrtdt * backend.dot(xshift, ltensor.reshape(nchol, -1)).reshape(xshift.shape[0], nao, nao)


def build_bp_left_states(ph: "Phaseless", trial, walkers, ltensor):
    """
    Returns per-walker left Slaters after back-propagation through the
    buffered BP window. Shapes follow propagator objects:
      - UHF: La: (nwalkers, nα, nao), Lb: (nwalkers, nβ, nao)
      - RHF: L:  (nwalkers, nocc, nao)
    Assumes trial holds right-trial Slaters; we conjugate-transpose to bras.

    TODO:
    replace split this function into one and HS part so that we can make different
    combination as the forward propagator!
    """

    win = ph._bpbuf.window() if (ph._bpbuf and ph.enable_bp) else []
    if len(win) == 0:
        raise RuntimeError("BP buffer is empty. Run a few steps with BP enabled.")
    # print("Debug-yz: length of back propagaiton:", len(win))


    nao = walkers.phiwa.shape[1]
    nwalk = walkers.nwalkers

    # Start from trial bras (Φ_T^†). Adapt if Trial object exposes them differently.
    if walkers.ncomponents > 1:
        # UHF: trial.psia: (nao, nalpha), trial.psib: (nao, nbeta)
        PhiTa = trial.psia   # right trial \alpha
        PhiTb = trial.psib   # right trial \beta
        La = backend.conj(PhiTa).T[None, ...].repeat(nwalk, axis=0)  # (w, nalpha, nao)
        Lb = backend.conj(PhiTb).T[None, ...].repeat(nwalk, axis=0)  # (w, nbeta, nao)
    else:
        # RHF (spin-orbital or restricted): trial.psia: (nao, nocc)
        PhiT = trial.psia
        L = backend.conj(PhiT).T[None, ...].repeat(nwalk, axis=0)    # (w, nocc, nao)


    # Replay steps oldest → newest, matching propagate_walkers():
    for xshift in win:

        # 1) one-body half step: e^{-dt/2 H1}
        if walkers.ncomponents > 1:
            La, Lb = left_apply_onebody_uhf(La, Lb, ph.exp_h1e[0], ph.exp_h1e[1])
        else:
            L = left_apply_onebody(L, ph.exp_h1e[0])

        # 2) HS step using recorded xshift
        eri_op = make_eri_op_from_xshift(ph.dt, ltensor, xshift)

        if walkers.ncomponents > 1:
            La = left_apply_HS_taylor(La, eri_op, ph.taylor_order)
            Lb = left_apply_HS_taylor(Lb, eri_op, ph.taylor_order)
        else:
            L = left_apply_HS_taylor(L, eri_op, ph.taylor_order)

        # 3) one-body half step again
        if walkers.ncomponents > 1:
            La, Lb = left_apply_onebody_uhf(La, Lb, ph.exp_h1e[0], ph.exp_h1e[1])
        else:
            L = left_apply_onebody(L, ph.exp_h1e[0])

        # Optional: stabilize every ~5–10 multiplies
        # Replace with QR/MGS stabilizers (TBA), e.g., trial.stabilize_left(...)
        # La, logdet_a, phase_a = stabilize_left(La)
        # Lb, logdet_b, phase_b = stabilize_left(Lb)

    return (La, Lb) if walkers.ncomponents > 1 else L


# green function meaurement with BP
def batched_biorthogonal_G(R, L):
    """
    R: (w, nao, nocc)
    L: (w, nocc, nao)
    returns G: (w, nao, nao)
    """
    M = backend.einsum("wip, wpj->wij", L.conj(), R)    # (w, nocc, nocc)
    Minv = backend.linalg.inv(M)
    # Minv_{zij} R_{qj> -> Ghalf_{zqi}, Ghalf_{qi} * L_{pi} -> G_{zpq}
    return backend.einsum("wij, wqj, wip->wpq", Minv, R, L.conj())



from openms.qmc.estimators import local_eng_elec_chol_new
from openms.qmc.estimators import e_rh1e_Ghalf, ecoul_rltensor_uhf,  exx_rltensor_Ghalf_kernel

# BP energy measurement
def bp_energy(ph: "Phaseless", trial, walkers, ltensor, h1e, enuc=0.0):
    """
    Returns (E_tot, E1, E2) with BP-corrected mixed estimator.
    """
    # 1) Build BP bras
    # print("Debug-yz: entering bp_energy")

    L = build_bp_left_states(ph, trial, walkers, ltensor)  # (w, nocc, nao) or (La, Lb)

    # 2) Right Slaters from walkers
    if walkers.ncomponents > 1:
        Ra = walkers.phiwa  # (w, nao, nα)
        Rb = walkers.phiwb  # (w, nao, nβ)
        Ga = batched_biorthogonal_G(Ra, L[0])
        Gb = batched_biorthogonal_G(Rb, L[1])
        # Assemble spin blocks as elsewhere

        ej = ecoul_rltensor_uhf(ltensor, Ga, ltensor, Gb)
        ek = exx_rltensor_Ghalf_kernel(ltensor, Ga)
        ek += exx_rltensor_Ghalf_kernel(ltensor, Gb)

        E1 = e_rh1e_Ghalf(h1e[0], Ga) + e_rh1e_Ghalf(h1e[1], Gb)
        E2 = ej - ek
    else:
        R = walkers.phiwa                    # (w, nao, nocc)
        G = batched_biorthogonal_G(R, L)     # (w, nao, nao)
        #
        E1 = 2.0 * e_rh1e_Ghalf(h1e[0], G)
        ej = 4.0 * ecoul_rltensor_uhf(ltensor, G)
        ek = 2.0 * exx_rltensor_Ghalf_kernel(ltensor, G)
        E2 = ej - ek

    Eloc = E1.real + E2.real + enuc
    w = walkers.weights
    norm = backend.sum(w)
    Etot = backend.dot(w, Eloc)
    E1  = backend.dot(w, E1.real)
    E2  = backend.dot(w, E2.real)

    if walkers._mpi.size > 1:
        norm = walkers._mpi.comm.allreduce(norm, op=MPI.SUM)
        e1 = walkers._mpi.comm.allreduce(e1, op=MPI.SUM)
        e2 = walkers._mpi.comm.allreduce(e2, op=MPI.SUM)
        etot = walkers._mpi.comm.allreduce(etot, op=MPI.SUM)

    return [Etot, norm, E1, E2]




@dataclass
class _BPBuffer:
    L_bp: int
    reortho_period: int = 5
    _af_stack: Deque[Array] = field(default_factory=deque)  # store AF
    _bias_stack: Deque[Array] = field(default_factory=deque)  # store bias potential
    _steps: int = 0

    def push(self, xk: Array, vbias: Array) -> None:
        """Push auxiliary fields :math:`x_k` (for the *forward* step k) into a stack.

        During measurement, we will pull the auxiliary field :math:`x_k` in reverse order to build
        the propagator :math:`B_k` and apply it to the trial WF.
        """
        self._af_stack.append(xk.copy())
        self._bias_stack.append(vbias.copy())
        if len(self._af_stack) > self.L_bp:
            # print("Debug-yz: maximum stack length reacheed, poping left")
            self._af_stack.popleft()
            self._bias_stack.popleft()
        self._steps += 1
        # print("Debug-yz: current BP stack length is:", len(self._af_stack))


    def window(self):
        # oldest -> newest; shape list of length ≤ Lbp, each (nwalkers, nfields)
        return list(self._af_stack)


    def construct_BK(self, xk, vbias):
        r"""re-construct the propagator :math:`B_k`

        may be moved into propagator class
        """
        return None


    def construct_left(self):
        r"""Construct left state
        """

        return None


# -------------------
# old code, not used!
# -------------------
#  @dataclass
#  class BPBuffer:
#      r"""
#      Need to decide whether to store B operator of the Auxiliary field.
#
#      Storing B operators make it easier to implement the code and is numerically efficient.
#      But it requires a lot of memories to store the B operators
#
#      Storing Auxiliary field solves the memory issue, but need to re-construct the B operatory,
#      which is numerically less efficient.
#      """
#      L_bp: int  # length of BP
#      reortho_freq: int = 5 #
#      _AF_stack: Deque[Array] = field(default_factory=deque)  # store auxiliary field (used to construct B_k)
#      _steps: int = 0
#
#
#      def push(self, Afield: Array) -> None:
#          """Push a new one-body propagator B_k (for the *forward* step k).
#          During measurement, we will apply B_k^\dagger in reverse order to the trial.
#          """
#          self._AF_stack.append(Afield.copy())
#          if len(self._B_stack) > self.L_bp:
#              self._B_stack.popleft()
#          self._steps += 1
#
#
#      def build_left(self, T: Array) -> Array:
#          r"""Return the left state :math:`\ket{L}` after back-propagating the trial T:
#
#          .. math::
#              \ket{L} = \prod_{k=n-L+1}^{n} \hat{B}_k \ket{T}
#
#          We construct :math:`\ket{L}`, with periodic re-orthonormalization.
#          """
#          # construct B_k from Afield
#          L = T.copy()
#          if not self._AF_stack:
#              # No back-propagation yet; just return the trial as left state
#              return thin_qr(L)
#          # Apply in reverse (most recent step first)
#          for idx, x in enumerate(reversed(self._AF_stack), start=1):
#              # build B propagation
#              B = None # = exp(-T*dt/2) exp(-Bx_k dt) * exp(-T*dt/2)
#              L = B.conj().T @ L
#              if (idx % self.reortho_freq) == 0:
#                  L = thin_qr(L)
#          # Final stabilization
#          L = thin_qr(L)
#          return L
