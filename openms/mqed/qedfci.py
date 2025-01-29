#!/usr/bin/env python
# Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Yu Zhang <zhy@lanl.gov>
#         Qiming Sun <osirpt.sun@gmail.com>
#

"""
FCI for electron-boson coupled system.


This file is modified from pyscf fci/direct_ep: the major changes are:
   1) lift the restriction of nmode = norb
   2) fock state for each mode can be different, controlled by boson_states


"""


import numpy
from pyscf import lib
from pyscf import ao2mo
from pyscf.fci import cistring
from pyscf.fci import rdm
from pyscf.fci.direct_spin1 import _unpack_nelec
import time
#                                mode-1,      ...,   mode-n
#                                v                    v
# ep_wfn, shape = (nstra,nstrb,boson_states[0],...,boson_states[n-1])
#               = (nstra,nstrb) + tuple(boson_states)
#       For each mode, {0,1,...,boson_states[imode]-1} gives nfock possible confs
# t for hopping, shape = (norb, norb)
# u
# g for the electorn-boson coupling
# Hbb for boson-boson interaction: (nmode, nmode)


def fboson_ci_shape(norb, nelec, nmode=0, boson_states=None):
    r"""Get the CI shape for Fermion-Boson mixture.

    Parameters:
        norb (int): Number of orbitals.
        nelec (int): Number of electrons (neleca, nelecb).
        nmode (int): Number of bosonic modes.
        boson_states (int or array): Number of Fock states for each mode.

    Returns:
        tuple: CI shape for the Fermion-Boson mixture.
    """
    if boson_states is not None:
        if isinstance(boson_states, int):
            boson_states = numpy.array([boson_states] * nmode)
        elif isinstance(boson_states, (list, numpy.ndarray)):
            boson_states = numpy.array(boson_states)
            assert nmode == len(
                boson_states
            ), f"nmode must match the length of boson_states"
        else:
            raise TypeError("boson_states must be an int or an array-like object")
    else:
        if nmode > 0:
            raise TypeError("boson_states must be set if nmode > 0")

    neleca, nelecb = _unpack_nelec(nelec)
    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)
    cishape = (na, nb)
    if nmode > 0:
        cishape = (na, nb) + tuple(boson_states + 1)
    return cishape


def contract_1e(h1e, fcivec, norb, nelec, nmode=0, boson_states=None):
    """
    Contract the 1e integral: E_{pq} |CI>
    """

    neleca, nelecb = _unpack_nelec(nelec)
    link_indexa = cistring.gen_linkstr_index(range(norb), neleca)
    link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)
    cishape = fboson_ci_shape(norb, nelec, nmode, boson_states)

    ci0 = fcivec.reshape(cishape)
    fcinew = numpy.zeros(cishape)
    for str0, tab in enumerate(link_indexa):
        for a, i, str1, sign in tab:
            fcinew[str1] += sign * ci0[str0] * h1e[a, i]
    for str0, tab in enumerate(link_indexb):
        for a, i, str1, sign in tab:
            fcinew[:, str1] += sign * ci0[:, str0] * h1e[a, i]
    return fcinew.reshape(fcivec.shape)


def contract_2e_spin0(eri, fcivec, norb, nelec, nmode, boson_states, out=None):
    """
    Compute E_{pq}E_{rs}|CI>
    """
    neleca, nelecb = _unpack_nelec(nelec)
    link_indexa = cistring.gen_linkstr_index(range(norb), neleca)
    link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)

    cishape = fboson_ci_shape(norb, nelec, nmode, boson_states)
    ci0 = fcivec.reshape(cishape)
    eri = eri.reshape(norb, norb, norb, norb)

    t1 = numpy.zeros((norb, norb,) + cishape)
    for str0, tab in enumerate(link_indexa):
        for a, i, str1, sign in tab:
            t1[a, i, str1] += sign * ci0[str0]
    for str0, tab in enumerate(link_indexb):
        for a, i, str1, sign in tab:
            t1[a, i, :, str1] += sign * ci0[:, str0]

    t1 = numpy.tensordot(eri, t1, axes=((2, 3), (0, 1)))

    if out is None:
        fcinew = numpy.zeros(cishape)
    else:
        fcinew = out.reshape(cishape)
    for str0, tab in enumerate(link_indexa):
        for a, i, str1, sign in tab:
            fcinew[str1] += sign * t1[a, i, str0]
    for str0, tab in enumerate(link_indexb):
        for a, i, str1, sign in tab:
            fcinew[:, str1] += sign * t1[a, i, :, str0]
    return fcinew.reshape(fcivec.shape)


# eri is a list of 2e hamiltonian (a for alpha, b for beta)
# [(aa|aa), (aa|bb), (bb|bb)]
def contract_2e(eri, fcivec, norb, nelec, nmode, boson_states):
    r"""
    contract 2e integral: E_{pq} E_{rs} |CI>
    """
    neleca, nelecb = _unpack_nelec(nelec)
    link_indexa = cistring.gen_linkstr_index(range(norb), neleca)
    link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)
    cishape = fboson_ci_shape(norb, nelec, nmode, boson_states)

    ci0 = fcivec.reshape(cishape)
    t1a = numpy.zeros((norb, norb) + cishape)
    t1b = numpy.zeros((norb, norb) + cishape)
    for str0, tab in enumerate(link_indexa):
        for a, i, str1, sign in tab:
            t1a[a, i, str1] += sign * ci0[str0]
    for str0, tab in enumerate(link_indexb):
        for a, i, str1, sign in tab:
            t1b[a, i, :, str1] += sign * ci0[:, str0]

    if isinstance(eri, list):
        g2e_aa = ao2mo.restore(1, eri[0], norb)
        g2e_ab = ao2mo.restore(1, eri[1], norb)
        g2e_bb = ao2mo.restore(1, eri[2], nsorb)

        t2a = numpy.dot(g2e_aa.reshape(norb**2, -1), t1a.reshape(norb**2, -1))
        t2a += numpy.dot(g2e_ab.reshape(norb**2, -1), t1b.reshape(norb**2, -1))
        t2b = numpy.dot(g2e_ab.reshape(norb**2, -1).T, t1a.reshape(norb**2, -1))
        t2b += numpy.dot(g2e_bb.reshape(norb**2, -1), t1b.reshape(norb**2, -1))
    else:
        # single eri tensor, i.e., (aa|aa) == (bb|bb) case
        t1 = t1a + t1b
        t2a = t2b = numpy.dot(eri.reshape(norb**2, -1), t1.reshape(norb**2, -1))

    t2a = t2a.reshape((norb, norb) + cishape)
    t2b = t2b.reshape((norb, norb) + cishape)

    fcinew = numpy.zeros(cishape)
    for str0, tab in enumerate(link_indexa):
        for a, i, str1, sign in tab:
            fcinew[str1] += sign * t2a[a, i, str0]
    for str0, tab in enumerate(link_indexb):
        for a, i, str1, sign in tab:
            fcinew[:, str1] += sign * t2b[a, i, :, str0]
    return fcinew.reshape(fcivec.shape)


def _unpack_u(u):
    if numpy.ndim(u) == 0:
        u_aa = u_ab = u_bb = u
    else:
        u_aa, u_ab, u_bb = u
    return u_aa, u_ab, u_bb


def contract_2e_hubbard(u, fcivec, norb, nelec, nmode, boson_states):
    r"""
    Contract two-body term within the hubbard-U model
    """
    neleca, nelecb = _unpack_nelec(nelec)
    # u_aa, u_ab, u_bb = _unpack_u(u)

    strsa = numpy.asarray(cistring.gen_strings4orblist(range(norb), neleca))
    strsb = numpy.asarray(cistring.gen_strings4orblist(range(norb), nelecb))
    cishape = fboson_ci_shape(norb, nelec, nmode, boson_states)
    ci0 = fcivec.reshape(cishape)
    fcinew = numpy.zeros(cishape)

    for i in range(norb):
        maska = (strsa & (1 << i)) > 0
        maskb = (strsb & (1 << i)) > 0
        fcinew[maska[:, None] & maskb] += u * ci0[maska[:, None] & maskb]
    return fcinew.reshape(fcivec.shape)


def absorb_h1e(h1e, eri, norb, nelec, fac=1):
    """Modify 2e Hamiltonian to include 1e Hamiltonian contribution."""
    if not isinstance(nelec, (int, numpy.integer)):
        nelec = sum(nelec)
    # h2e = ao2mo.restore(1, eri.copy(), norb).astype(h1e.dtype, copy=False)
    if eri.size == norb ** 4:
        h2e = numpy.array(eri, copy=True).reshape(norb, norb, norb, norb)
    else:
        h2e = ao2mo.restore(1, eri, norb)

    f1e = h1e - numpy.einsum("jiik->jk", h2e) * 0.5
    f1e = f1e * (1.0 / (nelec + 1e-100))
    for k in range(norb):
        h2e[k, k, :, :] += f1e
        h2e[:, :, k, k] += f1e
    return h2e * fac


def contract_all(H1, H2, Heb, Hbb, cin, norb, nelec, nmode, boson_states):
    r""" """

    # electronic part
    cout = numpy.zeros_like(cin)
    #print("Debug: c before contract_e is ", cin)
    t0 = time.time()
    # print("Debug-yz: contracting 1e part")
    # cout += contract_1e(H1, cin, norb, nelec, nmode, boson_states)
    t1 = time.time()

    # print("Debug-yz: contracting 2e part")
    if H2.ndim > 1:
        cout += contract_2e(H2, cin, norb, nelec, nmode, boson_states)
    else:
        cout += contract_2e_hubbard(H2, cin, norb, nelec, nmode, boson_states)
    # print("Debug: c after contract_e is ", cout)
    t2 = time.time()

    # e-b coupling term
    # print("Debug-yz: contracting e-b coupling part")
    cout += contract_eb(Heb, cin, norb, nelec, nmode, boson_states)
    t3 = time.time()

    # bosonic term
    # print("Debug-yz: contracting bosonic interaction part")
    cout += contract_bb(Hbb, cin, norb, nelec, nmode, boson_states)
    # print("Debug: c after contract_bb is ", cout)
    t4 = time.time()

    # print(f"Times of contract: {t1-t0:10.5f} {t2-t1:10.5f} {t3-t2:10.5f} {t4-t3:10.5f}\n")

    return cout


def slices_for(imode, nmode, fock_id, idxab=None):
    r"""Slice for the CI vector

    return: (idx_a, idx_b, ..., boson_id, ...)
    """
    slices = [slice(None, None, None)] * (2 + nmode)  # +2 for electron indices
    slices[2 + imode] = fock_id

    if idxab is not None:
        ia, ib = idxab
        if ia is not None:
            slices[0] = ia
        if ib is not None:
            slices[1] = ib
    return tuple(slices)


def slices_for_cre(imode, nmode, fock_id, idxab=None):
    return slices_for(imode, nmode, fock_id + 1, idxab)


def slices_for_des(imode, nmode, fock_id, idxab=None):
    return slices_for(imode, nmode, fock_id - 1, idxab)


# Contract to one phonon creation operator
def cre_phonon(fcivec, norb, nelec, nmode, boson_states, imode):
    cishape = fboson_ci_shape(norb, nelec, nmode, boson_states)
    ci0 = fcivec.reshape(cishape)
    fcinew = numpy.zeros(cishape)

    nboson = boson_states[imode]
    boson_cre = numpy.sqrt(numpy.arange(1, nboson + 1))
    for ip in range(nboson):
        slices1 = slices_for_cre(site_id, nmode, ip)
        slices0 = slices_for(site_id, nmode, ip)
        fcinew[slices1] += boson_cre[ip] * ci0[slices0]
    return fcinew.reshape(fcivec.shape)


# Contract to one phonon annihilation operator
def des_phonon(fcivec, nsite, nelec, nphonon, site_id):
    cishape = fboson_ci_shape(norb, nelec, nmode, boson_states)
    ci0 = fcivec.reshape(cishape)
    fcinew = numpy.zeros(cishape)

    boson_cre = numpy.sqrt(numpy.arange(1, nphonon + 1))
    for ip in range(nphonon):
        slices1 = slices_for_cre(site_id, nmode, ip)
        slices0 = slices_for(site_id, nmode, ip)
        fcinew[slices0] += boson_cre[ip] * ci0[slices1]
    return fcinew.reshape(fcivec.shape)


# boson_boson coupling
def contract_bb(Hbb, fcivec, norb, nelec, nmode, boson_states):
    cishape = fboson_ci_shape(norb, nelec, nmode, boson_states)
    ci0 = fcivec.reshape(cishape)
    fcinew = numpy.zeros(cishape)

    t1 = numpy.zeros((norb,) + cishape)
    for imode in range(nmode):
        nboson = boson_states[imode]
        boson_cre = numpy.sqrt(numpy.arange(1, nboson + 1))
        for i in range(nboson):
            slices1 = slices_for_cre(imode, nmode, i)
            slices0 = slices_for(imode, nmode, i)
            t1[(imode,) + slices0] += ci0[slices1] * boson_cre[i]  # annihilation

    t1 = lib.dot(Hbb, t1.reshape(nmode, -1)).reshape(t1.shape)

    for imode in range(nmode):
        nboson = boson_states[imode]
        boson_cre = numpy.sqrt(numpy.arange(1, nboson + 1))
        for i in range(nboson):
            slices1 = slices_for_cre(imode, nmode, i)
            slices0 = slices_for(imode, nmode, i)
            fcinew[slices1] += t1[(imode,) + slices0] * boson_cre[i]  # creation
    return fcinew.reshape(fcivec.shape)


# N_alpha N_beta * \sum_{p} (p^+ + p)
# N_alpha, N_beta are particle number operator, p^+ and p are phonon creation annihilation operator
def contract_eb_new(g, fcivec, norb, nelec, nmode, boson_states):
    r"""
    Contract the electron-boson coupling part:

    .. math::
         c = &

    not working yet!
    """
    neleca, nelecb = _unpack_nelec(nelec)
    strsa = numpy.asarray(cistring.gen_strings4orblist(range(norb), neleca))
    strsb = numpy.asarray(cistring.gen_strings4orblist(range(norb), nelecb))
    cishape = fboson_ci_shape(norb, nelec, nmode, boson_states)
    na, nb = cishape[:2]

    ci0 = fcivec.reshape(cishape)
    fcinew = numpy.zeros(cishape)

    print("Debug: na/nb   = ", na, nb)
    print("Debug: cishape = ", cishape)

    for i in range(nmode):
        nboson = boson_states[i]
        boson_cre = numpy.sqrt(numpy.arange(1, nboson + 1))
        maska = (strsa & (1 << i)) > 0
        maskb = (strsb & (1 << i)) > 0
        e_part = numpy.zeros((na, nb))
        e_part[maska, :] += 1
        e_part[:, maskb] += 1
        e_part[:] -= float(neleca + nelecb) / nmode

        print("Debug: e_part.shape = ", e_part.shape)
        print("Debug: g[i].shape = ", g[i].shape)
        # FXIME: map g[i] into CI shape, then multiply it to boson_cre and e_part

        for ip in range(nboson):
            slices1 = slices_for_cre(i, nmode, ip)
            slices0 = slices_for(i, nmode, ip)
            fcinew[slices1] += numpy.einsum(
                "ij...,ij...->ij...", g[i] * boson_cre[ip] * e_part, ci0[slices0]
            )
            fcinew[slices0] += numpy.einsum(
                "ij...,ij...->ij...", g[i] * boson_cre[ip] * e_part, ci0[slices1]
            )
    return fcinew.reshape(fcivec.shape)


def contract_eb(g, fcivec, norb, nelec, nmode, boson_states, out=None):
    """
    Contract the electron-boson coupling part.
    """
    neleca, nelecb = _unpack_nelec(nelec)
    link_indexa = cistring.gen_linkstr_index(range(norb), neleca)
    link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)
    cishape = fboson_ci_shape(norb, nelec, nmode, boson_states)

    ci0 = fcivec.reshape(cishape)
    if out is None:
        fcinew = numpy.zeros(cishape, dtype=ci0.dtype)
    else:
        fcinew = out.reshape(cishape)

    for imode in range(nmode):
        nboson = boson_states[imode]
        boson_cre = numpy.sqrt(numpy.arange(1, nboson + 1))
        for ip in range(nboson):
            for str0, tab in enumerate(link_indexa):
                for a, i, str1, sign in tab:
                    # b^+
                    slices1 = slices_for_cre(imode, nmode, ip, idxab=(str1, None))
                    slices0 = slices_for(imode, nmode, ip, idxab=(str0, None))
                    fcinew[slices1] += (g[imode, a, i] * boson_cre[ip] * sign) * ci0[
                        slices0
                    ]
                    # b
                    slices0 = slices_for(imode, nmode, ip, idxab=(str1, None))
                    slices1 = slices_for_cre(imode, nmode, ip, idxab=(str0, None))
                    fcinew[slices0] += (g[imode, a, i] * boson_cre[ip] * sign) * ci0[
                        slices1
                    ]
            for str0, tab in enumerate(link_indexb):
                for a, i, str1, sign in tab:
                    # b^+
                    slices1 = slices_for_cre(imode, nmode, ip, idxab=(None, str1))
                    slices0 = slices_for(imode, nmode, ip, idxab=(None, str0))
                    fcinew[slices1] += (g[imode, a, i] * boson_cre[ip] * sign) * ci0[
                        slices0
                    ]
                    # b
                    slices0 = slices_for(imode, nmode, ip, idxab=(None, str1))
                    slices1 = slices_for_cre(imode, nmode, ip, idxab=(None, str0))
                    fcinew[slices0] += (g[imode, a, i] * boson_cre[ip] * sign) * ci0[
                        slices1
                    ]

    return fcinew.reshape(fcivec.shape)


def make_hdiag(h1e, eri, g, Hbb, norb, nelec, nmode, boson_states):
    neleca, nelecb = _unpack_nelec(nelec)

    ci_shape = fboson_ci_shape(norb, nelec, nmode, boson_states)
    hdiag = numpy.zeros(ci_shape)

    try:
        occslista = cistring.gen_occslst(range(norb), neleca)
        occslistb = cistring.gen_occslst(range(norb), nelecb)
    except AttributeError:
        occslista = cistring._gen_occslst(range(norb), neleca)
        occslistb = cistring._gen_occslst(range(norb), nelecb)
    eri = ao2mo.restore(1, eri, norb)
    diagj = numpy.einsum("iijj->ij", eri)
    diagk = numpy.einsum("ijji->ij", eri)
    for ia, aocc in enumerate(occslista):
        for ib, bocc in enumerate(occslistb):
            e1 = h1e[aocc, aocc].sum() + h1e[bocc, bocc].sum()
            e2 = (
                diagj[aocc][:, aocc].sum() + diagj[aocc][:, bocc].sum()
              + diagj[bocc][:, aocc].sum() + diagj[bocc][:, bocc].sum()
              - diagk[aocc][:, aocc].sum() - diagk[bocc][:, bocc].sum()
            )
            hdiag[ia, ib] = e1 + e2 * 0.5

    for imode in range(nmode):
        for i in range(boson_states[imode]+1):
            slices0 = slices_for(imode, nmode, i)
            if Hbb.ndim == 1:
                hdiag[slices0] += max(i * Hbb[imode], 0.5)
            else:
                hdiag[slices0] += max(i * Hbb[imode, imode], 0.5)

    return hdiag.ravel()


def make_hdiag_hubbard(h1e, eri, Hbb, norb, nelec, nmode, nph, opt=None):
    neleca, nelecb = _unpack_nelec(nelec)
    nelec_tot = neleca + nelecb
    link_indexa = cistring.gen_linkstr_index(range(norb), neleca)
    link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)
    occslista = [tab[:neleca, 0] for tab in link_indexa]
    occslistb = [tab[:nelecb, 0] for tab in link_indexb]
    # electron part
    cishape = fboson_ci_shape(norb, nelec, nmode, boson_states)
    hdiag = numpy.zeros(cishape)

    u_aa, u_ab, u_bb = _unpack_u(eri)

    for ia, aocc in enumerate(occslista):
        for ib, bocc in enumerate(occslistb):
            e1 = h1e[aocc, aocc].sum() + h1e[bocc, bocc].sum()
            e2 = u_ab * nelec_tot
            hdiag[ia, ib] = e1 + e2

    # TODO: electron-phonon part

    # phonon part
    for imode in range(nmode):
        nboson = boson_states[imode]
        for i in range(nboson + 1):
            slices0 = slices_for(imode, nmode, i)
            # hdiag[slices0] += i+1
            if Hbb.ndim == 1:
                hdiag[slices0] += max(i * Hbb[imode], 0.5)
            else:
                hdiag[slices0] += max(i * Hbb[imode, imode], 0.5)

    return hdiag.ravel()


def kernel(
    H1,
    H2,
    norb,
    nelec,
    nmode,
    boson_states,
    Heb,
    Hbb,
    shift_vac=True,
    system="model",
    verbose=5,
    ecore=0,
    tol=1e-10,
    lindep=1.e-14,
    max_cycle=100,
    nroots=1,
    **kwargs,
):
    r"""
    QEDFCI kernel

    Parameters
    ----------

    H1: ndarray
       The one-electron Hamiltonain matrix in the orthogonal representation (OAO or MO)
    H2: ndarray
       The two-electron Hamiltonian tensor in the orthogonal representation
    norb: int
       Number of orbitals
    nelec: [int]
       Number of alpha and beta electrons
    nmode: int
       Number of bosonic modes
    boson_states: 1D array
       Number of fock states for each mode
    Heb: ndarray
       Electron-boson coupling matrix
    Hbb: ndarray (1d or 2d)
       Hamiltonian matrix for bosons
    tol: float
       convergence tolerance
    max_cycle: int
       Maximum cycle
    ecore: float
       Energy of core orbitals and nuclei

    Returns
    -------
    e: 1D array
       Low-lying energies
    c: ndarray:
       Low-lying eigenstates
    """

    # check consistency among input integrals
    assert norb == H1.shape[-1]  # we may also directly get norb from H1 shape
    assert norb == H2.shape[-1]
    assert norb == Heb.shape[-1]
    assert nmode == Hbb.shape[0]
    assert nmode == Heb.shape[0]

    # Hbb is supposed to be a 2D array, convert it to 2D if it is 1D
    if Hbb.ndim == 1:
        # Convert 1D array to 2D diagonal matrix
        Hbb = numpy.diag(Hbb)

    numpy.random.seed(1)

    coherent_state = kwargs.get("coherent_state", False)
    max_space = kwargs.get("max_space", 16)

    #
    # coherence state:
    # TODO: we may move the following out of the fci kernel as we can have different
    # pre-processing of the Hamiltonains like coherent state, VLF ansatz, and squeezed states
    if True:
        from pyscf.fci import direct_spin1
        print("\n******Computing FCI energy of the electronic part************\n")
        e0, c = direct_spin1.kernel(H1, H2, norb, nelec, ecore=ecore,
                           verbose=verbose,
                           tol=tol, lindep=lindep, max_cycle=max_cycle,
                           max_space=max_space, nroots=nroots)
        print ("FCI electronic : ", e0)

    if coherent_state:
        if nroots > 1:
            dm0 = direct_spin1.make_rdm1(c[0], norb, nelec)
        else:
            dm0 = direct_spin1.make_rdm1(c, norb, nelec)
        # TODO: modify the H1, H2, Heb, Hbb with coherent state representation


    # QED-FCI calculations
    cishape = fboson_ci_shape(norb, nelec, nmode, boson_states)


    # get/set initial guess
    ci0 = kwargs.get("init_ci", None)

    if ci0 is None:
        ci0 = numpy.zeros(cishape)
        ci0.__setitem__((0, 0) + (0,) * nmode, 1)

        # Add noise for initial guess, remove it if problematic
        ci0 += (numpy.random.random(cishape) - 0.5) * 1e-5

        # ci0[0, :] += numpy.random.random(ci0[0, :].shape) * 1e-6
        # ci0[:, 0] += numpy.random.random(ci0[:, 0].shape) * 1e-6
    else:
        assert (
            ci0.shape == cishape
        ), "the shape of provided init guess does not match the dimension of ci"

    #
    # according to the system type (model or abinit), we use different contraction functions
    H2mod =  absorb_h1e(H1, H2, norb, nelec, 0.5)
    print("Debug: norm of modified H2 =", numpy.linalg.norm(H2mod))
    def hop(c):
        hc = contract_all(H1, H2mod, Heb, Hbb, c, norb, nelec, nmode, boson_states)
        # hc = contract_all(H1, H2, Heb, Hbb, c, norb, nelec, nmode, boson_states)
        return hc.reshape(-1)

    hdiag = make_hdiag(H1, H2, Heb, Hbb, norb, nelec, nmode, boson_states)
    # precond = lambda x, e, *args: x/(hdiag-e+1e-4) # same as the lib
    precond = lib.make_diag_precond(hdiag, level_shift=1e-3)

    e, c = lib.davidson(
        hop,
        ci0.reshape(-1),
        precond,
        tol=tol,
        max_cycle=max_cycle,
        verbose=verbose,
        nroots=nroots,
        lindep=lindep
    )

    return e0, e + ecore, c


def make_rdm1e(fcivec, nsite, nelec):
    """1-electron density matrix dm_pq = <|p^+ q|>"""
    neleca, nelecb = _unpack_nelec(nelec)
    link_indexa = cistring.gen_linkstr_index(range(nsite), neleca)
    link_indexb = cistring.gen_linkstr_index(range(nsite), nelecb)
    na = cistring.num_strings(nsite, neleca)
    nb = cistring.num_strings(nsite, nelecb)

    rdm1 = numpy.zeros((nsite, nsite))
    ci0 = fcivec.reshape(na, -1)
    for str0, tab in enumerate(link_indexa):
        for a, i, str1, sign in tab:
            rdm1[a, i] += sign * numpy.dot(ci0[str1], ci0[str0])

    ci0 = fcivec.reshape(na, nb, -1)
    for str0, tab in enumerate(link_indexb):
        for a, i, str1, sign in tab:
            rdm1[a, i] += sign * numpy.einsum("ax,ax->", ci0[:, str1], ci0[:, str0])
    return rdm1


def make_rdm12e(fcivec, norb, nelec):
    """1-electron and 2-electron density matrices
    dm_pq = <|p^+ q|>
    dm_{pqrs} = <|p^+ r^+ q s|>  (note 2pdm is ordered in chemist notation)
    """
    neleca, nelecb = _unpack_nelec(nelec)
    link_indexa = cistring.gen_linkstr_index(range(norb), neleca)
    link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)
    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)

    ci0 = fcivec.reshape(na, nb, -1)
    rdm1 = numpy.zeros((norb, norb))
    rdm2 = numpy.zeros((norb, norb, norb, norb))
    for str0 in range(na):
        t1 = numpy.zeros((norb, norb, nb) + ci0.shape[2:])
        for a, i, str1, sign in link_indexa[str0]:
            t1[i, a, :] += sign * ci0[str1, :]

        for k, tab in enumerate(link_indexb):
            for a, i, str1, sign in tab:
                t1[i, a, k] += sign * ci0[str0, str1]

        rdm1 += numpy.einsum("mp,ijmp->ij", ci0[str0], t1)
        # i^+ j|0> => <0|j^+ i, so swap i and j
        #:rdm2 += numpy.einsum('ijmp,klmp->jikl', t1, t1)
        tmp = lib.dot(t1.reshape(norb**2, -1), t1.reshape(norb**2, -1).T)
        rdm2 += tmp.reshape((norb,) * 4).transpose(1, 0, 2, 3)
    rdm1, rdm2 = rdm.reorder_rdm(rdm1, rdm2, True)
    return rdm1, rdm2


def make_rdm1p(fcivec, norb, nelec, nmode, boson_states):
    """1-phonon density matrix dm_pq = <|p^+ q|>"""
    cishape = fboson_ci_shape(norb, nelec, nmode, boson_states)
    ci0 = fcivec.reshape(cishape)

    t1 = numpy.zeros((norb,) + cishape)
    for imode in range(nmode):
        nboson = boson_states[imode]
        boson_cre = numpy.sqrt(numpy.arange(1, nboson + 1))
        for i in range(nboson):
            slices1 = slices_for_cre(imode, nmode, i)
            slices0 = slices_for(imode, nmode, i)
            t1[(imode,) + slices0] += ci0[slices1] * boson_cre[i]

    rdm1 = lib.dot(t1.reshape(nsite, -1), t1.reshape(nsite, -1).T)
    return rdm1


def test_on_molecule():

    from functools import reduce
    from pyscf import gto
    from pyscf import scf

    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None
    mol.atom = [
        ["H", (1.0, -1.0, 0.0)],
        ["H", (0.0, -1.0, -1.0)],
        ["H", (1.0, -0.5, -1.0)],
        ["H", (0.0, -0.0, -1.0)],
        ["H", (1.0, -0.5, 0.0)],
        ["H", (0.0, 1.0, 1.0)],
    ]
    mol.basis = "sto-3g"
    mol.build()

    m = scf.RHF(mol)
    m.kernel()
    norb = m.mo_coeff.shape[1]
    nelec = mol.nelectron - 2

    nfock = 2
    nmode = 2

    cishape = fboson_ci_shape(norb, nelec, nmode=nmode, boson_states=nfock)
    print("Debug: ci_shape = ", cishape)

    h1e = reduce(numpy.dot, (m.mo_coeff.T, m.get_hcore(), m.mo_coeff))
    eri = ao2mo.kernel(m._eri, m.mo_coeff, compact=False)
    eri = eri.reshape(norb, norb, norb, norb)

    e1 = kernel(h1e, eri, norb, nelec)
    print(e1, e1 - -7.9766331504361414)


if __name__ == "__main__":
    nsite = 2
    nelec = 2
    nmode = 2
    boson_states = numpy.array([3 for _ in range(nmode)])

    t = numpy.zeros((nsite, nsite))
    idx = numpy.arange(nsite - 1)
    t[idx + 1, idx] = t[idx, idx + 1] = -1
    # t[:] = 0
    u = 1.5
    g = 0.5
    hpp = numpy.eye(nsite) * 1.1
    hpp[idx + 1, idx] = hpp[idx, idx + 1] = 0.1
    # hpp[:] = 0
    print("nelec = ", nelec)
    print("nphonon = ", boson_states)
    print("t =\n", t)
    print("u =", u)
    print("g =", g)
    print("hpp =\n", hpp)

    es = []
    nelecs = [(ia, ib) for ia in range(nsite + 1) for ib in range(ia + 1)]
    print("electronic configurations =", nelecs)
    for nelec in nelecs:
        e, c = kernel(
            t,
            u,
            g,
            hpp,
            nsite,
            nelec,
            nmode,
            boson_states,
            tol=1e-10,
            verbose=0,
            nroots=1,
        )
        print("nelec =", nelec, "E =", e)
        es.append(e)

    es = numpy.hstack(es)
    idx = numpy.argsort(es)
    print(es[idx])

    print("\nGround state is")
    nelec = nelecs[idx[0]]
    e, c = kernel(t, u, g, hpp, nsite, nelec, nphonon, tol=1e-10, verbose=0, nroots=1)
    print("nelec =", nelec, "E =", e)
    dm1 = make_rdm1e(c, nsite, nelec)
    print("electron DM")
    print(dm1)

    exit()

    dm1a, dm2 = make_rdm12e(c, nsite, nelec)
    print("check 1e DM", numpy.allclose(dm1, dm1a))
    print(
        "check 2e DM",
        numpy.allclose(dm1, numpy.einsum("ijkk->ij", dm2) / (sum(nelec) - 1.0)),
    )
    print(
        "check 2e DM",
        numpy.allclose(dm1, numpy.einsum("kkij->ij", dm2) / (sum(nelec) - 1.0)),
    )

    print("phonon DM")
    dm1 = make_rdm1p(c, nsite, nelec, nmode, boson_states)
    print(dm1)

    dm1a = numpy.empty_like(dm1)
    for i in range(nsite):
        for j in range(nsite):
            c1 = des_phonon(c, nsite, nelec, nphonon, j)
            c1 = cre_phonon(c1, nsite, nelec, nphonon, i)
            dm1a[i, j] = numpy.dot(c.ravel(), c1.ravel())
    print("check phonon DM", numpy.allclose(dm1, dm1a))

    cishape = make_shape(nsite, nelec, nphonon)
    eri = numpy.zeros((nsite, nsite, nsite, nsite))
    for i in range(nsite):
        eri[i, i, i, i] = u
    numpy.random.seed(3)
    ci0 = numpy.random.random(cishape)
    ci1 = contract_2e([eri * 0, eri * 0.5, eri * 0], ci0, nsite, nelec, nphonon)
    ci2 = contract_2e_hubbard(u, ci0, nsite, nelec, nphonon)
    print("Check contract_2e", abs(ci1 - ci2).sum())
