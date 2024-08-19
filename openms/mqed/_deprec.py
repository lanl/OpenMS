import copy
import numpy

from pyscf import lib, scf

from openms import __config__

TIGHT_GRAD_CONV_TOL = getattr(__config__, "TIGHT_GRAD_CONV_TOL", True)

# depreciated standalone qedhf function
def qedrhf(model, options):
    # restricted qed hf
    # make a copy for qedhf
    mf = copy.copy(model.mf)
    conv_tol = 1.0e-10
    conv_tol_grad = None
    dump_chk = False
    callback = None
    conv_check = False
    noscf = False
    if "noscf" in options:
        noscf = options["noscf"]

    # converged bare HF coefficients
    na = int(mf.mo_occ.sum() // 2)
    ca = mf.mo_coeff
    dm = 2.0 * numpy.einsum("ai,bi->ab", ca[:, :na], ca[:, :na])
    mu_ao = model.dmat

    if conv_tol_grad is None:
        conv_tol_grad = numpy.sqrt(conv_tol)
    mol = mf.mol

    # initial guess
    h1e = mf.get_hcore(mol)
    vhf = mf.get_veff(mol, dm)

    e_tot = mf.energy_tot(dm, h1e, vhf)
    nuc = mf.energy_nuc()
    scf_conv = False
    mo_energy = mo_coeff = mo_occ = None
    s1e = mf.get_ovlp(mol)
    cond = lib.cond(s1e)

    # Skip SCF iterations. Compute only the total energy of the initial density
    if mf.max_cycle <= 0:
        fock = mf.get_fock(h1e, s1e, vhf, dm)  # = h1e + vhf, no DIIS
        mo_energy, mo_coeff = mf.eig(fock, s1e)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        return scf_conv, e_tot, mo_energy, mo_coeff, mo_occ

    if isinstance(mf.diis, lib.diis.DIIS):
        mf_diis = mf.diis
    elif mf.diis:
        assert issubclass(mf.DIIS, lib.diis.DIIS)
        mf_diis = mf.DIIS(mf, mf.diis_file)
        mf_diis.space = mf.diis_space
        mf_diis.rollback = mf.diis_space_rollback

        # We get the used orthonormalized AO basis from any old eigendecomposition.
        # Since the ingredients for the Fock matrix has already been built, we can
        # just go ahead and use it to determine the orthonormal basis vectors.
        fock = mf.get_fock(h1e, s1e, vhf, dm)
        _, mf_diis.Corth = mf.eig(fock, s1e)
    else:
        mf_diis = None

    if dump_chk and mf.chkfile:
        # Explicit overwrite the mol object in chkfile
        # Note in pbc.scf, mf.mol == mf.cell, cell is saved under key "mol"
        scf.hf.chkfile.save_mol(mol, mf.chkfile)

    # A preprocessing hook before the SCF iteration
    mf.pre_kernel(locals())

    print("converged Tr[D]=", numpy.trace(dm) / 2.0)
    nmode = model.vec.shape[0]

    for cycle in range(mf.max_cycle):
        dm_last = dm
        last_hf_e = e_tot

        # fock = mf.get_fock(h1e, s1e, vhf, dm)
        fock = h1e + vhf

        """
        fock = mf.get_fock(h1e, s1e, vhf, dm, cycle, mf_diis)
        """

        mu_mo = lib.einsum("pq, Xpq ->X", dm, mu_ao)

        scaled_mu = 0.0
        z_alpha = 0.0
        for imode in range(nmode):
            z_alpha -= numpy.dot(mu_mo, model.vec[imode])
            scaled_mu += numpy.einsum("pq, pq ->", dm, model.gmat[imode])

        dse = 0.5 * z_alpha * z_alpha

        # oei = numpy.zeros((h1e.shape[0], h1e.shape[1]))
        oei = model.gmat * z_alpha
        oei -= model.qd2
        oei = numpy.sum(oei, axis=0)
        fock += oei

        #  <>
        for imode in range(nmode):
            fock += scaled_mu * model.gmat[imode]
        fock -= 0.5 * numpy.einsum("Xpr, Xqs, rs -> pq", model.gmat, model.gmat, dm)

        # e_tot = mf.energy_tot(dm, h1e, fock - h1e + oei) + dse
        e_tot = 0.5 * numpy.einsum("pq,pq->", (oei + h1e + fock), dm) + nuc + dse

        mo_energy, mo_coeff = mf.eig(fock, s1e)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)

        # factor of 2 is applied (via mo_occ)
        dm = mf.make_rdm1(mo_coeff, mo_occ)
        # attach mo_coeff and mo_occ to dm to improve DFT get_veff efficiency
        dm = lib.tag_array(dm, mo_coeff=mo_coeff, mo_occ=mo_occ)
        vhf = mf.get_veff(mol, dm, dm_last, vhf)

        """
      # Here Fock matrix is h1e + vhf, without DIIS.  Calling get_fock
      # instead of the statement "fock = h1e + vhf" because Fock matrix may
      # be modified in some methods.

      fock = mf.get_fock(h1e, s1e, vhf, dm)  # = h1e + vhf, no DIIS
      oei = model.gmat * model.gmat
      oei -= model.qd2
      oei = numpy.sum(oei, axis=0)

      fock += oei

      mu_mo = lib.einsum('pq, Xpq ->X', 2 * dm, mu_ao)
      z_alpha = 0.0
      scaled_mu = 0.0
      for imode in range(nmode):
          z_alpha += -numpy.dot(mu_mo, model.vec[imode])
          scaled_mu += numpy.einsum('pq, pq ->', dm, model.gmat[imode])
      dse = 0.5 * z_alpha * z_alpha

      for imode in range(nmode):
          fock += 2 * scaled_mu * model.gmat[imode]
      fock -= numpy.einsum('Xpr, Xqs, rs -> pq', model.gmat, model.gmat, dm)
      """

        norm_gorb = numpy.linalg.norm(mf.get_grad(mo_coeff, mo_occ, fock))
        if not TIGHT_GRAD_CONV_TOL:
            norm_gorb = norm_gorb / numpy.sqrt(norm_gorb.size)
        norm_ddm = numpy.linalg.norm(dm - dm_last)
        print(
            "cycle= %3d E= %.12g  delta_E= %4.3g |g|= %4.3g |ddm|= %4.3g |d*u|= %4.3g dse= %4.3g"
            % (cycle + 1, e_tot, e_tot - last_hf_e, norm_gorb, norm_ddm, z_alpha, dse)
        )
        if noscf:
            break

        if callable(mf.check_convergence):
            scf_conv = mf.check_convergence(locals())
        elif abs(e_tot - last_hf_e) < conv_tol and norm_gorb < conv_tol_grad:
            scf_conv = True

        if dump_chk:
            mf.dump_chk(locals())

        if callable(callback):
            callback(locals())

        if scf_conv:
            break

    # to be updated
    if scf_conv and conv_check:
        # An extra diagonalization, to remove level shift
        # fock = mf.get_fock(h1e, s1e, vhf, dm)  # = h1e + vhf
        mo_energy, mo_coeff = mf.eig(fock, s1e)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        dm, dm_last = mf.make_rdm1(mo_coeff, mo_occ), dm
        dm = lib.tag_array(dm, mo_coeff=mo_coeff, mo_occ=mo_occ)
        vhf = mf.get_veff(mol, dm, dm_last, vhf)
        # e_tot, last_hf_e = mf.energy_tot(dm, h1e, vhf), e_tot
        e_tot, last_hf_e = mf.energy_tot(dm, h1e, fock - h1e + oei) + dse, e_tot

        fock = mf.get_fock(h1e, s1e, vhf, dm)
        norm_gorb = numpy.linalg.norm(mf.get_grad(mo_coeff, mo_occ, fock))
        if not TIGHT_GRAD_CONV_TOL:
            norm_gorb = norm_gorb / numpy.sqrt(norm_gorb.size)
        norm_ddm = numpy.linalg.norm(dm - dm_last)

        conv_tol = conv_tol * 10
        conv_tol_grad = conv_tol_grad * 3
        if callable(mf.check_convergence):
            scf_conv = mf.check_convergence(locals())
        elif abs(e_tot - last_hf_e) < conv_tol or norm_gorb < conv_tol_grad:
            scf_conv = True
        print(
            "Extra cycle  E= %.15g  delta_E= %4.3g  |g|= %4.3g  |ddm|= %4.3g",
            e_tot,
            e_tot - last_hf_e,
            norm_gorb,
            norm_ddm,
        )
        if dump_chk:
            mf.dump_chk(locals())

    # A post-processing hook before return
    mf.post_kernel(locals())
    print("HOMO-LUMO gap=", mo_energy[na] - mo_energy[na - 1])
    print("QEDHF energy=", e_tot)

    return scf_conv, e_tot, dse, mo_energy, mo_coeff, mo_occ
