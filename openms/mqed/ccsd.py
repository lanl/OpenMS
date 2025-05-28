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
QED coupled-cluster (CC)-U1/2n-Sn
"""

import numpy
import sys
from cqcpy import cc_equations
from cqcpy import cc_energy
from . import epcc_equations
from . import qedcc_equations
#from . import myqedcc_equations
from . import myqedcc_equations_opt as myqedcc_equations
#from . import epcc_equations_many
#from . import epcc_equations_gen

from pyscf import lib
from pyscf.lib import logger
from pyscf import __config__
from pyscf.mp.mp2 import get_nocc, get_nmo, get_frozen_mask, get_e_hf, _mo_without_core

from pyscf import lib
einsum = lib.einsum

def eph_energy(t1,s1,u11,g,G):
    Eeph = 0.0
    Eeph = einsum('I,I->',G,s1)
    Eeph += einsum('Iia,Iai->',g,u11)
    Eeph += einsum('Iia,ai,I->',g,t1,s1)
    return Eeph

def epcc_energy(t1, t2, s1, u11, f, eri, w, g ,G):
    Ecc = cc_energy.cc_energy(t1,t2,f,eri)
    if g is not None:
        Eeph = eph_energy(t1,s1,u11,g,G)
        return Ecc + Eeph
    return Ecc

def getDsn(w, nfock):

    if nfock < 1: return None

    Dsn = [None] * nfock
    Dsn[0] = - w

    for i in range(1, nfock):
        Dsn[i] = Dsn[i-1][...,None] - numpy.expand_dims(w, axis=tuple(range(i)))

    return Dsn

def getD1p(eo, ev, w, nfock):
    """
    denominator for U1n
    """

    if nfock < 1: return None

    D1p = [None] * nfock
    D1p[0] = eo[None, None, :] - ev[None, :, None] - w[:, None, None]
    for i in range(1, nfock):
        D1p[i] = D1p[i-1][None,...] - numpy.expand_dims(w, axis=tuple(range(i+2)))
    return D1p

def getD2p(eo, ev, w, nfock):
    """
    denominator for U2n
    """

    if nfock < 1: return None
    D2p = [None] * nfock

    D2p[0] = -ev[None,:,None,None,None] - ev[None,None,:,None,None] \
        + eo[None,None,None,:,None] + eo[None,None,None,None,:] \
        - w[:,None,None,None,None]

    for i in range(1, nfock):
        D2p[i] = D2p[i-1][None,...] - numpy.expand_dims(w, axis=tuple(range(i+4)))
    return D2p


def kernel(mycc, eris=None, t1=None, t2=None, Sn=None, Un=None,
           theory='polariton',
           ret=False):

    """
    nfock1 = order of photon operator (b^\dag) in pure photonic excitaiton (T_p)
    nfock2 = order of photon operator in coupled excitaiton T_ep
    """
    if isinstance(mycc._qed, list): # mamny molecule case
        return epcc_nfock_many(mycc, ret, theory)

    nfock = nfock1 = mycc.nfock1
    nfock2 = mycc.nfock2
    ethresh = mycc.ethresh
    tthresh = mycc.tthresh
    useslow = mycc.useslow
    damp = mycc.damp

    if isinstance(mycc.diis, lib.diis.DIIS):
        adiis = mycc.diis
    elif mycc.diis:
        adiis = lib.diis.DIIS(mycc, mycc.diis_file, incore=mycc.incore_complete)
        adiis.space = mycc.diis_space
    else:
        adiis = None

    # ---------------- -----------------------------------------
    # get ERIs
    F, I, w, g, h, G, H = mycc.qed_eris()

    # orbital energies, denominators, and Fock matrix in spin-orbital basis
    eo = F.oo.diagonal()
    ev = F.vv.diagonal()

    F.oo = F.oo - numpy.diag(eo)
    F.vv = F.vv - numpy.diag(ev)

    no = eo.shape[0]
    nv = ev.shape[0]
    np = 0
    if w is not None:
        np = w.shape[0]

    # get HF energy
    Ehf = mycc.e_hf
    logger.info(mycc, "Hartree-Fock energy: %.10f", mycc.e_hf)
    Ehf2 = mycc._qed.hf_energy()
    logger.info(mycc, "Hartree-Fock energy2: %.10f", mycc.e_hf)

    D1 = eo[:,None] - ev[None,:]
    D2 = eo[:,None,None,None] + eo[None,:,None,None]\
            - ev[None,None,:,None] - ev[None,None,None,:]

    D1p = getD1p(eo, ev, w, nfock2)
    if mycc.add_U2n:
        D2p = getD2p(eo, ev, w, nfock2)
    Dsn = getDsn(w, nfock1)

    if np > 0: G = H = numpy.zeros(np)
    # ----------------------------------------------------------

    # DIIS
    # build MP2 T,S,U amplitudes
    #amps = mycc.init_amps(F, I, w, g, D1, D2, D1p, D2p, nfock1, nfock2)

    T1old = F.vo/D1.transpose((1,0))
    T2old = I.vvoo/D2.transpose((2,3,0,1))

    S1old = None
    U11old = None
    if nfock1 > 0:
        S1old = -G/w
    if nfock2 > 0: U11old = h.vo/D1p[0] #.transpose((0,2,1))

    diis_T1 = [T1old.copy()]
    diis_T2 = [T2old.copy()]

    Eold = 0.0
    if g is None:
        Eccsd = epcc_energy(T1old,T2old,S1old,U11old, F.ov, I.oovv, None, None, None)
    else:
        Eccsd = epcc_energy(T1old,T2old,S1old,U11old, F.ov, I.oovv, w, g.ov,G)
    if mycc.verbose > 1:
        print("Guess energy: {:.8f}".format(Eccsd))

    # sqare root of numbers of elements
    st1 = numpy.sqrt(T1old.size)
    st2 = numpy.sqrt(T2old.size)

    Snold = None  # None if nfock1 <= 0
    U1nold = None # None if nfock2 <= 0
    U2nold = None # None if nfock2 <= 0
    U2n = None
    if nfock1 > 0:
        # ssn and su1n
        ssn = [None] * nfock1
        su1n = [None] * nfock2
        Snold = [None] * nfock1
        U1nold = [None] * nfock2
        U2nold = [None] * nfock2

        diis_sn = []
        diis_un = []
        for k in range(nfock1):
            if k == 0:
                Snold[k] = -H/w
            else:
                shape  = [np for j in range(k+1)]
                Snold[k] = numpy.zeros(tuple(shape))
            ssn[k] = numpy.sqrt(Snold[k].size)
            diis_sn.append(Snold[k].copy())

        for k in range(nfock2):
            if k == 0:
                U1nold[k] =  U11old #h.vo/D1p[k].transpose((0,2,1))
                if mycc.add_U2n:
                    U2nold[k] =  numpy.zeros((np, nv, nv, no, no))
            else:
                shape  = [np for j in range(k+1)] + [nv, no]
                U1nold[k] =  numpy.zeros(tuple(shape))

                shape  = [np for j in range(k+1)] + [nv, nv, no, no]
                if mycc.add_U2n:
                    U2nold[k] =  numpy.zeros(tuple(shape))

            su1n[k] = numpy.sqrt(U1nold[k].size)
            diis_un.append(U1nold[k].copy())


    # coupled cluster iterations
    converged = False
    istep = 0
    while istep < mycc.max_cycle and not converged:

        amps = (T1old, T2old, Snold, U1nold)
        #if mycc.add_U2n:
        amps = (T1old, T2old, Snold, U1nold, U2nold)

        T1,T2, Sn, U1n, U2n = myqedcc_equations.new_qedccsd_sn_u2n(
        #T1,T2,Sn,U1n = epcc_equations.qed_ccsd_sn_u1n_opt(
                    F, I, w, g, h, G, H, nfock1, nfock2, amps)

        T1 /= D1.transpose((1,0))
        T2 /= D2.transpose((2,3,0,1))

        for k in range(nfock1):
            Sn[k] /= Dsn[k]

        for k in range(nfock2):
            U1n[k] /= D1p[k] #.transpose((0,2,1))
            if mycc.add_U2n:
                U2n[k] /= D2p[k] #.transpose((...,2,3,0,1))

        res = numpy.linalg.norm(T1old - T1) #/st1
        res += numpy.linalg.norm(T2old - T2) #/st2

        # res of sn
        for k in range(nfock1):
            res += numpy.linalg.norm(Snold[k] - Sn[k]) #/ssn[k]
        # res of U1n / U2n
        for k in range(nfock2):
            res += numpy.linalg.norm(U1nold[k] - U1n[k]) #/su1n[k] #
            if mycc.add_U2n:
                res += numpy.linalg.norm(U2nold[k] - U2n[k])

        # for diis
        tmpvec = mycc.amplitudes2vector(T1, T2, Sn, U1n, U2n)
        tmpvec -= mycc.amplitudes2vector(T1old, T2old, Snold, U1nold, U2nold)
        normt = numpy.linalg.norm(tmpvec)
        tmpvec = None

        # Linear mixer
        if damp < 1.0:
            T1 = damp*T1old + (1.0 - damp)*T1
            T2 = damp*T2old + (1.0 - damp)*T2
            for k in range(nfock1):
                Sn[k] = damp*Snold[k] + (1.0 - damp)*Sn[k]
            for k in range(nfock2):
                U1n[k] = damp*U1nold[k] + (1.0 - damp)*U1n[k]
                if mycc.add_U2n:
                    U2n[k] = damp*U2nold[k] + (1.0 - damp)*U2n[k]

        T1old, T2old, Snold, U1nold, U2nold = T1, T2, Sn, U1n, U2n
        amps = (T1old, T2old, Snold, U1nold, U2nold)
        T1old, T2old, Snold, U1nold, U2nold = mycc.run_diis(amps, istep, normt, Eccsd-Eold, adiis)

        if Sn is not None:
            Eccsd = epcc_energy(T1old,T2old,Snold[0],U1nold[0],F.ov,I.oovv,w,g.ov,G)
        else:
            Eccsd = epcc_energy(T1old,T2old,None, None,F.ov,I.oovv,None, None, None)

        if g is not None:
            nocc = boson_occ(Sn, U1n[0], g.ov, nfock)

        Ediff = abs(Eccsd - Eold)
        if mycc.verbose > 0:
            if g is not None:
                print(' {:2d}  {:.10f}   {:.4E}  {:.4E}  {:.10f}'.format(istep+1, Eccsd, res, normt, nocc[0]), flush=True)
            else:
                print(' {:2d}  {:.10f}   {:.4E}  {:.4E}'.format(istep+1, Eccsd, res, normt), flush=True)
        if Ediff < ethresh and res < tthresh:
            converged = True

        #sys.exit()
        Eold = Eccsd
        istep = istep + 1

    mycc.e_corr = Eold
    if mycc.verbose > 0:
        if not converged:
            print("WARNING: ep-CCSD did not converge!")
    if ret:
        return (Ehf + Eold, Eold, (T1,T2,Sn,U1n))
    else:
        return (Ehf + Eold, Eold)


class CCSD(lib.StreamObject):
    r"""

    """
    max_cycle = getattr(__config__, 'cc_ccsd_CCSD_max_cycle', 50)
    conv_tol = getattr(__config__, 'cc_ccsd_CCSD_conv_tol', 1e-7)
    iterative_damping = getattr(__config__, 'cc_ccsd_CCSD_iterative_damping', 1.0)
    conv_tol_normt = getattr(__config__, 'cc_ccsd_CCSD_conv_tol_normt', 1e-5)

    diis = getattr(__config__, 'cc_ccsd_CCSD_diis', True)
    diis_space = getattr(__config__, 'cc_ccsd_CCSD_diis_space', 6)
    diis_file = None
    diis_start_cycle = getattr(__config__, 'cc_ccsd_CCSD_diis_start_cycle', 0)
    # FIXME: Should we avoid DIIS starting early?
    diis_start_energy_diff = getattr(__config__, 'cc_ccsd_CCSD_diis_start_energy_diff', 1e9)

    direct = getattr(__config__, 'cc_ccsd_CCSD_direct', False)
    async_io = getattr(__config__, 'cc_ccsd_CCSD_async_io', True)
    incore_complete = getattr(__config__, 'cc_ccsd_CCSD_incore_complete', False)

    def __init__(self, qed, frozen=None, mo_coeff=None, mo_occ=None, **kwargs):

        from pyscf.scf import hf
        if isinstance(qed, hf.KohnShamDFT) or isinstance(qed, hf.SCF):
            raise RuntimeError('QED-CCSD Warning: The first argument qed is a DFT or SCF object. '
                               'QED-CCSD calculation should be initialized with Boson object.\n')

        mf = qed._mf
        self._qed = qed

        if mo_coeff is None: mo_coeff = mf.mo_coeff
        if mo_occ is None: mo_occ = mf.mo_occ

        self.mol = mf.mol
        self._scf = mf
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.max_memory = mf.max_memory

        self.frozen = frozen
        self.incore_complete = self.incore_complete or self.mol.incore_anyway
        self.level_shift = 0

        nfock1 = kwargs['nfock1'] if 'nfock1' in kwargs else 1
        nfock2 = kwargs['nfock2'] if 'nfock2' in kwargs else 1

        self.add_U2n = kwargs['add_U2n'] if 'add_U2n' in kwargs else False
        damp = kwargs['damp'] if 'damp' in kwargs else 0.4
        ethresh = kwargs['ethresh'] if 'ethresh' in kwargs else 1.e-7
        tthresh = kwargs['tthresh'] if 'tthresh' in kwargs else 1.e-6

        self.useslow = kwargs['useslow'] if 'useslow' in kwargs else False

##################################################
# don't modify the following attributes, they are not input options
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self.converged = False
        self.converged_lambda = False
        self.emp2 = None
        self.e_hf = None
        self.e_corr = None
        self.t1 = None
        self.t2 = None
        self.l1 = None
        self.l2 = None
        self._nocc = None
        self._nmo = None
        self.chkfile = mf.chkfile
        self.callback = None

        self.nfock1 = nfock1
        self.nfock2 = nfock2
        self.damp = damp
        self.ethresh = ethresh
        self.tthresh = tthresh

        keys = set(('max_cycle', 'conv_tol', 'iterative_damping',
                    'conv_tol_normt', 'diis', 'diis_space', 'diis_file',
                    'diis_start_cycle', 'diis_start_energy_diff', 'direct',
                    'async_io', 'incore_complete', 'cc2', 'nfock1', 'nfock2',
                    'damp', 'ethresh', 'tthresh'))
        self._keys = set(self.__dict__.keys()).union(keys)

    @property
    def ecc(self):
        return self.e_corr

    @property
    def e_tot(self):
        return self.e_hf + self.e_corr

    get_e_hf = get_e_hf

    def dump_flags(self):
        logger.info(self, '')
        logger.info(self, '----ccsd code in OpenMS-----')
        logger.info(self, '******** Flags of  %s ********', self.__class__)
        logger.info(self, 'nfock1            = %d', self.nfock1)
        logger.info(self, 'nfock2            = %d', self.nfock2)
        logger.info(self, 'Add U2n term?     = %s', self.add_U2n)
        if self.frozen is not None:
            logger.info(self, 'frozen orbitals?  = %s', self.frozen)
        logger.info(self, 'max_cycle         = %d', self.max_cycle)
        logger.info(self, 'direct            = %d', self.direct)
        logger.info(self, 'conv_tol          = %g', self.conv_tol)
        logger.info(self, 'conv_tol_normt    = %s', self.conv_tol_normt)
        logger.info(self, 'diis_space        = %d', self.diis_space)
        #log.info('diis_file = %s', self.diis_file)
        logger.info(self, 'diis_start_cycle  = %d', self.diis_start_cycle)
        logger.info(self, 'diis_start_energy_diff = %g', self.diis_start_energy_diff)
        logger.info(self, 'max_memory        = %d MB ', self.max_memory)
        logger.info(self, 'used memory       = %d MB)', lib.current_memory()[0])
        if (self.verbose >= logger.DEBUG1 and
            self.__class__ == CCSD):
            nocc = self.nocc
            nvir = self.nmo - self.nocc
            flops = _flops(nocc, nvir)
            logger.debug1(self, 'total FLOPs %s', flops)
        logger.info(self, '***************end of dumpling flags ****************\n')
        return self

    def qed_eris(self):
        r"""get F, I, g, w, H
        """
        self._qed.kernel()
        F = self._qed.g_fock()
        I = self._qed.get_I()
        w = None
        g = None
        h = None
        G = None
        H = None
        if self.nfock1 > 0:
            # get normal mode energies
            w = self._qed.get_omega()
            # get elec-phon matrix elements
            g, h = self._qed.gint()
            G, H = self._qed.mfG()
        return F, I, w, g, h, G, H

    def init_amps(self, F, I, w, g, D1, D2, D1p, D2p, nfock1, nfock2):
        r"""
        Initialize ccsd amplitudes
        """
        pass

    def amplitudes2vector(self, T1, T2, Sn, U1n, U2n, out=None):
        nocc, nvir = T1.shape
        nov = nocc * nvir
        size = nov + nov*(nov+1)//2
        idx = size

        Sn_size = 0
        if Sn is not None:
            Sn_size = sum([S.size for S in Sn])

        U1n_size = 0
        if U1n is not None:
            U1n_size = sum([U.size for U in U1n])
        U2n_size = 0
        if self.add_U2n and U2n is not None:
            U2n_size = sum([U.size for U in U2n])
        size += Sn_size + U1n_size + U2n_size
        vector = numpy.ndarray(size, T1.dtype, buffer=out)

        # T1 + T2
        vector[:nov] = T1.ravel()
        lib.pack_tril(T2.transpose(0,2,1,3).reshape(nov,nov), out=vector[nov:])
        if Sn_size > 0:
            for S in Sn:
                vector[idx:idx+S.size] = S.ravel()
                idx += S.size
        if U1n_size > 0:
            for U in U1n:
                vector[idx:idx+U.size] = U.ravel()
                idx += U.size
        if U2n_size > 0:
            for U in U2n:
                vector[idx:idx+U.size] = U.ravel()
                idx += U.size
        return vector

    def vector2amplitudes(self, vector, amps):
        T1, T2, Sn, U1n, U2n = amps

        nocc, nvir = T1.shape
        nov = nocc * nvir
        size = nov + nov*(nov+1)//2
        idx = size
        T1 = vector[:nov].copy().reshape((nocc,nvir))
        T2 = lib.unpack_tril(vector[nov:size], filltriu=lib.SYMMETRIC)
        T2 = T2.reshape(nocc,nvir,nocc,nvir).transpose(0,2,1,3)
        T2 = numpy.asarray(T2, order='C')

        if Sn is not None:
            for S in Sn:
                S = vector[idx:idx+S.size].reshape(S.shape)
                idx += S.size
        if U1n is not None:
            for S in U1n:
                S = vector[idx:idx+S.size].reshape(S.shape)
                idx += S.size
        if self.add_U2n and U2n is not None:
            for S in U2n:
                S = vector[idx:idx+S.size].reshape(S.shape)
                idx += S.size

        return (T1, T2, Sn, U1n, U2n)

    def run_diis(self, amps, istep, normt, de, adiis):
        T1, T2, Sn, U1n, U2n = amps

        if (adiis and istep >= self.diis_start_cycle and
            abs(de) < self.diis_start_energy_diff):
            vec = self.amplitudes2vector(T1, T2, Sn, U1n, U2n)
            T1, T2, Sn, U1n, U2n = self.vector2amplitudes(adiis.update(vec), amps)
            logger.debug1(self, 'DIIS for step %d', istep)
        return (T1, T2, Sn, U1n, U2n)

    def kernel(self, t1=None, t2=None, Sn=None, Un = None, eris=None, ret=False):
        return self.ccsd_nfock(t1, t2, Sn, Un, eris, ret)

    def ccsd_nfock(self, t1=None, t2=None, Sn=None, Un=None, eris=None, ret=False):
        self.dump_flags()

        self.e_hf = self.get_e_hf(mo_coeff=self.mo_coeff)

        results = kernel(self, t1, t2, Sn, Un, eris, ret)

        return results


if __name__ == "__main__":
    from openms.lib.boson import Photon
    # will add a loop
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
        verbose=5,
    )
    print("mol coordinates=\n", mol.atom_coords())

    hf = scf.HF(mol)
    hf.max_cycle = 200
    hf.conv_tol = 1.0e-8
    hf.diis_space = 10
    hf.polariton = True
    mf = hf.run()

    nmodes = 1
    omega = numpy.zeros(nmodes)
    gfac = numpy.zeros(nmodes)
    vec = numpy.zeros((nmodes, 3))
    gfac[0] = 0.05
    omega[0] = 0.5
    vec[0, :] = [0.0, 0.0, 1.0]  # [1.0, 1.0, 1.0]
    vec[0, :] = vec[0, :] / numpy.sqrt(numpy.dot(vec[0], vec[0]))

    qed = Photon(mol, mf=mf, omega=omega, vec=vec, gfac=gfac)
    qed.kernel()
    qed.get_mos()

    qedccsd = CCSD(qed, nfock1=1, nfock2=1)
    qedccsd.kernel()
