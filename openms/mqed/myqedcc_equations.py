#this code performs QED-CCSD calculations with complete double excitations
#t = T1old + S1old + T2old + S2old + U11old + U21old + U12old + U22old
#     T1      S1      T2      S2      U11     U21     U12     U22
import sys
import numpy
import math
from numba import jit

import time
import math
einsum = numpy.einsum

r"""
Use the symmetry to optimize the code
1) Sn/U1n/U2n: boson indices are permutation invariant
   i.e., IJK = IJK = JIK = JKI = KIJ = KJI
2) eri symmetry:

contraction (gov, T2old, Snold) is similar to (gov, U2nold)

"""

class Imds(object):
    r"""
    build intermediate variables
    """
    def __init__(self, F, I, w, g, h, G, H, nfock1, nfock2, amps):
        T1old, T2old, Snold, U1nold, U2nold = amps
        self.T2A = None
        self.Xov = None
        self.Foo = None
        self.Woovv = None
        self.F = F
        self.I = I
        self.w = w
        self.g = g
        self.T1old = T1old
        self.T2old = T2old
        self.Snold = Snold
        self.U1nold = U1nold
        self.U2nold = U2nold

    def build(self, F, I, w, g, h, G, H, nfock1, nfock2, amps):
        T1old, T2old, Snold, U1nold, U2nold = amps

        self.T2A = T2old.copy()
        self.T1T1 = einsum('ai,bj->abij', T1old, T1old)
        self.T2A += 0.5 * self.T1T1
        self.T2A -= 0.5*einsum('bi,aj->abij', T1old, T1old)

        self.Xov = einsum('ijab,xbj->xia', I.oovv, U1nold[0])

        self.Fvv = F.vv.copy()
        # 0.5 comes from the spliting into two (otehr part moved to Fvv)
        self.Fvv -= 0.5*einsum('jb,aj->ab',F.ov, T1old)
        self.Fvv -= einsum('ajcb,cj->ab',I.vovv, T1old)
        self.Fvv -= 0.5*einsum('jkbc,acjk->ab',I.oovv,self.T2A)
        if nfock1 > 0:
            self.Fvv += einsum('xab,x->ab', g.vv, Snold[0])
        if nfock2> 0:
            self.Fvv -= einsum('yjb,yaj->ab', g.ov, U1nold[0])

        self.Foo = F.oo.copy()
        # 0.5 comes from the spliting into two (other part moved to Fvv)
        self.Foo += 0.5*einsum('jb,bi->ji',F.ov, T1old)
        self.Foo += einsum('jkib,bk->ji',I.ooov, T1old)
        self.Foo += 0.5*einsum('jkbc,bcik->ji',I.oovv,self.T2A)
        if nfock1 > 0:
            self.Foo += einsum('xji,x->ji', g.oo, Snold[0])
        if nfock2> 0:
            self.Foo += einsum('yjb,ybi->ji', g.ov, U1nold[0])

        self.Wovvo = -I.vovo.transpose((1,0,2,3))
        self.Wovvo -= einsum('bkcd,dj->kbcj',I.vovv, T1old)
        self.Wovvo += einsum('kljc,bl->kbcj',I.ooov, T1old)
        temp = 0.5*T2old + einsum('dj,bl->dbjl', T1old, T1old)
        self.Wovvo -= einsum('klcd,dbjl->kbcj',I.oovv, temp)

        self.gU1xy = 1.0*einsum('Jia,Iai->JI', g.ov, U1nold[0], optimize=True)
        # xia, ai -> x
        self.gtm = einsum('xia,ai->x', g.ov, T1old)

        self.gsov = einsum('yia,y->ia', g.ov, Snold[0])
        #if nfock1 > 1:
        #    self.gsov += einsum('yia,xy->ia', g.ov, Snold[1])

        self.Fov = F.ov.copy()
        self.Fov += einsum('jkbc,ck->jb',I.oovv,T1old)
        self.Fov += self.gsov

        self.gT1oo = einsum('Ijb,bi->Iji', g.ov, T1old, optimize=True)
        self.gT1T1ov = einsum('Iji,aj->Iai', self.gT1oo, T1old, optimize=True)

        self.gvoU11_oovv = einsum('Iai,Ibj->abij', g.vo, U1nold[0])

def new_qedccsd_sn_u2n(F, I, w, g, h, G, H, nfock1, nfock2, amps):
    imds = Imds(F, I, w, g, h, G, H, nfock1, nfock2, amps)
    imds.build(F, I, w, g, h, G, H, nfock1, nfock2, amps)

    #T1 = qedccsd_T1(F, I, w, g, h, G, H, nfock1, nfock2, amps, imds)
    #T2 = qedccsd_T2(F, I, w, g, h, G, H, nfock1, nfock2, amps, imds)
    T1, T2 = qedccsd_T1T2_opt(F, I, w, g, h, G, H, nfock1, nfock2, amps)

    Sn = qedccsd_Sn(F, I, w, g, h, G, H, nfock1, nfock2, amps, imds)
    #U1n = qedccsd_U1n(F, I, w, g, h, G, H, nfock1, nfock2, amps, imds)
    #U2n = qedccsd_U2n(F, I, w, g, h, G, H, nfock1, nfock2, amps, imds)
    U1n = qedccsd_U1n_opt(F, I, w, g, h, G, H, nfock1, nfock2, amps, imds)
    U2n = qedccsd_U2n_opt(F, I, w, g, h, G, H, nfock1, nfock2, amps, imds)

    return T1, T2, Sn, U1n, U2n

def qedccsd_T1(F, I, w, g, h, G, H, nfock1, nfock2, amps, imds):

    T1old, T2old, Snold, U1nold, U2nold = amps

    nvir = T2old.shape[0]
    nocc = T2old.shape[2]

    eps_occ = F.oo.diagonal()
    eps_vir = F.vv.diagonal()
    #e_denom = 1 / (eps_occ.reshape(-1, 1) - eps_vir)

    T1 = F.vo.copy()

    # T1 from T1old
    #T1 -= einsum('ji,aj->ai', imds.Foo, T1old)
    T1 += -1.0*einsum('ji,aj->ai', F.oo, T1old)
    T1 += -1.0*einsum('Iji,aj,I->ai', g.oo, T1old, Snold[0], optimize=True)
    T1 += -1.0*einsum('Ijb,bi,Iaj->ai', g.ov, T1old, U1nold[0], optimize=True)
    T1 += -1.0*einsum('jkib,aj,bk->ai', I.ooov, T1old, T1old, optimize=True)

    #T1 += 1.0*einsum('ab,bi->ai', imds.Fvv, T1old)
    T1 += 1.0*einsum('ab,bi->ai', F.vv, T1old)
    T1 += 1.0*einsum('Iab,bi,I->ai', g.vv, T1old, Snold[0], optimize=True)
    T1 += -1.0*einsum('jb,bi,aj->ai', F.ov, T1old, T1old, optimize=True)
    T1 += -1.0*einsum('Ijb,aj,Ibi->ai', g.ov, T1old, U1nold[0], optimize=True)
    T1 += 1.0*einsum('jabc,ci,bj->ai', I.ovvv, T1old, T1old, optimize=True)

    T1 += -1.0*einsum('jaib,bj->ai', I.ovov, T1old, optimize=True)

    # From T2old
    Fov = F.ov.copy()
    Fov += einsum('jkbc,ck->jb',I.oovv, T1old)
    T1 += einsum('jb,abij->ai',Fov, T2old)
    #T1 += -1.0*einsum('jb,abji->ai', F.ov, T2old, optimize=True)
    #T1 += 1.0*einsum('jkbc,cj,abki->ai', I.oovv, T1old, T2old, optimize=True)

    T1 += 0.5*einsum('jkib,abkj->ai', I.ooov, T2old, optimize=True)
    T1 += -0.5*einsum('jabc,cbji->ai', I.ovvv, T2old, optimize=True)

    # below are absorbed into Fvv and Foo via T2A
    T1 += -0.5*einsum('jkbc,aj,cbki->ai', I.oovv, T1old, T2old, optimize=True)
    T1 += -0.5*einsum('jkbc,ci,abkj->ai', I.oovv, T1old, T2old, optimize=True)
    T1 += 1.0*einsum('jkbc,ci,aj,bk->ai', I.oovv, T1old, T1old, T1old, optimize=True)

    # from Sn
    T1 += 1.0*einsum('Iai,I->ai', g.vo, Snold[0])  # gsov
    #T1 += 1.0 * einsum('I,Iai->ai', G, U1nold[0], optimize=True)
    T1 += -1.0*einsum('Ijb,abji,I->ai', g.ov, T2old, Snold[0], optimize=True)

    T1 += -1.0*einsum('I,Iai->ai', Snold[0], imds.gT1T1ov, optimize=True)
    #T1 += -1.0*einsum('I,Ijb,bi,aj->ai', Snold[0], g.ov, T1old, T1old, optimize=True)

    # from U1n
    T1 += -1.0*einsum('Iji,Iaj->ai', g.oo, U1nold[0], optimize=True)
    T1 += 1.0*einsum('Iab,Ibi->ai', g.vv, U1nold[0], optimize=True)
    T1 += 1.0*einsum('I,Iai->ai', imds.gtm, U1nold[0], optimize=True)
    #T1 += 1.0*einsum('Ijb,bj,Iai->ai', g.ov, T1old, U1nold[0], optimize=True)

    if U2nold[0] is not None:
        # basically, this is can be combined with T2old * Snold[0]
        T1 += -1.0*einsum('Ijb,Iabji->ai', g.ov, U2nold[0], optimize=True)

    #T1old += numpy.einsum('ai,ia -> ai', res_T1old, e_denom, optimize=True)

    return T1

def qedccsd_T2(F, I, w, g, h, G, H, nfock1, nfock2, amps, imds):

    T1old, T2old, Snold, U1nold, U2nold = amps

    nvir = T2old.shape[0]
    nocc = T2old.shape[2]

    eps_occ = F.oo.diagonal()
    eps_vir = F.vv.diagonal()

    #e_denom = 1 / (eps_occ.reshape(-1, 1, 1, 1) + eps_occ.reshape(-1, 1) - eps_vir.reshape(-1, 1, 1) - eps_vir)

    T2 = I.vvoo.copy()
    #T2 += 1.0*einsum('baji->abij', I.vvoo)
    #T2 += 1.0 * einsum('I,baji->abijI', G, U21old, optimize=True)

    Foo = F.oo.copy()

    abij_temp = einsum('ki,bakj->abij', F.oo, T2old, optimize=True)
    T2 += 1.0 * abij_temp
    T2 += -1.0 * abij_temp.transpose((0,1,3,2))
    abij_temp = None
    #T2 += -1.0*einsum('kj,baki->abij', F.oo, T2old, optimize=True)

    Fvv = F.vv.copy()
    abij_temp = einsum('ac,bcji->abij', F.vv, T2old, optimize=True)
    T2 += 1.0* abij_temp
    T2 += -1.0* abij_temp.transpose((1,0,2,3))
    abij_temp = None
    #T2 += -1.0*einsum('bc,acji->abij', F.vv, T2old, optimize=True)

    T2 += imds.gvoU11_oovv
    T2 -= imds.gvoU11_oovv.transpose((1,0,2,3))
    T2 -= imds.gvoU11_oovv.transpose((0,1,3,2))
    T2 += imds.gvoU11_oovv.transpose((1,0,3,2))

    #T2 += 1.0*einsum('Iai,Ibj->abij', g.vo, U1nold[0], optimize=True)
    #T2 += -1.0*einsum('Iaj,Ibi->abij', g.vo, U1nold[0], optimize=True)
    #T2 += -1.0*einsum('Ibi,Iaj->abij', g.vo, U1nold[0], optimize=True)
    #T2 += 1.0*einsum('Ibj,Iai->abij', g.vo, U1nold[0], optimize=True)

    #-------------------------------------------------------------#
    T2B = T2old.copy()
    T2B += einsum('ai,bj->abij',T1old, T1old)
    T2B -= einsum('bi,aj->abij',T1old, T1old)

    Woooo = I.oooo.copy()
    #Woooo += einsum('klic,cj->klij',I.ooov,T1old)
    #Woooo -= einsum('kljc,ci->klij',I.ooov,T1old)
    #Woooo += 0.25*einsum('klcd,cdij->klij',I.oovv,T2B)
    T2 += 0.5*einsum('klij,abkl->abij',Woooo,T2B)
    Woooo = None

    # (I.oooo, T2B)
    #T2 += -0.5*einsum('klji,balk->abij', I.oooo, T2old, optimize=True)
    #T2 += 1.0*einsum('klji,bk,al->abij', I.oooo, T1old, T1old, optimize=True)

    # (I.ooov T1old) * T2B
    T2 += 0.5*einsum('klic,cj,balk->abij', I.ooov, T1old, T2old, optimize=True)
    T2 += -0.5*einsum('kljc,ci,balk->abij', I.ooov, T1old, T2old, optimize=True)
    T2 += 1.0*einsum('kljc,ci,bk,al->abij', I.ooov, T1old, T1old, T1old, optimize=True)
    T2 += -1.0*einsum('klic,cj,bk,al->abij', I.ooov, T1old, T1old, T1old, optimize=True)

    T2 += 1.0*einsum('klic,ak,bclj->abij', I.ooov, T1old, T2old, optimize=True)
    T2 += -1.0*einsum('klic,bk,aclj->abij', I.ooov, T1old, T2old, optimize=True)
    T2 += -1.0*einsum('kljc,ak,bcli->abij', I.ooov, T1old, T2old, optimize=True)
    T2 += 1.0*einsum('kljc,bk,acli->abij', I.ooov, T1old, T2old, optimize=True)
    T2 += -1.0*einsum('klic,ck,balj->abij', I.ooov, T1old, T2old, optimize=True)
    T2 += 1.0*einsum('kljc,ck,bali->abij', I.ooov, T1old, T2old, optimize=True)

    # the following contraction: 0.25 -> Woooo
    T2 += 1.0*einsum('klcd,di,cj,bk,al->abij', I.oovv, T1old, T1old, T1old, T1old, optimize=True)

    T2 += -1.0*einsum('kaji,bk->abij', I.ovoo, T1old, optimize=True)
    T2 += 1.0*einsum('kbji,ak->abij', I.ovoo, T1old, optimize=True)
    T2 += -1.0*einsum('baic,cj->abij', I.vvov, T1old, optimize=True)
    T2 += 1.0*einsum('bajc,ci->abij', I.vvov, T1old, optimize=True)
    T2 += 1.0*einsum('kaic,bckj->abij', I.ovov, T2old, optimize=True)
    T2 += -1.0*einsum('kajc,bcki->abij', I.ovov, T2old, optimize=True)
    T2 += -1.0*einsum('kbic,ackj->abij', I.ovov, T2old, optimize=True)
    T2 += 1.0*einsum('kbjc,acki->abij', I.ovov, T2old, optimize=True)
    T2 += -0.5*einsum('bacd,dcji->abij', I.vvvv, T2old, optimize=True)

    # moved into Foo Fvv ?
    T2 += -1.0*einsum('kc,ak,bcji->abij', F.ov, T1old, T2old, optimize=True)
    T2 += 1.0*einsum('kc,bk,acji->abij', F.ov, T1old, T2old, optimize=True)
    T2 += 1.0*einsum('kc,ci,bakj->abij', F.ov, T1old, T2old, optimize=True)
    T2 += -1.0*einsum('kc,cj,baki->abij', F.ov, T1old, T2old, optimize=True)

    T2 += 1.0*einsum('Iki,bakj,I->abij', g.oo, T2old, Snold[0], optimize=True)
    T2 += -1.0*einsum('Ikj,baki,I->abij', g.oo, T2old, Snold[0], optimize=True)
    T2 += 1.0*einsum('Iac,bcji,I->abij', g.vv, T2old, Snold[0], optimize=True)
    T2 += -1.0*einsum('Ibc,acji,I->abij', g.vv, T2old, Snold[0], optimize=True)

    T2 += -1.0*einsum('Iki,ak,Ibj->abij', g.oo, T1old, U1nold[0], optimize=True)
    T2 += 1.0*einsum('Iki,bk,Iaj->abij', g.oo, T1old, U1nold[0], optimize=True)
    T2 += 1.0*einsum('Ikj,ak,Ibi->abij', g.oo, T1old, U1nold[0], optimize=True)
    T2 += -1.0*einsum('Ikj,bk,Iai->abij', g.oo, T1old, U1nold[0], optimize=True)
    T2 += 1.0*einsum('Iac,ci,Ibj->abij', g.vv, T1old, U1nold[0], optimize=True)
    T2 += -1.0*einsum('Iac,cj,Ibi->abij', g.vv, T1old, U1nold[0], optimize=True)
    T2 += -1.0*einsum('Ibc,ci,Iaj->abij', g.vv, T1old, U1nold[0], optimize=True)
    T2 += 1.0*einsum('Ibc,cj,Iai->abij', g.vv, T1old, U1nold[0], optimize=True)
    T2 += 1.0*einsum('kaic,cj,bk->abij', I.ovov, T1old, T1old, optimize=True)
    T2 += -1.0*einsum('kajc,ci,bk->abij', I.ovov, T1old, T1old, optimize=True)
    T2 += -1.0*einsum('kbic,cj,ak->abij', I.ovov, T1old, T1old, optimize=True)
    T2 += 1.0*einsum('kbjc,ci,ak->abij', I.ovov, T1old, T1old, optimize=True)
    T2 += 1.0*einsum('bacd,di,cj->abij', I.vvvv, T1old, T1old, optimize=True)

    T2 += 1.0*einsum('Ikc,acji,Ibk->abij', g.ov, T2old, U1nold[0], optimize=True)
    T2 += -1.0*einsum('Ikc,bcji,Iak->abij', g.ov, T2old, U1nold[0], optimize=True)
    T2 += -1.0*einsum('Ikc,baki,Icj->abij', g.ov, T2old, U1nold[0], optimize=True)
    T2 += 1.0*einsum('Ikc,bakj,Ici->abij', g.ov, T2old, U1nold[0], optimize=True)

    # use transpose (gov, U11) -> abij and then contract with T2old
    T2 += -1.0*einsum('Ikc,acki,Ibj->abij', g.ov, T2old, U1nold[0], optimize=True)
    T2 += 1.0*einsum('Ikc,ackj,Ibi->abij', g.ov, T2old, U1nold[0], optimize=True)
    T2 += 1.0*einsum('Ikc,bcki,Iaj->abij', g.ov, T2old, U1nold[0], optimize=True)
    T2 += -1.0*einsum('Ikc,bckj,Iai->abij', g.ov, T2old, U1nold[0], optimize=True)

    T2 += 0.5*einsum('kacd,bk,dcji->abij', I.ovvv, T1old, T2old, optimize=True)
    T2 += -1.0*einsum('kacd,di,bckj->abij', I.ovvv, T1old, T2old, optimize=True)
    T2 += 1.0*einsum('kacd,dj,bcki->abij', I.ovvv, T1old, T2old, optimize=True)
    T2 += -1.0*einsum('kacd,dk,bcji->abij', I.ovvv, T1old, T2old, optimize=True)
    T2 += -0.5*einsum('kbcd,ak,dcji->abij', I.ovvv, T1old, T2old, optimize=True)
    T2 += 1.0*einsum('kbcd,di,ackj->abij', I.ovvv, T1old, T2old, optimize=True)
    T2 += -1.0*einsum('kbcd,dj,acki->abij', I.ovvv, T1old, T2old, optimize=True)
    T2 += 1.0*einsum('kbcd,dk,acji->abij', I.ovvv, T1old, T2old, optimize=True)
    T2 += 0.5*einsum('klcd,adji,bclk->abij', I.oovv, T2old, T2old, optimize=True)
    T2 += -1.0*einsum('klcd,adki,bclj->abij', I.oovv, T2old, T2old, optimize=True)
    T2 += -0.5*einsum('klcd,baki,dclj->abij', I.oovv, T2old, T2old, optimize=True)
    T2 += -0.5*einsum('klcd,bdji,aclk->abij', I.oovv, T2old, T2old, optimize=True)
    T2 += 1.0*einsum('klcd,bdki,aclj->abij', I.oovv, T2old, T2old, optimize=True)
    T2 += 0.25*einsum('klcd,dcji,balk->abij', I.oovv, T2old, T2old, optimize=True)
    T2 += -0.5*einsum('klcd,dcki,balj->abij', I.oovv, T2old, T2old, optimize=True)
    T2 += -1.0*einsum('Ikc,ak,bcji,I->abij', g.ov, T1old, T2old, Snold[0], optimize=True)
    T2 += 1.0*einsum('Ikc,bk,acji,I->abij', g.ov, T1old, T2old, Snold[0], optimize=True)

    #temp_abij = einsum('Iai,Ibj->abij', imds.gT1T1ov, U1nold[0], optimize=True)
    #TODO: alternatively, contract (gov, U11) and (T1, T1old), which can be comibined with
    # (gov, T2old, U1nold) contraction and use transpose for four different contractions

    T2 += -1.0*einsum('Iai,Ibj->abij', imds.gT1T1ov, U1nold[0], optimize=True)
    T2 += 1.0*einsum('Ibi,Iaj->abij', imds.gT1T1ov, U1nold[0], optimize=True)
    T2 += 1.0*einsum('Iaj,Ibi->abij', imds.gT1T1ov, U1nold[0], optimize=True)
    T2 += -1.0*einsum('Ibj,Iai->abij', imds.gT1T1ov, U1nold[0], optimize=True)

    #T2 += -1.0*einsum('Ikc,ci,ak,Ibj->abij', g.ov, T1old, T1old, U1nold[0], optimize=True)
    #T2 += 1.0*einsum('Ikc,ci,bk,Iaj->abij', g.ov, T1old, T1old, U1nold[0], optimize=True)
    #T2 += 1.0*einsum('Ikc,cj,ak,Ibi->abij', g.ov, T1old, T1old, U1nold[0], optimize=True)
    #T2 += -1.0*einsum('Ikc,cj,bk,Iai->abij', g.ov, T1old, T1old, U1nold[0], optimize=True)

    T2 += 1.0*einsum('Iki,bakj,I->abij', imds.gT1oo, T2old, Snold[0], optimize=True)
    T2 += -1.0*einsum('Ikj,baki,I->abij', imds.gT1oo, T2old, Snold[0], optimize=True)
    #T2 += 1.0*einsum('Ikc,ci,bakj,I->abij', g.ov, T1old, T2old, Snold[0], optimize=True)
    #T2 += -1.0*einsum('Ikc,cj,baki,I->abij', g.ov, T1old, T2old, Snold[0], optimize=True)

    T2 += -1.0*einsum('kacd,di,cj,bk->abij', I.ovvv, T1old, T1old, T1old, optimize=True)
    T2 += 1.0*einsum('kbcd,di,cj,ak->abij', I.ovvv, T1old, T1old, T1old, optimize=True)

    T2 += -1.0*einsum('klcd,ak,dl,bcji->abij', I.oovv, T1old, T1old, T2old, optimize=True)
    T2 += -0.5*einsum('klcd,bk,al,dcji->abij', I.oovv, T1old, T1old, T2old, optimize=True)
    T2 += 1.0*einsum('klcd,bk,dl,acji->abij', I.oovv, T1old, T1old, T2old, optimize=True)
    T2 += -1.0*einsum('klcd,di,ak,bclj->abij', I.oovv, T1old, T1old, T2old, optimize=True)
    T2 += 1.0*einsum('klcd,di,bk,aclj->abij', I.oovv, T1old, T1old, T2old, optimize=True)
    T2 += -0.5*einsum('klcd,di,cj,balk->abij', I.oovv, T1old, T1old, T2old, optimize=True)
    T2 += 1.0*einsum('klcd,di,ck,balj->abij', I.oovv, T1old, T1old, T2old, optimize=True)
    T2 += 1.0*einsum('klcd,dj,ak,bcli->abij', I.oovv, T1old, T1old, T2old, optimize=True)
    T2 += -1.0*einsum('klcd,dj,bk,acli->abij', I.oovv, T1old, T1old, T2old, optimize=True)
    T2 += -1.0*einsum('klcd,dj,ck,bali->abij', I.oovv, T1old, T1old, T2old, optimize=True)

    if U2nold[0] is not None:
        T2 += 1.0*einsum('Iki,Ibakj->abij', g.oo, U2nold[0], optimize=True)
        T2 += -1.0*einsum('Ikj,Ibaki->abij', g.oo, U2nold[0], optimize=True)
        T2 += 1.0*einsum('Iac,Ibcji->abij', g.vv, U2nold[0], optimize=True)
        T2 += -1.0*einsum('Ibc,Iacji->abij', g.vv, U2nold[0], optimize=True)

        #abij_temp = einsum('Iki,Ibakj->abij', imds.gT1oo, U2nold[0], optimize=True)
        T2 += -1.0*einsum('Ikc,ak,Ibcji->abij', g.ov, T1old, U2nold[0], optimize=True)
        T2 += 1.0*einsum('Ikc,bk,Iacji->abij', g.ov, T1old, U2nold[0], optimize=True)
        T2 += 1.0*einsum('Ikc,ci,Ibakj->abij', g.ov, T1old, U2nold[0], optimize=True)
        T2 += -1.0*einsum('Ikc,cj,Ibaki->abij', g.ov, T1old, U2nold[0], optimize=True)
        T2 += 1.0*einsum('I,Ibaji->abij', imds.gtm, U2nold[0], optimize=True)

    #T2 += numpy.einsum('abij,iajb -> abij', res_T2old, e_denom, optimize=True)

    return T2


def qedccsd_T1T2_opt(F, I, w, g, h, G, H, nfock1, nfock2, amps):

    T1old, T2old, Snold, U1nold, U2nold = amps

    nvir = T2old.shape[0]
    nocc = T2old.shape[2]

    #-------------------------------------------------------------------------
    # copy of  ccsd_opt (since some intermediate avaiable will be used later)
    # CCSD equations
    if nfock1 > 0:
        U11old = U1nold[0]
        S1old = Snold[0]

    T1 = F.vo.copy()
    if nfock1 > 0:
        T1 += einsum('xai,x->ai', g.vo, S1old)
        T1 += einsum('x,xai->ai',G, U11old)

    T2 = I.vvoo.copy()
    abij_temp = einsum('xai,xbj->abij', g.vo, U11old)
    T2 += abij_temp
    T2 -= abij_temp.transpose((1,0,2,3))
    T2 -= abij_temp.transpose((0,1,3,2))
    T2 += abij_temp.transpose((1,0,3,2))
    abij_temp = None

    T2A = T2old.copy()
    T1T1 = einsum('ai,bj->abij',T1old, T1old)

    T2A += 0.5*T1T1 #einsum('ai,bj->abij',T1old, T1old)
    T2A -= 0.5*einsum('bi,aj->abij',T1old, T1old)

    Fvv = F.vv.copy()
    Fvv -= 0.5*einsum('jb,aj->ab',F.ov,T1old)
    Fvv -= einsum('ajcb,cj->ab',I.vovv,T1old)
    Fvv -= 0.5*einsum('jkbc,acjk->ab',I.oovv,T2A)
    if nfock1 > 0:
        Fvv += einsum('xab,x->ab', g.vv, S1old)
        Fvv -= einsum('yjb,yaj->ab', g.ov, U11old)

    Foo = F.oo.copy()
    Foo += 0.5*einsum('jb,bi->ji',F.ov,T1old)
    Foo += einsum('jkib,bk->ji',I.ooov,T1old)
    Foo += 0.5*einsum('jkbc,bcik->ji',I.oovv,T2A)
    if nfock1 > 0:
        Foo += einsum('xji,x->ji', g.oo, S1old)
        Foo += einsum('yjb,ybi->ji', g.ov, U11old)
    T2A = None

    gsov = einsum('yia,y->ia', g.ov, S1old)
    gtm = einsum('xjb,bj->x', g.ov, T1old)
    Fov = F.ov.copy()
    Fov += einsum('jkbc,ck->jb',I.oovv,T1old)
    Fov += gsov

    T1 += einsum('ab,bi->ai',Fvv,T1old)
    T1 -= einsum('ji,aj->ai',Foo,T1old)
    T1 += einsum('jb,abij->ai',Fov,T2old)
    T1 -= einsum('ajbi,bj->ai',I.vovo,T1old)
    T1 += 0.5*einsum('ajbc,bcij->ai',I.vovv,T2old)
    T1 -= 0.5*einsum('jkib,abjk->ai',I.ooov,T2old)

    T2B = T2old.copy()
    T2B += einsum('ai,bj->abij',T1old, T1old)
    T2B -= einsum('bi,aj->abij',T1old, T1old)

    Woooo = I.oooo.copy()
    Woooo += einsum('klic,cj->klij',I.ooov,T1old)
    Woooo -= einsum('kljc,ci->klij',I.ooov,T1old)
    Woooo += 0.25*einsum('klcd,cdij->klij',I.oovv,T2B)
    T2 += 0.5*einsum('klij,abkl->abij',Woooo,T2B)
    Woooo = None

    Wvvvv = I.vvvv.copy()
    Wvvvv -= einsum('akcd,bk->abcd',I.vovv,T1old)
    Wvvvv += einsum('bkcd,ak->abcd',I.vovv,T1old)
    Wvvvv += 0.25*einsum('klcd,abkl->abcd',I.oovv,T2B)
    T2 += 0.5*einsum('abcd,cdij->abij',Wvvvv,T2B)
    T2B = None
    Wvvvv = None

    Wovvo = -I.vovo.transpose((1,0,2,3))
    Wovvo -= einsum('bkcd,dj->kbcj',I.vovv,T1old)
    Wovvo += einsum('kljc,bl->kbcj',I.ooov,T1old)
    temp = 0.5*T2old + einsum('dj,bl->dbjl',T1old, T1old)
    Wovvo -= einsum('klcd,dbjl->kbcj',I.oovv,temp)
    temp = einsum('kbcj,acik->abij',Wovvo,T2old)
    temp += einsum('bkcj,ci,ak->abij',I.vovo,T1old, T1old)
    T2 += temp
    T2 -= temp.transpose((0,1,3,2))
    T2 -= temp.transpose((1,0,2,3))
    T2 += temp.transpose((1,0,3,2))
    temp = None
    #Wovvo = None

    Ftemp = Fvv - 0.5*einsum('jb,aj->ab',Fov,T1old)
    temp_ab = einsum('bc,acij->abij',Ftemp,T2old)
    temp_ab += einsum('bkij,ak->abij',I.vooo,T1old)
    T2 += temp_ab
    T2 -= temp_ab.transpose((1,0,2,3))
    temp_ab = None

    Ftemp = Foo + 0.5*einsum('jb,bi->ji',Fov,T1old)
    temp_ij = -einsum('kj,abik->abij',Ftemp,T2old)
    temp_ij += einsum('abcj,ci->abij',I.vvvo,T1old)
    T2 += temp_ij
    T2 -= temp_ij.transpose((0,1,3,2))
    temp_ij = None

    # remaining T1 terms
    T1 += einsum('xab,xbi->ai', g.vv, U11old)
    T1 -= einsum('xji,xaj->ai',g.oo, U11old)

    T1 += einsum('x,xai->ai', gtm, U11old)
    ttt = einsum('jb,bi->ji', gsov, T1old)
    T1 -= einsum('ji,aj->ai', ttt, T1old)

    # Remaining T2 terms
    gtov = einsum('xac,ci->xai', g.vv, T1old)
    gtov -= einsum('xki,ak->xai', g.oo, T1old)
    temp_abij = einsum('xai,xbj->abij', gtov, U11old)
    T2temp = T2old - einsum('bi,aj->abij', T1old, T1old)
    Wmvo = einsum('xkc,acik->xai', g.ov, T2temp)
    temp_abij += einsum('xai,xbj->abij', Wmvo, U11old)
    T2 += temp_abij
    T2 -= temp_abij.transpose((1,0,2,3))
    T2 -= temp_abij.transpose((0,1,3,2))
    T2 += temp_abij.transpose((1,0,3,2))
    temp_abij = None

    gstv = einsum('kc,bk->bc', gsov, T1old)
    temp_ab = -0.5*einsum('bc,acij->abij', gstv, T2old)
    T2 += temp_ab
    T2 -= temp_ab.transpose((1,0,2,3))
    temp_ab = None

    gsto = einsum('kc,cj->kj', gsov, T1old)
    temp_ij = -0.5*einsum('kj,abik->abij', gsto, T2old)
    T2 += temp_ij
    T2 -= temp_ij.transpose((0,1,3,2))
    temp_ij = None

    if U2nold[0] is not None:
        T1 += -1.0*einsum('Ijb,Iabji->ai', g.ov, U2nold[0], optimize=True)

        gT2_ovvv = 1.0 * einsum('Ikc,Ibcij->bkij', g.ov, U2nold[0], optimize=True)

        T2 += 1.0*einsum('Iki,Ibakj->abij', g.oo, U2nold[0], optimize=True)
        T2 += -1.0*einsum('Ikj,Ibaki->abij', g.oo, U2nold[0], optimize=True)
        T2 += 1.0*einsum('Iac,Ibcji->abij', g.vv, U2nold[0], optimize=True)
        T2 += -1.0*einsum('Ibc,Iacji->abij', g.vv, U2nold[0], optimize=True)

        T2 += -1.0*einsum('Ikc,ak,Ibcji->abij', g.ov, T1old, U2nold[0], optimize=True)
        T2 += 1.0*einsum('Ikc,bk,Iacji->abij', g.ov, T1old, U2nold[0], optimize=True)
        T2 += 1.0*einsum('Ikc,ci,Ibakj->abij', g.ov, T1old, U2nold[0], optimize=True)
        T2 += -1.0*einsum('Ikc,cj,Ibaki->abij', g.ov, T1old, U2nold[0], optimize=True)
        T2 += 1.0*einsum('Ikc,ck,Ibaji->abij', g.ov, T1old, U2nold[0], optimize=True)

    return T1, T2


def qedccsd_Sn(F, I, w, g, h, G, H, nfock1, nfock2, amps, imds):

    T1old, T2old, Snold, U1nold, U2nold = amps

    Sn = [None]*nfock1
    if nfock1 < 1: return Sn

    nocc = T2old.shape[2]
    nm = w.shape[0]

    eps_occ = F.oo.diagonal()
    eps_vir = F.vv.diagonal()

    S1 = H.copy() * 0.0
    #S1 += 1.0*einsum('I->I', H)
    #S1 += 1.0*einsum('J,IJ->I', G, S2old)
    S1 += 1.0*einsum('I,I->I', w, Snold[0])
    S1 += 1.0*einsum('ia,Iai->I', F.ov, U1nold[0], optimize=True)
    S1 += 1.0*einsum('Iia,ai->I', g.ov, T1old, optimize=True)
    # Jia, Iai -> IJ # (g, U) contraction
    S1 += 1.0*einsum('JI,J->I', imds.gU1xy, Snold[0], optimize=True)
    #S1 += 1.0*einsum('Jia,J,Iai->I', g.ov, Snold[0], U1nold[0], optimize=True)
    S1 += einsum('xia,ai->x', imds.Xov, T1old)
    #S1 += -1.0*einsum('ijab,bi,Iaj->I', I.oovv, T1old, U1nold[0], optimize=True)
    if U2nold[0] is not None:
        S1 += 0.25*einsum('ijab,Ibaji->I', I.oovv, U2nold[0], optimize=True)
    if nfock1 > 1:
        S1 += einsum('J, IJ->I', imds.gtm, Snold[1])
        #S1 += 1.0*einsum('Jia,ai,IJ->I', g.ov, T1old, Snold[1], optimize=True)
    if nfock2 > 1:
        S1 += 1.0*einsum('Jia,IJai->I', g.ov, U1nold[1], optimize=True)

    for i in range(nm):
        if w[i] == 0:
            S1[i] = 0.0
    Sn[0] = S1

    if nfock1 < 2: return Sn

    # S2
    S2 = numpy.zeros((nm, nm))

    S2 += 1.0*einsum('I,JI->IJ', w, Snold[1], optimize=True)
    S2 += 1.0*einsum('J,IJ->IJ', w, Snold[1], optimize=True)
    S2 += 1.0*einsum('Iia,Jai->IJ', g.ov, U1nold[0], optimize=True)
    S2 += 1.0*einsum('Jia,Iai->IJ', g.ov, U1nold[0], optimize=True)
    S2 += 1.0*einsum('Kia,IK,Jai->IJ', g.ov, Snold[1], U1nold[0], optimize=True)
    S2 += 1.0*einsum('Kia,JK,Iai->IJ', g.ov, Snold[1], U1nold[0], optimize=True)
    S2 += -1.0*einsum('ijab,Ibi,Jaj->IJ', I.oovv, U1nold[0], U1nold[0], optimize=True)
    if nfock2 > 1:
        S2 += 1.0*einsum('ia,IJai->IJ', F.ov, U1nold[1], optimize=True)
        if U2nold[0] is not None:
            S2 += 0.25*einsum('ijab,IJbaji->IJ', I.oovv, U2nold[1], optimize=True)
        S2 += 1.0*einsum('Kia,K,IJai->IJ', g.ov, Snold[0], U1nold[1], optimize=True)
        S2 += -1.0*einsum('ijab,bi,IJaj->IJ', I.oovv, T1old, U1nold[1], optimize=True)

    for i in range(nm):
        if w[i] == 0:
            S2[i,i] = 0.0

    Sn[1] = S2

    return Sn


def qedccsd_U1n(F, I, w, g, h, G, H, nfock1, nfock2, amps, imds):

    T1old, T2old, Snold, U1nold, U2nold = amps

    nv = T2old.shape[0]
    no = T2old.shape[2]
    nm = w.shape[0]

    eps_occ = F.oo.diagonal()
    eps_vir = F.vv.diagonal()

    #e_denom = 1 / (eps_occ.reshape(-1, 1) - eps_vir)

    U1n = [None]*nfock2
    if nfock2 < 1: return U1n

    #if nm == 1: # if single mode, we use faster single mode code (reducing indexing overhead)
    #    return single_qedccsd_U1n(F, I, w, g, h, G, H, nfock1, nfock2, amps)

    U1n[0] = g.vo.copy()
    U1n[0] += 1.0*einsum('I,Iai->Iai', w, U1nold[0])

    U1n[0] += 1.0*einsum('Iab,bi->Iai', g.vv, T1old)
    U1n[0] -= 1.0*einsum('Iji,aj->Iai', g.oo, T1old)

    T2temp = T2old - einsum('bi,aj->abij', T1old, T1old)
    U1n[0] += einsum('xjb,abij->xai', h.ov, T2temp)
    #U1n[0] += -1.0*einsum('Ijb,abji->Iai', g.ov, T2old)
    #U1n[0] += -1.0*einsum('Ijb,bi,aj->Iai', g.ov, T1old, T1old, optimize=True)

    # using intermediate variables
    #U1n[0] += 1.0*einsum('ab,Ibi->Iai', imds.Fvv, U1nold[0])
    U1n[0] += 1.0*einsum('ab,Ibi->Iai', F.vv, U1nold[0])
    U1n[0] += 1.0*einsum('Jab,J,Ibi->Iai', g.vv, Snold[0], U1nold[0], optimize=True)
    U1n[0] += -1.0*einsum('jb,aj,Ibi->Iai', F.ov, T1old, U1nold[0], optimize=True) # in Fvv
    U1n[0] += -1.0*einsum('Jjb,Iaj,Jbi->Iai', g.ov, U1nold[0], U1nold[0], optimize=True)
    U1n[0] += -1.0*einsum('jabc,cj,Ibi->Iai', I.ovvv, T1old, U1nold[0], optimize=True) # in Fvv

    #U1n[0] -= einsum('ji,xaj->xai', imds.Foo, U1nold[0])
    U1n[0] += -1.0*einsum('ji,Iaj->Iai', F.oo, U1nold[0])
    U1n[0] += -1.0*einsum('Jji,J,Iaj->Iai', g.oo, Snold[0], U1nold[0], optimize=True)
    U1n[0] += -1.0*einsum('jb,bi,Iaj->Iai', F.ov, T1old, U1nold[0], optimize=True) # in Foo
    U1n[0] += -1.0*einsum('Jjb,Ibi,Jaj->Iai', g.ov, U1nold[0], U1nold[0], optimize=True)
    U1n[0] += 1.0*einsum('jkib,bj,Iak->Iai', I.ooov, T1old, U1nold[0], optimize=True) # in Foo

    U1n[0] += -1.0*einsum('jaib,Ibj->Iai', I.ovov, U1nold[0], optimize=True)

    U1n[0] += 1.0*einsum('Jjb,Ibj,Jai->Iai', g.ov, U1nold[0], U1nold[0], optimize=True)

    # in Foo, Fvv
    U1n[0] += -1.0*einsum('jkib,aj,Ibk->Iai', I.ooov, T1old, U1nold[0], optimize=True)
    U1n[0] += 1.0*einsum('jabc,ci,Ibj->Iai', I.ovvv, T1old, U1nold[0], optimize=True)

    # absorbed into Foo, Fvv
    U1n[0] += 1.0*einsum('jkbc,acji,Ibk->Iai', I.oovv, T2old, U1nold[0], optimize=True)
    U1n[0] += 0.5*einsum('jkbc,ackj,Ibi->Iai', I.oovv, T2old, U1nold[0], optimize=True)
    U1n[0] += 0.5*einsum('jkbc,cbji,Iak->Iai', I.oovv, T2old, U1nold[0], optimize=True)

    U1n[0] += 1.0*einsum('jkbc,ci,aj,Ibk->Iai', I.oovv, T1old, T1old, U1nold[0], optimize=True)
    U1n[0] += -1.0*einsum('jkbc,aj,ck,Ibi->Iai', I.oovv, T1old, T1old, U1nold[0], optimize=True)
    U1n[0] += -1.0*einsum('jkbc,ci,bj,Iak->Iai', I.oovv, T1old, T1old, U1nold[0], optimize=True)

    U1n[0] += -1.0*einsum('Jjb,aj,J,Ibi->Iai', g.ov, T1old, Snold[0], U1nold[0], optimize=True)
    U1n[0] += -1.0*einsum('Jjb,bi,J,Iaj->Iai', g.ov, T1old, Snold[0], U1nold[0], optimize=True)

    if U2nold[0] is not None:
        U1n[0] += -1.0*einsum('jb,Iabji->Iai', F.ov, U2nold[0])

    if nfock2 > 1:
        U1n[0] += -1.0*einsum('Jji,IJaj->Iai', g.oo, U1nold[1])
        U1n[0] += 1.0*einsum('Jab,IJbi->Iai', g.vv, U1nold[1])
        U1n[0] += -1.0*einsum('Jjb,aj,IJbi->Iai', g.ov, T1old, U1nold[1], optimize=True)
        U1n[0] += -1.0*einsum('Jjb,bi,IJaj->Iai', g.ov, T1old, U1nold[1], optimize=True)
        U1n[0] += 1.0*einsum('J,IJai->Iai', imds.gtm, U1nold[1], optimize=True)
        #U1n[0] += 1.0*einsum('Jjb,bj,IJai->Iai', g.ov, T1old, U1nold[1], optimize=True)
        if U2nold[0] is not None:
            U1n[0] += -1.0*einsum('Jjb,IJabji->Iai', g.ov, U2nold[1], optimize=True)
            #U1n[0] += -0.625*einsum('Jjb,IJabji->Iai', g.ov, U2nold[1], optimize=True)
            #U1n[0] += 0.375*einsum('Jjb, JIbaji->Iai', g.ov, U2nold[1], optimize=True)

    if U2nold[0] is not None:
        U1n[0] += 0.5*einsum('jkib,Iabkj->Iai', I.ooov, U2nold[0], optimize=True)
        U1n[0] += -0.5*einsum('jabc,Icbji->Iai', I.ovvv, U2nold[0], optimize=True)
        U1n[0] += -1.0*einsum('Jjb,J,Iabji->Iai', g.ov, Snold[0], U2nold[0], optimize=True)
        U1n[0] += -0.5*einsum('jkbc,aj,Icbki->Iai', I.oovv, T1old, U2nold[0], optimize=True)
        U1n[0] += -0.5*einsum('jkbc,ci,Iabkj->Iai', I.oovv, T1old, U2nold[0], optimize=True)
        U1n[0] += 1.0*einsum('jkbc,cj,Iabki->Iai', I.oovv, T1old, U2nold[0], optimize=True)

    if nfock1 > 1:
        U1n[0] += 1.0*einsum('Jai,IJ->Iai', g.vo, Snold[1])
        U1n[0] += -1.0*einsum('Jji,aj,IJ->Iai', g.oo, T1old, Snold[1])
        U1n[0] += 1.0*einsum('Jab,bi,IJ->Iai', g.vv, T1old, Snold[1])
        U1n[0] += -1.0*einsum('Jjb,abji,IJ->Iai', g.ov, T2old, Snold[1])
        U1n[0] += -1.0*einsum('Jai,IJ->Iai', imds.gT1T1ov, Snold[1])
        #U1n[0] += -1.0*einsum('Jjb,bi,aj,IJ->Iai', g.ov, T1old, T1old, Snold[1])

    #U11old += einsum('ai,ia -> ai', res_U11old, e_denom, optimize=True)

    if nfock2 < 2: return U1n

    # U12
    U1n[1] = numpy.zeros((nm, nm, nv, no))

    U1n[1] += -1.0*einsum('ji,IJaj->IJai', F.oo, U1nold[1])
    U1n[1] += 1.0*einsum('ab,IJbi->IJai', F.vv, U1nold[1])

    U1n[1] += 2.0*einsum('I,JIai->IJai', w, U1nold[1])
    #U1n[1] += 1.0*einsum('I,JIai->IJai', w, U1nold[1])
    #U1n[1] += 1.0*einsum('J,IJai->IJai', w, U1nold[1])

    U1n[1] += -1.0*einsum('Iji,Jaj->IJai', g.oo, U1nold[0])
    U1n[1] += 1.0*einsum('Iab,Jbi->IJai', g.vv, U1nold[0])
    U1n[1] += -1.0*einsum('Jji,Iaj->IJai', g.oo, U1nold[0])
    U1n[1] += 1.0*einsum('Jab,Ibi->IJai', g.vv, U1nold[0])
    U1n[1] += -1.0*einsum('jaib,IJbj->IJai', I.ovov, U1nold[1], optimize = True)
    if U2nold[0] is not None:
        U1n[1] += -1.0*einsum('jb,IJabji->IJai', F.ov, U2nold[1])
        U1n[1] += -1.0*einsum('Ijb,Jabji->IJai', g.ov, U2nold[0])
        U1n[1] += -1.0*einsum('Jjb,Iabji->IJai', g.ov, U2nold[0])

        U1n[1] += 0.5*einsum('jkib,IJabkj->IJai', I.ooov, U2nold[1], optimize = True)
        U1n[1] += -0.5*einsum('jabc,IJcbji->IJai', I.ovvv, U2nold[1], optimize = True)

        U1n[1] += -1.0*einsum('Kjb,K,IJabji->IJai', g.ov, Snold[0], U2nold[1], optimize = True)
        U1n[1] += -0.5*einsum('jkbc,aj,IJcbki->IJai', I.oovv, T1old, U2nold[1], optimize = True)
        #U1n[1] += 0.125*einsum('jkbc,aj,JIcbik->IJai', I.oovv, T1old, U2nold[1], optimize = True)
        U1n[1] += -0.5*einsum('jkbc,ci,IJabkj->IJai', I.oovv, T1old, U2nold[1], optimize = True)
        #U1n[1] += 0.125*einsum('jkbc,ci,JIbakj->IJai', I.oovv, T1old, U2nold[1], optimize = True)
        U1n[1] += 1.0*einsum('jkbc,cj,IJabki->IJai', I.oovv, T1old, U2nold[1], optimize = True)
        #U1n[1] += -0.375*einsum('jkbc,cj,JIbaki->IJai', I.oovv, T1old, U2nold[1], optimize = True)
        U1n[1] += -0.5*einsum('jkbc,Iaj,Jcbki->IJai', I.oovv, U1nold[0], U2nold[0], optimize = True)
        U1n[1] += -0.5*einsum('jkbc,Ici,Jabkj->IJai', I.oovv, U1nold[0], U2nold[0], optimize = True)
        U1n[1] += 1.0*einsum('jkbc,Icj,Jabki->IJai', I.oovv, U1nold[0], U2nold[0], optimize = True)
        U1n[1] += -0.5*einsum('jkbc,Jaj,Icbki->IJai', I.oovv, U1nold[0], U2nold[0], optimize = True)
        U1n[1] += -0.5*einsum('jkbc,Jci,Iabkj->IJai', I.oovv, U1nold[0], U2nold[0], optimize = True)
        U1n[1] += 1.0*einsum('jkbc,Jcj,Iabki->IJai', I.oovv, U1nold[0], U2nold[0], optimize = True)
        U1n[1] += -0.5*einsum('jkbc,ci,IJabkj->IJai', I.oovv, T1old, U2nold[1], optimize = True)
        #U1n[1] += 0.125*einsum('jkbc,ci,JIbakj->IJai', I.oovv, T1old, U2nold[1], optimize = True)
        U1n[1] += 1.0*einsum('jkbc,cj,IJabki->IJai', I.oovv, T1old, U2nold[1], optimize = True)
        #U1n[1] += -0.375*einsum('jkbc,cj,JIbaki->IJai', I.oovv, T1old, U2nold[1], optimize = True)
        U1n[1] += -0.5*einsum('jkbc,Iaj,Jcbki->IJai', I.oovv, U1nold[0], U2nold[0], optimize = True)
        U1n[1] += -0.5*einsum('jkbc,Ici,Jabkj->IJai', I.oovv, U1nold[0], U2nold[0], optimize = True)
        U1n[1] += 1.0*einsum('jkbc,Icj,Jabki->IJai', I.oovv, U1nold[0], U2nold[0], optimize = True)
        U1n[1] += -0.5*einsum('jkbc,Jaj,Icbki->IJai', I.oovv, U1nold[0], U2nold[0], optimize = True)
        U1n[1] += -0.5*einsum('jkbc,Jci,Iabkj->IJai', I.oovv, U1nold[0], U2nold[0], optimize = True)
        U1n[1] += 1.0*einsum('jkbc,Jcj,Iabki->IJai', I.oovv, U1nold[0], U2nold[0], optimize = True)
        if nfock1 > 1:
            U1n[1] += -1.0*einsum('Kjb,IK,Jabji->IJai', g.ov, Snold[1], U2nold[0], optimize = True)
            U1n[1] += -1.0*einsum('Kjb,JK,Iabji->IJai', g.ov, Snold[1], U2nold[0], optimize = True)
    U1n[1] += -1.0*einsum('jb,aj,IJbi->IJai', F.ov, T1old, U1nold[1], optimize = True)
    U1n[1] += -1.0*einsum('jb,bi,IJaj->IJai', F.ov, T1old, U1nold[1], optimize = True)
    U1n[1] += -1.0*einsum('jb,Iaj,Jbi->IJai', F.ov, U1nold[0], U1nold[0], optimize = True)
    U1n[1] += -1.0*einsum('jb,Ibi,Jaj->IJai', F.ov, U1nold[0], U1nold[0], optimize = True)
    U1n[1] += -1.0*einsum('Ijb,aj,Jbi->IJai', g.ov, T1old, U1nold[0], optimize = True)
    U1n[1] += -1.0*einsum('Ijb,bi,Jaj->IJai', g.ov, T1old, U1nold[0], optimize = True)
    U1n[1] += -1.0*einsum('Jjb,aj,Ibi->IJai', g.ov, T1old, U1nold[0], optimize = True)
    U1n[1] += -1.0*einsum('Jjb,bi,Iaj->IJai', g.ov, T1old, U1nold[0], optimize = True)
    U1n[1] += -1.0*einsum('Kji,K,IJaj->IJai', g.oo, Snold[0], U1nold[1], optimize = True)
    if nfock1 > 1:
        U1n[1] += -1.0*einsum('Kji,IK,Jaj->IJai', g.oo, Snold[1], U1nold[0], optimize = True)
        U1n[1] += -1.0*einsum('Kji,JK,Iaj->IJai', g.oo, Snold[1], U1nold[0], optimize = True)
        U1n[1] += 1.0*einsum('Kab,IK,Jbi->IJai', g.vv, Snold[1], U1nold[0], optimize = True)
        U1n[1] += 1.0*einsum('Kab,JK,Ibi->IJai', g.vv, Snold[1], U1nold[0], optimize = True)
        U1n[1] += -1.0*einsum('Kjb,aj,IK,Jbi->IJai', g.ov, T1old, Snold[1], U1nold[0], optimize = True)
        U1n[1] += -1.0*einsum('Kjb,aj,JK,Ibi->IJai', g.ov, T1old, Snold[1], U1nold[0], optimize = True)
        U1n[1] += -1.0*einsum('Kjb,bi,IK,Jaj->IJai', g.ov, T1old, Snold[1], U1nold[0], optimize = True)
        U1n[1] += -1.0*einsum('Kjb,bi,JK,Iaj->IJai', g.ov, T1old, Snold[1], U1nold[0], optimize = True)
    U1n[1] += 1.0*einsum('Kab,K,IJbi->IJai', g.vv, Snold[0], U1nold[1], optimize = True)
    U1n[1] += -1.0*einsum('Kjb,Iaj,JKbi->IJai', g.ov, U1nold[0], U1nold[1], optimize = True)
    U1n[1] += -1.0*einsum('Kjb,Ibi,JKaj->IJai', g.ov, U1nold[0], U1nold[1], optimize = True)
    U1n[1] += 1.0*einsum('Kjb,Ibj,JKai->IJai', g.ov, U1nold[0], U1nold[1], optimize = True)
    U1n[1] += -1.0*einsum('Kjb,Jaj,IKbi->IJai', g.ov, U1nold[0], U1nold[1], optimize = True)
    U1n[1] += -1.0*einsum('Kjb,Jbi,IKaj->IJai', g.ov, U1nold[0], U1nold[1], optimize = True)
    U1n[1] += 1.0*einsum('Kjb,Jbj,IKai->IJai', g.ov, U1nold[0], U1nold[1], optimize = True)
    U1n[1] += 1.0*einsum('Kjb,Kai,IJbj->IJai', g.ov, U1nold[0], U1nold[1], optimize = True)
    U1n[1] += -1.0*einsum('Kjb,Kaj,IJbi->IJai', g.ov, U1nold[0], U1nold[1], optimize = True)
    U1n[1] += -1.0*einsum('Kjb,Kbi,IJaj->IJai', g.ov, U1nold[0], U1nold[1], optimize = True)
    U1n[1] += -1.0*einsum('jkib,aj,IJbk->IJai', I.ooov, T1old, U1nold[1], optimize = True)
    U1n[1] += 1.0*einsum('jkib,bj,IJak->IJai', I.ooov, T1old, U1nold[1], optimize = True)
    U1n[1] += -1.0*einsum('jkib,Iaj,Jbk->IJai', I.ooov, U1nold[0], U1nold[0], optimize = True)
    U1n[1] += 1.0*einsum('jkib,Ibj,Jak->IJai', I.ooov, U1nold[0], U1nold[0], optimize = True)
    U1n[1] += 1.0*einsum('jabc,ci,IJbj->IJai', I.ovvv, T1old, U1nold[1], optimize = True)
    U1n[1] += -1.0*einsum('jabc,cj,IJbi->IJai', I.ovvv, T1old, U1nold[1], optimize = True)
    U1n[1] += 1.0*einsum('jabc,Ici,Jbj->IJai', I.ovvv, U1nold[0], U1nold[0], optimize = True)
    U1n[1] += -1.0*einsum('jabc,Icj,Jbi->IJai', I.ovvv, U1nold[0], U1nold[0], optimize = True)
    U1n[1] += 1.0*einsum('jkbc,acji,IJbk->IJai', I.oovv, T2old, U1nold[1], optimize = True)
    U1n[1] += 0.5*einsum('jkbc,ackj,IJbi->IJai', I.oovv, T2old, U1nold[1], optimize = True)
    U1n[1] += 0.5*einsum('jkbc,cbji,IJak->IJai', I.oovv, T2old, U1nold[1], optimize = True)
    U1n[1] += -1.0*einsum('Kjb,K,Iaj,Jbi->IJai', g.ov, Snold[0], U1nold[0], U1nold[0], optimize = True)
    U1n[1] += -1.0*einsum('Kjb,K,Ibi,Jaj->IJai', g.ov, Snold[0], U1nold[0], U1nold[0], optimize = True)
    U1n[1] += -1.0*einsum('Kjb,aj,K,IJbi->IJai', g.ov, T1old, Snold[0], U1nold[1], optimize = True)
    U1n[1] += -1.0*einsum('Kjb,bi,K,IJaj->IJai', g.ov, T1old, Snold[0], U1nold[1], optimize = True)
    U1n[1] += -1.0*einsum('jkbc,aj,ck,IJbi->IJai', I.oovv, T1old, T1old, U1nold[1], optimize = True)
    U1n[1] += 1.0*einsum('jkbc,aj,Ici,Jbk->IJai', I.oovv, T1old, U1nold[0], U1nold[0], optimize = True)
    U1n[1] += -1.0*einsum('jkbc,aj,Ick,Jbi->IJai', I.oovv, T1old, U1nold[0], U1nold[0], optimize = True)
    U1n[1] += 1.0*einsum('jkbc,ci,aj,IJbk->IJai', I.oovv, T1old, T1old, U1nold[1], optimize = True)
    U1n[1] += -1.0*einsum('jkbc,ci,bj,IJak->IJai', I.oovv, T1old, T1old, U1nold[1], optimize = True)
    U1n[1] += 1.0*einsum('jkbc,ci,Iaj,Jbk->IJai', I.oovv, T1old, U1nold[0], U1nold[0], optimize = True)
    U1n[1] += -1.0*einsum('jkbc,ci,Ibj,Jak->IJai', I.oovv, T1old, U1nold[0], U1nold[0], optimize = True)
    U1n[1] += 1.0*einsum('jkbc,cj,Iak,Jbi->IJai', I.oovv, T1old, U1nold[0], U1nold[0], optimize = True)
    U1n[1] += 1.0*einsum('jkbc,cj,Ibi,Jak->IJai', I.oovv, T1old, U1nold[0], U1nold[0], optimize = True)

    return U1n

def qedccsd_U2n(F, I, w, g, h, G, H, nfock1, nfock2, amps, imds):

    T1old, T2old, Snold, U1nold, U2nold = amps

    nvir = T2old.shape[0]
    nocc = T2old.shape[2]
    nm = w.shape[0]

    eps_occ = F.oo.diagonal()
    eps_vir = F.vv.diagonal()
    eps_vir_p_w = eps_vir + w

    #e_denom = 1 / (eps_occ.reshape(-1, 1, 1, 1) + eps_occ.reshape(-1, 1) - eps_vir.reshape(-1, 1, 1) - eps_vir_p_w)

    #if nm == 1: # if single mode, we use faster single mode code (reducing indexing overhead)
    #    return single_qedccsd_U2n(F, I, w, g, h, G, H, nfock1, nfock2, amps)

    U2n = [None]*nfock2
    if U2nold[0] is None: return U2n
    if nfock2 < 1: return U2n

    U2n[0] = numpy.zeros((nm, nvir, nvir, nocc, nocc))

    U2n[0] += 1.0*einsum('ki,Ibakj->Iabij', F.oo, U2nold[0], optimize = True)
    U2n[0] += -1.0*einsum('kj,Ibaki->Iabij', F.oo, U2nold[0], optimize = True)
    U2n[0] += 1.0*einsum('ac,Ibcji->Iabij', F.vv, U2nold[0], optimize = True)
    U2n[0] += -1.0*einsum('bc,Iacji->Iabij', F.vv, U2nold[0], optimize = True)
    U2n[0] += 1.0*einsum('I,Ibaji->Iabij', w, U2nold[0], optimize = True)
    U2n[0] += 1.0*einsum('Iki,bakj->Iabij', g.oo, T2old, optimize = True)
    U2n[0] += -1.0*einsum('Ikj,baki->Iabij', g.oo, T2old, optimize = True)
    U2n[0] += 1.0*einsum('Iac,bcji->Iabij', g.vv, T2old, optimize = True)
    U2n[0] += -1.0*einsum('Ibc,acji->Iabij', g.vv, T2old, optimize = True)
    U2n[0] += -1.0*einsum('kaji,Ibk->Iabij', I.ovoo, U1nold[0], optimize = True)
    U2n[0] += 1.0*einsum('kbji,Iak->Iabij', I.ovoo, U1nold[0], optimize = True)
    U2n[0] += -1.0*einsum('baic,Icj->Iabij', I.vvov, U1nold[0], optimize = True)
    U2n[0] += 1.0*einsum('bajc,Ici->Iabij', I.vvov, U1nold[0], optimize = True)
    U2n[0] += -0.5*einsum('klji,Ibalk->Iabij', I.oooo, U2nold[0], optimize = True)
    U2n[0] += 1.0*einsum('kaic,Ibckj->Iabij', I.ovov, U2nold[0], optimize = True)
    U2n[0] += -1.0*einsum('kajc,Ibcki->Iabij', I.ovov, U2nold[0], optimize = True)
    U2n[0] += -1.0*einsum('kbic,Iackj->Iabij', I.ovov, U2nold[0], optimize = True)
    U2n[0] += 1.0*einsum('kbjc,Iacki->Iabij', I.ovov, U2nold[0], optimize = True)
    U2n[0] += -0.5*einsum('bacd,Idcji->Iabij', I.vvvv, U2nold[0], optimize = True)
    U2n[0] += -1.0*einsum('kc,ak,Ibcji->Iabij', F.ov, T1old, U2nold[0], optimize = True)
    U2n[0] += 1.0*einsum('kc,bk,Iacji->Iabij', F.ov, T1old, U2nold[0], optimize = True)
    U2n[0] += 1.0*einsum('kc,ci,Ibakj->Iabij', F.ov, T1old, U2nold[0], optimize = True)
    U2n[0] += -1.0*einsum('kc,cj,Ibaki->Iabij', F.ov, T1old, U2nold[0], optimize = True)
    U2n[0] += 1.0*einsum('kc,acji,Ibk->Iabij', F.ov, T2old, U1nold[0], optimize = True)
    U2n[0] += -1.0*einsum('kc,baki,Icj->Iabij', F.ov, T2old, U1nold[0], optimize = True)
    U2n[0] += 1.0*einsum('kc,bakj,Ici->Iabij', F.ov, T2old, U1nold[0], optimize = True)
    U2n[0] += -1.0*einsum('kc,bcji,Iak->Iabij', F.ov, T2old, U1nold[0], optimize = True)
    U2n[0] += -1.0*einsum('Ikc,ak,bcji->Iabij', g.ov, T1old, T2old, optimize = True)
    U2n[0] += 1.0*einsum('Ikc,bk,acji->Iabij', g.ov, T1old, T2old, optimize = True)
    U2n[0] += 1.0*einsum('Ikc,ci,bakj->Iabij', g.ov, T1old, T2old, optimize = True)
    U2n[0] += -1.0*einsum('Ikc,cj,baki->Iabij', g.ov, T1old, T2old, optimize = True)
    U2n[0] += 1.0*einsum('Jki,J,Ibakj->Iabij', g.oo, Snold[0], U2nold[0], optimize = True)
    U2n[0] += -1.0*einsum('Jki,Iak,Jbj->Iabij', g.oo, U1nold[0], U1nold[0], optimize = True)
    U2n[0] += 1.0*einsum('Jki,Ibk,Jaj->Iabij', g.oo, U1nold[0], U1nold[0], optimize = True)
    U2n[0] += -1.0*einsum('Jkj,J,Ibaki->Iabij', g.oo, Snold[0], U2nold[0], optimize = True)
    U2n[0] += 1.0*einsum('Jkj,Iak,Jbi->Iabij', g.oo, U1nold[0], U1nold[0], optimize = True)
    U2n[0] += -1.0*einsum('Jkj,Ibk,Jai->Iabij', g.oo, U1nold[0], U1nold[0], optimize = True)
    U2n[0] += 1.0*einsum('Jac,J,Ibcji->Iabij', g.vv, Snold[0], U2nold[0], optimize = True)
    U2n[0] += 1.0*einsum('Jac,Ici,Jbj->Iabij', g.vv, U1nold[0], U1nold[0], optimize = True)
    U2n[0] += -1.0*einsum('Jac,Icj,Jbi->Iabij', g.vv, U1nold[0], U1nold[0], optimize = True)
    U2n[0] += -1.0*einsum('Jbc,J,Iacji->Iabij', g.vv, Snold[0], U2nold[0], optimize = True)
    U2n[0] += -1.0*einsum('Jbc,Ici,Jaj->Iabij', g.vv, U1nold[0], U1nold[0], optimize = True)
    U2n[0] += 1.0*einsum('Jbc,Icj,Jai->Iabij', g.vv, U1nold[0], U1nold[0], optimize = True)
    U2n[0] += -1.0*einsum('klji,ak,Ibl->Iabij', I.oooo, T1old, U1nold[0], optimize = True)
    U2n[0] += 1.0*einsum('klji,bk,Ial->Iabij', I.oooo, T1old, U1nold[0], optimize = True)
    U2n[0] += 1.0*einsum('kaic,bk,Icj->Iabij', I.ovov, T1old, U1nold[0], optimize = True)
    U2n[0] += 1.0*einsum('kaic,cj,Ibk->Iabij', I.ovov, T1old, U1nold[0], optimize = True)
    U2n[0] += -1.0*einsum('kajc,bk,Ici->Iabij', I.ovov, T1old, U1nold[0], optimize = True)
    U2n[0] += -1.0*einsum('kajc,ci,Ibk->Iabij', I.ovov, T1old, U1nold[0], optimize = True)
    U2n[0] += -1.0*einsum('kbic,ak,Icj->Iabij', I.ovov, T1old, U1nold[0], optimize = True)
    U2n[0] += -1.0*einsum('kbic,cj,Iak->Iabij', I.ovov, T1old, U1nold[0], optimize = True)
    U2n[0] += 1.0*einsum('kbjc,ak,Ici->Iabij', I.ovov, T1old, U1nold[0], optimize = True)
    U2n[0] += 1.0*einsum('kbjc,ci,Iak->Iabij', I.ovov, T1old, U1nold[0], optimize = True)
    U2n[0] += 1.0*einsum('bacd,di,Icj->Iabij', I.vvvv, T1old, U1nold[0], optimize = True)
    U2n[0] += -1.0*einsum('bacd,dj,Ici->Iabij', I.vvvv, T1old, U1nold[0], optimize = True)
    U2n[0] += -1.0*einsum('Jkc,Iak,Jbcji->Iabij', g.ov, U1nold[0], U2nold[0], optimize = True)
    U2n[0] += 1.0*einsum('Jkc,Ibk,Jacji->Iabij', g.ov, U1nold[0], U2nold[0], optimize = True)
    U2n[0] += 1.0*einsum('Jkc,Ici,Jbakj->Iabij', g.ov, U1nold[0], U2nold[0], optimize = True)
    U2n[0] += -1.0*einsum('Jkc,Icj,Jbaki->Iabij', g.ov, U1nold[0], U2nold[0], optimize = True)
    U2n[0] += 1.0*einsum('Jkc,Ick,Jbaji->Iabij', g.ov, U1nold[0], U2nold[0], optimize = True)
    U2n[0] += -1.0*einsum('Jkc,Jai,Ibckj->Iabij', g.ov, U1nold[0], U2nold[0], optimize = True)
    U2n[0] += 1.0*einsum('Jkc,Jaj,Ibcki->Iabij', g.ov, U1nold[0], U2nold[0], optimize = True)
    U2n[0] += -1.0*einsum('Jkc,Jak,Ibcji->Iabij', g.ov, U1nold[0], U2nold[0], optimize = True)
    U2n[0] += 1.0*einsum('Jkc,Jbi,Iackj->Iabij', g.ov, U1nold[0], U2nold[0], optimize = True)
    U2n[0] += -1.0*einsum('Jkc,Jbj,Iacki->Iabij', g.ov, U1nold[0], U2nold[0], optimize = True)
    U2n[0] += 1.0*einsum('Jkc,Jbk,Iacji->Iabij', g.ov, U1nold[0], U2nold[0], optimize = True)
    U2n[0] += 1.0*einsum('Jkc,Jci,Ibakj->Iabij', g.ov, U1nold[0], U2nold[0], optimize = True)
    U2n[0] += -1.0*einsum('Jkc,Jcj,Ibaki->Iabij', g.ov, U1nold[0], U2nold[0], optimize = True)
    U2n[0] += 1.0*einsum('klic,ak,Ibclj->Iabij', I.ooov, T1old, U2nold[0], optimize = True)
    U2n[0] += -1.0*einsum('klic,bk,Iaclj->Iabij', I.ooov, T1old, U2nold[0], optimize = True)
    U2n[0] += 0.5*einsum('klic,cj,Ibalk->Iabij', I.ooov, T1old, U2nold[0], optimize = True)
    U2n[0] += -1.0*einsum('klic,ck,Ibalj->Iabij', I.ooov, T1old, U2nold[0], optimize = True)
    U2n[0] += 1.0*einsum('klic,ackj,Ibl->Iabij', I.ooov, T2old, U1nold[0], optimize = True)
    U2n[0] += 1.0*einsum('klic,bakj,Icl->Iabij', I.ooov, T2old, U1nold[0], optimize = True)
    U2n[0] += 0.5*einsum('klic,balk,Icj->Iabij', I.ooov, T2old, U1nold[0], optimize = True)
    U2n[0] += -1.0*einsum('klic,bckj,Ial->Iabij', I.ooov, T2old, U1nold[0], optimize = True)
    U2n[0] += -1.0*einsum('kljc,ak,Ibcli->Iabij', I.ooov, T1old, U2nold[0], optimize = True)
    U2n[0] += 1.0*einsum('kljc,bk,Iacli->Iabij', I.ooov, T1old, U2nold[0], optimize = True)
    U2n[0] += -0.5*einsum('kljc,ci,Ibalk->Iabij', I.ooov, T1old, U2nold[0], optimize = True)
    U2n[0] += 1.0*einsum('kljc,ck,Ibali->Iabij', I.ooov, T1old, U2nold[0], optimize = True)
    U2n[0] += -1.0*einsum('kljc,acki,Ibl->Iabij', I.ooov, T2old, U1nold[0], optimize = True)
    U2n[0] += -1.0*einsum('kljc,baki,Icl->Iabij', I.ooov, T2old, U1nold[0], optimize = True)
    U2n[0] += -0.5*einsum('kljc,balk,Ici->Iabij', I.ooov, T2old, U1nold[0], optimize = True)
    U2n[0] += 1.0*einsum('kljc,bcki,Ial->Iabij', I.ooov, T2old, U1nold[0], optimize = True)
    U2n[0] += 0.5*einsum('kacd,bk,Idcji->Iabij', I.ovvv, T1old, U2nold[0], optimize = True)
    U2n[0] += -1.0*einsum('kacd,di,Ibckj->Iabij', I.ovvv, T1old, U2nold[0], optimize = True)
    U2n[0] += 1.0*einsum('kacd,dj,Ibcki->Iabij', I.ovvv, T1old, U2nold[0], optimize = True)
    U2n[0] += -1.0*einsum('kacd,dk,Ibcji->Iabij', I.ovvv, T1old, U2nold[0], optimize = True)
    U2n[0] += 1.0*einsum('kacd,bdji,Ick->Iabij', I.ovvv, T2old, U1nold[0], optimize = True)
    U2n[0] += -1.0*einsum('kacd,bdki,Icj->Iabij', I.ovvv, T2old, U1nold[0], optimize = True)
    U2n[0] += 1.0*einsum('kacd,bdkj,Ici->Iabij', I.ovvv, T2old, U1nold[0], optimize = True)
    U2n[0] += 0.5*einsum('kacd,dcji,Ibk->Iabij', I.ovvv, T2old, U1nold[0], optimize = True)
    U2n[0] += -0.5*einsum('kbcd,ak,Idcji->Iabij', I.ovvv, T1old, U2nold[0], optimize = True)
    U2n[0] += 1.0*einsum('kbcd,di,Iackj->Iabij', I.ovvv, T1old, U2nold[0], optimize = True)
    U2n[0] += -1.0*einsum('kbcd,dj,Iacki->Iabij', I.ovvv, T1old, U2nold[0], optimize = True)
    U2n[0] += 1.0*einsum('kbcd,dk,Iacji->Iabij', I.ovvv, T1old, U2nold[0], optimize = True)
    U2n[0] += -1.0*einsum('kbcd,adji,Ick->Iabij', I.ovvv, T2old, U1nold[0], optimize = True)
    U2n[0] += 1.0*einsum('kbcd,adki,Icj->Iabij', I.ovvv, T2old, U1nold[0], optimize = True)
    U2n[0] += -1.0*einsum('kbcd,adkj,Ici->Iabij', I.ovvv, T2old, U1nold[0], optimize = True)
    U2n[0] += -0.5*einsum('kbcd,dcji,Iak->Iabij', I.ovvv, T2old, U1nold[0], optimize = True)
    U2n[0] += 0.5*einsum('klcd,adji,Ibclk->Iabij', I.oovv, T2old, U2nold[0], optimize = True)
    U2n[0] += -1.0*einsum('klcd,adki,Ibclj->Iabij', I.oovv, T2old, U2nold[0], optimize = True)
    U2n[0] += 1.0*einsum('klcd,adkj,Ibcli->Iabij', I.oovv, T2old, U2nold[0], optimize = True)
    U2n[0] += 0.5*einsum('klcd,adlk,Ibcji->Iabij', I.oovv, T2old, U2nold[0], optimize = True)
    U2n[0] += -0.5*einsum('klcd,baki,Idclj->Iabij', I.oovv, T2old, U2nold[0], optimize = True)
    U2n[0] += 0.5*einsum('klcd,bakj,Idcli->Iabij', I.oovv, T2old, U2nold[0], optimize = True)
    U2n[0] += 0.25*einsum('klcd,balk,Idcji->Iabij', I.oovv, T2old, U2nold[0], optimize = True)
    U2n[0] += -0.5*einsum('klcd,bdji,Iaclk->Iabij', I.oovv, T2old, U2nold[0], optimize = True)
    U2n[0] += 1.0*einsum('klcd,bdki,Iaclj->Iabij', I.oovv, T2old, U2nold[0], optimize = True)
    U2n[0] += -1.0*einsum('klcd,bdkj,Iacli->Iabij', I.oovv, T2old, U2nold[0], optimize = True)
    U2n[0] += -0.5*einsum('klcd,bdlk,Iacji->Iabij', I.oovv, T2old, U2nold[0], optimize = True)
    U2n[0] += 0.25*einsum('klcd,dcji,Ibalk->Iabij', I.oovv, T2old, U2nold[0], optimize = True)
    U2n[0] += -0.5*einsum('klcd,dcki,Ibalj->Iabij', I.oovv, T2old, U2nold[0], optimize = True)
    U2n[0] += 0.5*einsum('klcd,dckj,Ibali->Iabij', I.oovv, T2old, U2nold[0], optimize = True)
    U2n[0] += -1.0*einsum('Jkc,ak,J,Ibcji->Iabij', g.ov, T1old, Snold[0], U2nold[0], optimize = True)
    U2n[0] += -1.0*einsum('Jkc,ak,Ici,Jbj->Iabij', g.ov, T1old, U1nold[0], U1nold[0], optimize = True)
    U2n[0] += 1.0*einsum('Jkc,ak,Icj,Jbi->Iabij', g.ov, T1old, U1nold[0], U1nold[0], optimize = True)
    U2n[0] += 1.0*einsum('Jkc,bk,J,Iacji->Iabij', g.ov, T1old, Snold[0], U2nold[0], optimize = True)
    U2n[0] += 1.0*einsum('Jkc,bk,Ici,Jaj->Iabij', g.ov, T1old, U1nold[0], U1nold[0], optimize = True)
    U2n[0] += -1.0*einsum('Jkc,bk,Icj,Jai->Iabij', g.ov, T1old, U1nold[0], U1nold[0], optimize = True)
    U2n[0] += 1.0*einsum('Jkc,ci,J,Ibakj->Iabij', g.ov, T1old, Snold[0], U2nold[0], optimize = True)
    U2n[0] += -1.0*einsum('Jkc,ci,Iak,Jbj->Iabij', g.ov, T1old, U1nold[0], U1nold[0], optimize = True)
    U2n[0] += 1.0*einsum('Jkc,ci,Ibk,Jaj->Iabij', g.ov, T1old, U1nold[0], U1nold[0], optimize = True)
    U2n[0] += -1.0*einsum('Jkc,cj,J,Ibaki->Iabij', g.ov, T1old, Snold[0], U2nold[0], optimize = True)
    U2n[0] += 1.0*einsum('Jkc,cj,Iak,Jbi->Iabij', g.ov, T1old, U1nold[0], U1nold[0], optimize = True)
    U2n[0] += -1.0*einsum('Jkc,cj,Ibk,Jai->Iabij', g.ov, T1old, U1nold[0], U1nold[0], optimize = True)
    U2n[0] += 1.0*einsum('Jkc,acji,J,Ibk->Iabij', g.ov, T2old, Snold[0], U1nold[0], optimize = True)
    U2n[0] += -1.0*einsum('Jkc,baki,J,Icj->Iabij', g.ov, T2old, Snold[0], U1nold[0], optimize = True)
    U2n[0] += 1.0*einsum('Jkc,bakj,J,Ici->Iabij', g.ov, T2old, Snold[0], U1nold[0], optimize = True)
    U2n[0] += -1.0*einsum('Jkc,bcji,J,Iak->Iabij', g.ov, T2old, Snold[0], U1nold[0], optimize = True)
    U2n[0] += -1.0*einsum('klic,bk,al,Icj->Iabij', I.ooov, T1old, T1old, U1nold[0], optimize = True)
    U2n[0] += 1.0*einsum('klic,cj,ak,Ibl->Iabij', I.ooov, T1old, T1old, U1nold[0], optimize = True)
    U2n[0] += -1.0*einsum('klic,cj,bk,Ial->Iabij', I.ooov, T1old, T1old, U1nold[0], optimize = True)
    U2n[0] += 1.0*einsum('kljc,bk,al,Ici->Iabij', I.ooov, T1old, T1old, U1nold[0], optimize = True)
    U2n[0] += -1.0*einsum('kljc,ci,ak,Ibl->Iabij', I.ooov, T1old, T1old, U1nold[0], optimize = True)
    U2n[0] += 1.0*einsum('kljc,ci,bk,Ial->Iabij', I.ooov, T1old, T1old, U1nold[0], optimize = True)
    U2n[0] += -1.0*einsum('kacd,di,bk,Icj->Iabij', I.ovvv, T1old, T1old, U1nold[0], optimize = True)
    U2n[0] += -1.0*einsum('kacd,di,cj,Ibk->Iabij', I.ovvv, T1old, T1old, U1nold[0], optimize = True)
    U2n[0] += 1.0*einsum('kacd,dj,bk,Ici->Iabij', I.ovvv, T1old, T1old, U1nold[0], optimize = True)
    U2n[0] += 1.0*einsum('kbcd,di,ak,Icj->Iabij', I.ovvv, T1old, T1old, U1nold[0], optimize = True)
    U2n[0] += 1.0*einsum('kbcd,di,cj,Iak->Iabij', I.ovvv, T1old, T1old, U1nold[0], optimize = True)
    U2n[0] += -1.0*einsum('kbcd,dj,ak,Ici->Iabij', I.ovvv, T1old, T1old, U1nold[0], optimize = True)
    U2n[0] += -1.0*einsum('klcd,ak,dl,Ibcji->Iabij', I.oovv, T1old, T1old, U2nold[0], optimize = True)
    U2n[0] += 1.0*einsum('klcd,ak,bdji,Icl->Iabij', I.oovv, T1old, T2old, U1nold[0], optimize = True)
    U2n[0] += -1.0*einsum('klcd,ak,bdli,Icj->Iabij', I.oovv, T1old, T2old, U1nold[0], optimize = True)
    U2n[0] += 1.0*einsum('klcd,ak,bdlj,Ici->Iabij', I.oovv, T1old, T2old, U1nold[0], optimize = True)
    U2n[0] += 0.5*einsum('klcd,ak,dcji,Ibl->Iabij', I.oovv, T1old, T2old, U1nold[0], optimize = True)
    U2n[0] += -0.5*einsum('klcd,bk,al,Idcji->Iabij', I.oovv, T1old, T1old, U2nold[0], optimize = True)
    U2n[0] += 1.0*einsum('klcd,bk,dl,Iacji->Iabij', I.oovv, T1old, T1old, U2nold[0], optimize = True)
    U2n[0] += -1.0*einsum('klcd,bk,adji,Icl->Iabij', I.oovv, T1old, T2old, U1nold[0], optimize = True)
    U2n[0] += 1.0*einsum('klcd,bk,adli,Icj->Iabij', I.oovv, T1old, T2old, U1nold[0], optimize = True)
    U2n[0] += -1.0*einsum('klcd,bk,adlj,Ici->Iabij', I.oovv, T1old, T2old, U1nold[0], optimize = True)
    U2n[0] += -0.5*einsum('klcd,bk,dcji,Ial->Iabij', I.oovv, T1old, T2old, U1nold[0], optimize = True)
    U2n[0] += -1.0*einsum('klcd,di,ak,Ibclj->Iabij', I.oovv, T1old, T1old, U2nold[0], optimize = True)
    U2n[0] += 1.0*einsum('klcd,di,bk,Iaclj->Iabij', I.oovv, T1old, T1old, U2nold[0], optimize = True)
    U2n[0] += -0.5*einsum('klcd,di,cj,Ibalk->Iabij', I.oovv, T1old, T1old, U2nold[0], optimize = True)
    U2n[0] += 1.0*einsum('klcd,di,ck,Ibalj->Iabij', I.oovv, T1old, T1old, U2nold[0], optimize = True)
    U2n[0] += -1.0*einsum('klcd,di,ackj,Ibl->Iabij', I.oovv, T1old, T2old, U1nold[0], optimize = True)
    U2n[0] += -1.0*einsum('klcd,di,bakj,Icl->Iabij', I.oovv, T1old, T2old, U1nold[0], optimize = True)
    U2n[0] += -0.5*einsum('klcd,di,balk,Icj->Iabij', I.oovv, T1old, T2old, U1nold[0], optimize = True)
    U2n[0] += 1.0*einsum('klcd,di,bckj,Ial->Iabij', I.oovv, T1old, T2old, U1nold[0], optimize = True)
    U2n[0] += 1.0*einsum('klcd,dj,ak,Ibcli->Iabij', I.oovv, T1old, T1old, U2nold[0], optimize = True)
    U2n[0] += -1.0*einsum('klcd,dj,bk,Iacli->Iabij', I.oovv, T1old, T1old, U2nold[0], optimize = True)
    U2n[0] += -1.0*einsum('klcd,dj,ck,Ibali->Iabij', I.oovv, T1old, T1old, U2nold[0], optimize = True)
    U2n[0] += 1.0*einsum('klcd,dj,acki,Ibl->Iabij', I.oovv, T1old, T2old, U1nold[0], optimize = True)
    U2n[0] += 1.0*einsum('klcd,dj,baki,Icl->Iabij', I.oovv, T1old, T2old, U1nold[0], optimize = True)
    U2n[0] += 0.5*einsum('klcd,dj,balk,Ici->Iabij', I.oovv, T1old, T2old, U1nold[0], optimize = True)
    U2n[0] += -1.0*einsum('klcd,dj,bcki,Ial->Iabij', I.oovv, T1old, T2old, U1nold[0], optimize = True)
    U2n[0] += -1.0*einsum('klcd,dk,acji,Ibl->Iabij', I.oovv, T1old, T2old, U1nold[0], optimize = True)
    U2n[0] += 1.0*einsum('klcd,dk,bali,Icj->Iabij', I.oovv, T1old, T2old, U1nold[0], optimize = True)
    U2n[0] += -1.0*einsum('klcd,dk,balj,Ici->Iabij', I.oovv, T1old, T2old, U1nold[0], optimize = True)
    U2n[0] += 1.0*einsum('klcd,dk,bcji,Ial->Iabij', I.oovv, T1old, T2old, U1nold[0], optimize = True)
    U2n[0] += 1.0*einsum('klcd,di,bk,al,Icj->Iabij', I.oovv, T1old, T1old, T1old, U1nold[0], optimize = True)
    U2n[0] += -1.0*einsum('klcd,di,cj,ak,Ibl->Iabij', I.oovv, T1old, T1old, T1old, U1nold[0], optimize = True)
    U2n[0] += 1.0*einsum('klcd,di,cj,bk,Ial->Iabij', I.oovv, T1old, T1old, T1old, U1nold[0], optimize = True)
    U2n[0] += -1.0*einsum('klcd,dj,bk,al,Ici->Iabij', I.oovv, T1old, T1old, T1old, U1nold[0], optimize = True)
    if nfock1 > 1:
        U2n[0] += 1.0*einsum('Jki,bakj,IJ->Iabij', g.oo, T2old, Snold[1], optimize = True)
        U2n[0] += -1.0*einsum('Jkj,baki,IJ->Iabij', g.oo, T2old, Snold[1], optimize = True)
        U2n[0] += 1.0*einsum('Jac,bcji,IJ->Iabij', g.vv, T2old, Snold[1], optimize = True)
        U2n[0] += -1.0*einsum('Jbc,acji,IJ->Iabij', g.vv, T2old, Snold[1], optimize = True)
        U2n[0] += -1.0*einsum('Jkc,ak,bcji,IJ->Iabij', g.ov, T1old, T2old, Snold[1], optimize = True)
        U2n[0] += 1.0*einsum('Jkc,bk,acji,IJ->Iabij', g.ov, T1old, T2old, Snold[1], optimize = True)
        U2n[0] += 1.0*einsum('Jkc,ci,bakj,IJ->Iabij', g.ov, T1old, T2old, Snold[1], optimize = True)
        U2n[0] += -1.0*einsum('Jkc,cj,baki,IJ->Iabij', g.ov, T1old, T2old, Snold[1], optimize = True)

    if nfock2 > 1:
        gvoU12_oovv = einsum('Jai,IJbj->Iabij', g.vo, U1nold[1])
        U2n[0] += gvoU12_oovv
        U2n[0] -= gvoU12_oovv.transpose((0,2,1,3,4))
        U2n[0] -= gvoU12_oovv.transpose((0,1,2,4,3))
        U2n[0] += gvoU12_oovv.transpose((0,2,1,4,3))
        gvoU12_oovv= None
        #U2n[0] += 1.0*einsum('Jai,IJbj->Iabij', g.vo, U1nold[1], optimize = True)
        #U2n[0] += -1.0*einsum('Jaj,IJbi->Iabij', g.vo, U1nold[1], optimize = True)
        #U2n[0] += -1.0*einsum('Jbi,IJaj->Iabij', g.vo, U1nold[1], optimize = True)
        #U2n[0] += 1.0*einsum('Jbj,IJai->Iabij', g.vo, U1nold[1], optimize = True)

        gtov = einsum('xac,ci->xai', g.vv, T1old)
        gtov -= einsum('xki,ak->xai', g.oo, T1old)
        temp_abij = einsum('xai,xybj->xabij', gtov, U1nold[1])
        #T2temp = - einsum('bi,aj->abij', T1old, T1old)
        T2temp = T2old - einsum('bi,aj->abij', T1old, T1old)
        Wmvo = einsum('xkc,acik->xai', g.ov, T2temp)
        temp_abij += einsum('xai,xybj->xabij', Wmvo, U1nold[1])
        U2n[0] += temp_abij
        U2n[0] -= temp_abij.transpose((0,2,1,3,4))
        U2n[0] -= temp_abij.transpose((0,1,2,4,3))
        U2n[0] += temp_abij.transpose((0,2,1,4,3))
        temp_abij = None

        #U2n[0] += -1.0*einsum('Jki,ak,IJbj->Iabij', g.oo, T1old, U1nold[1], optimize = True)
        #U2n[0] += 1.0*einsum('Jki,bk,IJaj->Iabij', g.oo, T1old, U1nold[1], optimize = True)
        #U2n[0] += 1.0*einsum('Jkj,ak,IJbi->Iabij', g.oo, T1old, U1nold[1], optimize = True)
        #U2n[0] += -1.0*einsum('Jkj,bk,IJai->Iabij', g.oo, T1old, U1nold[1], optimize = True)
        #U2n[0] += 1.0*einsum('Jac,ci,IJbj->Iabij', g.vv, T1old, U1nold[1], optimize = True)
        #U2n[0] += -1.0*einsum('Jac,cj,IJbi->Iabij', g.vv, T1old, U1nold[1], optimize = True)
        #U2n[0] += -1.0*einsum('Jbc,ci,IJaj->Iabij', g.vv, T1old, U1nold[1], optimize = True)
        #U2n[0] += 1.0*einsum('Jbc,cj,IJai->Iabij', g.vv, T1old, U1nold[1], optimize = True)
        #U2n[0] += -1.0*einsum('Jkc,acki,IJbj->Iabij', g.ov, T2old, U1nold[1], optimize = True)
        #U2n[0] += 1.0*einsum('Jkc,ackj,IJbi->Iabij', g.ov, T2old, U1nold[1], optimize = True)
        U2n[0] += 1.0*einsum('Jkc,acji,IJbk->Iabij', g.ov, T2old, U1nold[1], optimize = True)
        U2n[0] += -1.0*einsum('Jkc,baki,IJcj->Iabij', g.ov, T2old, U1nold[1], optimize = True)
        U2n[0] += 1.0*einsum('Jkc,bakj,IJci->Iabij', g.ov, T2old, U1nold[1], optimize = True)
        U2n[0] += -1.0*einsum('Jkc,bcji,IJak->Iabij', g.ov, T2old, U1nold[1], optimize = True)
        #U2n[0] += 1.0*einsum('Jkc,bcki,IJaj->Iabij', g.ov, T2old, U1nold[1], optimize = True)
        #U2n[0] += -1.0*einsum('Jkc,bckj,IJai->Iabij', g.ov, T2old, U1nold[1], optimize = True)
        #U2n[0] += -1.0*einsum('Jkc,ci,ak,IJbj->Iabij', g.ov, T1old, T1old, U1nold[1], optimize = True)
        #U2n[0] += 1.0*einsum('Jkc,ci,bk,IJaj->Iabij', g.ov, T1old, T1old, U1nold[1], optimize = True)
        #U2n[0] += 1.0*einsum('Jkc,cj,ak,IJbi->Iabij', g.ov, T1old, T1old, U1nold[1], optimize = True)
        #U2n[0] += -1.0*einsum('Jkc,cj,bk,IJai->Iabij', g.ov, T1old, T1old, U1nold[1], optimize = True)

        U2n[0] += 1.0*einsum('Jki,IJbakj->Iabij', g.oo, U2nold[1], optimize = True)
        U2n[0] += -1.0*einsum('Jkj,IJbaki->Iabij', g.oo, U2nold[1], optimize = True)
        U2n[0] += 1.0*einsum('Jac,IJbcji->Iabij', g.vv, U2nold[1], optimize = True)
        U2n[0] += -1.0*einsum('Jbc,IJacji->Iabij', g.vv, U2nold[1], optimize = True)
        U2n[0] += -1.0*einsum('Jkc,ak,IJbcji->Iabij', g.ov, T1old, U2nold[1], optimize = True)
        U2n[0] += 1.0*einsum('Jkc,bk,IJacji->Iabij', g.ov, T1old, U2nold[1], optimize = True)
        U2n[0] += 1.0*einsum('Jkc,ci,IJbakj->Iabij', g.ov, T1old, U2nold[1], optimize = True)
        U2n[0] += -1.0*einsum('Jkc,cj,IJbaki->Iabij', g.ov, T1old, U2nold[1], optimize = True)
        U2n[0] += 1.0*einsum('Jkc,ck,IJbaji->Iabij', g.ov, T1old, U2nold[1], optimize = True)

    #U21old += einsum('abij,iajb -> abij', res_U21old, e_denom, optimize=True)

    if nfock2 < 2: return U2n

    eps_occ = F.oo.diagonal()
    eps_vir = F.vv.diagonal()
    eps_vir_p_2w = eps_vir + 2.0 * w

    #e_denom = 1 / (eps_occ.reshape(-1, 1, 1, 1) + eps_occ.reshape(-1, 1) - eps_vir.reshape(-1, 1, 1) - eps_vir_p_2w)

    U2n[1] = numpy.zeros((nm, nm, nvir, nvir, nocc, nocc))

    # works for single mode at this moment
    U22 = numpy.zeros((nvir, nvir, nocc, nocc))
    U11old = U1nold[0][0]   # single mode
    U12old = U1nold[1][0,0] # single mode
    U21old = U2nold[0][0]   # single mode
    U22old = U2nold[1][0,0] # single mode
    S1old = Snold[0][0]     # single mode
    if nfock1 > 1:
        S2old = Snold[1][0,0]   # single mode

    U22 += 1.0 * einsum('ki,bakj->abij', F.oo, U22old, optimize = True)
    U22 += -1.0 * einsum('kj,baki->abij', F.oo, U22old, optimize = True)
    U22 += -1.0 * einsum('bc,acji->abij', F.vv, U22old, optimize = True)
    U22 += 1.0 * einsum('ac,bcji->abij', F.vv, U22old, optimize = True)
    U22 += 1.0 * einsum('kbji,ak->abij', I.ovoo, U12old, optimize = True)
    U22 += -1.0 * einsum('kaji,bk->abij', I.ovoo, U12old, optimize = True)
    U22 += -1.0 * einsum('baic,cj->abij', I.vvov, U12old, optimize = True)
    U22 += 1.0 * einsum('bajc,ci->abij', I.vvov, U12old, optimize = True)
    U22 += 2.0 * einsum('J,IJbaji->abij', w, U2nold[1], optimize = True) # IJ
    #U22 += 1.0 * einsum('J,IJbaji->abij', w, U2nold[1], optimize = True) # IJ
    #U22 += 1.0 * einsum('I,IJbaji->abij', w, U2nold[1], optimize = True) # IJ
    U22 += 1.0 * einsum('ki,bakj->abij', g.oo[0], U21old, optimize = True)
    U22 += -1.0 * einsum('kj,baki->abij', g.oo[0], U21old, optimize = True)
    U22 += 1.0 * einsum('ki,bakj->abij', g.oo[0], U21old, optimize = True)
    U22 += -1.0 * einsum('kj,baki->abij', g.oo[0], U21old, optimize = True)
    U22 += -1.0 * einsum('bc,acji->abij', g.vv[0], U21old, optimize = True)
    U22 += 1.0 * einsum('ac,bcji->abij', g.vv[0], U21old, optimize = True)
    U22 += -1.0 * einsum('bc,acji->abij', g.vv[0], U21old, optimize = True)
    U22 += 1.0 * einsum('ac,bcji->abij', g.vv[0], U21old, optimize = True)
    U22 += -0.5 * einsum('klji,balk->abij', I.oooo, U22old, optimize = True)
    U22 += -1.0 * einsum('kbic,ackj->abij', I.ovov, U22old, optimize = True)
    U22 += 1.0 * einsum('kaic,bckj->abij', I.ovov, U22old, optimize = True)
    U22 += 1.0 * einsum('kbjc,acki->abij', I.ovov, U22old, optimize = True)
    U22 += -1.0 * einsum('kajc,bcki->abij', I.ovov, U22old, optimize = True)
    U22 += -0.5 * einsum('bacd,dcji->abij', I.vvvv, U22old, optimize = True)
    U22 += 1.0 * einsum('kc,ci,bakj->abij', F.ov, T1old, U22old, optimize = True)
    U22 += -1.0 * einsum('kc,cj,baki->abij', F.ov, T1old, U22old, optimize = True)
    U22 += 1.0 * einsum('kc,bk,acji->abij', F.ov, T1old, U22old, optimize = True)
    U22 += -1.0 * einsum('kc,ak,bcji->abij', F.ov, T1old, U22old, optimize = True)
    U22 += -1.0 * einsum('kc,bcji,ak->abij', F.ov, T2old, U12old, optimize = True)
    U22 += 1.0 * einsum('kc,acji,bk->abij', F.ov, T2old, U12old, optimize = True)
    U22 += -1.0 * einsum('kc,baki,cj->abij', F.ov, T2old, U12old, optimize = True)
    U22 += 1.0 * einsum('kc,bakj,ci->abij', F.ov, T2old, U12old, optimize = True)
    U22 += 1.0 * einsum('klji,bk,al->abij', I.oooo, T1old, U12old, optimize = True)
    U22 += -1.0 * einsum('klji,ak,bl->abij', I.oooo, T1old, U12old, optimize = True)
    U22 += -1.0 * einsum('kbic,cj,ak->abij', I.ovov, T1old, U12old, optimize = True)
    U22 += 1.0 * einsum('kaic,cj,bk->abij', I.ovov, T1old, U12old, optimize = True)
    U22 += -1.0 * einsum('kbic,ak,cj->abij', I.ovov, T1old, U12old, optimize = True)
    U22 += 1.0 * einsum('kaic,bk,cj->abij', I.ovov, T1old, U12old, optimize = True)
    U22 += 1.0 * einsum('kbjc,ci,ak->abij', I.ovov, T1old, U12old, optimize = True)
    U22 += -1.0 * einsum('kajc,ci,bk->abij', I.ovov, T1old, U12old, optimize = True)
    U22 += 1.0 * einsum('kbjc,ak,ci->abij', I.ovov, T1old, U12old, optimize = True)
    U22 += -1.0 * einsum('kajc,bk,ci->abij', I.ovov, T1old, U12old, optimize = True)
    U22 += 1.0 * einsum('bacd,di,cj->abij', I.vvvv, T1old, U12old, optimize = True)
    U22 += -1.0 * einsum('bacd,dj,ci->abij', I.vvvv, T1old, U12old, optimize = True)
    U22 += 1.0 * S1old * einsum('ki,bakj->abij', g.oo[0], U22old, optimize = True)
    U22 += -1.0 * S1old * einsum('kj,baki->abij', g.oo[0], U22old, optimize = True)
    if nfock1 > 1:
        U22 += 1.0 * S2old * einsum('ki,bakj->abij', g.oo[0], U21old, optimize = True)
        U22 += -1.0 * S2old * einsum('kj,baki->abij', g.oo[0], U21old, optimize = True)
        U22 += 1.0 * S2old * einsum('ki,bakj->abij', g.oo[0], U21old, optimize = True)
        U22 += -1.0 * S2old * einsum('kj,baki->abij', g.oo[0], U21old, optimize = True)
        U22 += -1.0 * S2old * einsum('bc,acji->abij', g.vv[0], U21old, optimize = True)
        U22 += 1.0 * S2old * einsum('ac,bcji->abij', g.vv[0], U21old, optimize = True)
        U22 += -1.0 * S2old * einsum('bc,acji->abij', g.vv[0], U21old, optimize = True)
        U22 += 1.0 * S2old * einsum('ac,bcji->abij', g.vv[0], U21old, optimize = True)
        U22 += 1.0 * S2old * einsum('kc,ci,bakj->abij', g.ov[0], T1old, U21old, optimize = True)
        U22 += -1.0 * S2old * einsum('kc,cj,baki->abij', g.ov[0], T1old, U21old, optimize = True)
        U22 += 1.0 * S2old * einsum('kc,bk,acji->abij', g.ov[0], T1old, U21old, optimize = True)
        U22 += -1.0 * S2old * einsum('kc,ak,bcji->abij', g.ov[0], T1old, U21old, optimize = True)
        U22 += 1.0 * S2old * einsum('kc,ci,bakj->abij', g.ov[0], T1old, U21old, optimize = True)
        U22 += -1.0 * S2old * einsum('kc,cj,baki->abij', g.ov[0], T1old, U21old, optimize = True)
        U22 += 1.0 * S2old * einsum('kc,bk,acji->abij', g.ov[0], T1old, U21old, optimize = True)
        U22 += -1.0 * S2old * einsum('kc,ak,bcji->abij', g.ov[0], T1old, U21old, optimize = True)
        U22 += -1.0 * S2old * einsum('kc,bcji,ak->abij', g.ov[0], T2old, U11old, optimize = True)
        U22 += 1.0 * S2old * einsum('kc,acji,bk->abij', g.ov[0], T2old, U11old, optimize = True)
        U22 += -1.0 * S2old * einsum('kc,baki,cj->abij', g.ov[0], T2old, U11old, optimize = True)
        U22 += 1.0 * S2old * einsum('kc,bakj,ci->abij', g.ov[0], T2old, U11old, optimize = True)
        U22 += -1.0 * S2old * einsum('kc,bcji,ak->abij', g.ov[0], T2old, U11old, optimize = True)
        U22 += 1.0 * S2old * einsum('kc,acji,bk->abij', g.ov[0], T2old, U11old, optimize = True)
        U22 += -1.0 * S2old * einsum('kc,baki,cj->abij', g.ov[0], T2old, U11old, optimize = True)
        U22 += 1.0 * S2old * einsum('kc,bakj,ci->abij', g.ov[0], T2old, U11old, optimize = True)
    U22 += 1.0 * einsum('kc,ci,bakj->abij', g.ov[0], T1old, U21old, optimize = True)
    U22 += -1.0 * einsum('kc,cj,baki->abij', g.ov[0], T1old, U21old, optimize = True)
    U22 += 1.0 * einsum('kc,bk,acji->abij', g.ov[0], T1old, U21old, optimize = True)
    U22 += -1.0 * einsum('kc,ak,bcji->abij', g.ov[0], T1old, U21old, optimize = True)
    U22 += 1.0 * einsum('kc,ci,bakj->abij', g.ov[0], T1old, U21old, optimize = True)
    U22 += -1.0 * einsum('kc,cj,baki->abij', g.ov[0], T1old, U21old, optimize = True)
    U22 += 1.0 * einsum('kc,bk,acji->abij', g.ov[0], T1old, U21old, optimize = True)
    U22 += -1.0 * einsum('kc,ak,bcji->abij', g.ov[0], T1old, U21old, optimize = True)
    U22 += -1.0 * einsum('kc,bcji,ak->abij', g.ov[0], T2old, U11old, optimize = True)
    U22 += 1.0 * einsum('kc,acji,bk->abij', g.ov[0], T2old, U11old, optimize = True)
    U22 += -1.0 * einsum('kc,baki,cj->abij', g.ov[0], T2old, U11old, optimize = True)
    U22 += 1.0 * einsum('kc,bakj,ci->abij', g.ov[0], T2old, U11old, optimize = True)
    U22 += -1.0 * einsum('kc,bcji,ak->abij', g.ov[0], T2old, U11old, optimize = True)
    U22 += 1.0 * einsum('kc,acji,bk->abij', g.ov[0], T2old, U11old, optimize = True)
    U22 += -1.0 * einsum('kc,baki,cj->abij', g.ov[0], T2old, U11old, optimize = True)
    U22 += 1.0 * einsum('kc,bakj,ci->abij', g.ov[0], T2old, U11old, optimize = True)
    U22 += -1.0 * S1old * einsum('bc,acji->abij', g.vv[0], U22old, optimize = True)
    U22 += 1.0 * S1old * einsum('ac,bcji->abij', g.vv[0], U22old, optimize = True)
    U22 += 0.5 * einsum('klic,cj,balk->abij', I.ooov, T1old, U22old, optimize = True)
    U22 += -1.0 * einsum('klic,bk,aclj->abij', I.ooov, T1old, U22old, optimize = True)
    U22 += 1.0 * einsum('klic,ak,bclj->abij', I.ooov, T1old, U22old, optimize = True)
    U22 += -1.0 * einsum('klic,ck,balj->abij', I.ooov, T1old, U22old, optimize = True)
    U22 += -0.5 * einsum('kljc,ci,balk->abij', I.ooov, T1old, U22old, optimize = True)
    U22 += 1.0 * einsum('kljc,bk,acli->abij', I.ooov, T1old, U22old, optimize = True)
    U22 += -1.0 * einsum('kljc,ak,bcli->abij', I.ooov, T1old, U22old, optimize = True)
    U22 += 1.0 * einsum('kljc,ck,bali->abij', I.ooov, T1old, U22old, optimize = True)
    U22 += 1.0 * einsum('klic,bakj,cl->abij', I.ooov, T2old, U12old, optimize = True)
    U22 += -1.0 * einsum('klic,bckj,al->abij', I.ooov, T2old, U12old, optimize = True)
    U22 += 1.0 * einsum('klic,ackj,bl->abij', I.ooov, T2old, U12old, optimize = True)
    U22 += 0.5 * einsum('klic,balk,cj->abij', I.ooov, T2old, U12old, optimize = True)
    U22 += -1.0 * einsum('kljc,baki,cl->abij', I.ooov, T2old, U12old, optimize = True)
    U22 += 1.0 * einsum('kljc,bcki,al->abij', I.ooov, T2old, U12old, optimize = True)
    U22 += -1.0 * einsum('kljc,acki,bl->abij', I.ooov, T2old, U12old, optimize = True)
    U22 += -0.5 * einsum('kljc,balk,ci->abij', I.ooov, T2old, U12old, optimize = True)
    U22 += 1.0 * einsum('kbcd,di,ackj->abij', I.ovvv, T1old, U22old, optimize = True)
    U22 += -1.0 * einsum('kacd,di,bckj->abij', I.ovvv, T1old, U22old, optimize = True)
    U22 += -1.0 * einsum('kbcd,dj,acki->abij', I.ovvv, T1old, U22old, optimize = True)
    U22 += 1.0 * einsum('kacd,dj,bcki->abij', I.ovvv, T1old, U22old, optimize = True)
    U22 += -0.5 * einsum('kbcd,ak,dcji->abij', I.ovvv, T1old, U22old, optimize = True)
    U22 += 1.0 * einsum('kbcd,dk,acji->abij', I.ovvv, T1old, U22old, optimize = True)
    U22 += 0.5 * einsum('kacd,bk,dcji->abij', I.ovvv, T1old, U22old, optimize = True)
    U22 += -1.0 * einsum('kacd,dk,bcji->abij', I.ovvv, T1old, U22old, optimize = True)
    U22 += -1.0 * einsum('kbcd,adji,ck->abij', I.ovvv, T2old, U12old, optimize = True)
    U22 += -0.5 * einsum('kbcd,dcji,ak->abij', I.ovvv, T2old, U12old, optimize = True)
    U22 += 1.0 * einsum('kacd,bdji,ck->abij', I.ovvv, T2old, U12old, optimize = True)
    U22 += 0.5 * einsum('kacd,dcji,bk->abij', I.ovvv, T2old, U12old, optimize = True)
    U22 += 1.0 * einsum('kbcd,adki,cj->abij', I.ovvv, T2old, U12old, optimize = True)
    U22 += -1.0 * einsum('kacd,bdki,cj->abij', I.ovvv, T2old, U12old, optimize = True)
    U22 += -1.0 * einsum('kbcd,adkj,ci->abij', I.ovvv, T2old, U12old, optimize = True)
    U22 += 1.0 * einsum('kacd,bdkj,ci->abij', I.ovvv, T2old, U12old, optimize = True)
    U22 += -0.5 * einsum('klcd,bdji,aclk->abij', I.oovv, T2old, U22old, optimize = True)
    U22 += 0.5 * einsum('klcd,adji,bclk->abij', I.oovv, T2old, U22old, optimize = True)
    U22 += 0.25 * einsum('klcd,dcji,balk->abij', I.oovv, T2old, U22old, optimize = True)
    U22 += -0.5 * einsum('klcd,baki,dclj->abij', I.oovv, T2old, U22old, optimize = True)
    U22 += 1.0 * einsum('klcd,bdki,aclj->abij', I.oovv, T2old, U22old, optimize = True)
    U22 += -1.0 * einsum('klcd,adki,bclj->abij', I.oovv, T2old, U22old, optimize = True)
    U22 += -0.5 * einsum('klcd,dcki,balj->abij', I.oovv, T2old, U22old, optimize = True)
    U22 += 0.5 * einsum('klcd,bakj,dcli->abij', I.oovv, T2old, U22old, optimize = True)
    U22 += -1.0 * einsum('klcd,bdkj,acli->abij', I.oovv, T2old, U22old, optimize = True)
    U22 += 1.0 * einsum('klcd,adkj,bcli->abij', I.oovv, T2old, U22old, optimize = True)
    U22 += 0.5 * einsum('klcd,dckj,bali->abij', I.oovv, T2old, U22old, optimize = True)
    U22 += 0.25 * einsum('klcd,balk,dcji->abij', I.oovv, T2old, U22old, optimize = True)
    U22 += -0.5 * einsum('klcd,bdlk,acji->abij', I.oovv, T2old, U22old, optimize = True)
    U22 += 0.5 * einsum('klcd,adlk,bcji->abij', I.oovv, T2old, U22old, optimize = True)
    U22 += 2.0 * einsum('kc,ci,bakj->abij', F.ov, U11old, U21old, optimize = True)
    U22 += -2.0 * einsum('kc,cj,baki->abij', F.ov, U11old, U21old, optimize = True)
    U22 += 2.0 * einsum('kc,bk,acji->abij', F.ov, U11old, U21old, optimize = True)
    U22 += -2.0 * einsum('kc,ak,bcji->abij', F.ov, U11old, U21old, optimize = True)
    U22 += 2.0 * einsum('klji,bk,al->abij', I.oooo, U11old, U11old, optimize = True)
    U22 += -2.0 * einsum('kbic,cj,ak->abij', I.ovov, U11old, U11old, optimize = True)
    U22 += 2.0 * einsum('kaic,cj,bk->abij', I.ovov, U11old, U11old, optimize = True)
    U22 += 2.0 * einsum('kbjc,ci,ak->abij', I.ovov, U11old, U11old, optimize = True)
    U22 += -2.0 * einsum('kajc,ci,bk->abij', I.ovov, U11old, U11old, optimize = True)
    U22 += 2.0 * einsum('bacd,di,cj->abij', I.vvvv, U11old, U11old, optimize = True)
    U22 += 1.0 * einsum('ki,bk,aj->abij', g.oo[0], U11old, U12old, optimize = True)
    U22 += -1.0 * einsum('ki,ak,bj->abij', g.oo[0], U11old, U12old, optimize = True)
    U22 += -1.0 * einsum('kj,bk,ai->abij', g.oo[0], U11old, U12old, optimize = True)
    U22 += 1.0 * einsum('kj,ak,bi->abij', g.oo[0], U11old, U12old, optimize = True)
    U22 += 1.0 * einsum('ki,bk,aj->abij', g.oo[0], U11old, U12old, optimize = True)
    U22 += -1.0 * einsum('ki,ak,bj->abij', g.oo[0], U11old, U12old, optimize = True)
    U22 += -1.0 * einsum('kj,bk,ai->abij', g.oo[0], U11old, U12old, optimize = True)
    U22 += 1.0 * einsum('kj,ak,bi->abij', g.oo[0], U11old, U12old, optimize = True)
    U22 += -1.0 * einsum('ki,bj,ak->abij', g.oo[0], U11old, U12old, optimize = True)
    U22 += 1.0 * einsum('ki,aj,bk->abij', g.oo[0], U11old, U12old, optimize = True)
    U22 += 1.0 * einsum('kj,bi,ak->abij', g.oo[0], U11old, U12old, optimize = True)
    U22 += -1.0 * einsum('kj,ai,bk->abij', g.oo[0], U11old, U12old, optimize = True)
    U22 += -1.0 * einsum('bc,ci,aj->abij', g.vv[0], U11old, U12old, optimize = True)
    U22 += 1.0 * einsum('ac,ci,bj->abij', g.vv[0], U11old, U12old, optimize = True)
    U22 += 1.0 * einsum('bc,cj,ai->abij', g.vv[0], U11old, U12old, optimize = True)
    U22 += -1.0 * einsum('ac,cj,bi->abij', g.vv[0], U11old, U12old, optimize = True)
    U22 += -1.0 * einsum('bc,ci,aj->abij', g.vv[0], U11old, U12old, optimize = True)
    U22 += 1.0 * einsum('ac,ci,bj->abij', g.vv[0], U11old, U12old, optimize = True)
    U22 += 1.0 * einsum('bc,cj,ai->abij', g.vv[0], U11old, U12old, optimize = True)
    U22 += -1.0 * einsum('ac,cj,bi->abij', g.vv[0], U11old, U12old, optimize = True)
    U22 += 1.0 * einsum('bc,ai,cj->abij', g.vv[0], U11old, U12old, optimize = True)
    U22 += -1.0 * einsum('ac,bi,cj->abij', g.vv[0], U11old, U12old, optimize = True)
    U22 += -1.0 * einsum('bc,aj,ci->abij', g.vv[0], U11old, U12old, optimize = True)
    U22 += 1.0 * einsum('ac,bj,ci->abij', g.vv[0], U11old, U12old, optimize = True)
    U22 += 1.0 * einsum('klic,cj,balk->abij', I.ooov, U11old, U21old, optimize = True)
    U22 += -2.0 * einsum('klic,bk,aclj->abij', I.ooov, U11old, U21old, optimize = True)
    U22 += 2.0 * einsum('klic,ak,bclj->abij', I.ooov, U11old, U21old, optimize = True)
    U22 += -2.0 * einsum('klic,ck,balj->abij', I.ooov, U11old, U21old, optimize = True)
    U22 += -1.0 * einsum('kljc,ci,balk->abij', I.ooov, U11old, U21old, optimize = True)
    U22 += 2.0 * einsum('kljc,bk,acli->abij', I.ooov, U11old, U21old, optimize = True)
    U22 += -2.0 * einsum('kljc,ak,bcli->abij', I.ooov, U11old, U21old, optimize = True)
    U22 += 2.0 * einsum('kljc,ck,bali->abij', I.ooov, U11old, U21old, optimize = True)
    U22 += 2.0 * einsum('kbcd,di,ackj->abij', I.ovvv, U11old, U21old, optimize = True)
    U22 += -2.0 * einsum('kacd,di,bckj->abij', I.ovvv, U11old, U21old, optimize = True)
    U22 += -2.0 * einsum('kbcd,dj,acki->abij', I.ovvv, U11old, U21old, optimize = True)
    U22 += 2.0 * einsum('kacd,dj,bcki->abij', I.ovvv, U11old, U21old, optimize = True)
    U22 += -1.0 * einsum('kbcd,ak,dcji->abij', I.ovvv, U11old, U21old, optimize = True)
    U22 += 2.0 * einsum('kbcd,dk,acji->abij', I.ovvv, U11old, U21old, optimize = True)
    U22 += 1.0 * einsum('kacd,bk,dcji->abij', I.ovvv, U11old, U21old, optimize = True)
    U22 += -2.0 * einsum('kacd,dk,bcji->abij', I.ovvv, U11old, U21old, optimize = True)
    U22 += 1.0 * einsum('kc,ci,bakj->abij', g.ov[0], U11old, U22old, optimize = True)
    U22 += -1.0 * einsum('kc,cj,baki->abij', g.ov[0], U11old, U22old, optimize = True)
    U22 += 1.0 * einsum('kc,bk,acji->abij', g.ov[0], U11old, U22old, optimize = True)
    U22 += -1.0 * einsum('kc,ak,bcji->abij', g.ov[0], U11old, U22old, optimize = True)
    U22 += 1.0 * einsum('kc,ck,baji->abij', g.ov[0], U11old, U22old, optimize = True)
    U22 += 1.0 * einsum('kc,ci,bakj->abij', g.ov[0], U11old, U22old, optimize = True)
    U22 += -1.0 * einsum('kc,cj,baki->abij', g.ov[0], U11old, U22old, optimize = True)
    U22 += 1.0 * einsum('kc,bk,acji->abij', g.ov[0], U11old, U22old, optimize = True)
    U22 += -1.0 * einsum('kc,ak,bcji->abij', g.ov[0], U11old, U22old, optimize = True)
    U22 += 1.0 * einsum('kc,ck,baji->abij', g.ov[0], U11old, U22old, optimize = True)
    U22 += 1.0 * einsum('kc,bi,ackj->abij', g.ov[0], U11old, U22old, optimize = True)
    U22 += -1.0 * einsum('kc,ai,bckj->abij', g.ov[0], U11old, U22old, optimize = True)
    U22 += 1.0 * einsum('kc,ci,bakj->abij', g.ov[0], U11old, U22old, optimize = True)
    U22 += -1.0 * einsum('kc,bj,acki->abij', g.ov[0], U11old, U22old, optimize = True)
    U22 += 1.0 * einsum('kc,aj,bcki->abij', g.ov[0], U11old, U22old, optimize = True)
    U22 += -1.0 * einsum('kc,cj,baki->abij', g.ov[0], U11old, U22old, optimize = True)
    U22 += 1.0 * einsum('kc,bk,acji->abij', g.ov[0], U11old, U22old, optimize = True)
    U22 += -1.0 * einsum('kc,ak,bcji->abij', g.ov[0], U11old, U22old, optimize = True)
    U22 += -1.0 * einsum('kc,bcji,ak->abij', g.ov[0], U21old, U12old, optimize = True)
    U22 += 1.0 * einsum('kc,acji,bk->abij', g.ov[0], U21old, U12old, optimize = True)
    U22 += -1.0 * einsum('kc,baki,cj->abij', g.ov[0], U21old, U12old, optimize = True)
    U22 += 1.0 * einsum('kc,bcki,aj->abij', g.ov[0], U21old, U12old, optimize = True)
    U22 += -1.0 * einsum('kc,acki,bj->abij', g.ov[0], U21old, U12old, optimize = True)
    U22 += 1.0 * einsum('kc,bakj,ci->abij', g.ov[0], U21old, U12old, optimize = True)
    U22 += -1.0 * einsum('kc,bckj,ai->abij', g.ov[0], U21old, U12old, optimize = True)
    U22 += 1.0 * einsum('kc,ackj,bi->abij', g.ov[0], U21old, U12old, optimize = True)
    U22 += -1.0 * einsum('kc,bcji,ak->abij', g.ov[0], U21old, U12old, optimize = True)
    U22 += 1.0 * einsum('kc,acji,bk->abij', g.ov[0], U21old, U12old, optimize = True)
    U22 += -1.0 * einsum('kc,baki,cj->abij', g.ov[0], U21old, U12old, optimize = True)
    U22 += 1.0 * einsum('kc,bcki,aj->abij', g.ov[0], U21old, U12old, optimize = True)
    U22 += -1.0 * einsum('kc,acki,bj->abij', g.ov[0], U21old, U12old, optimize = True)
    U22 += 1.0 * einsum('kc,bakj,ci->abij', g.ov[0], U21old, U12old, optimize = True)
    U22 += -1.0 * einsum('kc,bckj,ai->abij', g.ov[0], U21old, U12old, optimize = True)
    U22 += 1.0 * einsum('kc,ackj,bi->abij', g.ov[0], U21old, U12old, optimize = True)
    U22 += 1.0 * einsum('kc,baji,ck->abij', g.ov[0], U21old, U12old, optimize = True)
    U22 += -1.0 * einsum('kc,bcji,ak->abij', g.ov[0], U21old, U12old, optimize = True)
    U22 += 1.0 * einsum('kc,acji,bk->abij', g.ov[0], U21old, U12old, optimize = True)
    U22 += -1.0 * einsum('kc,baki,cj->abij', g.ov[0], U21old, U12old, optimize = True)
    U22 += 1.0 * einsum('kc,bakj,ci->abij', g.ov[0], U21old, U12old, optimize = True)
    U22 += -1.0 * einsum('klic,cj,bk,al->abij', I.ooov, T1old, T1old, U12old, optimize = True)
    U22 += 1.0 * einsum('klic,cj,ak,bl->abij', I.ooov, T1old, T1old, U12old, optimize = True)
    U22 += -1.0 * einsum('klic,bk,al,cj->abij', I.ooov, T1old, T1old, U12old, optimize = True)
    U22 += 1.0 * einsum('kljc,ci,bk,al->abij', I.ooov, T1old, T1old, U12old, optimize = True)
    U22 += -1.0 * einsum('kljc,ci,ak,bl->abij', I.ooov, T1old, T1old, U12old, optimize = True)
    U22 += 1.0 * einsum('kljc,bk,al,ci->abij', I.ooov, T1old, T1old, U12old, optimize = True)
    U22 += 1.0 * einsum('kbcd,di,cj,ak->abij', I.ovvv, T1old, T1old, U12old, optimize = True)
    U22 += -1.0 * einsum('kacd,di,cj,bk->abij', I.ovvv, T1old, T1old, U12old, optimize = True)
    U22 += 1.0 * einsum('kbcd,di,ak,cj->abij', I.ovvv, T1old, T1old, U12old, optimize = True)
    U22 += -1.0 * einsum('kacd,di,bk,cj->abij', I.ovvv, T1old, T1old, U12old, optimize = True)
    U22 += -1.0 * einsum('kbcd,dj,ak,ci->abij', I.ovvv, T1old, T1old, U12old, optimize = True)
    U22 += 1.0 * einsum('kacd,dj,bk,ci->abij', I.ovvv, T1old, T1old, U12old, optimize = True)
    U22 += 1.0 * S1old * einsum('kc,ci,bakj->abij', g.ov[0], T1old, U22old, optimize = True)
    U22 += -1.0 * S1old * einsum('kc,cj,baki->abij', g.ov[0], T1old, U22old, optimize = True)
    U22 += 1.0 * S1old * einsum('kc,bk,acji->abij', g.ov[0], T1old, U22old, optimize = True)
    U22 += -1.0 * S1old * einsum('kc,ak,bcji->abij', g.ov[0], T1old, U22old, optimize = True)
    U22 += -1.0 * S1old * einsum('kc,bcji,ak->abij', g.ov[0], T2old, U12old, optimize = True)
    U22 += 1.0 * S1old * einsum('kc,acji,bk->abij', g.ov[0], T2old, U12old, optimize = True)
    U22 += -1.0 * S1old * einsum('kc,baki,cj->abij', g.ov[0], T2old, U12old, optimize = True)
    U22 += 1.0 * S1old * einsum('kc,bakj,ci->abij', g.ov[0], T2old, U12old, optimize = True)
    U22 += -1.0 * einsum('klcd,bdji,aclk->abij', I.oovv, U21old, U21old, optimize = True)
    U22 += 1.0 * einsum('klcd,adji,bclk->abij', I.oovv, U21old, U21old, optimize = True)
    U22 += 0.5 * einsum('klcd,dcji,balk->abij', I.oovv, U21old, U21old, optimize = True)
    U22 += -1.0 * einsum('klcd,baki,dclj->abij', I.oovv, U21old, U21old, optimize = True)
    U22 += 2.0 * einsum('klcd,bdki,aclj->abij', I.oovv, U21old, U21old, optimize = True)
    U22 += -2.0 * einsum('klcd,adki,bclj->abij', I.oovv, U21old, U21old, optimize = True)
    U22 += -1.0 * einsum('klcd,dcki,balj->abij', I.oovv, U21old, U21old, optimize = True)
    U22 += -0.5 * einsum('klcd,di,cj,balk->abij', I.oovv, T1old, T1old, U22old, optimize = True)
    U22 += 1.0 * einsum('klcd,di,bk,aclj->abij', I.oovv, T1old, T1old, U22old, optimize = True)
    U22 += -1.0 * einsum('klcd,di,ak,bclj->abij', I.oovv, T1old, T1old, U22old, optimize = True)
    U22 += 1.0 * einsum('klcd,di,ck,balj->abij', I.oovv, T1old, T1old, U22old, optimize = True)
    U22 += -1.0 * einsum('klcd,dj,bk,acli->abij', I.oovv, T1old, T1old, U22old, optimize = True)
    U22 += 1.0 * einsum('klcd,dj,ak,bcli->abij', I.oovv, T1old, T1old, U22old, optimize = True)
    U22 += -1.0 * einsum('klcd,dj,ck,bali->abij', I.oovv, T1old, T1old, U22old, optimize = True)
    U22 += -0.5 * einsum('klcd,bk,al,dcji->abij', I.oovv, T1old, T1old, U22old, optimize = True)
    U22 += 1.0 * einsum('klcd,bk,dl,acji->abij', I.oovv, T1old, T1old, U22old, optimize = True)
    U22 += -1.0 * einsum('klcd,ak,dl,bcji->abij', I.oovv, T1old, T1old, U22old, optimize = True)
    U22 += -1.0 * einsum('klcd,di,bakj,cl->abij', I.oovv, T1old, T2old, U12old, optimize = True)
    U22 += 1.0 * einsum('klcd,di,bckj,al->abij', I.oovv, T1old, T2old, U12old, optimize = True)
    U22 += -1.0 * einsum('klcd,di,ackj,bl->abij', I.oovv, T1old, T2old, U12old, optimize = True)
    U22 += -0.5 * einsum('klcd,di,balk,cj->abij', I.oovv, T1old, T2old, U12old, optimize = True)
    U22 += 1.0 * einsum('klcd,dj,baki,cl->abij', I.oovv, T1old, T2old, U12old, optimize = True)
    U22 += -1.0 * einsum('klcd,dj,bcki,al->abij', I.oovv, T1old, T2old, U12old, optimize = True)
    U22 += 1.0 * einsum('klcd,dj,acki,bl->abij', I.oovv, T1old, T2old, U12old, optimize = True)
    U22 += -1.0 * einsum('klcd,bk,adji,cl->abij', I.oovv, T1old, T2old, U12old, optimize = True)
    U22 += -0.5 * einsum('klcd,bk,dcji,al->abij', I.oovv, T1old, T2old, U12old, optimize = True)
    U22 += 1.0 * einsum('klcd,ak,bdji,cl->abij', I.oovv, T1old, T2old, U12old, optimize = True)
    U22 += 1.0 * einsum('klcd,dk,bcji,al->abij', I.oovv, T1old, T2old, U12old, optimize = True)
    U22 += 0.5 * einsum('klcd,ak,dcji,bl->abij', I.oovv, T1old, T2old, U12old, optimize = True)
    U22 += -1.0 * einsum('klcd,dk,acji,bl->abij', I.oovv, T1old, T2old, U12old, optimize = True)
    U22 += 1.0 * einsum('klcd,bk,adli,cj->abij', I.oovv, T1old, T2old, U12old, optimize = True)
    U22 += -1.0 * einsum('klcd,ak,bdli,cj->abij', I.oovv, T1old, T2old, U12old, optimize = True)
    U22 += 1.0 * einsum('klcd,dk,bali,cj->abij', I.oovv, T1old, T2old, U12old, optimize = True)
    U22 += 0.5 * einsum('klcd,dj,balk,ci->abij', I.oovv, T1old, T2old, U12old, optimize = True)
    U22 += -1.0 * einsum('klcd,bk,adlj,ci->abij', I.oovv, T1old, T2old, U12old, optimize = True)
    U22 += 1.0 * einsum('klcd,ak,bdlj,ci->abij', I.oovv, T1old, T2old, U12old, optimize = True)
    U22 += -1.0 * einsum('klcd,dk,balj,ci->abij', I.oovv, T1old, T2old, U12old, optimize = True)
    U22 += -2.0 * einsum('klic,cj,bk,al->abij', I.ooov, T1old, U11old, U11old, optimize = True)
    U22 += -2.0 * einsum('klic,bk,cj,al->abij', I.ooov, T1old, U11old, U11old, optimize = True)
    U22 += 2.0 * einsum('klic,ak,cj,bl->abij', I.ooov, T1old, U11old, U11old, optimize = True)
    U22 += 2.0 * einsum('kljc,ci,bk,al->abij', I.ooov, T1old, U11old, U11old, optimize = True)
    U22 += 2.0 * einsum('kljc,bk,ci,al->abij', I.ooov, T1old, U11old, U11old, optimize = True)
    U22 += -2.0 * einsum('kljc,ak,ci,bl->abij', I.ooov, T1old, U11old, U11old, optimize = True)
    U22 += 2.0 * einsum('kbcd,di,cj,ak->abij', I.ovvv, T1old, U11old, U11old, optimize = True)
    U22 += -2.0 * einsum('kacd,di,cj,bk->abij', I.ovvv, T1old, U11old, U11old, optimize = True)
    U22 += -2.0 * einsum('kbcd,dj,ci,ak->abij', I.ovvv, T1old, U11old, U11old, optimize = True)
    U22 += 2.0 * einsum('kacd,dj,ci,bk->abij', I.ovvv, T1old, U11old, U11old, optimize = True)
    U22 += 2.0 * einsum('kbcd,ak,di,cj->abij', I.ovvv, T1old, U11old, U11old, optimize = True)
    U22 += -2.0 * einsum('kacd,bk,di,cj->abij', I.ovvv, T1old, U11old, U11old, optimize = True)
    U22 += 1.0 * einsum('kc,ci,bk,aj->abij', g.ov[0], T1old, U11old, U12old, optimize = True)
    U22 += -1.0 * einsum('kc,ci,ak,bj->abij', g.ov[0], T1old, U11old, U12old, optimize = True)
    U22 += 1.0 * einsum('kc,bk,ci,aj->abij', g.ov[0], T1old, U11old, U12old, optimize = True)
    U22 += -1.0 * einsum('kc,ak,ci,bj->abij', g.ov[0], T1old, U11old, U12old, optimize = True)
    U22 += -1.0 * einsum('kc,cj,bk,ai->abij', g.ov[0], T1old, U11old, U12old, optimize = True)
    U22 += 1.0 * einsum('kc,cj,ak,bi->abij', g.ov[0], T1old, U11old, U12old, optimize = True)
    U22 += -1.0 * einsum('kc,bk,cj,ai->abij', g.ov[0], T1old, U11old, U12old, optimize = True)
    U22 += 1.0 * einsum('kc,ak,cj,bi->abij', g.ov[0], T1old, U11old, U12old, optimize = True)
    U22 += 1.0 * einsum('kc,ci,bk,aj->abij', g.ov[0], T1old, U11old, U12old, optimize = True)
    U22 += -1.0 * einsum('kc,ci,ak,bj->abij', g.ov[0], T1old, U11old, U12old, optimize = True)
    U22 += 1.0 * einsum('kc,bk,ci,aj->abij', g.ov[0], T1old, U11old, U12old, optimize = True)
    U22 += -1.0 * einsum('kc,ak,ci,bj->abij', g.ov[0], T1old, U11old, U12old, optimize = True)
    U22 += -1.0 * einsum('kc,cj,bk,ai->abij', g.ov[0], T1old, U11old, U12old, optimize = True)
    U22 += 1.0 * einsum('kc,cj,ak,bi->abij', g.ov[0], T1old, U11old, U12old, optimize = True)
    U22 += -1.0 * einsum('kc,bk,cj,ai->abij', g.ov[0], T1old, U11old, U12old, optimize = True)
    U22 += 1.0 * einsum('kc,ak,cj,bi->abij', g.ov[0], T1old, U11old, U12old, optimize = True)
    U22 += -1.0 * einsum('kc,ci,bj,ak->abij', g.ov[0], T1old, U11old, U12old, optimize = True)
    U22 += 1.0 * einsum('kc,ci,aj,bk->abij', g.ov[0], T1old, U11old, U12old, optimize = True)
    U22 += 1.0 * einsum('kc,cj,bi,ak->abij', g.ov[0], T1old, U11old, U12old, optimize = True)
    U22 += -1.0 * einsum('kc,cj,ai,bk->abij', g.ov[0], T1old, U11old, U12old, optimize = True)
    U22 += -1.0 * einsum('kc,bk,ai,cj->abij', g.ov[0], T1old, U11old, U12old, optimize = True)
    U22 += 1.0 * einsum('kc,ak,bi,cj->abij', g.ov[0], T1old, U11old, U12old, optimize = True)
    U22 += 1.0 * einsum('kc,bk,aj,ci->abij', g.ov[0], T1old, U11old, U12old, optimize = True)
    U22 += -1.0 * einsum('kc,ak,bj,ci->abij', g.ov[0], T1old, U11old, U12old, optimize = True)
    U22 += 2.0 * S1old * einsum('kc,ci,bakj->abij', g.ov[0], U11old, U21old, optimize = True)
    U22 += -2.0 * S1old * einsum('kc,cj,baki->abij', g.ov[0], U11old, U21old, optimize = True)
    U22 += 2.0 * S1old * einsum('kc,bk,acji->abij', g.ov[0], U11old, U21old, optimize = True)
    U22 += -2.0 * S1old * einsum('kc,ak,bcji->abij', g.ov[0], U11old, U21old, optimize = True)
    U22 += -1.0 * einsum('klcd,di,cj,balk->abij', I.oovv, T1old, U11old, U21old, optimize = True)
    U22 += 2.0 * einsum('klcd,di,bk,aclj->abij', I.oovv, T1old, U11old, U21old, optimize = True)
    U22 += -2.0 * einsum('klcd,di,ak,bclj->abij', I.oovv, T1old, U11old, U21old, optimize = True)
    U22 += 2.0 * einsum('klcd,di,ck,balj->abij', I.oovv, T1old, U11old, U21old, optimize = True)
    U22 += 1.0 * einsum('klcd,dj,ci,balk->abij', I.oovv, T1old, U11old, U21old, optimize = True)
    U22 += 2.0 * einsum('klcd,bk,di,aclj->abij', I.oovv, T1old, U11old, U21old, optimize = True)
    U22 += -2.0 * einsum('klcd,ak,di,bclj->abij', I.oovv, T1old, U11old, U21old, optimize = True)
    U22 += -2.0 * einsum('klcd,dk,ci,balj->abij', I.oovv, T1old, U11old, U21old, optimize = True)
    U22 += -2.0 * einsum('klcd,dj,bk,acli->abij', I.oovv, T1old, U11old, U21old, optimize = True)
    U22 += 2.0 * einsum('klcd,dj,ak,bcli->abij', I.oovv, T1old, U11old, U21old, optimize = True)
    U22 += -2.0 * einsum('klcd,dj,ck,bali->abij', I.oovv, T1old, U11old, U21old, optimize = True)
    U22 += -2.0 * einsum('klcd,bk,dj,acli->abij', I.oovv, T1old, U11old, U21old, optimize = True)
    U22 += 2.0 * einsum('klcd,ak,dj,bcli->abij', I.oovv, T1old, U11old, U21old, optimize = True)
    U22 += 2.0 * einsum('klcd,dk,cj,bali->abij', I.oovv, T1old, U11old, U21old, optimize = True)
    U22 += -1.0 * einsum('klcd,bk,al,dcji->abij', I.oovv, T1old, U11old, U21old, optimize = True)
    U22 += 2.0 * einsum('klcd,bk,dl,acji->abij', I.oovv, T1old, U11old, U21old, optimize = True)
    U22 += 1.0 * einsum('klcd,ak,bl,dcji->abij', I.oovv, T1old, U11old, U21old, optimize = True)
    U22 += -2.0 * einsum('klcd,dk,bl,acji->abij', I.oovv, T1old, U11old, U21old, optimize = True)
    U22 += -2.0 * einsum('klcd,ak,dl,bcji->abij', I.oovv, T1old, U11old, U21old, optimize = True)
    U22 += 2.0 * einsum('klcd,dk,al,bcji->abij', I.oovv, T1old, U11old, U21old, optimize = True)
    U22 += 2.0 * einsum('klcd,bdji,ak,cl->abij', I.oovv, T2old, U11old, U11old, optimize = True)
    U22 += -2.0 * einsum('klcd,adji,bk,cl->abij', I.oovv, T2old, U11old, U11old, optimize = True)
    U22 += -1.0 * einsum('klcd,dcji,bk,al->abij', I.oovv, T2old, U11old, U11old, optimize = True)
    U22 += 2.0 * einsum('klcd,baki,dj,cl->abij', I.oovv, T2old, U11old, U11old, optimize = True)
    U22 += 2.0 * einsum('klcd,bdki,cj,al->abij', I.oovv, T2old, U11old, U11old, optimize = True)
    U22 += -2.0 * einsum('klcd,adki,cj,bl->abij', I.oovv, T2old, U11old, U11old, optimize = True)
    U22 += -2.0 * einsum('klcd,bakj,di,cl->abij', I.oovv, T2old, U11old, U11old, optimize = True)
    U22 += -2.0 * einsum('klcd,bdkj,ci,al->abij', I.oovv, T2old, U11old, U11old, optimize = True)
    U22 += 2.0 * einsum('klcd,adkj,ci,bl->abij', I.oovv, T2old, U11old, U11old, optimize = True)
    U22 += -1.0 * einsum('klcd,balk,di,cj->abij', I.oovv, T2old, U11old, U11old, optimize = True)
    U22 += 2.0 * einsum('kc,bi,cj,ak->abij', g.ov[0], U11old, U11old, U11old, optimize = True)
    U22 += -2.0 * einsum('kc,ci,bj,ak->abij', g.ov[0], U11old, U11old, U11old, optimize = True)
    U22 += -2.0 * einsum('kc,ai,cj,bk->abij', g.ov[0], U11old, U11old, U11old, optimize = True)
    U22 += 2.0 * einsum('kc,ci,aj,bk->abij', g.ov[0], U11old, U11old, U11old, optimize = True)


    U2n[1][0,0] = U22

    return U2n


def single_qedccsd_U1n(F, I, w, g, h, G, H, nfock1, nfock2, amps):

    T1old, T2old, Snold, U1nold, U2nold = amps

    nvir = T2old.shape[0]
    nocc = T2old.shape[2]
    nm = w.shape[0]

    U1n = [None]*nfock2
    if nfock2 < 1: return U1n

    # works for single mode at this moment
    U11old = U1nold[0][0]
    if nfock2 > 1:
        U12old = U1nold[1][0,0]

    if U2nold[0] is not None:
        U21old = U2nold[0][0]
        if nfock2> 1:
            U22old = U2nold[1][0,0]

    if nfock1 > 0:
        S1old = Snold[0][0]
    if nfock1 > 1:
        S2old = Snold[1][0,0]

    U11 = numpy.zeros((nvir, nocc))

    #U11 += 1.0 * G * einsum('ai->ai', U12old, optimize = True)
    U11 += 1.0 * einsum('ai->ai', g.vo[0], optimize = True)
    U11 += -1.0 * einsum('ji,aj->ai', g.oo[0], T1old, optimize = True)
    U11 += 1.0 * einsum('ab,bi->ai', g.vv[0], T1old, optimize = True)
    U11 += -1.0 * einsum('jb,abji->ai', g.ov[0], T2old, optimize = True)
    U11 += -1.0 * einsum('ji,aj->ai', F.oo, U11old, optimize = True)
    U11 += 1.0 * einsum('ab,bi->ai', F.vv, U11old, optimize = True)
    U11 += 1.0 * w * einsum('ai->ai', U11old, optimize = True)
    U11 += -1.0 * einsum('jaib,bj->ai', I.ovov, U11old, optimize = True)
    if nfock1 > 1:
        U11 += 1.0 * S2old * einsum('ai->ai', g.vo[0], optimize = True)
        U11 += -1.0 * S2old * einsum('ji,aj->ai', g.oo[0], T1old, optimize = True)
        U11 += 1.0 * S2old * einsum('ab,bi->ai', g.vv[0], T1old, optimize = True)
        U11 += -1.0 * S2old * einsum('jb,abji->ai', g.ov[0], T2old, optimize = True)
        U11 += -1.0 * S2old * einsum('jb,bi,aj->ai', g.ov[0], T1old, T1old, optimize = True)
    U11 += -1.0 * einsum('jb,bi,aj->ai', g.ov[0], T1old, T1old, optimize = True)
    U11 += -1.0 * einsum('jb,bi,aj->ai', F.ov, T1old, U11old, optimize = True)
    U11 += -1.0 * einsum('jb,aj,bi->ai', F.ov, T1old, U11old, optimize = True)
    U11 += -1.0 * S1old * einsum('ji,aj->ai', g.oo[0], U11old, optimize = True)
    U11 += 1.0 * S1old * einsum('ab,bi->ai', g.vv[0], U11old, optimize = True)
    U11 += -1.0 * einsum('jkib,aj,bk->ai', I.ooov, T1old, U11old, optimize = True)
    U11 += 1.0 * einsum('jkib,bj,ak->ai', I.ooov, T1old, U11old, optimize = True)
    U11 += 1.0 * einsum('jabc,ci,bj->ai', I.ovvv, T1old, U11old, optimize = True)
    U11 += -1.0 * einsum('jabc,cj,bi->ai', I.ovvv, T1old, U11old, optimize = True)
    if nfock2 > 1:
        U11 += -1.0 * einsum('ji,aj->ai', g.oo[0], U12old, optimize = True)
        U11 += 1.0 * einsum('ab,bi->ai', g.vv[0], U12old, optimize = True)
        U11 += -1.0 * einsum('jb,bi,aj->ai', g.ov[0], T1old, U12old, optimize = True)
        U11 += -1.0 * einsum('jb,aj,bi->ai', g.ov[0], T1old, U12old, optimize = True)
        U11 += 1.0 * einsum('jb,bj,ai->ai', g.ov[0], T1old, U12old, optimize = True)
    if U2nold[0] is not None:
        U11 += -1.0 * einsum('jb,abji->ai', F.ov, U21old, optimize = True)
        U11 += 0.5 * einsum('jkib,abkj->ai', I.ooov, U21old, optimize = True)
        U11 += -0.5 * einsum('jabc,cbji->ai', I.ovvv, U21old, optimize = True)
        U11 += -1.0 * S1old * einsum('jb,abji->ai', g.ov[0], U21old, optimize = True)
        U11 += -0.5 * einsum('jkbc,ci,abkj->ai', I.oovv, T1old, U21old, optimize = True)
        U11 += -0.5 * einsum('jkbc,aj,cbki->ai', I.oovv, T1old, U21old, optimize = True)
        U11 += 1.0 * einsum('jkbc,cj,abki->ai', I.oovv, T1old, U21old, optimize = True)
        if nfock2 > 1:
            U11 += -1.0 * einsum('jb,abji->ai', g.ov[0], U22old, optimize = True)
    U11 += 1.0 * einsum('jkbc,acji,bk->ai', I.oovv, T2old, U11old, optimize = True)
    U11 += 0.5 * einsum('jkbc,cbji,ak->ai', I.oovv, T2old, U11old, optimize = True)
    U11 += 0.5 * einsum('jkbc,ackj,bi->ai', I.oovv, T2old, U11old, optimize = True)
    U11 += 1.0 * einsum('jb,ai,bj->ai', g.ov[0], U11old, U11old, optimize = True)
    U11 += -2.0 * einsum('jb,bi,aj->ai', g.ov[0], U11old, U11old, optimize = True)
    U11 += -1.0 * S1old * einsum('jb,bi,aj->ai', g.ov[0], T1old, U11old, optimize = True)
    U11 += -1.0 * S1old * einsum('jb,aj,bi->ai', g.ov[0], T1old, U11old, optimize = True)
    U11 += 1.0 * einsum('jkbc,ci,aj,bk->ai', I.oovv, T1old, T1old, U11old, optimize = True)
    U11 += -1.0 * einsum('jkbc,ci,bj,ak->ai', I.oovv, T1old, T1old, U11old, optimize = True)
    U11 += -1.0 * einsum('jkbc,aj,ck,bi->ai', I.oovv, T1old, T1old, U11old, optimize = True)

    U1n[0] = numpy.zeros((nm, nvir, nocc))
    U1n[0][0] = U11

    if nfock2 < 2: return U1n

    U12 = numpy.zeros((nvir, nocc))

    U12 += -1.0 * einsum('ji,aj->ai', F.oo, U12old, optimize = True)
    U12 += 1.0 * einsum('ab,bi->ai', F.vv, U12old, optimize = True)
    U12 += 1.0 * w * einsum('ai->ai', U12old, optimize = True)
    U12 += 1.0 * w * einsum('ai->ai', U12old, optimize = True)
    U12 += -1.0 * einsum('ji,aj->ai', g.oo[0], U11old, optimize = True)
    U12 += -1.0 * einsum('ji,aj->ai', g.oo[0], U11old, optimize = True)
    U12 += 1.0 * einsum('ab,bi->ai', g.vv[0], U11old, optimize = True)
    U12 += 1.0 * einsum('ab,bi->ai', g.vv[0], U11old, optimize = True)
    U12 += -1.0 * einsum('jaib,bj->ai', I.ovov, U12old, optimize = True)
    U12 += -1.0 * einsum('jb,bi,aj->ai', F.ov, T1old, U12old, optimize = True)
    U12 += -1.0 * einsum('jb,aj,bi->ai', F.ov, T1old, U12old, optimize = True)
    U12 += -1.0 * S1old * einsum('ji,aj->ai', g.oo[0], U12old, optimize = True)
    U12 += -1.0 * einsum('jb,bi,aj->ai', g.ov[0], T1old, U11old, optimize = True)
    U12 += -1.0 * einsum('jb,aj,bi->ai', g.ov[0], T1old, U11old, optimize = True)
    U12 += -1.0 * einsum('jb,bi,aj->ai', g.ov[0], T1old, U11old, optimize = True)
    U12 += -1.0 * einsum('jb,aj,bi->ai', g.ov[0], T1old, U11old, optimize = True)
    U12 += 1.0 * S1old * einsum('ab,bi->ai', g.vv[0], U12old, optimize = True)
    U12 += -1.0 * einsum('jkib,aj,bk->ai', I.ooov, T1old, U12old, optimize = True)
    U12 += 1.0 * einsum('jkib,bj,ak->ai', I.ooov, T1old, U12old, optimize = True)
    U12 += 1.0 * einsum('jabc,ci,bj->ai', I.ovvv, T1old, U12old, optimize = True)
    U12 += -1.0 * einsum('jabc,cj,bi->ai', I.ovvv, T1old, U12old, optimize = True)
    U12 += 0.5 * einsum('jkbc,cbji,ak->ai', I.oovv, T2old, U12old, optimize = True)
    U12 += 0.5 * einsum('jkbc,ackj,bi->ai', I.oovv, T2old, U12old, optimize = True)
    U12 += -2.0 * einsum('jb,bi,aj->ai', F.ov, U11old, U11old, optimize = True)
    U12 += -2.0 * einsum('jkib,aj,bk->ai', I.ooov, U11old, U11old, optimize = True)
    U12 += 2.0 * einsum('jabc,ci,bj->ai', I.ovvv, U11old, U11old, optimize = True)
    U12 += -1.0 * einsum('jb,bi,aj->ai', g.ov[0], U11old, U12old, optimize = True)
    U12 += -1.0 * einsum('jb,aj,bi->ai', g.ov[0], U11old, U12old, optimize = True)
    U12 += 1.0 * einsum('jb,bj,ai->ai', g.ov[0], U11old, U12old, optimize = True)
    U12 += -1.0 * einsum('jb,bi,aj->ai', g.ov[0], U11old, U12old, optimize = True)
    U12 += -1.0 * einsum('jb,aj,bi->ai', g.ov[0], U11old, U12old, optimize = True)
    U12 += 1.0 * einsum('jb,bj,ai->ai', g.ov[0], U11old, U12old, optimize = True)
    U12 += 1.0 * einsum('jb,ai,bj->ai', g.ov[0], U11old, U12old, optimize = True)
    U12 += -1.0 * einsum('jb,bi,aj->ai', g.ov[0], U11old, U12old, optimize = True)
    U12 += -1.0 * einsum('jb,aj,bi->ai', g.ov[0], U11old, U12old, optimize = True)
    U12 += -1.0 * S1old * einsum('jb,bi,aj->ai', g.ov[0], T1old, U12old, optimize = True)
    U12 += -1.0 * S1old * einsum('jb,aj,bi->ai', g.ov[0], T1old, U12old, optimize = True)
    if nfock1 > 1:
        U12 += -1.0 * S2old * einsum('ji,aj->ai', g.oo[0], U11old, optimize = True)
        U12 += -1.0 * S2old * einsum('ji,aj->ai', g.oo[0], U11old, optimize = True)
        U12 += 1.0 * S2old * einsum('ab,bi->ai', g.vv[0], U11old, optimize = True)
        U12 += 1.0 * S2old * einsum('ab,bi->ai', g.vv[0], U11old, optimize = True)
        U12 += -1.0 * S2old * einsum('jb,bi,aj->ai', g.ov[0], T1old, U11old, optimize = True)
        U12 += -1.0 * S2old * einsum('jb,aj,bi->ai', g.ov[0], T1old, U11old, optimize = True)
        U12 += -1.0 * S2old * einsum('jb,bi,aj->ai', g.ov[0], T1old, U11old, optimize = True)
        U12 += -1.0 * S2old * einsum('jb,aj,bi->ai', g.ov[0], T1old, U11old, optimize = True)
    if U2nold[0] is not None: # when reach here, nfock is at least 2
        U12 += -1.0 * einsum('jb,abji->ai', g.ov[0], U21old, optimize = True)
        U12 += -1.0 * einsum('jb,abji->ai', g.ov[0], U21old, optimize = True)
        U12 += -1.0 * einsum('jkbc,ci,abkj->ai', I.oovv, U11old, U21old, optimize = True)
        U12 += -1.0 * einsum('jkbc,aj,cbki->ai', I.oovv, U11old, U21old, optimize = True)
        U12 += 2.0 * einsum('jkbc,cj,abki->ai', I.oovv, U11old, U21old, optimize = True)
        U12 += -1.0 * einsum('jb,abji->ai', F.ov, U22old, optimize = True)
        U12 += 0.5 * einsum('jkib,abkj->ai', I.ooov, U22old, optimize = True)
        U12 += -0.5 * einsum('jabc,cbji->ai', I.ovvv, U22old, optimize = True)
        U12 += -1.0 * S1old * einsum('jb,abji->ai', g.ov[0], U22old, optimize = True)
        U12 += -0.5 * einsum('jkbc,ci,abkj->ai', I.oovv, T1old, U22old, optimize = True)
        U12 += -0.5 * einsum('jkbc,aj,cbki->ai', I.oovv, T1old, U22old, optimize = True)
        U12 += 1.0 * einsum('jkbc,cj,abki->ai', I.oovv, T1old, U22old, optimize = True)
        if nfock1 > 1:
            U12 += -1.0 * S2old * einsum('jb,abji->ai', g.ov[0], U21old, optimize = True)
            U12 += -1.0 * S2old * einsum('jb,abji->ai', g.ov[0], U21old, optimize = True)
    U12 += 1.0 * einsum('jkbc,acji,bk->ai', I.oovv, T2old, U12old, optimize = True)
    U12 += 1.0 * einsum('jkbc,ci,aj,bk->ai', I.oovv, T1old, T1old, U12old, optimize = True)
    U12 += -1.0 * einsum('jkbc,ci,bj,ak->ai', I.oovv, T1old, T1old, U12old, optimize = True)
    U12 += -1.0 * einsum('jkbc,aj,ck,bi->ai', I.oovv, T1old, T1old, U12old, optimize = True)
    U12 += -2.0 * S1old * einsum('jb,bi,aj->ai', g.ov[0], U11old, U11old, optimize = True)
    U12 += 2.0 * einsum('jkbc,ci,aj,bk->ai', I.oovv, T1old, U11old, U11old, optimize = True)
    U12 += 2.0 * einsum('jkbc,aj,ci,bk->ai', I.oovv, T1old, U11old, U11old, optimize = True)
    U12 += 2.0 * einsum('jkbc,cj,bi,ak->ai', I.oovv, T1old, U11old, U11old, optimize = True)

    U1n[1] = numpy.zeros((nm, nm, nvir, nocc))
    U1n[1][0][0] = U12
    return U1n


def single_qedccsd_U2n(F, I, w, g, h, G, H, nfock1, nfock2, amps):

    T1old, T2old, Snold, U1nold, U2nold = amps

    nvir = T2old.shape[0]
    nocc = T2old.shape[2]
    nm = w.shape[0]

    eps_occ = F.oo.diagonal()
    eps_vir = F.vv.diagonal()
    eps_vir_p_w = eps_vir + w

    U2n = [None]*nfock2
    if U2nold[0] is None: return U2n
    if nfock2 < 1: return U2n

    U2n[0] = numpy.zeros((nm, nvir, nvir, nocc, nocc))

    # works for single mode at this moment
    U11old = U1nold[0][0]   # single mode
    U21old = U2nold[0][0]   # single mode
    if nfock2 > 1:
        U12old = U1nold[1][0,0] # single mode
        U22old = U2nold[1][0,0] # single mode

    if nfock1 > 0:
        S1old = Snold[0][0]     # single mode
    if nfock1 > 1:
        S2old = Snold[1][0,0]   # single mode

    U21 = numpy.zeros((nvir, nvir, nocc, nocc))
    U22 = numpy.zeros((nvir, nvir, nocc, nocc))

    #res_U2d += 1.0 * G * einsum('J,baji->abij', U22old, optimize = True)
    U21 += 1.0 * einsum('ki,bakj->abij', g.oo[0], T2old, optimize = True)
    U21 += -1.0 * einsum('kj,baki->abij', g.oo[0], T2old, optimize = True)
    U21 += -1.0 * einsum('bc,acji->abij', g.vv[0], T2old, optimize = True)
    U21 += 1.0 * einsum('ac,bcji->abij', g.vv[0], T2old, optimize = True)

    U21 += 1.0 * einsum('kbji,ak->abij', I.ovoo, U11old, optimize = True)
    U21 += -1.0 * einsum('kaji,bk->abij', I.ovoo, U11old, optimize = True)
    U21 += -1.0 * einsum('baic,cj->abij', I.vvov, U11old, optimize = True)
    U21 += 1.0 * einsum('bajc,ci->abij', I.vvov, U11old, optimize = True)
    U21 += -1.0 * einsum('kc,bcji,ak->abij', F.ov, T2old, U11old, optimize = True)
    U21 += 1.0 * einsum('kc,acji,bk->abij', F.ov, T2old, U11old, optimize = True)
    U21 += -1.0 * einsum('kc,baki,cj->abij', F.ov, T2old, U11old, optimize = True)
    U21 += 1.0 * einsum('kc,bakj,ci->abij', F.ov, T2old, U11old, optimize = True)
    U21 += 1.0 * einsum('klji,bk,al->abij', I.oooo, T1old, U11old, optimize = True)
    U21 += -1.0 * einsum('klji,ak,bl->abij', I.oooo, T1old, U11old, optimize = True)
    U21 += -1.0 * einsum('kbic,cj,ak->abij', I.ovov, T1old, U11old, optimize = True)
    U21 += 1.0 * einsum('kaic,cj,bk->abij', I.ovov, T1old, U11old, optimize = True)
    U21 += -1.0 * einsum('kbic,ak,cj->abij', I.ovov, T1old, U11old, optimize = True)
    U21 += 1.0 * einsum('kaic,bk,cj->abij', I.ovov, T1old, U11old, optimize = True)
    U21 += 1.0 * einsum('kbjc,ci,ak->abij', I.ovov, T1old, U11old, optimize = True)
    U21 += -1.0 * einsum('kajc,ci,bk->abij', I.ovov, T1old, U11old, optimize = True)
    U21 += 1.0 * einsum('kbjc,ak,ci->abij', I.ovov, T1old, U11old, optimize = True)
    U21 += -1.0 * einsum('kajc,bk,ci->abij', I.ovov, T1old, U11old, optimize = True)
    U21 += 1.0 * einsum('bacd,di,cj->abij', I.vvvv, T1old, U11old, optimize = True)
    U21 += -1.0 * einsum('bacd,dj,ci->abij', I.vvvv, T1old, U11old, optimize = True)

    U21 += 1.0 * einsum('ki,bakj->abij', F.oo, U21old, optimize = True)
    U21 += -1.0 * einsum('kj,baki->abij', F.oo, U21old, optimize = True)
    U21 += -1.0 * einsum('bc,acji->abij', F.vv, U21old, optimize = True)
    U21 += 1.0 * einsum('ac,bcji->abij', F.vv, U21old, optimize = True)
    U21 += 1.0 * w * einsum('baji->abij', U21old, optimize = True)
    U21 += -0.5 * einsum('klji,balk->abij', I.oooo, U21old, optimize = True)
    U21 += -1.0 * einsum('kbic,ackj->abij', I.ovov, U21old, optimize = True)
    U21 += 1.0 * einsum('kaic,bckj->abij', I.ovov, U21old, optimize = True)
    U21 += 1.0 * einsum('kbjc,acki->abij', I.ovov, U21old, optimize = True)
    U21 += -1.0 * einsum('kajc,bcki->abij', I.ovov, U21old, optimize = True)
    U21 += -0.5 * einsum('bacd,dcji->abij', I.vvvv, U21old, optimize = True)
    if nfock2 > 1:
        U21 += 1.0 * einsum('ki,bakj->abij', g.oo[0], U22old, optimize = True)
        U21 += -1.0 * einsum('kj,baki->abij', g.oo[0], U22old, optimize = True)
        U21 += -1.0 * einsum('bc,acji->abij', g.vv[0], U22old, optimize = True)
        U21 += 1.0 * einsum('ac,bcji->abij', g.vv[0], U22old, optimize = True)
        U21 += 1.0 * einsum('kc,ci,bakj->abij', g.ov[0], T1old, U22old, optimize = True)
        U21 += -1.0 * einsum('kc,cj,baki->abij', g.ov[0], T1old, U22old, optimize = True)
        U21 += 1.0 * einsum('kc,bk,acji->abij', g.ov[0], T1old, U22old, optimize = True)
        U21 += -1.0 * einsum('kc,ak,bcji->abij', g.ov[0], T1old, U22old, optimize = True)
        U21 += 1.0 * einsum('kc,ck,baji->abij', g.ov[0], T1old, U22old, optimize = True)
    if nfock1 > 1:
        U21 += 1.0 * S2old * einsum('ki,bakj->abij', g.oo[0], T2old, optimize = True)
        U21 += -1.0 * S2old * einsum('kj,baki->abij', g.oo[0], T2old, optimize = True)
        U21 += -1.0 * S2old * einsum('bc,acji->abij', g.vv[0], T2old, optimize = True)
        U21 += 1.0 * S2old * einsum('ac,bcji->abij', g.vv[0], T2old, optimize = True)
        U21 += 1.0 * S2old * einsum('kc,ci,bakj->abij', g.ov[0], T1old, T2old, optimize = True)
        U21 += -1.0 * S2old * einsum('kc,cj,baki->abij', g.ov[0], T1old, T2old, optimize = True)
        U21 += 1.0 * S2old * einsum('kc,bk,acji->abij', g.ov[0], T1old, T2old, optimize = True)
        U21 += -1.0 * S2old * einsum('kc,ak,bcji->abij', g.ov[0], T1old, T2old, optimize = True)
    U21 += -0.5 * einsum('klcd,bdji,aclk->abij', I.oovv, T2old, U21old, optimize = True)
    U21 += 1.0 * einsum('kc,ci,bakj->abij', g.ov[0], T1old, T2old, optimize = True)
    U21 += -1.0 * einsum('kc,cj,baki->abij', g.ov[0], T1old, T2old, optimize = True)
    U21 += 1.0 * einsum('kc,bk,acji->abij', g.ov[0], T1old, T2old, optimize = True)
    U21 += -1.0 * einsum('kc,ak,bcji->abij', g.ov[0], T1old, T2old, optimize = True)
    U21 += 1.0 * einsum('kc,ci,bakj->abij', F.ov, T1old, U21old, optimize = True)
    U21 += -1.0 * einsum('kc,cj,baki->abij', F.ov, T1old, U21old, optimize = True)
    U21 += 1.0 * einsum('kc,bk,acji->abij', F.ov, T1old, U21old, optimize = True)
    U21 += -1.0 * einsum('kc,ak,bcji->abij', F.ov, T1old, U21old, optimize = True)

    if nfock2 > 1:
        U21 += -1.0 * einsum('bi,aj->abij', g.vo[0], U12old, optimize = True)
        U21 += 1.0 * einsum('ai,bj->abij', g.vo[0], U12old, optimize = True)
        U21 += 1.0 * einsum('bj,ai->abij', g.vo[0], U12old, optimize = True)
        U21 += -1.0 * einsum('aj,bi->abij', g.vo[0], U12old, optimize = True)
        U21 += 1.0 * einsum('ki,bk,aj->abij', g.oo[0], T1old, U12old, optimize = True)
        U21 += -1.0 * einsum('ki,ak,bj->abij', g.oo[0], T1old, U12old, optimize = True)
        U21 += -1.0 * einsum('kj,bk,ai->abij', g.oo[0], T1old, U12old, optimize = True)
        U21 += 1.0 * einsum('kj,ak,bi->abij', g.oo[0], T1old, U12old, optimize = True)
        U21 += -1.0 * einsum('bc,ci,aj->abij', g.vv[0], T1old, U12old, optimize = True)
        U21 += 1.0 * einsum('ac,ci,bj->abij', g.vv[0], T1old, U12old, optimize = True)
        U21 += 1.0 * einsum('bc,cj,ai->abij', g.vv[0], T1old, U12old, optimize = True)
        U21 += -1.0 * einsum('ac,cj,bi->abij', g.vv[0], T1old, U12old, optimize = True)
        U21 += -1.0 * einsum('kc,bcji,ak->abij', g.ov[0], T2old, U12old, optimize = True)
        U21 += 1.0 * einsum('kc,acji,bk->abij', g.ov[0], T2old, U12old, optimize = True)
        U21 += -1.0 * einsum('kc,baki,cj->abij', g.ov[0], T2old, U12old, optimize = True)
        U21 += 1.0 * einsum('kc,bcki,aj->abij', g.ov[0], T2old, U12old, optimize = True)
        U21 += -1.0 * einsum('kc,acki,bj->abij', g.ov[0], T2old, U12old, optimize = True)
        U21 += 1.0 * einsum('kc,bakj,ci->abij', g.ov[0], T2old, U12old, optimize = True)
        U21 += -1.0 * einsum('kc,bckj,ai->abij', g.ov[0], T2old, U12old, optimize = True)
        U21 += 1.0 * einsum('kc,ackj,bi->abij', g.ov[0], T2old, U12old, optimize = True)
        U21 += 1.0 * einsum('kc,ci,bk,aj->abij', g.ov[0], T1old, T1old, U12old, optimize = True)
        U21 += -1.0 * einsum('kc,ci,ak,bj->abij', g.ov[0], T1old, T1old, U12old, optimize = True)
        U21 += -1.0 * einsum('kc,cj,bk,ai->abij', g.ov[0], T1old, T1old, U12old, optimize = True)
        U21 += 1.0 * einsum('kc,cj,ak,bi->abij', g.ov[0], T1old, T1old, U12old, optimize = True)
    U21 += 1.0 * S1old * einsum('ki,bakj->abij', g.oo[0], U21old, optimize = True)
    U21 += -1.0 * S1old * einsum('kj,baki->abij', g.oo[0], U21old, optimize = True)
    U21 += -1.0 * S1old * einsum('bc,acji->abij', g.vv[0], U21old, optimize = True)
    U21 += 1.0 * S1old * einsum('ac,bcji->abij', g.vv[0], U21old, optimize = True)
    U21 += 0.5 * einsum('klic,cj,balk->abij', I.ooov, T1old, U21old, optimize = True)
    U21 += -1.0 * einsum('klic,bk,aclj->abij', I.ooov, T1old, U21old, optimize = True)
    U21 += 1.0 * einsum('klic,ak,bclj->abij', I.ooov, T1old, U21old, optimize = True)
    U21 += -1.0 * einsum('klic,ck,balj->abij', I.ooov, T1old, U21old, optimize = True)
    U21 += -0.5 * einsum('kljc,ci,balk->abij', I.ooov, T1old, U21old, optimize = True)
    U21 += 1.0 * einsum('kljc,bk,acli->abij', I.ooov, T1old, U21old, optimize = True)
    U21 += -1.0 * einsum('kljc,ak,bcli->abij', I.ooov, T1old, U21old, optimize = True)
    U21 += 1.0 * einsum('kljc,ck,bali->abij', I.ooov, T1old, U21old, optimize = True)

    U21 += 1.0 * einsum('klic,bakj,cl->abij', I.ooov, T2old, U11old, optimize = True)
    U21 += -1.0 * einsum('klic,bckj,al->abij', I.ooov, T2old, U11old, optimize = True)
    U21 += 1.0 * einsum('klic,ackj,bl->abij', I.ooov, T2old, U11old, optimize = True)
    U21 += 0.5 * einsum('klic,balk,cj->abij', I.ooov, T2old, U11old, optimize = True)
    U21 += -1.0 * einsum('kljc,baki,cl->abij', I.ooov, T2old, U11old, optimize = True)
    U21 += 1.0 * einsum('kljc,bcki,al->abij', I.ooov, T2old, U11old, optimize = True)
    U21 += -1.0 * einsum('kljc,acki,bl->abij', I.ooov, T2old, U11old, optimize = True)
    U21 += -0.5 * einsum('kljc,balk,ci->abij', I.ooov, T2old, U11old, optimize = True)

    U21 += 1.0 * einsum('kbcd,di,ackj->abij', I.ovvv, T1old, U21old, optimize = True)
    U21 += -1.0 * einsum('kacd,di,bckj->abij', I.ovvv, T1old, U21old, optimize = True)
    U21 += -1.0 * einsum('kbcd,dj,acki->abij', I.ovvv, T1old, U21old, optimize = True)
    U21 += 1.0 * einsum('kacd,dj,bcki->abij', I.ovvv, T1old, U21old, optimize = True)
    U21 += -0.5 * einsum('kbcd,ak,dcji->abij', I.ovvv, T1old, U21old, optimize = True)
    U21 += 1.0 * einsum('kbcd,dk,acji->abij', I.ovvv, T1old, U21old, optimize = True)
    U21 += 0.5 * einsum('kacd,bk,dcji->abij', I.ovvv, T1old, U21old, optimize = True)
    U21 += -1.0 * einsum('kacd,dk,bcji->abij', I.ovvv, T1old, U21old, optimize = True)

    U21 += -1.0 * einsum('kbcd,adji,ck->abij', I.ovvv, T2old, U11old, optimize = True)
    U21 += -0.5 * einsum('kbcd,dcji,ak->abij', I.ovvv, T2old, U11old, optimize = True)
    U21 += 1.0 * einsum('kacd,bdji,ck->abij', I.ovvv, T2old, U11old, optimize = True)
    U21 += 0.5 * einsum('kacd,dcji,bk->abij', I.ovvv, T2old, U11old, optimize = True)
    U21 += 1.0 * einsum('kbcd,adki,cj->abij', I.ovvv, T2old, U11old, optimize = True)
    U21 += -1.0 * einsum('kacd,bdki,cj->abij', I.ovvv, T2old, U11old, optimize = True)
    U21 += -1.0 * einsum('kbcd,adkj,ci->abij', I.ovvv, T2old, U11old, optimize = True)
    U21 += 1.0 * einsum('kacd,bdkj,ci->abij', I.ovvv, T2old, U11old, optimize = True)

    U21 += 0.5 * einsum('klcd,adji,bclk->abij', I.oovv, T2old, U21old, optimize = True)
    U21 += 0.25 * einsum('klcd,dcji,balk->abij', I.oovv, T2old, U21old, optimize = True)
    U21 += -0.5 * einsum('klcd,baki,dclj->abij', I.oovv, T2old, U21old, optimize = True)
    U21 += 1.0 * einsum('klcd,bdki,aclj->abij', I.oovv, T2old, U21old, optimize = True)
    U21 += -1.0 * einsum('klcd,adki,bclj->abij', I.oovv, T2old, U21old, optimize = True)
    U21 += -0.5 * einsum('klcd,dcki,balj->abij', I.oovv, T2old, U21old, optimize = True)
    U21 += 0.5 * einsum('klcd,bakj,dcli->abij', I.oovv, T2old, U21old, optimize = True)
    U21 += -1.0 * einsum('klcd,bdkj,acli->abij', I.oovv, T2old, U21old, optimize = True)
    U21 += 1.0 * einsum('klcd,adkj,bcli->abij', I.oovv, T2old, U21old, optimize = True)
    U21 += 0.5 * einsum('klcd,dckj,bali->abij', I.oovv, T2old, U21old, optimize = True)
    U21 += 0.25 * einsum('klcd,balk,dcji->abij', I.oovv, T2old, U21old, optimize = True)
    U21 += -0.5 * einsum('klcd,bdlk,acji->abij', I.oovv, T2old, U21old, optimize = True)
    U21 += 0.5 * einsum('klcd,adlk,bcji->abij', I.oovv, T2old, U21old, optimize = True)

    U21 += -1.0 * einsum('ki,bj,ak->abij', g.oo[0], U11old, U11old, optimize = True)
    U21 += 1.0 * einsum('ki,aj,bk->abij', g.oo[0], U11old, U11old, optimize = True)
    U21 += 1.0 * einsum('kj,bi,ak->abij', g.oo[0], U11old, U11old, optimize = True)
    U21 += -1.0 * einsum('kj,ai,bk->abij', g.oo[0], U11old, U11old, optimize = True)
    U21 += 1.0 * einsum('bc,ai,cj->abij', g.vv[0], U11old, U11old, optimize = True)
    U21 += -1.0 * einsum('bc,ci,aj->abij', g.vv[0], U11old, U11old, optimize = True)
    U21 += -1.0 * einsum('ac,bi,cj->abij', g.vv[0], U11old, U11old, optimize = True)
    U21 += 1.0 * einsum('ac,ci,bj->abij', g.vv[0], U11old, U11old, optimize = True)

    U21 += 1.0 * einsum('kc,bi,ackj->abij', g.ov[0], U11old, U21old, optimize = True)
    U21 += -1.0 * einsum('kc,ai,bckj->abij', g.ov[0], U11old, U21old, optimize = True)
    U21 += 2.0 * einsum('kc,ci,bakj->abij', g.ov[0], U11old, U21old, optimize = True)
    U21 += -1.0 * einsum('kc,bj,acki->abij', g.ov[0], U11old, U21old, optimize = True)
    U21 += 1.0 * einsum('kc,aj,bcki->abij', g.ov[0], U11old, U21old, optimize = True)
    U21 += -2.0 * einsum('kc,cj,baki->abij', g.ov[0], U11old, U21old, optimize = True)
    U21 += 2.0 * einsum('kc,bk,acji->abij', g.ov[0], U11old, U21old, optimize = True)
    U21 += -2.0 * einsum('kc,ak,bcji->abij', g.ov[0], U11old, U21old, optimize = True)
    U21 += 1.0 * einsum('kc,ck,baji->abij', g.ov[0], U11old, U21old, optimize = True)

    U21 += -1.0 * einsum('klic,cj,bk,al->abij', I.ooov, T1old, T1old, U11old, optimize = True)
    U21 += 1.0 * einsum('klic,cj,ak,bl->abij', I.ooov, T1old, T1old, U11old, optimize = True)
    U21 += -1.0 * einsum('klic,bk,al,cj->abij', I.ooov, T1old, T1old, U11old, optimize = True)
    U21 += 1.0 * einsum('kljc,ci,bk,al->abij', I.ooov, T1old, T1old, U11old, optimize = True)
    U21 += -1.0 * einsum('kljc,ci,ak,bl->abij', I.ooov, T1old, T1old, U11old, optimize = True)
    U21 += 1.0 * einsum('kljc,bk,al,ci->abij', I.ooov, T1old, T1old, U11old, optimize = True)
    U21 += 1.0 * einsum('kbcd,di,cj,ak->abij', I.ovvv, T1old, T1old, U11old, optimize = True)
    U21 += -1.0 * einsum('kacd,di,cj,bk->abij', I.ovvv, T1old, T1old, U11old, optimize = True)
    U21 += 1.0 * einsum('kbcd,di,ak,cj->abij', I.ovvv, T1old, T1old, U11old, optimize = True)
    U21 += -1.0 * einsum('kacd,di,bk,cj->abij', I.ovvv, T1old, T1old, U11old, optimize = True)
    U21 += -1.0 * einsum('kbcd,dj,ak,ci->abij', I.ovvv, T1old, T1old, U11old, optimize = True)
    U21 += 1.0 * einsum('kacd,dj,bk,ci->abij', I.ovvv, T1old, T1old, U11old, optimize = True)

    U21 += 1.0 * S1old * einsum('kc,ci,bakj->abij', g.ov[0], T1old, U21old, optimize = True)
    U21 += -1.0 * S1old * einsum('kc,cj,baki->abij', g.ov[0], T1old, U21old, optimize = True)
    U21 += 1.0 * S1old * einsum('kc,bk,acji->abij', g.ov[0], T1old, U21old, optimize = True)
    U21 += -1.0 * S1old * einsum('kc,ak,bcji->abij', g.ov[0], T1old, U21old, optimize = True)

    U21 += -1.0 * S1old * einsum('kc,bcji,ak->abij', g.ov[0], T2old, U11old, optimize = True)
    U21 += 1.0 * S1old * einsum('kc,acji,bk->abij', g.ov[0], T2old, U11old, optimize = True)
    U21 += -1.0 * S1old * einsum('kc,baki,cj->abij', g.ov[0], T2old, U11old, optimize = True)
    U21 += 1.0 * S1old * einsum('kc,bakj,ci->abij', g.ov[0], T2old, U11old, optimize = True)

    U21 += -0.5 * einsum('klcd,di,cj,balk->abij', I.oovv, T1old, T1old, U21old, optimize = True)
    U21 += 1.0 * einsum('klcd,di,bk,aclj->abij', I.oovv, T1old, T1old, U21old, optimize = True)
    U21 += -1.0 * einsum('klcd,di,ak,bclj->abij', I.oovv, T1old, T1old, U21old, optimize = True)
    U21 += 1.0 * einsum('klcd,di,ck,balj->abij', I.oovv, T1old, T1old, U21old, optimize = True)
    U21 += -1.0 * einsum('klcd,dj,bk,acli->abij', I.oovv, T1old, T1old, U21old, optimize = True)
    U21 += 1.0 * einsum('klcd,dj,ak,bcli->abij', I.oovv, T1old, T1old, U21old, optimize = True)
    U21 += -1.0 * einsum('klcd,dj,ck,bali->abij', I.oovv, T1old, T1old, U21old, optimize = True)
    U21 += -0.5 * einsum('klcd,bk,al,dcji->abij', I.oovv, T1old, T1old, U21old, optimize = True)
    U21 += 1.0 * einsum('klcd,bk,dl,acji->abij', I.oovv, T1old, T1old, U21old, optimize = True)
    U21 += -1.0 * einsum('klcd,ak,dl,bcji->abij', I.oovv, T1old, T1old, U21old, optimize = True)

    U21 += -1.0 * einsum('klcd,di,bakj,cl->abij', I.oovv, T1old, T2old, U11old, optimize = True)
    U21 += 1.0 * einsum('klcd,di,bckj,al->abij', I.oovv, T1old, T2old, U11old, optimize = True)
    U21 += -1.0 * einsum('klcd,di,ackj,bl->abij', I.oovv, T1old, T2old, U11old, optimize = True)
    U21 += -0.5 * einsum('klcd,di,balk,cj->abij', I.oovv, T1old, T2old, U11old, optimize = True)
    U21 += 1.0 * einsum('klcd,dj,baki,cl->abij', I.oovv, T1old, T2old, U11old, optimize = True)
    U21 += -1.0 * einsum('klcd,dj,bcki,al->abij', I.oovv, T1old, T2old, U11old, optimize = True)
    U21 += 1.0 * einsum('klcd,dj,acki,bl->abij', I.oovv, T1old, T2old, U11old, optimize = True)
    U21 += -1.0 * einsum('klcd,bk,adji,cl->abij', I.oovv, T1old, T2old, U11old, optimize = True)
    U21 += -0.5 * einsum('klcd,bk,dcji,al->abij', I.oovv, T1old, T2old, U11old, optimize = True)
    U21 += 1.0 * einsum('klcd,ak,bdji,cl->abij', I.oovv, T1old, T2old, U11old, optimize = True)
    U21 += 1.0 * einsum('klcd,dk,bcji,al->abij', I.oovv, T1old, T2old, U11old, optimize = True)
    U21 += 0.5 * einsum('klcd,ak,dcji,bl->abij', I.oovv, T1old, T2old, U11old, optimize = True)
    U21 += -1.0 * einsum('klcd,dk,acji,bl->abij', I.oovv, T1old, T2old, U11old, optimize = True)
    U21 += 1.0 * einsum('klcd,bk,adli,cj->abij', I.oovv, T1old, T2old, U11old, optimize = True)
    U21 += -1.0 * einsum('klcd,ak,bdli,cj->abij', I.oovv, T1old, T2old, U11old, optimize = True)
    U21 += 1.0 * einsum('klcd,dk,bali,cj->abij', I.oovv, T1old, T2old, U11old, optimize = True)
    U21 += 0.5 * einsum('klcd,dj,balk,ci->abij', I.oovv, T1old, T2old, U11old, optimize = True)
    U21 += -1.0 * einsum('klcd,bk,adlj,ci->abij', I.oovv, T1old, T2old, U11old, optimize = True)
    U21 += 1.0 * einsum('klcd,ak,bdlj,ci->abij', I.oovv, T1old, T2old, U11old, optimize = True)
    U21 += -1.0 * einsum('klcd,dk,balj,ci->abij', I.oovv, T1old, T2old, U11old, optimize = True)

    U21 += -1.0 * einsum('kc,ci,bj,ak->abij', g.ov[0], T1old, U11old, U11old, optimize = True)
    U21 += 1.0 * einsum('kc,ci,aj,bk->abij', g.ov[0], T1old, U11old, U11old, optimize = True)
    U21 += 1.0 * einsum('kc,cj,bi,ak->abij', g.ov[0], T1old, U11old, U11old, optimize = True)
    U21 += -1.0 * einsum('kc,cj,ai,bk->abij', g.ov[0], T1old, U11old, U11old, optimize = True)
    U21 += -1.0 * einsum('kc,bk,ai,cj->abij', g.ov[0], T1old, U11old, U11old, optimize = True)
    U21 += 1.0 * einsum('kc,bk,ci,aj->abij', g.ov[0], T1old, U11old, U11old, optimize = True)
    U21 += 1.0 * einsum('kc,ak,bi,cj->abij', g.ov[0], T1old, U11old, U11old, optimize = True)
    U21 += -1.0 * einsum('kc,ak,ci,bj->abij', g.ov[0], T1old, U11old, U11old, optimize = True)

    U2n[0][0] = U21

    if nfock2 < 2: return U2n

    U22 += 1.0 * einsum('ki,bakj->abij', F.oo, U22old, optimize = True)
    U22 += -1.0 * einsum('kj,baki->abij', F.oo, U22old, optimize = True)
    U22 += -1.0 * einsum('bc,acji->abij', F.vv, U22old, optimize = True)
    U22 += 1.0 * einsum('ac,bcji->abij', F.vv, U22old, optimize = True)
    U22 += 1.0 * einsum('kbji,ak->abij', I.ovoo, U12old, optimize = True)
    U22 += -1.0 * einsum('kaji,bk->abij', I.ovoo, U12old, optimize = True)
    U22 += -1.0 * einsum('baic,cj->abij', I.vvov, U12old, optimize = True)
    U22 += 1.0 * einsum('bajc,ci->abij', I.vvov, U12old, optimize = True)
    U22 += 1.0 * w * einsum('baji->abij', U22old, optimize = True)
    U22 += 1.0 * w * einsum('baji->abij', U22old, optimize = True)
    U22 += 1.0 * einsum('ki,bakj->abij', g.oo[0], U21old, optimize = True)
    U22 += -1.0 * einsum('kj,baki->abij', g.oo[0], U21old, optimize = True)
    U22 += 1.0 * einsum('ki,bakj->abij', g.oo[0], U21old, optimize = True)
    U22 += -1.0 * einsum('kj,baki->abij', g.oo[0], U21old, optimize = True)
    U22 += -1.0 * einsum('bc,acji->abij', g.vv[0], U21old, optimize = True)
    U22 += 1.0 * einsum('ac,bcji->abij', g.vv[0], U21old, optimize = True)
    U22 += -1.0 * einsum('bc,acji->abij', g.vv[0], U21old, optimize = True)
    U22 += 1.0 * einsum('ac,bcji->abij', g.vv[0], U21old, optimize = True)
    U22 += -0.5 * einsum('klji,balk->abij', I.oooo, U22old, optimize = True)
    U22 += -1.0 * einsum('kbic,ackj->abij', I.ovov, U22old, optimize = True)
    U22 += 1.0 * einsum('kaic,bckj->abij', I.ovov, U22old, optimize = True)
    U22 += 1.0 * einsum('kbjc,acki->abij', I.ovov, U22old, optimize = True)
    U22 += -1.0 * einsum('kajc,bcki->abij', I.ovov, U22old, optimize = True)
    U22 += -0.5 * einsum('bacd,dcji->abij', I.vvvv, U22old, optimize = True)
    U22 += 1.0 * einsum('kc,ci,bakj->abij', F.ov, T1old, U22old, optimize = True)
    U22 += -1.0 * einsum('kc,cj,baki->abij', F.ov, T1old, U22old, optimize = True)
    U22 += 1.0 * einsum('kc,bk,acji->abij', F.ov, T1old, U22old, optimize = True)
    U22 += -1.0 * einsum('kc,ak,bcji->abij', F.ov, T1old, U22old, optimize = True)
    U22 += -1.0 * einsum('kc,bcji,ak->abij', F.ov, T2old, U12old, optimize = True)
    U22 += 1.0 * einsum('kc,acji,bk->abij', F.ov, T2old, U12old, optimize = True)
    U22 += -1.0 * einsum('kc,baki,cj->abij', F.ov, T2old, U12old, optimize = True)
    U22 += 1.0 * einsum('kc,bakj,ci->abij', F.ov, T2old, U12old, optimize = True)
    U22 += 1.0 * einsum('klji,bk,al->abij', I.oooo, T1old, U12old, optimize = True)
    U22 += -1.0 * einsum('klji,ak,bl->abij', I.oooo, T1old, U12old, optimize = True)
    U22 += -1.0 * einsum('kbic,cj,ak->abij', I.ovov, T1old, U12old, optimize = True)
    U22 += 1.0 * einsum('kaic,cj,bk->abij', I.ovov, T1old, U12old, optimize = True)
    U22 += -1.0 * einsum('kbic,ak,cj->abij', I.ovov, T1old, U12old, optimize = True)
    U22 += 1.0 * einsum('kaic,bk,cj->abij', I.ovov, T1old, U12old, optimize = True)
    U22 += 1.0 * einsum('kbjc,ci,ak->abij', I.ovov, T1old, U12old, optimize = True)
    U22 += -1.0 * einsum('kajc,ci,bk->abij', I.ovov, T1old, U12old, optimize = True)
    U22 += 1.0 * einsum('kbjc,ak,ci->abij', I.ovov, T1old, U12old, optimize = True)
    U22 += -1.0 * einsum('kajc,bk,ci->abij', I.ovov, T1old, U12old, optimize = True)
    U22 += 1.0 * einsum('bacd,di,cj->abij', I.vvvv, T1old, U12old, optimize = True)
    U22 += -1.0 * einsum('bacd,dj,ci->abij', I.vvvv, T1old, U12old, optimize = True)
    U22 += 1.0 * S1old * einsum('ki,bakj->abij', g.oo[0], U22old, optimize = True)
    U22 += -1.0 * S1old * einsum('kj,baki->abij', g.oo[0], U22old, optimize = True)
    if nfock1 > 2:
        U22 += 1.0 * S2old * einsum('ki,bakj->abij', g.oo[0], U21old, optimize = True)
        U22 += -1.0 * S2old * einsum('kj,baki->abij', g.oo[0], U21old, optimize = True)
        U22 += 1.0 * S2old * einsum('ki,bakj->abij', g.oo[0], U21old, optimize = True)
        U22 += -1.0 * S2old * einsum('kj,baki->abij', g.oo[0], U21old, optimize = True)
        U22 += -1.0 * S2old * einsum('bc,acji->abij', g.vv[0], U21old, optimize = True)
        U22 += 1.0 * S2old * einsum('ac,bcji->abij', g.vv[0], U21old, optimize = True)
        U22 += -1.0 * S2old * einsum('bc,acji->abij', g.vv[0], U21old, optimize = True)
        U22 += 1.0 * S2old * einsum('ac,bcji->abij', g.vv[0], U21old, optimize = True)
        U22 += 1.0 * S2old * einsum('kc,ci,bakj->abij', g.ov[0], T1old, U21old, optimize = True)
        U22 += -1.0 * S2old * einsum('kc,cj,baki->abij', g.ov[0], T1old, U21old, optimize = True)
        U22 += 1.0 * S2old * einsum('kc,bk,acji->abij', g.ov[0], T1old, U21old, optimize = True)
        U22 += -1.0 * S2old * einsum('kc,ak,bcji->abij', g.ov[0], T1old, U21old, optimize = True)
        U22 += 1.0 * S2old * einsum('kc,ci,bakj->abij', g.ov[0], T1old, U21old, optimize = True)
        U22 += -1.0 * S2old * einsum('kc,cj,baki->abij', g.ov[0], T1old, U21old, optimize = True)
        U22 += 1.0 * S2old * einsum('kc,bk,acji->abij', g.ov[0], T1old, U21old, optimize = True)
        U22 += -1.0 * S2old * einsum('kc,ak,bcji->abij', g.ov[0], T1old, U21old, optimize = True)
        U22 += -1.0 * S2old * einsum('kc,bcji,ak->abij', g.ov[0], T2old, U11old, optimize = True)
        U22 += 1.0 * S2old * einsum('kc,acji,bk->abij', g.ov[0], T2old, U11old, optimize = True)
        U22 += -1.0 * S2old * einsum('kc,baki,cj->abij', g.ov[0], T2old, U11old, optimize = True)
        U22 += 1.0 * S2old * einsum('kc,bakj,ci->abij', g.ov[0], T2old, U11old, optimize = True)
        U22 += -1.0 * S2old * einsum('kc,bcji,ak->abij', g.ov[0], T2old, U11old, optimize = True)
        U22 += 1.0 * S2old * einsum('kc,acji,bk->abij', g.ov[0], T2old, U11old, optimize = True)
        U22 += -1.0 * S2old * einsum('kc,baki,cj->abij', g.ov[0], T2old, U11old, optimize = True)
        U22 += 1.0 * S2old * einsum('kc,bakj,ci->abij', g.ov[0], T2old, U11old, optimize = True)
    U22 += 1.0 * einsum('kc,ci,bakj->abij', g.ov[0], T1old, U21old, optimize = True)
    U22 += -1.0 * einsum('kc,cj,baki->abij', g.ov[0], T1old, U21old, optimize = True)
    U22 += 1.0 * einsum('kc,bk,acji->abij', g.ov[0], T1old, U21old, optimize = True)
    U22 += -1.0 * einsum('kc,ak,bcji->abij', g.ov[0], T1old, U21old, optimize = True)
    U22 += 1.0 * einsum('kc,ci,bakj->abij', g.ov[0], T1old, U21old, optimize = True)
    U22 += -1.0 * einsum('kc,cj,baki->abij', g.ov[0], T1old, U21old, optimize = True)
    U22 += 1.0 * einsum('kc,bk,acji->abij', g.ov[0], T1old, U21old, optimize = True)
    U22 += -1.0 * einsum('kc,ak,bcji->abij', g.ov[0], T1old, U21old, optimize = True)
    U22 += -1.0 * einsum('kc,bcji,ak->abij', g.ov[0], T2old, U11old, optimize = True)
    U22 += 1.0 * einsum('kc,acji,bk->abij', g.ov[0], T2old, U11old, optimize = True)
    U22 += -1.0 * einsum('kc,baki,cj->abij', g.ov[0], T2old, U11old, optimize = True)
    U22 += 1.0 * einsum('kc,bakj,ci->abij', g.ov[0], T2old, U11old, optimize = True)
    U22 += -1.0 * einsum('kc,bcji,ak->abij', g.ov[0], T2old, U11old, optimize = True)
    U22 += 1.0 * einsum('kc,acji,bk->abij', g.ov[0], T2old, U11old, optimize = True)
    U22 += -1.0 * einsum('kc,baki,cj->abij', g.ov[0], T2old, U11old, optimize = True)
    U22 += 1.0 * einsum('kc,bakj,ci->abij', g.ov[0], T2old, U11old, optimize = True)
    U22 += -1.0 * S1old * einsum('bc,acji->abij', g.vv[0], U22old, optimize = True)
    U22 += 1.0 * S1old * einsum('ac,bcji->abij', g.vv[0], U22old, optimize = True)
    U22 += 0.5 * einsum('klic,cj,balk->abij', I.ooov, T1old, U22old, optimize = True)
    U22 += -1.0 * einsum('klic,bk,aclj->abij', I.ooov, T1old, U22old, optimize = True)
    U22 += 1.0 * einsum('klic,ak,bclj->abij', I.ooov, T1old, U22old, optimize = True)
    U22 += -1.0 * einsum('klic,ck,balj->abij', I.ooov, T1old, U22old, optimize = True)
    U22 += -0.5 * einsum('kljc,ci,balk->abij', I.ooov, T1old, U22old, optimize = True)
    U22 += 1.0 * einsum('kljc,bk,acli->abij', I.ooov, T1old, U22old, optimize = True)
    U22 += -1.0 * einsum('kljc,ak,bcli->abij', I.ooov, T1old, U22old, optimize = True)
    U22 += 1.0 * einsum('kljc,ck,bali->abij', I.ooov, T1old, U22old, optimize = True)
    U22 += 1.0 * einsum('klic,bakj,cl->abij', I.ooov, T2old, U12old, optimize = True)
    U22 += -1.0 * einsum('klic,bckj,al->abij', I.ooov, T2old, U12old, optimize = True)
    U22 += 1.0 * einsum('klic,ackj,bl->abij', I.ooov, T2old, U12old, optimize = True)
    U22 += 0.5 * einsum('klic,balk,cj->abij', I.ooov, T2old, U12old, optimize = True)
    U22 += -1.0 * einsum('kljc,baki,cl->abij', I.ooov, T2old, U12old, optimize = True)
    U22 += 1.0 * einsum('kljc,bcki,al->abij', I.ooov, T2old, U12old, optimize = True)
    U22 += -1.0 * einsum('kljc,acki,bl->abij', I.ooov, T2old, U12old, optimize = True)
    U22 += -0.5 * einsum('kljc,balk,ci->abij', I.ooov, T2old, U12old, optimize = True)
    U22 += 1.0 * einsum('kbcd,di,ackj->abij', I.ovvv, T1old, U22old, optimize = True)
    U22 += -1.0 * einsum('kacd,di,bckj->abij', I.ovvv, T1old, U22old, optimize = True)
    U22 += -1.0 * einsum('kbcd,dj,acki->abij', I.ovvv, T1old, U22old, optimize = True)
    U22 += 1.0 * einsum('kacd,dj,bcki->abij', I.ovvv, T1old, U22old, optimize = True)
    U22 += -0.5 * einsum('kbcd,ak,dcji->abij', I.ovvv, T1old, U22old, optimize = True)
    U22 += 1.0 * einsum('kbcd,dk,acji->abij', I.ovvv, T1old, U22old, optimize = True)
    U22 += 0.5 * einsum('kacd,bk,dcji->abij', I.ovvv, T1old, U22old, optimize = True)
    U22 += -1.0 * einsum('kacd,dk,bcji->abij', I.ovvv, T1old, U22old, optimize = True)
    U22 += -1.0 * einsum('kbcd,adji,ck->abij', I.ovvv, T2old, U12old, optimize = True)
    U22 += -0.5 * einsum('kbcd,dcji,ak->abij', I.ovvv, T2old, U12old, optimize = True)
    U22 += 1.0 * einsum('kacd,bdji,ck->abij', I.ovvv, T2old, U12old, optimize = True)
    U22 += 0.5 * einsum('kacd,dcji,bk->abij', I.ovvv, T2old, U12old, optimize = True)
    U22 += 1.0 * einsum('kbcd,adki,cj->abij', I.ovvv, T2old, U12old, optimize = True)
    U22 += -1.0 * einsum('kacd,bdki,cj->abij', I.ovvv, T2old, U12old, optimize = True)
    U22 += -1.0 * einsum('kbcd,adkj,ci->abij', I.ovvv, T2old, U12old, optimize = True)
    U22 += 1.0 * einsum('kacd,bdkj,ci->abij', I.ovvv, T2old, U12old, optimize = True)
    U22 += -0.5 * einsum('klcd,bdji,aclk->abij', I.oovv, T2old, U22old, optimize = True)
    U22 += 0.5 * einsum('klcd,adji,bclk->abij', I.oovv, T2old, U22old, optimize = True)
    U22 += 0.25 * einsum('klcd,dcji,balk->abij', I.oovv, T2old, U22old, optimize = True)
    U22 += -0.5 * einsum('klcd,baki,dclj->abij', I.oovv, T2old, U22old, optimize = True)
    U22 += 1.0 * einsum('klcd,bdki,aclj->abij', I.oovv, T2old, U22old, optimize = True)
    U22 += -1.0 * einsum('klcd,adki,bclj->abij', I.oovv, T2old, U22old, optimize = True)
    U22 += -0.5 * einsum('klcd,dcki,balj->abij', I.oovv, T2old, U22old, optimize = True)
    U22 += 0.5 * einsum('klcd,bakj,dcli->abij', I.oovv, T2old, U22old, optimize = True)
    U22 += -1.0 * einsum('klcd,bdkj,acli->abij', I.oovv, T2old, U22old, optimize = True)
    U22 += 1.0 * einsum('klcd,adkj,bcli->abij', I.oovv, T2old, U22old, optimize = True)
    U22 += 0.5 * einsum('klcd,dckj,bali->abij', I.oovv, T2old, U22old, optimize = True)
    U22 += 0.25 * einsum('klcd,balk,dcji->abij', I.oovv, T2old, U22old, optimize = True)
    U22 += -0.5 * einsum('klcd,bdlk,acji->abij', I.oovv, T2old, U22old, optimize = True)
    U22 += 0.5 * einsum('klcd,adlk,bcji->abij', I.oovv, T2old, U22old, optimize = True)
    U22 += 2.0 * einsum('kc,ci,bakj->abij', F.ov, U11old, U21old, optimize = True)
    U22 += -2.0 * einsum('kc,cj,baki->abij', F.ov, U11old, U21old, optimize = True)
    U22 += 2.0 * einsum('kc,bk,acji->abij', F.ov, U11old, U21old, optimize = True)
    U22 += -2.0 * einsum('kc,ak,bcji->abij', F.ov, U11old, U21old, optimize = True)
    U22 += 2.0 * einsum('klji,bk,al->abij', I.oooo, U11old, U11old, optimize = True)
    U22 += -2.0 * einsum('kbic,cj,ak->abij', I.ovov, U11old, U11old, optimize = True)
    U22 += 2.0 * einsum('kaic,cj,bk->abij', I.ovov, U11old, U11old, optimize = True)
    U22 += 2.0 * einsum('kbjc,ci,ak->abij', I.ovov, U11old, U11old, optimize = True)
    U22 += -2.0 * einsum('kajc,ci,bk->abij', I.ovov, U11old, U11old, optimize = True)
    U22 += 2.0 * einsum('bacd,di,cj->abij', I.vvvv, U11old, U11old, optimize = True)
    U22 += 1.0 * einsum('ki,bk,aj->abij', g.oo[0], U11old, U12old, optimize = True)
    U22 += -1.0 * einsum('ki,ak,bj->abij', g.oo[0], U11old, U12old, optimize = True)
    U22 += -1.0 * einsum('kj,bk,ai->abij', g.oo[0], U11old, U12old, optimize = True)
    U22 += 1.0 * einsum('kj,ak,bi->abij', g.oo[0], U11old, U12old, optimize = True)
    U22 += 1.0 * einsum('ki,bk,aj->abij', g.oo[0], U11old, U12old, optimize = True)
    U22 += -1.0 * einsum('ki,ak,bj->abij', g.oo[0], U11old, U12old, optimize = True)
    U22 += -1.0 * einsum('kj,bk,ai->abij', g.oo[0], U11old, U12old, optimize = True)
    U22 += 1.0 * einsum('kj,ak,bi->abij', g.oo[0], U11old, U12old, optimize = True)
    U22 += -1.0 * einsum('ki,bj,ak->abij', g.oo[0], U11old, U12old, optimize = True)
    U22 += 1.0 * einsum('ki,aj,bk->abij', g.oo[0], U11old, U12old, optimize = True)
    U22 += 1.0 * einsum('kj,bi,ak->abij', g.oo[0], U11old, U12old, optimize = True)
    U22 += -1.0 * einsum('kj,ai,bk->abij', g.oo[0], U11old, U12old, optimize = True)
    U22 += -1.0 * einsum('bc,ci,aj->abij', g.vv[0], U11old, U12old, optimize = True)
    U22 += 1.0 * einsum('ac,ci,bj->abij', g.vv[0], U11old, U12old, optimize = True)
    U22 += 1.0 * einsum('bc,cj,ai->abij', g.vv[0], U11old, U12old, optimize = True)
    U22 += -1.0 * einsum('ac,cj,bi->abij', g.vv[0], U11old, U12old, optimize = True)
    U22 += -1.0 * einsum('bc,ci,aj->abij', g.vv[0], U11old, U12old, optimize = True)
    U22 += 1.0 * einsum('ac,ci,bj->abij', g.vv[0], U11old, U12old, optimize = True)
    U22 += 1.0 * einsum('bc,cj,ai->abij', g.vv[0], U11old, U12old, optimize = True)
    U22 += -1.0 * einsum('ac,cj,bi->abij', g.vv[0], U11old, U12old, optimize = True)
    U22 += 1.0 * einsum('bc,ai,cj->abij', g.vv[0], U11old, U12old, optimize = True)
    U22 += -1.0 * einsum('ac,bi,cj->abij', g.vv[0], U11old, U12old, optimize = True)
    U22 += -1.0 * einsum('bc,aj,ci->abij', g.vv[0], U11old, U12old, optimize = True)
    U22 += 1.0 * einsum('ac,bj,ci->abij', g.vv[0], U11old, U12old, optimize = True)
    U22 += 1.0 * einsum('klic,cj,balk->abij', I.ooov, U11old, U21old, optimize = True)
    U22 += -2.0 * einsum('klic,bk,aclj->abij', I.ooov, U11old, U21old, optimize = True)
    U22 += 2.0 * einsum('klic,ak,bclj->abij', I.ooov, U11old, U21old, optimize = True)
    U22 += -2.0 * einsum('klic,ck,balj->abij', I.ooov, U11old, U21old, optimize = True)
    U22 += -1.0 * einsum('kljc,ci,balk->abij', I.ooov, U11old, U21old, optimize = True)
    U22 += 2.0 * einsum('kljc,bk,acli->abij', I.ooov, U11old, U21old, optimize = True)
    U22 += -2.0 * einsum('kljc,ak,bcli->abij', I.ooov, U11old, U21old, optimize = True)
    U22 += 2.0 * einsum('kljc,ck,bali->abij', I.ooov, U11old, U21old, optimize = True)
    U22 += 2.0 * einsum('kbcd,di,ackj->abij', I.ovvv, U11old, U21old, optimize = True)
    U22 += -2.0 * einsum('kacd,di,bckj->abij', I.ovvv, U11old, U21old, optimize = True)
    U22 += -2.0 * einsum('kbcd,dj,acki->abij', I.ovvv, U11old, U21old, optimize = True)
    U22 += 2.0 * einsum('kacd,dj,bcki->abij', I.ovvv, U11old, U21old, optimize = True)
    U22 += -1.0 * einsum('kbcd,ak,dcji->abij', I.ovvv, U11old, U21old, optimize = True)
    U22 += 2.0 * einsum('kbcd,dk,acji->abij', I.ovvv, U11old, U21old, optimize = True)
    U22 += 1.0 * einsum('kacd,bk,dcji->abij', I.ovvv, U11old, U21old, optimize = True)
    U22 += -2.0 * einsum('kacd,dk,bcji->abij', I.ovvv, U11old, U21old, optimize = True)
    U22 += 1.0 * einsum('kc,ci,bakj->abij', g.ov[0], U11old, U22old, optimize = True)
    U22 += -1.0 * einsum('kc,cj,baki->abij', g.ov[0], U11old, U22old, optimize = True)
    U22 += 1.0 * einsum('kc,bk,acji->abij', g.ov[0], U11old, U22old, optimize = True)
    U22 += -1.0 * einsum('kc,ak,bcji->abij', g.ov[0], U11old, U22old, optimize = True)
    U22 += 1.0 * einsum('kc,ck,baji->abij', g.ov[0], U11old, U22old, optimize = True)
    U22 += 1.0 * einsum('kc,ci,bakj->abij', g.ov[0], U11old, U22old, optimize = True)
    U22 += -1.0 * einsum('kc,cj,baki->abij', g.ov[0], U11old, U22old, optimize = True)
    U22 += 1.0 * einsum('kc,bk,acji->abij', g.ov[0], U11old, U22old, optimize = True)
    U22 += -1.0 * einsum('kc,ak,bcji->abij', g.ov[0], U11old, U22old, optimize = True)
    U22 += 1.0 * einsum('kc,ck,baji->abij', g.ov[0], U11old, U22old, optimize = True)
    U22 += 1.0 * einsum('kc,bi,ackj->abij', g.ov[0], U11old, U22old, optimize = True)
    U22 += -1.0 * einsum('kc,ai,bckj->abij', g.ov[0], U11old, U22old, optimize = True)
    U22 += 1.0 * einsum('kc,ci,bakj->abij', g.ov[0], U11old, U22old, optimize = True)
    U22 += -1.0 * einsum('kc,bj,acki->abij', g.ov[0], U11old, U22old, optimize = True)
    U22 += 1.0 * einsum('kc,aj,bcki->abij', g.ov[0], U11old, U22old, optimize = True)
    U22 += -1.0 * einsum('kc,cj,baki->abij', g.ov[0], U11old, U22old, optimize = True)
    U22 += 1.0 * einsum('kc,bk,acji->abij', g.ov[0], U11old, U22old, optimize = True)
    U22 += -1.0 * einsum('kc,ak,bcji->abij', g.ov[0], U11old, U22old, optimize = True)
    U22 += -1.0 * einsum('kc,bcji,ak->abij', g.ov[0], U21old, U12old, optimize = True)
    U22 += 1.0 * einsum('kc,acji,bk->abij', g.ov[0], U21old, U12old, optimize = True)
    U22 += -1.0 * einsum('kc,baki,cj->abij', g.ov[0], U21old, U12old, optimize = True)
    U22 += 1.0 * einsum('kc,bcki,aj->abij', g.ov[0], U21old, U12old, optimize = True)
    U22 += -1.0 * einsum('kc,acki,bj->abij', g.ov[0], U21old, U12old, optimize = True)
    U22 += 1.0 * einsum('kc,bakj,ci->abij', g.ov[0], U21old, U12old, optimize = True)
    U22 += -1.0 * einsum('kc,bckj,ai->abij', g.ov[0], U21old, U12old, optimize = True)
    U22 += 1.0 * einsum('kc,ackj,bi->abij', g.ov[0], U21old, U12old, optimize = True)
    U22 += -1.0 * einsum('kc,bcji,ak->abij', g.ov[0], U21old, U12old, optimize = True)
    U22 += 1.0 * einsum('kc,acji,bk->abij', g.ov[0], U21old, U12old, optimize = True)
    U22 += -1.0 * einsum('kc,baki,cj->abij', g.ov[0], U21old, U12old, optimize = True)
    U22 += 1.0 * einsum('kc,bcki,aj->abij', g.ov[0], U21old, U12old, optimize = True)
    U22 += -1.0 * einsum('kc,acki,bj->abij', g.ov[0], U21old, U12old, optimize = True)
    U22 += 1.0 * einsum('kc,bakj,ci->abij', g.ov[0], U21old, U12old, optimize = True)
    U22 += -1.0 * einsum('kc,bckj,ai->abij', g.ov[0], U21old, U12old, optimize = True)
    U22 += 1.0 * einsum('kc,ackj,bi->abij', g.ov[0], U21old, U12old, optimize = True)
    U22 += 1.0 * einsum('kc,baji,ck->abij', g.ov[0], U21old, U12old, optimize = True)
    U22 += -1.0 * einsum('kc,bcji,ak->abij', g.ov[0], U21old, U12old, optimize = True)
    U22 += 1.0 * einsum('kc,acji,bk->abij', g.ov[0], U21old, U12old, optimize = True)
    U22 += -1.0 * einsum('kc,baki,cj->abij', g.ov[0], U21old, U12old, optimize = True)
    U22 += 1.0 * einsum('kc,bakj,ci->abij', g.ov[0], U21old, U12old, optimize = True)
    U22 += -1.0 * einsum('klic,cj,bk,al->abij', I.ooov, T1old, T1old, U12old, optimize = True)
    U22 += 1.0 * einsum('klic,cj,ak,bl->abij', I.ooov, T1old, T1old, U12old, optimize = True)
    U22 += -1.0 * einsum('klic,bk,al,cj->abij', I.ooov, T1old, T1old, U12old, optimize = True)
    U22 += 1.0 * einsum('kljc,ci,bk,al->abij', I.ooov, T1old, T1old, U12old, optimize = True)
    U22 += -1.0 * einsum('kljc,ci,ak,bl->abij', I.ooov, T1old, T1old, U12old, optimize = True)
    U22 += 1.0 * einsum('kljc,bk,al,ci->abij', I.ooov, T1old, T1old, U12old, optimize = True)
    U22 += 1.0 * einsum('kbcd,di,cj,ak->abij', I.ovvv, T1old, T1old, U12old, optimize = True)
    U22 += -1.0 * einsum('kacd,di,cj,bk->abij', I.ovvv, T1old, T1old, U12old, optimize = True)
    U22 += 1.0 * einsum('kbcd,di,ak,cj->abij', I.ovvv, T1old, T1old, U12old, optimize = True)
    U22 += -1.0 * einsum('kacd,di,bk,cj->abij', I.ovvv, T1old, T1old, U12old, optimize = True)
    U22 += -1.0 * einsum('kbcd,dj,ak,ci->abij', I.ovvv, T1old, T1old, U12old, optimize = True)
    U22 += 1.0 * einsum('kacd,dj,bk,ci->abij', I.ovvv, T1old, T1old, U12old, optimize = True)
    U22 += 1.0 * S1old * einsum('kc,ci,bakj->abij', g.ov[0], T1old, U22old, optimize = True)
    U22 += -1.0 * S1old * einsum('kc,cj,baki->abij', g.ov[0], T1old, U22old, optimize = True)
    U22 += 1.0 * S1old * einsum('kc,bk,acji->abij', g.ov[0], T1old, U22old, optimize = True)
    U22 += -1.0 * S1old * einsum('kc,ak,bcji->abij', g.ov[0], T1old, U22old, optimize = True)
    U22 += -1.0 * S1old * einsum('kc,bcji,ak->abij', g.ov[0], T2old, U12old, optimize = True)
    U22 += 1.0 * S1old * einsum('kc,acji,bk->abij', g.ov[0], T2old, U12old, optimize = True)
    U22 += -1.0 * S1old * einsum('kc,baki,cj->abij', g.ov[0], T2old, U12old, optimize = True)
    U22 += 1.0 * S1old * einsum('kc,bakj,ci->abij', g.ov[0], T2old, U12old, optimize = True)
    U22 += -1.0 * einsum('klcd,bdji,aclk->abij', I.oovv, U21old, U21old, optimize = True)
    U22 += 1.0 * einsum('klcd,adji,bclk->abij', I.oovv, U21old, U21old, optimize = True)
    U22 += 0.5 * einsum('klcd,dcji,balk->abij', I.oovv, U21old, U21old, optimize = True)
    U22 += -1.0 * einsum('klcd,baki,dclj->abij', I.oovv, U21old, U21old, optimize = True)
    U22 += 2.0 * einsum('klcd,bdki,aclj->abij', I.oovv, U21old, U21old, optimize = True)
    U22 += -2.0 * einsum('klcd,adki,bclj->abij', I.oovv, U21old, U21old, optimize = True)
    U22 += -1.0 * einsum('klcd,dcki,balj->abij', I.oovv, U21old, U21old, optimize = True)
    U22 += -0.5 * einsum('klcd,di,cj,balk->abij', I.oovv, T1old, T1old, U22old, optimize = True)
    U22 += 1.0 * einsum('klcd,di,bk,aclj->abij', I.oovv, T1old, T1old, U22old, optimize = True)
    U22 += -1.0 * einsum('klcd,di,ak,bclj->abij', I.oovv, T1old, T1old, U22old, optimize = True)
    U22 += 1.0 * einsum('klcd,di,ck,balj->abij', I.oovv, T1old, T1old, U22old, optimize = True)
    U22 += -1.0 * einsum('klcd,dj,bk,acli->abij', I.oovv, T1old, T1old, U22old, optimize = True)
    U22 += 1.0 * einsum('klcd,dj,ak,bcli->abij', I.oovv, T1old, T1old, U22old, optimize = True)
    U22 += -1.0 * einsum('klcd,dj,ck,bali->abij', I.oovv, T1old, T1old, U22old, optimize = True)
    U22 += -0.5 * einsum('klcd,bk,al,dcji->abij', I.oovv, T1old, T1old, U22old, optimize = True)
    U22 += 1.0 * einsum('klcd,bk,dl,acji->abij', I.oovv, T1old, T1old, U22old, optimize = True)
    U22 += -1.0 * einsum('klcd,ak,dl,bcji->abij', I.oovv, T1old, T1old, U22old, optimize = True)
    U22 += -1.0 * einsum('klcd,di,bakj,cl->abij', I.oovv, T1old, T2old, U12old, optimize = True)
    U22 += 1.0 * einsum('klcd,di,bckj,al->abij', I.oovv, T1old, T2old, U12old, optimize = True)
    U22 += -1.0 * einsum('klcd,di,ackj,bl->abij', I.oovv, T1old, T2old, U12old, optimize = True)
    U22 += -0.5 * einsum('klcd,di,balk,cj->abij', I.oovv, T1old, T2old, U12old, optimize = True)
    U22 += 1.0 * einsum('klcd,dj,baki,cl->abij', I.oovv, T1old, T2old, U12old, optimize = True)
    U22 += -1.0 * einsum('klcd,dj,bcki,al->abij', I.oovv, T1old, T2old, U12old, optimize = True)
    U22 += 1.0 * einsum('klcd,dj,acki,bl->abij', I.oovv, T1old, T2old, U12old, optimize = True)
    U22 += -1.0 * einsum('klcd,bk,adji,cl->abij', I.oovv, T1old, T2old, U12old, optimize = True)
    U22 += -0.5 * einsum('klcd,bk,dcji,al->abij', I.oovv, T1old, T2old, U12old, optimize = True)
    U22 += 1.0 * einsum('klcd,ak,bdji,cl->abij', I.oovv, T1old, T2old, U12old, optimize = True)
    U22 += 1.0 * einsum('klcd,dk,bcji,al->abij', I.oovv, T1old, T2old, U12old, optimize = True)
    U22 += 0.5 * einsum('klcd,ak,dcji,bl->abij', I.oovv, T1old, T2old, U12old, optimize = True)
    U22 += -1.0 * einsum('klcd,dk,acji,bl->abij', I.oovv, T1old, T2old, U12old, optimize = True)
    U22 += 1.0 * einsum('klcd,bk,adli,cj->abij', I.oovv, T1old, T2old, U12old, optimize = True)
    U22 += -1.0 * einsum('klcd,ak,bdli,cj->abij', I.oovv, T1old, T2old, U12old, optimize = True)
    U22 += 1.0 * einsum('klcd,dk,bali,cj->abij', I.oovv, T1old, T2old, U12old, optimize = True)
    U22 += 0.5 * einsum('klcd,dj,balk,ci->abij', I.oovv, T1old, T2old, U12old, optimize = True)
    U22 += -1.0 * einsum('klcd,bk,adlj,ci->abij', I.oovv, T1old, T2old, U12old, optimize = True)
    U22 += 1.0 * einsum('klcd,ak,bdlj,ci->abij', I.oovv, T1old, T2old, U12old, optimize = True)
    U22 += -1.0 * einsum('klcd,dk,balj,ci->abij', I.oovv, T1old, T2old, U12old, optimize = True)
    U22 += -2.0 * einsum('klic,cj,bk,al->abij', I.ooov, T1old, U11old, U11old, optimize = True)
    U22 += -2.0 * einsum('klic,bk,cj,al->abij', I.ooov, T1old, U11old, U11old, optimize = True)
    U22 += 2.0 * einsum('klic,ak,cj,bl->abij', I.ooov, T1old, U11old, U11old, optimize = True)
    U22 += 2.0 * einsum('kljc,ci,bk,al->abij', I.ooov, T1old, U11old, U11old, optimize = True)
    U22 += 2.0 * einsum('kljc,bk,ci,al->abij', I.ooov, T1old, U11old, U11old, optimize = True)
    U22 += -2.0 * einsum('kljc,ak,ci,bl->abij', I.ooov, T1old, U11old, U11old, optimize = True)
    U22 += 2.0 * einsum('kbcd,di,cj,ak->abij', I.ovvv, T1old, U11old, U11old, optimize = True)
    U22 += -2.0 * einsum('kacd,di,cj,bk->abij', I.ovvv, T1old, U11old, U11old, optimize = True)
    U22 += -2.0 * einsum('kbcd,dj,ci,ak->abij', I.ovvv, T1old, U11old, U11old, optimize = True)
    U22 += 2.0 * einsum('kacd,dj,ci,bk->abij', I.ovvv, T1old, U11old, U11old, optimize = True)
    U22 += 2.0 * einsum('kbcd,ak,di,cj->abij', I.ovvv, T1old, U11old, U11old, optimize = True)
    U22 += -2.0 * einsum('kacd,bk,di,cj->abij', I.ovvv, T1old, U11old, U11old, optimize = True)
    U22 += 1.0 * einsum('kc,ci,bk,aj->abij', g.ov[0], T1old, U11old, U12old, optimize = True)
    U22 += -1.0 * einsum('kc,ci,ak,bj->abij', g.ov[0], T1old, U11old, U12old, optimize = True)
    U22 += 1.0 * einsum('kc,bk,ci,aj->abij', g.ov[0], T1old, U11old, U12old, optimize = True)
    U22 += -1.0 * einsum('kc,ak,ci,bj->abij', g.ov[0], T1old, U11old, U12old, optimize = True)
    U22 += -1.0 * einsum('kc,cj,bk,ai->abij', g.ov[0], T1old, U11old, U12old, optimize = True)
    U22 += 1.0 * einsum('kc,cj,ak,bi->abij', g.ov[0], T1old, U11old, U12old, optimize = True)
    U22 += -1.0 * einsum('kc,bk,cj,ai->abij', g.ov[0], T1old, U11old, U12old, optimize = True)
    U22 += 1.0 * einsum('kc,ak,cj,bi->abij', g.ov[0], T1old, U11old, U12old, optimize = True)
    U22 += 1.0 * einsum('kc,ci,bk,aj->abij', g.ov[0], T1old, U11old, U12old, optimize = True)
    U22 += -1.0 * einsum('kc,ci,ak,bj->abij', g.ov[0], T1old, U11old, U12old, optimize = True)
    U22 += 1.0 * einsum('kc,bk,ci,aj->abij', g.ov[0], T1old, U11old, U12old, optimize = True)
    U22 += -1.0 * einsum('kc,ak,ci,bj->abij', g.ov[0], T1old, U11old, U12old, optimize = True)
    U22 += -1.0 * einsum('kc,cj,bk,ai->abij', g.ov[0], T1old, U11old, U12old, optimize = True)
    U22 += 1.0 * einsum('kc,cj,ak,bi->abij', g.ov[0], T1old, U11old, U12old, optimize = True)
    U22 += -1.0 * einsum('kc,bk,cj,ai->abij', g.ov[0], T1old, U11old, U12old, optimize = True)
    U22 += 1.0 * einsum('kc,ak,cj,bi->abij', g.ov[0], T1old, U11old, U12old, optimize = True)
    U22 += -1.0 * einsum('kc,ci,bj,ak->abij', g.ov[0], T1old, U11old, U12old, optimize = True)
    U22 += 1.0 * einsum('kc,ci,aj,bk->abij', g.ov[0], T1old, U11old, U12old, optimize = True)
    U22 += 1.0 * einsum('kc,cj,bi,ak->abij', g.ov[0], T1old, U11old, U12old, optimize = True)
    U22 += -1.0 * einsum('kc,cj,ai,bk->abij', g.ov[0], T1old, U11old, U12old, optimize = True)
    U22 += -1.0 * einsum('kc,bk,ai,cj->abij', g.ov[0], T1old, U11old, U12old, optimize = True)
    U22 += 1.0 * einsum('kc,ak,bi,cj->abij', g.ov[0], T1old, U11old, U12old, optimize = True)
    U22 += 1.0 * einsum('kc,bk,aj,ci->abij', g.ov[0], T1old, U11old, U12old, optimize = True)
    U22 += -1.0 * einsum('kc,ak,bj,ci->abij', g.ov[0], T1old, U11old, U12old, optimize = True)
    U22 += 2.0 * S1old * einsum('kc,ci,bakj->abij', g.ov[0], U11old, U21old, optimize = True)
    U22 += -2.0 * S1old * einsum('kc,cj,baki->abij', g.ov[0], U11old, U21old, optimize = True)
    U22 += 2.0 * S1old * einsum('kc,bk,acji->abij', g.ov[0], U11old, U21old, optimize = True)
    U22 += -2.0 * S1old * einsum('kc,ak,bcji->abij', g.ov[0], U11old, U21old, optimize = True)
    U22 += -1.0 * einsum('klcd,di,cj,balk->abij', I.oovv, T1old, U11old, U21old, optimize = True)
    U22 += 2.0 * einsum('klcd,di,bk,aclj->abij', I.oovv, T1old, U11old, U21old, optimize = True)
    U22 += -2.0 * einsum('klcd,di,ak,bclj->abij', I.oovv, T1old, U11old, U21old, optimize = True)
    U22 += 2.0 * einsum('klcd,di,ck,balj->abij', I.oovv, T1old, U11old, U21old, optimize = True)
    U22 += 1.0 * einsum('klcd,dj,ci,balk->abij', I.oovv, T1old, U11old, U21old, optimize = True)
    U22 += 2.0 * einsum('klcd,bk,di,aclj->abij', I.oovv, T1old, U11old, U21old, optimize = True)
    U22 += -2.0 * einsum('klcd,ak,di,bclj->abij', I.oovv, T1old, U11old, U21old, optimize = True)
    U22 += -2.0 * einsum('klcd,dk,ci,balj->abij', I.oovv, T1old, U11old, U21old, optimize = True)
    U22 += -2.0 * einsum('klcd,dj,bk,acli->abij', I.oovv, T1old, U11old, U21old, optimize = True)
    U22 += 2.0 * einsum('klcd,dj,ak,bcli->abij', I.oovv, T1old, U11old, U21old, optimize = True)
    U22 += -2.0 * einsum('klcd,dj,ck,bali->abij', I.oovv, T1old, U11old, U21old, optimize = True)
    U22 += -2.0 * einsum('klcd,bk,dj,acli->abij', I.oovv, T1old, U11old, U21old, optimize = True)
    U22 += 2.0 * einsum('klcd,ak,dj,bcli->abij', I.oovv, T1old, U11old, U21old, optimize = True)
    U22 += 2.0 * einsum('klcd,dk,cj,bali->abij', I.oovv, T1old, U11old, U21old, optimize = True)
    U22 += -1.0 * einsum('klcd,bk,al,dcji->abij', I.oovv, T1old, U11old, U21old, optimize = True)
    U22 += 2.0 * einsum('klcd,bk,dl,acji->abij', I.oovv, T1old, U11old, U21old, optimize = True)
    U22 += 1.0 * einsum('klcd,ak,bl,dcji->abij', I.oovv, T1old, U11old, U21old, optimize = True)
    U22 += -2.0 * einsum('klcd,dk,bl,acji->abij', I.oovv, T1old, U11old, U21old, optimize = True)
    U22 += -2.0 * einsum('klcd,ak,dl,bcji->abij', I.oovv, T1old, U11old, U21old, optimize = True)
    U22 += 2.0 * einsum('klcd,dk,al,bcji->abij', I.oovv, T1old, U11old, U21old, optimize = True)
    U22 += 2.0 * einsum('klcd,bdji,ak,cl->abij', I.oovv, T2old, U11old, U11old, optimize = True)
    U22 += -2.0 * einsum('klcd,adji,bk,cl->abij', I.oovv, T2old, U11old, U11old, optimize = True)
    U22 += -1.0 * einsum('klcd,dcji,bk,al->abij', I.oovv, T2old, U11old, U11old, optimize = True)
    U22 += 2.0 * einsum('klcd,baki,dj,cl->abij', I.oovv, T2old, U11old, U11old, optimize = True)
    U22 += 2.0 * einsum('klcd,bdki,cj,al->abij', I.oovv, T2old, U11old, U11old, optimize = True)
    U22 += -2.0 * einsum('klcd,adki,cj,bl->abij', I.oovv, T2old, U11old, U11old, optimize = True)
    U22 += -2.0 * einsum('klcd,bakj,di,cl->abij', I.oovv, T2old, U11old, U11old, optimize = True)
    U22 += -2.0 * einsum('klcd,bdkj,ci,al->abij', I.oovv, T2old, U11old, U11old, optimize = True)
    U22 += 2.0 * einsum('klcd,adkj,ci,bl->abij', I.oovv, T2old, U11old, U11old, optimize = True)
    U22 += -1.0 * einsum('klcd,balk,di,cj->abij', I.oovv, T2old, U11old, U11old, optimize = True)
    U22 += 2.0 * einsum('kc,bi,cj,ak->abij', g.ov[0], U11old, U11old, U11old, optimize = True)
    U22 += 2.0 * einsum('kc,ci,aj,bk->abij', g.ov[0], U11old, U11old, U11old, optimize = True)
    U22 += -2.0 * einsum('kc,ci,bj,ak->abij', g.ov[0], U11old, U11old, U11old, optimize = True)
    U22 += -2.0 * einsum('kc,ai,cj,bk->abij', g.ov[0], U11old, U11old, U11old, optimize = True)

    U2n[1] = numpy.zeros((nm, nm, nvir, nvir, nocc, nocc))

    U2n[1][0,0] = U22

    return U2n

#===================optimized u1n and u2n code----------
def qedccsd_U1n_opt(F, I, w, g, h, G, H, nfock1, nfock2, amps, imds):

    T1old, T2old, Snold, U1nold, U2nold = amps

    nv = T2old.shape[0]
    no = T2old.shape[2]
    nm = w.shape[0]

    eps_occ = F.oo.diagonal()
    eps_vir = F.vv.diagonal()

    #e_denom = 1 / (eps_occ.reshape(-1, 1) - eps_vir)

    U1n = [None]*nfock2
    if nfock2 < 1: return U1n

    #if nm == 1: # if single mode, we use faster single mode code (reducing indexing overhead)
    #    return single_qedccsd_U1n(F, I, w, g, h, G, H, nfock1, nfock2, amps)

    U1n[0] = g.vo.copy()
    U1n[0] += 1.0*einsum('I,Iai->Iai', w, U1nold[0])

    Xov = einsum('ijab,xbj->xia', I.oovv, U1nold[0])
    U1n[0] += 0.5*einsum('xjb,abij->xai', Xov, T2old)

    U1n[0] += 1.0*einsum('Iab,bi->Iai', g.vv, T1old)
    U1n[0] -= 1.0*einsum('Iji,aj->Iai', g.oo, T1old)

    T2temp = T2old - einsum('bi,aj->abij', T1old, T1old)
    U1n[0] += einsum('xjb,abij->xai', h.ov, T2temp)
    #U1n[0] += -1.0*einsum('Ijb,abji->Iai', g.ov, T2old)
    #U1n[0] += -1.0*einsum('Ijb,bi,aj->Iai', g.ov, T1old, T1old, optimize=True)

    #Fvv = F.vv.copy()
    #Fvv += einsum('xab,x->ab', g.vv, Snold[0])
    #Fvv -= 0.5*einsum('jb,aj->ab',F.ov,T1old)
    #Fvv -= einsum('ajcb,cj->ab',I.vovv,T1old)
    #Fvv -= 0.5*einsum('jkbc,acjk->ab',I.oovv, imds.T2A)
    #Fvv -= einsum('yjb,yaj->ab', g.ov, U1nold[0])

    #Foo = F.oo.copy()
    #Foo += einsum('xji,x->ji', g.oo, Snold[0])
    #Foo += 0.5*einsum('jb,bi->ji',F.ov,T1old)
    #Foo += einsum('jkib,bk->ji',I.ooov,T1old)
    #Foo += 0.5*einsum('jkbc,bcik->ji',I.oovv, imds.T2A)
    #Foo += einsum('yjb,ybi->ji', g.ov, U1nold[0])

    # using intermediate variables
    U1n[0] += 1.0*einsum('ab,Ibi->Iai', imds.Fvv, U1nold[0])
    #U1n[0] += 1.0*einsum('ab,Ibi->Iai', F.vv, U1nold[0])
    #U1n[0] += 1.0*einsum('Jab,J,Ibi->Iai', g.vv, Snold[0], U1nold[0], optimize=True)
    #U1n[0] += -1.0*einsum('jb,aj,Ibi->Iai', F.ov, T1old, U1nold[0], optimize=True) # in Fvv
    #U1n[0] += -1.0*einsum('Jjb,Iaj,Jbi->Iai', g.ov, U1nold[0], U1nold[0], optimize=True)
    #U1n[0] += -1.0*einsum('jabc,cj,Ibi->Iai', I.ovvv, T1old, U1nold[0], optimize=True) # in Fvv
    #U1n[0] += 0.5*einsum('jkbc,ackj,Ibi->Iai', I.oovv, T2old, U1nold[0], optimize=True)
    #U1n[0] += -1.0*einsum('jkbc,aj,ck,Ibi->Iai', I.oovv, T1old, T1old, U1nold[0], optimize=True)

    U1n[0] -= einsum('ji,xaj->xai', imds.Foo, U1nold[0])
    #U1n[0] += -1.0*einsum('ji,Iaj->Iai', F.oo, U1nold[0])
    #U1n[0] += -1.0*einsum('Jji,J,Iaj->Iai', g.oo, Snold[0], U1nold[0], optimize=True)
    #U1n[0] += -1.0*einsum('jb,bi,Iaj->Iai', F.ov, T1old, U1nold[0], optimize=True) # in Foo
    #U1n[0] += -1.0*einsum('Jjb,Ibi,Jaj->Iai', g.ov, U1nold[0], U1nold[0], optimize=True)
    #U1n[0] += 1.0*einsum('jkib,bj,Iak->Iai', I.ooov, T1old, U1nold[0], optimize=True) # in Foo
    #U1n[0] += 0.5*einsum('jkbc,cbji,Iak->Iai', I.oovv, T2old, U1nold[0], optimize=True)
    #U1n[0] += -1.0*einsum('jkbc,ci,bj,Iak->Iai', I.oovv, T1old, T1old, U1nold[0], optimize=True)

    U1n[0] += einsum('jabi,xbj->xai', imds.Wovvo, U1nold[0])
    Xvv = einsum('jb,aj->ab', imds.Fov + imds.gsov, T1old)
    U1n[0] -= 0.5*einsum('ab,xbi->xai', Xvv, U1nold[0])

    Xoo = einsum('jb,bi->ji', imds.Fov + imds.gsov, T1old)
    U1n[0] -= 0.5*einsum('ji,xaj->xai', Xoo, U1nold[0])

    Xmm = einsum('yjb,xbj->xy', g.ov, U1nold[0])
    U1n[0] += einsum('xy,yai->xai', Xmm, U1nold[0])

    # in Wovvo
    #U1n[0] += -1.0*einsum('jaib,Ibj->Iai', I.ovov, U1nold[0], optimize=True)
    #U1n[0] += -1.0*einsum('jkib,aj,Ibk->Iai', I.ooov, T1old, U1nold[0], optimize=True)
    #U1n[0] += 1.0*einsum('jabc,ci,Ibj->Iai', I.ovvv, T1old, U1nold[0], optimize=True)
    #U1n[0] += 1.0*einsum('jkbc,acji,Ibk->Iai', I.oovv, T2old, U1nold[0], optimize=True)
    #U1n[0] += 1.0*einsum('jkbc,ci,aj,Ibk->Iai', I.oovv, T1old, T1old, U1nold[0], optimize=True)

    # in Xoo, Xvv
    #U1n[0] += -1.0*einsum('Jjb,aj,J,Ibi->Iai', g.ov, T1old, Snold[0], U1nold[0], optimize=True)
    #U1n[0] += -1.0*einsum('Jjb,bi,J,Iaj->Iai', g.ov, T1old, Snold[0], U1nold[0], optimize=True)

    # in Xmm
    #U1n[0] += 1.0*einsum('Jjb,Ibj,Jai->Iai', g.ov, U1nold[0], U1nold[0], optimize=True)

    if U2nold[0] is not None:
        U1n[0] += -1.0*einsum('jb,Iabji->Iai', F.ov, U2nold[0])

    if nfock2 > 1:
        U1n[0] += -1.0*einsum('Jji,IJaj->Iai', g.oo, U1nold[1])
        U1n[0] += 1.0*einsum('Jab,IJbi->Iai', g.vv, U1nold[1])
        U1n[0] += -1.0*einsum('Jjb,aj,IJbi->Iai', g.ov, T1old, U1nold[1], optimize=True)
        U1n[0] += -1.0*einsum('Jjb,bi,IJaj->Iai', g.ov, T1old, U1nold[1], optimize=True)
        U1n[0] += 1.0*einsum('J,IJai->Iai', imds.gtm, U1nold[1], optimize=True)
        #U1n[0] += 1.0*einsum('Jjb,bj,IJai->Iai', g.ov, T1old, U1nold[1], optimize=True)
        if U2nold[0] is not None:
            U1n[0] += -1.0*einsum('Jjb,IJabji->Iai', g.ov, U2nold[1], optimize=True)
            #U1n[0] += -0.625*einsum('Jjb,IJabji->Iai', g.ov, U2nold[1], optimize=True)
            #U1n[0] += 0.375*einsum('Jjb, JIbaji->Iai', g.ov, U2nold[1], optimize=True)

    if U2nold[0] is not None:
        U1n[0] += 0.5*einsum('jkib,Iabkj->Iai', I.ooov, U2nold[0], optimize=True)
        U1n[0] += -0.5*einsum('jabc,Icbji->Iai', I.ovvv, U2nold[0], optimize=True)
        U1n[0] += -1.0*einsum('Jjb,J,Iabji->Iai', g.ov, Snold[0], U2nold[0], optimize=True)
        U1n[0] += -0.5*einsum('jkbc,aj,Icbki->Iai', I.oovv, T1old, U2nold[0], optimize=True)
        U1n[0] += -0.5*einsum('jkbc,ci,Iabkj->Iai', I.oovv, T1old, U2nold[0], optimize=True)
        U1n[0] += 1.0*einsum('jkbc,cj,Iabki->Iai', I.oovv, T1old, U2nold[0], optimize=True)

    if nfock1 > 1:
        U1n[0] += 1.0*einsum('Jai,IJ->Iai', g.vo, Snold[1])
        U1n[0] += -1.0*einsum('Jji,aj,IJ->Iai', g.oo, T1old, Snold[1])
        U1n[0] += 1.0*einsum('Jab,bi,IJ->Iai', g.vv, T1old, Snold[1])
        U1n[0] += -1.0*einsum('Jjb,abji,IJ->Iai', g.ov, T2old, Snold[1])
        U1n[0] += -1.0*einsum('Jai,IJ->Iai', imds.gT1T1ov, Snold[1])
        #U1n[0] += -1.0*einsum('Jjb,bi,aj,IJ->Iai', g.ov, T1old, T1old, Snold[1])

    #U11old += einsum('ai,ia -> ai', res_U11old, e_denom, optimize=True)

    if nfock2 < 2: return U1n

    # U12
    U1n[1] = numpy.zeros((nm, nm, nv, no))

    # similar to g * T - > U11
    U1n[1] += -2.0*einsum('Iji,Jaj->IJai', g.oo, U1nold[0])
    #U1n[1] += -1.0*einsum('Iji,Jaj->IJai', g.oo, U1nold[0])
    #U1n[1] += -1.0*einsum('Jji,Iaj->IJai', g.oo, U1nold[0])

    U1n[1] += 2.0*einsum('Iab,Jbi->IJai', g.vv, U1nold[0])
    #U1n[1] += 1.0*einsum('Iab,Jbi->IJai', g.vv, U1nold[0])
    #U1n[1] += 1.0*einsum('Jab,Ibi->IJai', g.vv, U1nold[0])

    # --------------------------------------------------------------------
    # this part is similar to U1n[0] from U1nold[0]
    U1n[1] += 2.0*einsum('I,JIai->IJai', w, U1nold[1])

    U1n[1] += 1.0*einsum('ab,IJbi->IJai', imds.Fvv, U1nold[1])
    U1n[1] -= einsum('ji,IJaj->IJai', imds.Foo, U1nold[1])

    # in Fvv
    #U1n[1] += 1.0*einsum('ab,IJbi->IJai', F.vv, U1nold[1])
    #U1n[1] += 1.0*einsum('Kab,K,IJbi->IJai', g.vv, Snold[0], U1nold[1], optimize = True)
    #U1n[1] += -1.0*einsum('jb,aj,IJbi->IJai', F.ov, T1old, U1nold[1], optimize = True)
    #U1n[1] += -1.0*einsum('Kjb,Kaj,IJbi->IJai', g.ov, U1nold[0], U1nold[1], optimize = True)
    #U1n[1] += -1.0*einsum('jabc,cj,IJbi->IJai', I.ovvv, T1old, U1nold[1], optimize = True)
    #U1n[1] += 0.5*einsum('jkbc,ackj,IJbi->IJai', I.oovv, T2old, U1nold[1], optimize = True)
    #U1n[1] += -1.0*einsum('jkbc,aj,ck,IJbi->IJai', I.oovv, T1old, T1old, U1nold[1], optimize = True)

    # this part is slightly different from U1nold[0] -> U1n[0],  since u1nold[1], U1nold[0] contraction has 2 possible ways
    gU11_vv = -1.0*einsum('Jjb,Iaj->JIab', g.ov, U1nold[0])
    U1n[1] += 2.0*einsum('KIab,JKbi->IJai', gU11_vv, U1nold[1], optimize = True)
    #U1n[1] += -1.0*einsum('Kjb,Iaj,JKbi->IJai', g.ov, U1nold[0], U1nold[1], optimize = True)
    #U1n[1] += -1.0*einsum('Kjb,Jaj,IKbi->IJai', g.ov, U1nold[0], U1nold[1], optimize = True)

    # in Foo
    #U1n[1] += -1.0*einsum('ji,IJaj->IJai', F.oo, U1nold[1])
    #U1n[1] += -1.0*einsum('Kji,K,IJaj->IJai', g.oo, Snold[0], U1nold[1], optimize = True)
    #U1n[1] += -1.0*einsum('jb,bi,IJaj->IJai', F.ov, T1old, U1nold[1], optimize = True)
    #U1n[1] += -1.0*einsum('Kjb,Kbi,IJaj->IJai', g.ov, U1nold[0], U1nold[1], optimize = True)
    #U1n[1] += 1.0*einsum('jkib,bj,IJak->IJai', I.ooov, T1old, U1nold[1], optimize = True)
    #U1n[1] += 0.5*einsum('jkbc,cbji,IJak->IJai', I.oovv, T2old, U1nold[1], optimize = True)
    #U1n[1] += -1.0*einsum('jkbc,ci,bj,IJak->IJai', I.oovv, T1old, T1old, U1nold[1], optimize = True)

    # this part is slightly different from U1nold[0] -> U1n[0],  since u1nold[1], U1nold[0] contraction has 2 possible ways
    gU11_oo = -1.0*einsum('Kjb,Ibi->KIij', g.ov, U1nold[0])
    U1n[1] += 2.0*einsum('KIij,JKaj->IJai', gU11_oo, U1nold[1], optimize = True)
    #U1n[1] += -1.0*einsum('Kjb,Ibi,JKaj->IJai', g.ov, U1nold[0], U1nold[1], optimize = True)
    #U1n[1] += -1.0*einsum('Kjb,Jbi,IKaj->IJai', g.ov, U1nold[0], U1nold[1], optimize = True)

    #----------------------------
    U1n[1] += einsum('jabi,IJbj->IJai', imds.Wovvo, U1nold[1])
    U1n[1] -= 0.5*einsum('ab,IJbi->IJai', Xvv, U1nold[1])
    U1n[1] -= 0.5*einsum('ji,IJaj->IJai', Xoo, U1nold[1])

    # in Wovvo
    #U1n[1] += -1.0*einsum('jaib,IJbj->IJai', I.ovov, U1nold[1], optimize = True)
    #U1n[1] += -1.0*einsum('jkib,aj,IJbk->IJai', I.ooov, T1old, U1nold[1], optimize = True)
    #U1n[1] += 1.0*einsum('jabc,ci,IJbj->IJai', I.ovvv, T1old, U1nold[1], optimize = True)
    #U1n[1] += 1.0*einsum('jkbc,acji,IJbk->IJai', I.oovv, T2old, U1nold[1], optimize = True)
    #U1n[1] += 1.0*einsum('jkbc,ci,aj,IJbk->IJai', I.oovv, T1old, T1old, U1nold[1], optimize = True)

    # in Xoo, Xvv
    #U1n[1] += -1.0*einsum('Kjb,aj,K,IJbi->IJai', g.ov, T1old, Snold[0], U1nold[1], optimize = True)
    #U1n[1] += -1.0*einsum('Kjb,bi,K,IJaj->IJai', g.ov, T1old, Snold[0], U1nold[1], optimize = True)

    # in Xmm
    U1n[1] += 2.0*einsum('IK,JKai->IJai', Xmm, U1nold[1])
    #U1n[1] += 1.0*einsum('Kjb,Ibj,JKai->IJai', g.ov, U1nold[0], U1nold[1], optimize = True)
    #U1n[1] += 1.0*einsum('Kjb,Jbj,IKai->IJai', g.ov, U1nold[0], U1nold[1], optimize = True)

    Xmmm =  1.0*einsum('Kjb,IJbj->IJK', g.ov, U1nold[1])
    U1n[1] += 1.0*einsum('Kai,IJK->IJai', U1nold[0], Xmmm)
    #U1n[1] += 1.0*einsum('Kjb,Kai,IJbj->IJai', g.ov, U1nold[0], U1nold[1], optimize = True)
    # --------------------------------------------------------------------

    # vv
    U1n[1] += -2.0*einsum('jb,Iaj,Jbi->IJai', F.ov, U1nold[0], U1nold[0], optimize = True)
    #U1n[1] += -1.0*einsum('jb,Iaj,Jbi->IJai', F.ov, U1nold[0], U1nold[0], optimize = True)
    #U1n[1] += -1.0*einsum('jb,Ibi,Jaj->IJai', F.ov, U1nold[0], U1nold[0], optimize = True)

    # vv
    U1n[1] += -2.0*einsum('Ijb,aj,Jbi->IJai', g.ov, T1old, U1nold[0], optimize = True)
    #U1n[1] += -1.0*einsum('Ijb,aj,Jbi->IJai', g.ov, T1old, U1nold[0], optimize = True)
    #U1n[1] += -1.0*einsum('Jjb,aj,Ibi->IJai', g.ov, T1old, U1nold[0], optimize = True)
    # oo
    U1n[1] += -2.0*einsum('Ijb,bi,Jaj->IJai', g.ov, T1old, U1nold[0], optimize = True)
    #U1n[1] += -1.0*einsum('Ijb,bi,Jaj->IJai', g.ov, T1old, U1nold[0], optimize = True)
    #U1n[1] += -1.0*einsum('Jjb,bi,Iaj->IJai', g.ov, T1old, U1nold[0], optimize = True)

    # vv
    U1n[1] += -1.0*einsum('jb,Iaj,Jbi->IJai', imds.gsov, U1nold[0], U1nold[0], optimize = True)
    #U1n[1] += -1.0*einsum('Kjb,K,Iaj,Jbi->IJai', g.ov, Snold[0], U1nold[0], U1nold[0], optimize = True)
    # oo
    U1n[1] += -1.0*einsum('jb,Ibi,Jaj->IJai', imds.gsov, U1nold[0], U1nold[0], optimize = True)
    #U1n[1] += -1.0*einsum('Kjb,K,Ibi,Jaj->IJai', g.ov, Snold[0], U1nold[0], U1nold[0], optimize = True)

    U1n[1] += -1.0*einsum('jkib,Iaj,Jbk->IJai', I.ooov, U1nold[0], U1nold[0], optimize = True)
    U1n[1] += 1.0*einsum('jkib,Ibj,Jak->IJai', I.ooov, U1nold[0], U1nold[0], optimize = True)

    # vv
    U1n[1] += -1.0*einsum('jabc,Icj,Jbi->IJai', I.ovvv, U1nold[0], U1nold[0], optimize = True)
    # ovvo
    U1n[1] += 1.0*einsum('jabc,Ici,Jbj->IJai', I.ovvv, U1nold[0], U1nold[0], optimize = True)

    # vovv
    U1n[1] += -1.0*einsum('jkbc,aj,Ick,Jbi->IJai', I.oovv, T1old, U1nold[0], U1nold[0], optimize = True)
    U1n[1] += 1.0*einsum('jkbc,aj,Ici,Jbk->IJai', I.oovv, T1old, U1nold[0], U1nold[0], optimize = True)
    # ovvo
    U1n[1] += -1.0*einsum('jkbc,ci,Ibj,Jak->IJai', I.oovv, T1old, U1nold[0], U1nold[0], optimize = True)
    U1n[1] += 1.0*einsum('jkbc,ci,Iaj,Jbk->IJai', I.oovv, T1old, U1nold[0], U1nold[0], optimize = True)

    # ov (kb) -> vv (ab)
    U1n[1] += 1.0*einsum('jkbc,cj,Iak,Jbi->IJai', I.oovv, T1old, U1nold[0], U1nold[0], optimize = True)
    # ov (kb) -> oo (ij)
    U1n[1] += 1.0*einsum('jkbc,cj,Ibi,Jak->IJai', I.oovv, T1old, U1nold[0], U1nold[0], optimize = True)

    if U2nold[0] is not None:
        tmpg = F.ov + imds.gsov
        U1n[1] += -1.0*einsum('jb,IJabji->IJai', tmpg, U2nold[1])
        #U1n[1] += -1.0*einsum('jb,IJabji->IJai', F.ov, U2nold[1])
        #U1n[1] += -1.0*einsum('Kjb,K,IJabji->IJai', g.ov, Snold[0], U2nold[1], optimize = True)

        U1n[1] += -2.0*einsum('Ijb,Jabji->IJai', g.ov, U2nold[0])
        #U1n[1] += -1.0*einsum('Ijb,Jabji->IJai', g.ov, U2nold[0])
        #U1n[1] += -1.0*einsum('Jjb,Iabji->IJai', g.ov, U2nold[0])

        U1n[1] += 0.5*einsum('jkib,IJabkj->IJai', I.ooov, U2nold[1], optimize = True)
        U1n[1] += -0.5*einsum('jabc,IJcbji->IJai', I.ovvv, U2nold[1], optimize = True)

        # vovv
        U1n[1] += -0.5*einsum('jkbc,aj,IJcbki->IJai', I.oovv, T1old, U2nold[1], optimize = True)
        # ovvo
        U1n[1] += -1.0*einsum('jkbc,ci,IJabkj->IJai', I.oovv, T1old, U2nold[1], optimize = True)
        #U1n[1] += -0.5*einsum('jkbc,ci,IJabkj->IJai', I.oovv, T1old, U2nold[1], optimize = True)
        #U1n[1] += -0.5*einsum('jkbc,ci,IJabkj->IJai', I.oovv, T1old, U2nold[1], optimize = True)
        # ov
        U1n[1] += 2.0*einsum('jkbc,cj,IJabki->IJai', I.oovv, T1old, U2nold[1], optimize = True)
        #U1n[1] += 1.0*einsum('jkbc,cj,IJabki->IJai', I.oovv, T1old, U2nold[1], optimize = True)
        #U1n[1] += 1.0*einsum('jkbc,cj,IJabki->IJai', I.oovv, T1old, U2nold[1], optimize = True)

        #vovv
        U1n[1] += -2.0*einsum('jkbc,Iaj,Jcbki->IJai', I.oovv, U1nold[0], U2nold[0], optimize = True)
        #U1n[1] += -0.5*einsum('jkbc,Iaj,Jcbki->IJai', I.oovv, U1nold[0], U2nold[0], optimize = True)
        #U1n[1] += -0.5*einsum('jkbc,Iaj,Jcbki->IJai', I.oovv, U1nold[0], U2nold[0], optimize = True)
        #U1n[1] += -0.5*einsum('jkbc,Jaj,Icbki->IJai', I.oovv, U1nold[0], U2nold[0], optimize = True)
        #U1n[1] += -0.5*einsum('jkbc,Jaj,Icbki->IJai', I.oovv, U1nold[0], U2nold[0], optimize = True)

        # ovvo
        U1n[1] += -2.0*einsum('jkbc,Ici,Jabkj->IJai', I.oovv, U1nold[0], U2nold[0], optimize = True)
        #U1n[1] += -0.5*einsum('jkbc,Ici,Jabkj->IJai', I.oovv, U1nold[0], U2nold[0], optimize = True)
        #U1n[1] += -0.5*einsum('jkbc,Ici,Jabkj->IJai', I.oovv, U1nold[0], U2nold[0], optimize = True)
        #U1n[1] += -0.5*einsum('jkbc,Jci,Iabkj->IJai', I.oovv, U1nold[0], U2nold[0], optimize = True)
        #U1n[1] += -0.5*einsum('jkbc,Jci,Iabkj->IJai', I.oovv, U1nold[0], U2nold[0], optimize = True)

        U1n[1] += 4.0*einsum('jkbc,Icj,Jabki->IJai', I.oovv, U1nold[0], U2nold[0], optimize = True)
        #U1n[1] += 1.0*einsum('jkbc,Icj,Jabki->IJai', I.oovv, U1nold[0], U2nold[0], optimize = True)
        #U1n[1] += 1.0*einsum('jkbc,Icj,Jabki->IJai', I.oovv, U1nold[0], U2nold[0], optimize = True)
        #U1n[1] += 1.0*einsum('jkbc,Jcj,Iabki->IJai', I.oovv, U1nold[0], U2nold[0], optimize = True)
        #U1n[1] += 1.0*einsum('jkbc,Jcj,Iabki->IJai', I.oovv, U1nold[0], U2nold[0], optimize = True)
        if nfock1 > 1:
            U1n[1] += -2.0*einsum('Kjb,IK,Jabji->IJai', g.ov, Snold[1], U2nold[0], optimize = True)
            #U1n[1] += -1.0*einsum('Kjb,IK,Jabji->IJai', g.ov, Snold[1], U2nold[0], optimize = True)
            #U1n[1] += -1.0*einsum('Kjb,JK,Iabji->IJai', g.ov, Snold[1], U2nold[0], optimize = True)

    if nfock1 > 1:
        U1n[1] += -2.0*einsum('Kji,IK,Jaj->IJai', g.oo, Snold[1], U1nold[0], optimize = True)
        #U1n[1] += -1.0*einsum('Kji,IK,Jaj->IJai', g.oo, Snold[1], U1nold[0], optimize = True)
        #U1n[1] += -1.0*einsum('Kji,JK,Iaj->IJai', g.oo, Snold[1], U1nold[0], optimize = True)

        U1n[1] += 2.0*einsum('Kab,IK,Jbi->IJai', g.vv, Snold[1], U1nold[0], optimize = True)
        #U1n[1] += 1.0*einsum('Kab,IK,Jbi->IJai', g.vv, Snold[1], U1nold[0], optimize = True)
        #U1n[1] += 1.0*einsum('Kab,JK,Ibi->IJai', g.vv, Snold[1], U1nold[0], optimize = True)
        # vv
        U1n[1] += -2.0*einsum('Kjb,aj,IK,Jbi->IJai', g.ov, T1old, Snold[1], U1nold[0], optimize = True)
        #U1n[1] += -1.0*einsum('Kjb,aj,IK,Jbi->IJai', g.ov, T1old, Snold[1], U1nold[0], optimize = True)
        #U1n[1] += -1.0*einsum('Kjb,aj,JK,Ibi->IJai', g.ov, T1old, Snold[1], U1nold[0], optimize = True)
        # oo
        U1n[1] += -2.0*einsum('Kjb,bi,IK,Jaj->IJai', g.ov, T1old, Snold[1], U1nold[0], optimize = True)
        #U1n[1] += -1.0*einsum('Kjb,bi,IK,Jaj->IJai', g.ov, T1old, Snold[1], U1nold[0], optimize = True)
        #U1n[1] += -1.0*einsum('Kjb,bi,JK,Iaj->IJai', g.ov, T1old, Snold[1], U1nold[0], optimize = True)

    return U1n


def qedccsd_U2n_opt(F, I, w, g, h, G, H, nfock1, nfock2, amps, imds):

    T1old, T2old, Snold, U1nold, U2nold = amps

    nvir = T2old.shape[0]
    nocc = T2old.shape[2]
    nm = w.shape[0]

    eps_occ = F.oo.diagonal()
    eps_vir = F.vv.diagonal()
    eps_vir_p_w = eps_vir + w

    #e_denom = 1 / (eps_occ.reshape(-1, 1, 1, 1) + eps_occ.reshape(-1, 1) - eps_vir.reshape(-1, 1, 1) - eps_vir_p_w)

    #if nm == 1: # if single mode, we use faster single mode code (reducing indexing overhead)
    #    return single_qedccsd_U2n(F, I, w, g, h, G, H, nfock1, nfock2, amps)

    U2n = [None]*nfock2
    if U2nold[0] is None: return U2n
    if nfock2 < 1: return U2n

    U2n[0] = numpy.zeros((nm, nvir, nvir, nocc, nocc))

    U2n[0] += -1.0*einsum('Ikc,ak,bcji->Iabij', g.ov, T1old, T2old, optimize = True)
    U2n[0] += 1.0*einsum('Ikc,bk,acji->Iabij', g.ov, T1old, T2old, optimize = True)
    U2n[0] += 1.0*einsum('Ikc,ci,bakj->Iabij', g.ov, T1old, T2old, optimize = True)
    U2n[0] += -1.0*einsum('Ikc,cj,baki->Iabij', g.ov, T1old, T2old, optimize = True)
    #
    U2n[0] += 1.0*einsum('Iki,bakj->Iabij', g.oo, T2old, optimize = True)
    U2n[0] += -1.0*einsum('Ikj,baki->Iabij', g.oo, T2old, optimize = True)

    U2n[0] += 1.0*einsum('Iac,bcji->Iabij', g.vv, T2old, optimize = True)
    U2n[0] += -1.0*einsum('Ibc,acji->Iabij', g.vv, T2old, optimize = True)

    U2n[0] += -1.0*einsum('kj,Ibaki->Iabij', F.oo, U2nold[0], optimize = True)
    U2n[0] += 1.0*einsum('ki,Ibakj->Iabij', F.oo, U2nold[0], optimize = True)

    U2n[0] += 1.0*einsum('ac,Ibcji->Iabij', F.vv, U2nold[0], optimize = True)
    U2n[0] += -1.0*einsum('bc,Iacji->Iabij', F.vv, U2nold[0], optimize = True)

    U2n[0] += 1.0*einsum('I,Ibaji->Iabij', w, U2nold[0], optimize = True)
    U2n[0] += -0.5*einsum('klji,Ibalk->Iabij', I.oooo, U2nold[0], optimize = True)

    U2n[0] += 1.0*einsum('kaic,Ibckj->Iabij', I.ovov, U2nold[0], optimize = True)
    U2n[0] += -1.0*einsum('kajc,Ibcki->Iabij', I.ovov, U2nold[0], optimize = True)
    U2n[0] += -1.0*einsum('kbic,Iackj->Iabij', I.ovov, U2nold[0], optimize = True)
    U2n[0] += 1.0*einsum('kbjc,Iacki->Iabij', I.ovov, U2nold[0], optimize = True)
    U2n[0] += -0.5*einsum('bacd,Idcji->Iabij', I.vvvv, U2nold[0], optimize = True)

    #vv
    U2n[0] += -1.0*einsum('kc,ak,Ibcji->Iabij', F.ov, T1old, U2nold[0], optimize = True)
    U2n[0] += 1.0*einsum('kc,bk,Iacji->Iabij', F.ov, T1old, U2nold[0], optimize = True)
    #oo
    U2n[0] += 1.0*einsum('kc,ci,Ibakj->Iabij', F.ov, T1old, U2nold[0], optimize = True)
    U2n[0] += -1.0*einsum('kc,cj,Ibaki->Iabij', F.ov, T1old, U2nold[0], optimize = True)


    U2n[0] += 1.0*einsum('Jki,J,Ibakj->Iabij', g.oo, Snold[0], U2nold[0], optimize = True)
    U2n[0] += -1.0*einsum('Jkj,J,Ibaki->Iabij', g.oo, Snold[0], U2nold[0], optimize = True)
    U2n[0] += 1.0*einsum('Jac,J,Ibcji->Iabij', g.vv, Snold[0], U2nold[0], optimize = True)
    U2n[0] += -1.0*einsum('Jbc,J,Iacji->Iabij', g.vv, Snold[0], U2nold[0], optimize = True)

    U2n[0] += -1.0*einsum('Jkc,Iak,Jbcji->Iabij', g.ov, U1nold[0], U2nold[0], optimize = True)
    U2n[0] += 1.0*einsum('Jkc,Ibk,Jacji->Iabij', g.ov, U1nold[0], U2nold[0], optimize = True)
    U2n[0] += 1.0*einsum('Jkc,Ici,Jbakj->Iabij', g.ov, U1nold[0], U2nold[0], optimize = True)
    U2n[0] += -1.0*einsum('Jkc,Icj,Jbaki->Iabij', g.ov, U1nold[0], U2nold[0], optimize = True)
    U2n[0] += 1.0*einsum('Jkc,Ick,Jbaji->Iabij', g.ov, U1nold[0], U2nold[0], optimize = True)
    U2n[0] += -1.0*einsum('Jkc,Jai,Ibckj->Iabij', g.ov, U1nold[0], U2nold[0], optimize = True)
    U2n[0] += 1.0*einsum('Jkc,Jaj,Ibcki->Iabij', g.ov, U1nold[0], U2nold[0], optimize = True)
    U2n[0] += -1.0*einsum('Jkc,Jak,Ibcji->Iabij', g.ov, U1nold[0], U2nold[0], optimize = True)
    U2n[0] += 1.0*einsum('Jkc,Jbi,Iackj->Iabij', g.ov, U1nold[0], U2nold[0], optimize = True)
    U2n[0] += -1.0*einsum('Jkc,Jbj,Iacki->Iabij', g.ov, U1nold[0], U2nold[0], optimize = True)
    U2n[0] += 1.0*einsum('Jkc,Jbk,Iacji->Iabij', g.ov, U1nold[0], U2nold[0], optimize = True)
    U2n[0] += 1.0*einsum('Jkc,Jci,Ibakj->Iabij', g.ov, U1nold[0], U2nold[0], optimize = True)
    U2n[0] += -1.0*einsum('Jkc,Jcj,Ibaki->Iabij', g.ov, U1nold[0], U2nold[0], optimize = True)

    U2n[0] += 1.0*einsum('klic,ak,Ibclj->Iabij', I.ooov, T1old, U2nold[0], optimize = True)
    U2n[0] += -1.0*einsum('klic,bk,Iaclj->Iabij', I.ooov, T1old, U2nold[0], optimize = True)
    U2n[0] += 0.5*einsum('klic,cj,Ibalk->Iabij', I.ooov, T1old, U2nold[0], optimize = True)
    U2n[0] += -1.0*einsum('klic,ck,Ibalj->Iabij', I.ooov, T1old, U2nold[0], optimize = True)

    U2n[0] += -1.0*einsum('kljc,ak,Ibcli->Iabij', I.ooov, T1old, U2nold[0], optimize = True)
    U2n[0] += 1.0*einsum('kljc,bk,Iacli->Iabij', I.ooov, T1old, U2nold[0], optimize = True)
    U2n[0] += -0.5*einsum('kljc,ci,Ibalk->Iabij', I.ooov, T1old, U2nold[0], optimize = True)
    U2n[0] += 1.0*einsum('kljc,ck,Ibali->Iabij', I.ooov, T1old, U2nold[0], optimize = True)

    U2n[0] += 0.5*einsum('kacd,bk,Idcji->Iabij', I.ovvv, T1old, U2nold[0], optimize = True)
    U2n[0] += -1.0*einsum('kacd,di,Ibckj->Iabij', I.ovvv, T1old, U2nold[0], optimize = True)
    U2n[0] += 1.0*einsum('kacd,dj,Ibcki->Iabij', I.ovvv, T1old, U2nold[0], optimize = True)
    U2n[0] += -1.0*einsum('kacd,dk,Ibcji->Iabij', I.ovvv, T1old, U2nold[0], optimize = True)

    U2n[0] += -0.5*einsum('kbcd,ak,Idcji->Iabij', I.ovvv, T1old, U2nold[0], optimize = True)
    U2n[0] += 1.0*einsum('kbcd,di,Iackj->Iabij', I.ovvv, T1old, U2nold[0], optimize = True)
    U2n[0] += -1.0*einsum('kbcd,dj,Iacki->Iabij', I.ovvv, T1old, U2nold[0], optimize = True)
    U2n[0] += 1.0*einsum('kbcd,dk,Iacji->Iabij', I.ovvv, T1old, U2nold[0], optimize = True)

    U2n[0] += 0.5*einsum('klcd,adji,Ibclk->Iabij', I.oovv, T2old, U2nold[0], optimize = True)
    U2n[0] += -1.0*einsum('klcd,adki,Ibclj->Iabij', I.oovv, T2old, U2nold[0], optimize = True)
    U2n[0] += 1.0*einsum('klcd,adkj,Ibcli->Iabij', I.oovv, T2old, U2nold[0], optimize = True)
    U2n[0] += 0.5*einsum('klcd,adlk,Ibcji->Iabij', I.oovv, T2old, U2nold[0], optimize = True)
    U2n[0] += -0.5*einsum('klcd,baki,Idclj->Iabij', I.oovv, T2old, U2nold[0], optimize = True)
    U2n[0] += 0.5*einsum('klcd,bakj,Idcli->Iabij', I.oovv, T2old, U2nold[0], optimize = True)
    U2n[0] += 0.25*einsum('klcd,balk,Idcji->Iabij', I.oovv, T2old, U2nold[0], optimize = True)
    U2n[0] += -0.5*einsum('klcd,bdji,Iaclk->Iabij', I.oovv, T2old, U2nold[0], optimize = True)
    U2n[0] += 1.0*einsum('klcd,bdki,Iaclj->Iabij', I.oovv, T2old, U2nold[0], optimize = True)
    U2n[0] += -1.0*einsum('klcd,bdkj,Iacli->Iabij', I.oovv, T2old, U2nold[0], optimize = True)
    U2n[0] += -0.5*einsum('klcd,bdlk,Iacji->Iabij', I.oovv, T2old, U2nold[0], optimize = True)
    U2n[0] += 0.25*einsum('klcd,dcji,Ibalk->Iabij', I.oovv, T2old, U2nold[0], optimize = True)
    U2n[0] += -0.5*einsum('klcd,dcki,Ibalj->Iabij', I.oovv, T2old, U2nold[0], optimize = True)
    U2n[0] += 0.5*einsum('klcd,dckj,Ibali->Iabij', I.oovv, T2old, U2nold[0], optimize = True)
    U2n[0] += -1.0*einsum('Jkc,ak,J,Ibcji->Iabij', g.ov, T1old, Snold[0], U2nold[0], optimize = True)
    U2n[0] += 1.0*einsum('Jkc,bk,J,Iacji->Iabij', g.ov, T1old, Snold[0], U2nold[0], optimize = True)
    U2n[0] += 1.0*einsum('Jkc,ci,J,Ibakj->Iabij', g.ov, T1old, Snold[0], U2nold[0], optimize = True)
    U2n[0] += -1.0*einsum('Jkc,cj,J,Ibaki->Iabij', g.ov, T1old, Snold[0], U2nold[0], optimize = True)
    U2n[0] += -1.0*einsum('klcd,ak,dl,Ibcji->Iabij', I.oovv, T1old, T1old, U2nold[0], optimize = True)
    U2n[0] += -0.5*einsum('klcd,bk,al,Idcji->Iabij', I.oovv, T1old, T1old, U2nold[0], optimize = True)
    U2n[0] += 1.0*einsum('klcd,bk,dl,Iacji->Iabij', I.oovv, T1old, T1old, U2nold[0], optimize = True)
    U2n[0] += -1.0*einsum('klcd,di,ak,Ibclj->Iabij', I.oovv, T1old, T1old, U2nold[0], optimize = True)
    U2n[0] += 1.0*einsum('klcd,di,bk,Iaclj->Iabij', I.oovv, T1old, T1old, U2nold[0], optimize = True)
    U2n[0] += -0.5*einsum('klcd,di,cj,Ibalk->Iabij', I.oovv, T1old, T1old, U2nold[0], optimize = True)
    U2n[0] += 1.0*einsum('klcd,di,ck,Ibalj->Iabij', I.oovv, T1old, T1old, U2nold[0], optimize = True)
    U2n[0] += 1.0*einsum('klcd,dj,ak,Ibcli->Iabij', I.oovv, T1old, T1old, U2nold[0], optimize = True)
    U2n[0] += -1.0*einsum('klcd,dj,bk,Iacli->Iabij', I.oovv, T1old, T1old, U2nold[0], optimize = True)
    U2n[0] += -1.0*einsum('klcd,dj,ck,Ibali->Iabij', I.oovv, T1old, T1old, U2nold[0], optimize = True)

    U2n[0] += -1.0*einsum('klji,ak,Ibl->Iabij', I.oooo, T1old, U1nold[0], optimize = True)
    U2n[0] += 1.0*einsum('klji,bk,Ial->Iabij', I.oooo, T1old, U1nold[0], optimize = True)

    U2n[0] += -1.0*einsum('kaji,Ibk->Iabij', I.ovoo, U1nold[0], optimize = True)
    U2n[0] += 1.0*einsum('kbji,Iak->Iabij', I.ovoo, U1nold[0], optimize = True)
    U2n[0] += -1.0*einsum('baic,Icj->Iabij', I.vvov, U1nold[0], optimize = True)
    U2n[0] += 1.0*einsum('bajc,Ici->Iabij', I.vvov, U1nold[0], optimize = True)


    U2n[0] += 1.0*einsum('kc,acji,Ibk->Iabij', F.ov, T2old, U1nold[0], optimize = True)
    U2n[0] += -1.0*einsum('kc,baki,Icj->Iabij', F.ov, T2old, U1nold[0], optimize = True)
    U2n[0] += 1.0*einsum('kc,bakj,Ici->Iabij', F.ov, T2old, U1nold[0], optimize = True)
    U2n[0] += -1.0*einsum('kc,bcji,Iak->Iabij', F.ov, T2old, U1nold[0], optimize = True)

    U2n[0] += -1.0*einsum('Jki,Iak,Jbj->Iabij', g.oo, U1nold[0], U1nold[0], optimize = True)
    U2n[0] += 1.0*einsum('Jki,Ibk,Jaj->Iabij', g.oo, U1nold[0], U1nold[0], optimize = True)
    U2n[0] += 1.0*einsum('Jkj,Iak,Jbi->Iabij', g.oo, U1nold[0], U1nold[0], optimize = True)
    U2n[0] += -1.0*einsum('Jkj,Ibk,Jai->Iabij', g.oo, U1nold[0], U1nold[0], optimize = True)
    U2n[0] += 1.0*einsum('Jac,Ici,Jbj->Iabij', g.vv, U1nold[0], U1nold[0], optimize = True)
    U2n[0] += -1.0*einsum('Jac,Icj,Jbi->Iabij', g.vv, U1nold[0], U1nold[0], optimize = True)
    U2n[0] += -1.0*einsum('Jbc,Ici,Jaj->Iabij', g.vv, U1nold[0], U1nold[0], optimize = True)
    U2n[0] += 1.0*einsum('Jbc,Icj,Jai->Iabij', g.vv, U1nold[0], U1nold[0], optimize = True)
    U2n[0] += 1.0*einsum('kaic,bk,Icj->Iabij', I.ovov, T1old, U1nold[0], optimize = True)
    U2n[0] += 1.0*einsum('kaic,cj,Ibk->Iabij', I.ovov, T1old, U1nold[0], optimize = True)
    U2n[0] += -1.0*einsum('kajc,bk,Ici->Iabij', I.ovov, T1old, U1nold[0], optimize = True)
    U2n[0] += -1.0*einsum('kajc,ci,Ibk->Iabij', I.ovov, T1old, U1nold[0], optimize = True)
    U2n[0] += -1.0*einsum('kbic,ak,Icj->Iabij', I.ovov, T1old, U1nold[0], optimize = True)
    U2n[0] += -1.0*einsum('kbic,cj,Iak->Iabij', I.ovov, T1old, U1nold[0], optimize = True)
    U2n[0] += 1.0*einsum('kbjc,ak,Ici->Iabij', I.ovov, T1old, U1nold[0], optimize = True)
    U2n[0] += 1.0*einsum('kbjc,ci,Iak->Iabij', I.ovov, T1old, U1nold[0], optimize = True)
    U2n[0] += 1.0*einsum('bacd,di,Icj->Iabij', I.vvvv, T1old, U1nold[0], optimize = True)
    U2n[0] += -1.0*einsum('bacd,dj,Ici->Iabij', I.vvvv, T1old, U1nold[0], optimize = True)

    U2n[0] += 1.0*einsum('klic,ackj,Ibl->Iabij', I.ooov, T2old, U1nold[0], optimize = True)
    U2n[0] += 1.0*einsum('klic,bakj,Icl->Iabij', I.ooov, T2old, U1nold[0], optimize = True)
    U2n[0] += 0.5*einsum('klic,balk,Icj->Iabij', I.ooov, T2old, U1nold[0], optimize = True)
    U2n[0] += -1.0*einsum('klic,bckj,Ial->Iabij', I.ooov, T2old, U1nold[0], optimize = True)

    U2n[0] += -1.0*einsum('kljc,acki,Ibl->Iabij', I.ooov, T2old, U1nold[0], optimize = True)
    U2n[0] += -1.0*einsum('kljc,baki,Icl->Iabij', I.ooov, T2old, U1nold[0], optimize = True)
    U2n[0] += -0.5*einsum('kljc,balk,Ici->Iabij', I.ooov, T2old, U1nold[0], optimize = True)
    U2n[0] += 1.0*einsum('kljc,bcki,Ial->Iabij', I.ooov, T2old, U1nold[0], optimize = True)

    U2n[0] += 1.0*einsum('kacd,bdji,Ick->Iabij', I.ovvv, T2old, U1nold[0], optimize = True)
    U2n[0] += -1.0*einsum('kacd,bdki,Icj->Iabij', I.ovvv, T2old, U1nold[0], optimize = True)
    U2n[0] += 1.0*einsum('kacd,bdkj,Ici->Iabij', I.ovvv, T2old, U1nold[0], optimize = True)
    U2n[0] += 0.5*einsum('kacd,dcji,Ibk->Iabij', I.ovvv, T2old, U1nold[0], optimize = True)

    U2n[0] += -1.0*einsum('kbcd,adji,Ick->Iabij', I.ovvv, T2old, U1nold[0], optimize = True)
    U2n[0] += 1.0*einsum('kbcd,adki,Icj->Iabij', I.ovvv, T2old, U1nold[0], optimize = True)
    U2n[0] += -1.0*einsum('kbcd,adkj,Ici->Iabij', I.ovvv, T2old, U1nold[0], optimize = True)
    U2n[0] += -0.5*einsum('kbcd,dcji,Iak->Iabij', I.ovvv, T2old, U1nold[0], optimize = True)

    U2n[0] += -1.0*einsum('Jkc,ak,Ici,Jbj->Iabij', g.ov, T1old, U1nold[0], U1nold[0], optimize = True)
    U2n[0] += 1.0*einsum('Jkc,ak,Icj,Jbi->Iabij', g.ov, T1old, U1nold[0], U1nold[0], optimize = True)
    U2n[0] += 1.0*einsum('Jkc,bk,Ici,Jaj->Iabij', g.ov, T1old, U1nold[0], U1nold[0], optimize = True)
    U2n[0] += -1.0*einsum('Jkc,bk,Icj,Jai->Iabij', g.ov, T1old, U1nold[0], U1nold[0], optimize = True)
    U2n[0] += -1.0*einsum('Jkc,ci,Iak,Jbj->Iabij', g.ov, T1old, U1nold[0], U1nold[0], optimize = True)
    U2n[0] += 1.0*einsum('Jkc,ci,Ibk,Jaj->Iabij', g.ov, T1old, U1nold[0], U1nold[0], optimize = True)
    U2n[0] += 1.0*einsum('Jkc,cj,Iak,Jbi->Iabij', g.ov, T1old, U1nold[0], U1nold[0], optimize = True)
    U2n[0] += -1.0*einsum('Jkc,cj,Ibk,Jai->Iabij', g.ov, T1old, U1nold[0], U1nold[0], optimize = True)
    U2n[0] += 1.0*einsum('Jkc,acji,J,Ibk->Iabij', g.ov, T2old, Snold[0], U1nold[0], optimize = True)
    U2n[0] += -1.0*einsum('Jkc,baki,J,Icj->Iabij', g.ov, T2old, Snold[0], U1nold[0], optimize = True)
    U2n[0] += 1.0*einsum('Jkc,bakj,J,Ici->Iabij', g.ov, T2old, Snold[0], U1nold[0], optimize = True)
    U2n[0] += -1.0*einsum('Jkc,bcji,J,Iak->Iabij', g.ov, T2old, Snold[0], U1nold[0], optimize = True)

    U2n[0] += -1.0*einsum('klic,bk,al,Icj->Iabij', I.ooov, T1old, T1old, U1nold[0], optimize = True)
    U2n[0] += 1.0*einsum('klic,cj,ak,Ibl->Iabij', I.ooov, T1old, T1old, U1nold[0], optimize = True)
    U2n[0] += -1.0*einsum('klic,cj,bk,Ial->Iabij', I.ooov, T1old, T1old, U1nold[0], optimize = True)
    U2n[0] += 1.0*einsum('kljc,bk,al,Ici->Iabij', I.ooov, T1old, T1old, U1nold[0], optimize = True)
    U2n[0] += -1.0*einsum('kljc,ci,ak,Ibl->Iabij', I.ooov, T1old, T1old, U1nold[0], optimize = True)
    U2n[0] += 1.0*einsum('kljc,ci,bk,Ial->Iabij', I.ooov, T1old, T1old, U1nold[0], optimize = True)

    U2n[0] += -1.0*einsum('kacd,di,bk,Icj->Iabij', I.ovvv, T1old, T1old, U1nold[0], optimize = True)
    U2n[0] += -1.0*einsum('kacd,di,cj,Ibk->Iabij', I.ovvv, T1old, T1old, U1nold[0], optimize = True)
    U2n[0] += 1.0*einsum('kacd,dj,bk,Ici->Iabij', I.ovvv, T1old, T1old, U1nold[0], optimize = True)
    U2n[0] += 1.0*einsum('kbcd,di,ak,Icj->Iabij', I.ovvv, T1old, T1old, U1nold[0], optimize = True)
    U2n[0] += 1.0*einsum('kbcd,di,cj,Iak->Iabij', I.ovvv, T1old, T1old, U1nold[0], optimize = True)
    U2n[0] += -1.0*einsum('kbcd,dj,ak,Ici->Iabij', I.ovvv, T1old, T1old, U1nold[0], optimize = True)
    U2n[0] += 1.0*einsum('klcd,ak,bdji,Icl->Iabij', I.oovv, T1old, T2old, U1nold[0], optimize = True)
    U2n[0] += -1.0*einsum('klcd,ak,bdli,Icj->Iabij', I.oovv, T1old, T2old, U1nold[0], optimize = True)
    U2n[0] += 1.0*einsum('klcd,ak,bdlj,Ici->Iabij', I.oovv, T1old, T2old, U1nold[0], optimize = True)
    U2n[0] += 0.5*einsum('klcd,ak,dcji,Ibl->Iabij', I.oovv, T1old, T2old, U1nold[0], optimize = True)
    U2n[0] += -1.0*einsum('klcd,bk,adji,Icl->Iabij', I.oovv, T1old, T2old, U1nold[0], optimize = True)
    U2n[0] += 1.0*einsum('klcd,bk,adli,Icj->Iabij', I.oovv, T1old, T2old, U1nold[0], optimize = True)
    U2n[0] += -1.0*einsum('klcd,bk,adlj,Ici->Iabij', I.oovv, T1old, T2old, U1nold[0], optimize = True)
    U2n[0] += -0.5*einsum('klcd,bk,dcji,Ial->Iabij', I.oovv, T1old, T2old, U1nold[0], optimize = True)
    U2n[0] += -1.0*einsum('klcd,di,ackj,Ibl->Iabij', I.oovv, T1old, T2old, U1nold[0], optimize = True)
    U2n[0] += -1.0*einsum('klcd,di,bakj,Icl->Iabij', I.oovv, T1old, T2old, U1nold[0], optimize = True)
    U2n[0] += -0.5*einsum('klcd,di,balk,Icj->Iabij', I.oovv, T1old, T2old, U1nold[0], optimize = True)
    U2n[0] += 1.0*einsum('klcd,di,bckj,Ial->Iabij', I.oovv, T1old, T2old, U1nold[0], optimize = True)
    U2n[0] += 1.0*einsum('klcd,dj,acki,Ibl->Iabij', I.oovv, T1old, T2old, U1nold[0], optimize = True)
    U2n[0] += 1.0*einsum('klcd,dj,baki,Icl->Iabij', I.oovv, T1old, T2old, U1nold[0], optimize = True)
    U2n[0] += 0.5*einsum('klcd,dj,balk,Ici->Iabij', I.oovv, T1old, T2old, U1nold[0], optimize = True)
    U2n[0] += -1.0*einsum('klcd,dj,bcki,Ial->Iabij', I.oovv, T1old, T2old, U1nold[0], optimize = True)
    U2n[0] += -1.0*einsum('klcd,dk,acji,Ibl->Iabij', I.oovv, T1old, T2old, U1nold[0], optimize = True)
    U2n[0] += 1.0*einsum('klcd,dk,bali,Icj->Iabij', I.oovv, T1old, T2old, U1nold[0], optimize = True)
    U2n[0] += -1.0*einsum('klcd,dk,balj,Ici->Iabij', I.oovv, T1old, T2old, U1nold[0], optimize = True)
    U2n[0] += 1.0*einsum('klcd,dk,bcji,Ial->Iabij', I.oovv, T1old, T2old, U1nold[0], optimize = True)
    U2n[0] += 1.0*einsum('klcd,di,bk,al,Icj->Iabij', I.oovv, T1old, T1old, T1old, U1nold[0], optimize = True)
    U2n[0] += -1.0*einsum('klcd,di,cj,ak,Ibl->Iabij', I.oovv, T1old, T1old, T1old, U1nold[0], optimize = True)
    U2n[0] += 1.0*einsum('klcd,di,cj,bk,Ial->Iabij', I.oovv, T1old, T1old, T1old, U1nold[0], optimize = True)
    U2n[0] += -1.0*einsum('klcd,dj,bk,al,Ici->Iabij', I.oovv, T1old, T1old, T1old, U1nold[0], optimize = True)

    if nfock1 > 1:
        U2n[0] += 1.0*einsum('Jki,bakj,IJ->Iabij', g.oo, T2old, Snold[1], optimize = True)
        U2n[0] += -1.0*einsum('Jkj,baki,IJ->Iabij', g.oo, T2old, Snold[1], optimize = True)
        U2n[0] += 1.0*einsum('Jac,bcji,IJ->Iabij', g.vv, T2old, Snold[1], optimize = True)
        U2n[0] += -1.0*einsum('Jbc,acji,IJ->Iabij', g.vv, T2old, Snold[1], optimize = True)
        U2n[0] += -1.0*einsum('Jkc,ak,bcji,IJ->Iabij', g.ov, T1old, T2old, Snold[1], optimize = True)
        U2n[0] += 1.0*einsum('Jkc,bk,acji,IJ->Iabij', g.ov, T1old, T2old, Snold[1], optimize = True)
        U2n[0] += 1.0*einsum('Jkc,ci,bakj,IJ->Iabij', g.ov, T1old, T2old, Snold[1], optimize = True)
        U2n[0] += -1.0*einsum('Jkc,cj,baki,IJ->Iabij', g.ov, T1old, T2old, Snold[1], optimize = True)

    if nfock2 > 1:
        gvoU12_oovv = einsum('Jai,IJbj->Iabij', g.vo, U1nold[1])
        U2n[0] += gvoU12_oovv
        U2n[0] -= gvoU12_oovv.transpose((0,2,1,3,4))
        U2n[0] -= gvoU12_oovv.transpose((0,1,2,4,3))
        U2n[0] += gvoU12_oovv.transpose((0,2,1,4,3))
        gvoU12_oovv= None
        #U2n[0] += 1.0*einsum('Jai,IJbj->Iabij', g.vo, U1nold[1], optimize = True)
        #U2n[0] += -1.0*einsum('Jaj,IJbi->Iabij', g.vo, U1nold[1], optimize = True)
        #U2n[0] += -1.0*einsum('Jbi,IJaj->Iabij', g.vo, U1nold[1], optimize = True)
        #U2n[0] += 1.0*einsum('Jbj,IJai->Iabij', g.vo, U1nold[1], optimize = True)

        gtov = einsum('xac,ci->xai', g.vv, T1old)
        gtov -= einsum('xki,ak->xai', g.oo, T1old)
        temp_abij = einsum('xai,xybj->xabij', gtov, U1nold[1])
        #T2temp = - einsum('bi,aj->abij', T1old, T1old)
        T2temp = T2old - einsum('bi,aj->abij', T1old, T1old)
        Wmvo = einsum('xkc,acik->xai', g.ov, T2temp)
        temp_abij += einsum('xai,xybj->xabij', Wmvo, U1nold[1])
        U2n[0] += temp_abij
        U2n[0] -= temp_abij.transpose((0,2,1,3,4))
        U2n[0] -= temp_abij.transpose((0,1,2,4,3))
        U2n[0] += temp_abij.transpose((0,2,1,4,3))
        temp_abij = None

        #U2n[0] += -1.0*einsum('Jki,ak,IJbj->Iabij', g.oo, T1old, U1nold[1], optimize = True)
        #U2n[0] += 1.0*einsum('Jki,bk,IJaj->Iabij', g.oo, T1old, U1nold[1], optimize = True)
        #U2n[0] += 1.0*einsum('Jkj,ak,IJbi->Iabij', g.oo, T1old, U1nold[1], optimize = True)
        #U2n[0] += -1.0*einsum('Jkj,bk,IJai->Iabij', g.oo, T1old, U1nold[1], optimize = True)
        #U2n[0] += 1.0*einsum('Jac,ci,IJbj->Iabij', g.vv, T1old, U1nold[1], optimize = True)
        #U2n[0] += -1.0*einsum('Jac,cj,IJbi->Iabij', g.vv, T1old, U1nold[1], optimize = True)
        #U2n[0] += -1.0*einsum('Jbc,ci,IJaj->Iabij', g.vv, T1old, U1nold[1], optimize = True)
        #U2n[0] += 1.0*einsum('Jbc,cj,IJai->Iabij', g.vv, T1old, U1nold[1], optimize = True)
        #U2n[0] += -1.0*einsum('Jkc,acki,IJbj->Iabij', g.ov, T2old, U1nold[1], optimize = True)
        #U2n[0] += 1.0*einsum('Jkc,ackj,IJbi->Iabij', g.ov, T2old, U1nold[1], optimize = True)
        U2n[0] += 1.0*einsum('Jkc,acji,IJbk->Iabij', g.ov, T2old, U1nold[1], optimize = True)
        U2n[0] += -1.0*einsum('Jkc,baki,IJcj->Iabij', g.ov, T2old, U1nold[1], optimize = True)
        U2n[0] += 1.0*einsum('Jkc,bakj,IJci->Iabij', g.ov, T2old, U1nold[1], optimize = True)
        U2n[0] += -1.0*einsum('Jkc,bcji,IJak->Iabij', g.ov, T2old, U1nold[1], optimize = True)
        #U2n[0] += 1.0*einsum('Jkc,bcki,IJaj->Iabij', g.ov, T2old, U1nold[1], optimize = True)
        #U2n[0] += -1.0*einsum('Jkc,bckj,IJai->Iabij', g.ov, T2old, U1nold[1], optimize = True)
        #U2n[0] += -1.0*einsum('Jkc,ci,ak,IJbj->Iabij', g.ov, T1old, T1old, U1nold[1], optimize = True)
        #U2n[0] += 1.0*einsum('Jkc,ci,bk,IJaj->Iabij', g.ov, T1old, T1old, U1nold[1], optimize = True)
        #U2n[0] += 1.0*einsum('Jkc,cj,ak,IJbi->Iabij', g.ov, T1old, T1old, U1nold[1], optimize = True)
        #U2n[0] += -1.0*einsum('Jkc,cj,bk,IJai->Iabij', g.ov, T1old, T1old, U1nold[1], optimize = True)

        U2n[0] += 1.0*einsum('Jki,IJbakj->Iabij', g.oo, U2nold[1], optimize = True)
        U2n[0] += -1.0*einsum('Jkj,IJbaki->Iabij', g.oo, U2nold[1], optimize = True)
        U2n[0] += 1.0*einsum('Jac,IJbcji->Iabij', g.vv, U2nold[1], optimize = True)
        U2n[0] += -1.0*einsum('Jbc,IJacji->Iabij', g.vv, U2nold[1], optimize = True)

        U2n[0] += -1.0*einsum('Jkc,ak,IJbcji->Iabij', g.ov, T1old, U2nold[1], optimize = True)
        U2n[0] += 1.0*einsum('Jkc,bk,IJacji->Iabij', g.ov, T1old, U2nold[1], optimize = True)
        U2n[0] += 1.0*einsum('Jkc,ci,IJbakj->Iabij', g.ov, T1old, U2nold[1], optimize = True)
        U2n[0] += -1.0*einsum('Jkc,cj,IJbaki->Iabij', g.ov, T1old, U2nold[1], optimize = True)
        U2n[0] += 1.0*einsum('Jkc,ck,IJbaji->Iabij', g.ov, T1old, U2nold[1], optimize = True)

    #U21old += einsum('abij,iajb -> abij', res_U21old, e_denom, optimize=True)

    if nfock2 < 2: return U2n

    eps_occ = F.oo.diagonal()
    eps_vir = F.vv.diagonal()
    eps_vir_p_2w = eps_vir + 2.0 * w

    #e_denom = 1 / (eps_occ.reshape(-1, 1, 1, 1) + eps_occ.reshape(-1, 1) - eps_vir.reshape(-1, 1, 1) - eps_vir_p_2w)

    U2n[1] = numpy.zeros((nm, nm, nvir, nvir, nocc, nocc))

    # works for single mode at this moment
    U22 = numpy.zeros((nvir, nvir, nocc, nocc))
    U11old = U1nold[0][0]   # single mode
    U12old = U1nold[1][0,0] # single mode
    U21old = U2nold[0][0]   # single mode
    U22old = U2nold[1][0,0] # single mode
    S1old = Snold[0][0]     # single mode
    if nfock1 > 1:
        S2old = Snold[1][0,0]   # single mode

    U22 += 1.0 * einsum('ki,bakj->abij', F.oo, U22old, optimize = True)
    U22 += -1.0 * einsum('kj,baki->abij', F.oo, U22old, optimize = True)
    U22 += -1.0 * einsum('bc,acji->abij', F.vv, U22old, optimize = True)
    U22 += 1.0 * einsum('ac,bcji->abij', F.vv, U22old, optimize = True)

    U22 += 1.0 * einsum('kbji,ak->abij', I.ovoo, U12old, optimize = True)
    U22 += -1.0 * einsum('kaji,bk->abij', I.ovoo, U12old, optimize = True)
    U22 += -1.0 * einsum('baic,cj->abij', I.vvov, U12old, optimize = True)
    U22 += 1.0 * einsum('bajc,ci->abij', I.vvov, U12old, optimize = True)
    U22 += 2.0 * einsum('J,IJbaji->abij', w, U2nold[1], optimize = True) # IJ
    #U22 += 1.0 * einsum('J,IJbaji->abij', w, U2nold[1], optimize = True) # IJ
    #U22 += 1.0 * einsum('I,IJbaji->abij', w, U2nold[1], optimize = True) # IJ
    U22 += 1.0 * einsum('ki,bakj->abij', g.oo[0], U21old, optimize = True)
    U22 += -1.0 * einsum('kj,baki->abij', g.oo[0], U21old, optimize = True)
    U22 += 1.0 * einsum('ki,bakj->abij', g.oo[0], U21old, optimize = True)
    U22 += -1.0 * einsum('kj,baki->abij', g.oo[0], U21old, optimize = True)
    U22 += -1.0 * einsum('bc,acji->abij', g.vv[0], U21old, optimize = True)
    U22 += 1.0 * einsum('ac,bcji->abij', g.vv[0], U21old, optimize = True)
    U22 += -1.0 * einsum('bc,acji->abij', g.vv[0], U21old, optimize = True)
    U22 += 1.0 * einsum('ac,bcji->abij', g.vv[0], U21old, optimize = True)

    U22 += -0.5 * einsum('klji,balk->abij', I.oooo, U22old, optimize = True)
    U22 += -1.0 * einsum('kbic,ackj->abij', I.ovov, U22old, optimize = True)
    U22 += 1.0 * einsum('kaic,bckj->abij', I.ovov, U22old, optimize = True)
    U22 += 1.0 * einsum('kbjc,acki->abij', I.ovov, U22old, optimize = True)
    U22 += -1.0 * einsum('kajc,bcki->abij', I.ovov, U22old, optimize = True)
    U22 += -0.5 * einsum('bacd,dcji->abij', I.vvvv, U22old, optimize = True)
    U22 += 1.0 * einsum('kc,ci,bakj->abij', F.ov, T1old, U22old, optimize = True)
    U22 += -1.0 * einsum('kc,cj,baki->abij', F.ov, T1old, U22old, optimize = True)
    U22 += 1.0 * einsum('kc,bk,acji->abij', F.ov, T1old, U22old, optimize = True)
    U22 += -1.0 * einsum('kc,ak,bcji->abij', F.ov, T1old, U22old, optimize = True)

    U22 += -1.0 * einsum('kc,bcji,ak->abij', F.ov, T2old, U12old, optimize = True)
    U22 += 1.0 * einsum('kc,acji,bk->abij', F.ov, T2old, U12old, optimize = True)
    U22 += -1.0 * einsum('kc,baki,cj->abij', F.ov, T2old, U12old, optimize = True)
    U22 += 1.0 * einsum('kc,bakj,ci->abij', F.ov, T2old, U12old, optimize = True)
    U22 += 1.0 * einsum('klji,bk,al->abij', I.oooo, T1old, U12old, optimize = True)
    U22 += -1.0 * einsum('klji,ak,bl->abij', I.oooo, T1old, U12old, optimize = True)
    U22 += -1.0 * einsum('kbic,cj,ak->abij', I.ovov, T1old, U12old, optimize = True)
    U22 += 1.0 * einsum('kaic,cj,bk->abij', I.ovov, T1old, U12old, optimize = True)
    U22 += -1.0 * einsum('kbic,ak,cj->abij', I.ovov, T1old, U12old, optimize = True)
    U22 += 1.0 * einsum('kaic,bk,cj->abij', I.ovov, T1old, U12old, optimize = True)
    U22 += 1.0 * einsum('kbjc,ci,ak->abij', I.ovov, T1old, U12old, optimize = True)
    U22 += -1.0 * einsum('kajc,ci,bk->abij', I.ovov, T1old, U12old, optimize = True)
    U22 += 1.0 * einsum('kbjc,ak,ci->abij', I.ovov, T1old, U12old, optimize = True)
    U22 += -1.0 * einsum('kajc,bk,ci->abij', I.ovov, T1old, U12old, optimize = True)
    U22 += 1.0 * einsum('bacd,di,cj->abij', I.vvvv, T1old, U12old, optimize = True)
    U22 += -1.0 * einsum('bacd,dj,ci->abij', I.vvvv, T1old, U12old, optimize = True)

    U22 += 1.0 * S1old * einsum('ki,bakj->abij', g.oo[0], U22old, optimize = True)
    U22 += -1.0 * S1old * einsum('kj,baki->abij', g.oo[0], U22old, optimize = True)

    if nfock1 > 1:
        U22 += 1.0 * S2old * einsum('ki,bakj->abij', g.oo[0], U21old, optimize = True)
        U22 += -1.0 * S2old * einsum('kj,baki->abij', g.oo[0], U21old, optimize = True)
        U22 += 1.0 * S2old * einsum('ki,bakj->abij', g.oo[0], U21old, optimize = True)
        U22 += -1.0 * S2old * einsum('kj,baki->abij', g.oo[0], U21old, optimize = True)
        U22 += -1.0 * S2old * einsum('bc,acji->abij', g.vv[0], U21old, optimize = True)
        U22 += 1.0 * S2old * einsum('ac,bcji->abij', g.vv[0], U21old, optimize = True)
        U22 += -1.0 * S2old * einsum('bc,acji->abij', g.vv[0], U21old, optimize = True)
        U22 += 1.0 * S2old * einsum('ac,bcji->abij', g.vv[0], U21old, optimize = True)
        U22 += 1.0 * S2old * einsum('kc,ci,bakj->abij', g.ov[0], T1old, U21old, optimize = True)
        U22 += -1.0 * S2old * einsum('kc,cj,baki->abij', g.ov[0], T1old, U21old, optimize = True)
        U22 += 1.0 * S2old * einsum('kc,bk,acji->abij', g.ov[0], T1old, U21old, optimize = True)
        U22 += -1.0 * S2old * einsum('kc,ak,bcji->abij', g.ov[0], T1old, U21old, optimize = True)
        U22 += 1.0 * S2old * einsum('kc,ci,bakj->abij', g.ov[0], T1old, U21old, optimize = True)
        U22 += -1.0 * S2old * einsum('kc,cj,baki->abij', g.ov[0], T1old, U21old, optimize = True)
        U22 += 1.0 * S2old * einsum('kc,bk,acji->abij', g.ov[0], T1old, U21old, optimize = True)
        U22 += -1.0 * S2old * einsum('kc,ak,bcji->abij', g.ov[0], T1old, U21old, optimize = True)
        U22 += -1.0 * S2old * einsum('kc,bcji,ak->abij', g.ov[0], T2old, U11old, optimize = True)
        U22 += 1.0 * S2old * einsum('kc,acji,bk->abij', g.ov[0], T2old, U11old, optimize = True)
        U22 += -1.0 * S2old * einsum('kc,baki,cj->abij', g.ov[0], T2old, U11old, optimize = True)
        U22 += 1.0 * S2old * einsum('kc,bakj,ci->abij', g.ov[0], T2old, U11old, optimize = True)
        U22 += -1.0 * S2old * einsum('kc,bcji,ak->abij', g.ov[0], T2old, U11old, optimize = True)
        U22 += 1.0 * S2old * einsum('kc,acji,bk->abij', g.ov[0], T2old, U11old, optimize = True)
        U22 += -1.0 * S2old * einsum('kc,baki,cj->abij', g.ov[0], T2old, U11old, optimize = True)
        U22 += 1.0 * S2old * einsum('kc,bakj,ci->abij', g.ov[0], T2old, U11old, optimize = True)
    U22 += 1.0 * einsum('kc,ci,bakj->abij', g.ov[0], T1old, U21old, optimize = True)
    U22 += -1.0 * einsum('kc,cj,baki->abij', g.ov[0], T1old, U21old, optimize = True)
    U22 += 1.0 * einsum('kc,bk,acji->abij', g.ov[0], T1old, U21old, optimize = True)
    U22 += -1.0 * einsum('kc,ak,bcji->abij', g.ov[0], T1old, U21old, optimize = True)
    U22 += 1.0 * einsum('kc,ci,bakj->abij', g.ov[0], T1old, U21old, optimize = True)
    U22 += -1.0 * einsum('kc,cj,baki->abij', g.ov[0], T1old, U21old, optimize = True)
    U22 += 1.0 * einsum('kc,bk,acji->abij', g.ov[0], T1old, U21old, optimize = True)
    U22 += -1.0 * einsum('kc,ak,bcji->abij', g.ov[0], T1old, U21old, optimize = True)
    U22 += -1.0 * einsum('kc,bcji,ak->abij', g.ov[0], T2old, U11old, optimize = True)
    U22 += 1.0 * einsum('kc,acji,bk->abij', g.ov[0], T2old, U11old, optimize = True)
    U22 += -1.0 * einsum('kc,baki,cj->abij', g.ov[0], T2old, U11old, optimize = True)
    U22 += 1.0 * einsum('kc,bakj,ci->abij', g.ov[0], T2old, U11old, optimize = True)
    U22 += -1.0 * einsum('kc,bcji,ak->abij', g.ov[0], T2old, U11old, optimize = True)
    U22 += 1.0 * einsum('kc,acji,bk->abij', g.ov[0], T2old, U11old, optimize = True)
    U22 += -1.0 * einsum('kc,baki,cj->abij', g.ov[0], T2old, U11old, optimize = True)
    U22 += 1.0 * einsum('kc,bakj,ci->abij', g.ov[0], T2old, U11old, optimize = True)

    U22 += -1.0 * S1old * einsum('bc,acji->abij', g.vv[0], U22old, optimize = True)
    U22 += 1.0 * S1old * einsum('ac,bcji->abij', g.vv[0], U22old, optimize = True)
    U22 += 0.5 * einsum('klic,cj,balk->abij', I.ooov, T1old, U22old, optimize = True)
    U22 += -1.0 * einsum('klic,bk,aclj->abij', I.ooov, T1old, U22old, optimize = True)
    U22 += 1.0 * einsum('klic,ak,bclj->abij', I.ooov, T1old, U22old, optimize = True)
    U22 += -1.0 * einsum('klic,ck,balj->abij', I.ooov, T1old, U22old, optimize = True)
    U22 += -0.5 * einsum('kljc,ci,balk->abij', I.ooov, T1old, U22old, optimize = True)
    U22 += 1.0 * einsum('kljc,bk,acli->abij', I.ooov, T1old, U22old, optimize = True)
    U22 += -1.0 * einsum('kljc,ak,bcli->abij', I.ooov, T1old, U22old, optimize = True)
    U22 += 1.0 * einsum('kljc,ck,bali->abij', I.ooov, T1old, U22old, optimize = True)

    U22 += 1.0 * einsum('klic,bakj,cl->abij', I.ooov, T2old, U12old, optimize = True)
    U22 += -1.0 * einsum('klic,bckj,al->abij', I.ooov, T2old, U12old, optimize = True)
    U22 += 1.0 * einsum('klic,ackj,bl->abij', I.ooov, T2old, U12old, optimize = True)
    U22 += 0.5 * einsum('klic,balk,cj->abij', I.ooov, T2old, U12old, optimize = True)
    U22 += -1.0 * einsum('kljc,baki,cl->abij', I.ooov, T2old, U12old, optimize = True)
    U22 += 1.0 * einsum('kljc,bcki,al->abij', I.ooov, T2old, U12old, optimize = True)
    U22 += -1.0 * einsum('kljc,acki,bl->abij', I.ooov, T2old, U12old, optimize = True)
    U22 += -0.5 * einsum('kljc,balk,ci->abij', I.ooov, T2old, U12old, optimize = True)

    U22 += 1.0 * einsum('kbcd,di,ackj->abij', I.ovvv, T1old, U22old, optimize = True)
    U22 += -1.0 * einsum('kacd,di,bckj->abij', I.ovvv, T1old, U22old, optimize = True)
    U22 += -1.0 * einsum('kbcd,dj,acki->abij', I.ovvv, T1old, U22old, optimize = True)
    U22 += 1.0 * einsum('kacd,dj,bcki->abij', I.ovvv, T1old, U22old, optimize = True)
    U22 += -0.5 * einsum('kbcd,ak,dcji->abij', I.ovvv, T1old, U22old, optimize = True)
    U22 += 1.0 * einsum('kbcd,dk,acji->abij', I.ovvv, T1old, U22old, optimize = True)
    U22 += 0.5 * einsum('kacd,bk,dcji->abij', I.ovvv, T1old, U22old, optimize = True)
    U22 += -1.0 * einsum('kacd,dk,bcji->abij', I.ovvv, T1old, U22old, optimize = True)

    U22 += -1.0 * einsum('kbcd,adji,ck->abij', I.ovvv, T2old, U12old, optimize = True)
    U22 += -0.5 * einsum('kbcd,dcji,ak->abij', I.ovvv, T2old, U12old, optimize = True)
    U22 += 1.0 * einsum('kacd,bdji,ck->abij', I.ovvv, T2old, U12old, optimize = True)
    U22 += 0.5 * einsum('kacd,dcji,bk->abij', I.ovvv, T2old, U12old, optimize = True)
    U22 += 1.0 * einsum('kbcd,adki,cj->abij', I.ovvv, T2old, U12old, optimize = True)
    U22 += -1.0 * einsum('kacd,bdki,cj->abij', I.ovvv, T2old, U12old, optimize = True)
    U22 += -1.0 * einsum('kbcd,adkj,ci->abij', I.ovvv, T2old, U12old, optimize = True)
    U22 += 1.0 * einsum('kacd,bdkj,ci->abij', I.ovvv, T2old, U12old, optimize = True)

    U22 += -0.5 * einsum('klcd,bdji,aclk->abij', I.oovv, T2old, U22old, optimize = True)
    U22 += 0.5 * einsum('klcd,adji,bclk->abij', I.oovv, T2old, U22old, optimize = True)
    U22 += 0.25 * einsum('klcd,dcji,balk->abij', I.oovv, T2old, U22old, optimize = True)
    U22 += -0.5 * einsum('klcd,baki,dclj->abij', I.oovv, T2old, U22old, optimize = True)
    U22 += 1.0 * einsum('klcd,bdki,aclj->abij', I.oovv, T2old, U22old, optimize = True)
    U22 += -1.0 * einsum('klcd,adki,bclj->abij', I.oovv, T2old, U22old, optimize = True)
    U22 += -0.5 * einsum('klcd,dcki,balj->abij', I.oovv, T2old, U22old, optimize = True)
    U22 += 0.5 * einsum('klcd,bakj,dcli->abij', I.oovv, T2old, U22old, optimize = True)
    U22 += -1.0 * einsum('klcd,bdkj,acli->abij', I.oovv, T2old, U22old, optimize = True)
    U22 += 1.0 * einsum('klcd,adkj,bcli->abij', I.oovv, T2old, U22old, optimize = True)
    U22 += 0.5 * einsum('klcd,dckj,bali->abij', I.oovv, T2old, U22old, optimize = True)
    U22 += 0.25 * einsum('klcd,balk,dcji->abij', I.oovv, T2old, U22old, optimize = True)
    U22 += -0.5 * einsum('klcd,bdlk,acji->abij', I.oovv, T2old, U22old, optimize = True)
    U22 += 0.5 * einsum('klcd,adlk,bcji->abij', I.oovv, T2old, U22old, optimize = True)

    U22 += 2.0 * einsum('kc,ci,bakj->abij', F.ov, U11old, U21old, optimize = True)
    U22 += -2.0 * einsum('kc,cj,baki->abij', F.ov, U11old, U21old, optimize = True)
    U22 += 2.0 * einsum('kc,bk,acji->abij', F.ov, U11old, U21old, optimize = True)
    U22 += -2.0 * einsum('kc,ak,bcji->abij', F.ov, U11old, U21old, optimize = True)
    U22 += 2.0 * einsum('klji,bk,al->abij', I.oooo, U11old, U11old, optimize = True)
    U22 += -2.0 * einsum('kbic,cj,ak->abij', I.ovov, U11old, U11old, optimize = True)
    U22 += 2.0 * einsum('kaic,cj,bk->abij', I.ovov, U11old, U11old, optimize = True)
    U22 += 2.0 * einsum('kbjc,ci,ak->abij', I.ovov, U11old, U11old, optimize = True)
    U22 += -2.0 * einsum('kajc,ci,bk->abij', I.ovov, U11old, U11old, optimize = True)
    U22 += 2.0 * einsum('bacd,di,cj->abij', I.vvvv, U11old, U11old, optimize = True)
    U22 += 1.0 * einsum('ki,bk,aj->abij', g.oo[0], U11old, U12old, optimize = True)
    U22 += -1.0 * einsum('ki,ak,bj->abij', g.oo[0], U11old, U12old, optimize = True)
    U22 += -1.0 * einsum('kj,bk,ai->abij', g.oo[0], U11old, U12old, optimize = True)
    U22 += 1.0 * einsum('kj,ak,bi->abij', g.oo[0], U11old, U12old, optimize = True)
    U22 += 1.0 * einsum('ki,bk,aj->abij', g.oo[0], U11old, U12old, optimize = True)
    U22 += -1.0 * einsum('ki,ak,bj->abij', g.oo[0], U11old, U12old, optimize = True)
    U22 += -1.0 * einsum('kj,bk,ai->abij', g.oo[0], U11old, U12old, optimize = True)
    U22 += 1.0 * einsum('kj,ak,bi->abij', g.oo[0], U11old, U12old, optimize = True)
    U22 += -1.0 * einsum('ki,bj,ak->abij', g.oo[0], U11old, U12old, optimize = True)
    U22 += 1.0 * einsum('ki,aj,bk->abij', g.oo[0], U11old, U12old, optimize = True)
    U22 += 1.0 * einsum('kj,bi,ak->abij', g.oo[0], U11old, U12old, optimize = True)
    U22 += -1.0 * einsum('kj,ai,bk->abij', g.oo[0], U11old, U12old, optimize = True)
    U22 += -1.0 * einsum('bc,ci,aj->abij', g.vv[0], U11old, U12old, optimize = True)
    U22 += 1.0 * einsum('ac,ci,bj->abij', g.vv[0], U11old, U12old, optimize = True)
    U22 += 1.0 * einsum('bc,cj,ai->abij', g.vv[0], U11old, U12old, optimize = True)
    U22 += -1.0 * einsum('ac,cj,bi->abij', g.vv[0], U11old, U12old, optimize = True)
    U22 += -1.0 * einsum('bc,ci,aj->abij', g.vv[0], U11old, U12old, optimize = True)
    U22 += 1.0 * einsum('ac,ci,bj->abij', g.vv[0], U11old, U12old, optimize = True)
    U22 += 1.0 * einsum('bc,cj,ai->abij', g.vv[0], U11old, U12old, optimize = True)
    U22 += -1.0 * einsum('ac,cj,bi->abij', g.vv[0], U11old, U12old, optimize = True)
    U22 += 1.0 * einsum('bc,ai,cj->abij', g.vv[0], U11old, U12old, optimize = True)
    U22 += -1.0 * einsum('ac,bi,cj->abij', g.vv[0], U11old, U12old, optimize = True)
    U22 += -1.0 * einsum('bc,aj,ci->abij', g.vv[0], U11old, U12old, optimize = True)
    U22 += 1.0 * einsum('ac,bj,ci->abij', g.vv[0], U11old, U12old, optimize = True)
    U22 += 1.0 * einsum('klic,cj,balk->abij', I.ooov, U11old, U21old, optimize = True)
    U22 += -2.0 * einsum('klic,bk,aclj->abij', I.ooov, U11old, U21old, optimize = True)
    U22 += 2.0 * einsum('klic,ak,bclj->abij', I.ooov, U11old, U21old, optimize = True)
    U22 += -2.0 * einsum('klic,ck,balj->abij', I.ooov, U11old, U21old, optimize = True)
    U22 += -1.0 * einsum('kljc,ci,balk->abij', I.ooov, U11old, U21old, optimize = True)
    U22 += 2.0 * einsum('kljc,bk,acli->abij', I.ooov, U11old, U21old, optimize = True)
    U22 += -2.0 * einsum('kljc,ak,bcli->abij', I.ooov, U11old, U21old, optimize = True)
    U22 += 2.0 * einsum('kljc,ck,bali->abij', I.ooov, U11old, U21old, optimize = True)
    U22 += 2.0 * einsum('kbcd,di,ackj->abij', I.ovvv, U11old, U21old, optimize = True)
    U22 += -2.0 * einsum('kacd,di,bckj->abij', I.ovvv, U11old, U21old, optimize = True)
    U22 += -2.0 * einsum('kbcd,dj,acki->abij', I.ovvv, U11old, U21old, optimize = True)
    U22 += 2.0 * einsum('kacd,dj,bcki->abij', I.ovvv, U11old, U21old, optimize = True)
    U22 += -1.0 * einsum('kbcd,ak,dcji->abij', I.ovvv, U11old, U21old, optimize = True)
    U22 += 2.0 * einsum('kbcd,dk,acji->abij', I.ovvv, U11old, U21old, optimize = True)
    U22 += 1.0 * einsum('kacd,bk,dcji->abij', I.ovvv, U11old, U21old, optimize = True)
    U22 += -2.0 * einsum('kacd,dk,bcji->abij', I.ovvv, U11old, U21old, optimize = True)

    U22 += 1.0 * einsum('kc,ci,bakj->abij', g.ov[0], U11old, U22old, optimize = True)
    U22 += -1.0 * einsum('kc,cj,baki->abij', g.ov[0], U11old, U22old, optimize = True)
    U22 += 1.0 * einsum('kc,bk,acji->abij', g.ov[0], U11old, U22old, optimize = True)
    U22 += -1.0 * einsum('kc,ak,bcji->abij', g.ov[0], U11old, U22old, optimize = True)
    U22 += 1.0 * einsum('kc,ck,baji->abij', g.ov[0], U11old, U22old, optimize = True)
    U22 += 1.0 * einsum('kc,ci,bakj->abij', g.ov[0], U11old, U22old, optimize = True)
    U22 += -1.0 * einsum('kc,cj,baki->abij', g.ov[0], U11old, U22old, optimize = True)
    U22 += 1.0 * einsum('kc,bk,acji->abij', g.ov[0], U11old, U22old, optimize = True)
    U22 += -1.0 * einsum('kc,ak,bcji->abij', g.ov[0], U11old, U22old, optimize = True)
    U22 += 1.0 * einsum('kc,ck,baji->abij', g.ov[0], U11old, U22old, optimize = True)
    U22 += 1.0 * einsum('kc,bi,ackj->abij', g.ov[0], U11old, U22old, optimize = True)
    U22 += -1.0 * einsum('kc,ai,bckj->abij', g.ov[0], U11old, U22old, optimize = True)
    U22 += 1.0 * einsum('kc,ci,bakj->abij', g.ov[0], U11old, U22old, optimize = True)
    U22 += -1.0 * einsum('kc,bj,acki->abij', g.ov[0], U11old, U22old, optimize = True)
    U22 += 1.0 * einsum('kc,aj,bcki->abij', g.ov[0], U11old, U22old, optimize = True)
    U22 += -1.0 * einsum('kc,cj,baki->abij', g.ov[0], U11old, U22old, optimize = True)
    U22 += 1.0 * einsum('kc,bk,acji->abij', g.ov[0], U11old, U22old, optimize = True)
    U22 += -1.0 * einsum('kc,ak,bcji->abij', g.ov[0], U11old, U22old, optimize = True)

    U22 += -1.0 * einsum('kc,bcji,ak->abij', g.ov[0], U21old, U12old, optimize = True)
    U22 += 1.0 * einsum('kc,acji,bk->abij', g.ov[0], U21old, U12old, optimize = True)
    U22 += -1.0 * einsum('kc,baki,cj->abij', g.ov[0], U21old, U12old, optimize = True)
    U22 += 1.0 * einsum('kc,bcki,aj->abij', g.ov[0], U21old, U12old, optimize = True)
    U22 += -1.0 * einsum('kc,acki,bj->abij', g.ov[0], U21old, U12old, optimize = True)
    U22 += 1.0 * einsum('kc,bakj,ci->abij', g.ov[0], U21old, U12old, optimize = True)
    U22 += -1.0 * einsum('kc,bckj,ai->abij', g.ov[0], U21old, U12old, optimize = True)
    U22 += 1.0 * einsum('kc,ackj,bi->abij', g.ov[0], U21old, U12old, optimize = True)
    U22 += -1.0 * einsum('kc,bcji,ak->abij', g.ov[0], U21old, U12old, optimize = True)
    U22 += 1.0 * einsum('kc,acji,bk->abij', g.ov[0], U21old, U12old, optimize = True)
    U22 += -1.0 * einsum('kc,baki,cj->abij', g.ov[0], U21old, U12old, optimize = True)
    U22 += 1.0 * einsum('kc,bcki,aj->abij', g.ov[0], U21old, U12old, optimize = True)
    U22 += -1.0 * einsum('kc,acki,bj->abij', g.ov[0], U21old, U12old, optimize = True)
    U22 += 1.0 * einsum('kc,bakj,ci->abij', g.ov[0], U21old, U12old, optimize = True)
    U22 += -1.0 * einsum('kc,bckj,ai->abij', g.ov[0], U21old, U12old, optimize = True)
    U22 += 1.0 * einsum('kc,ackj,bi->abij', g.ov[0], U21old, U12old, optimize = True)
    U22 += 1.0 * einsum('kc,baji,ck->abij', g.ov[0], U21old, U12old, optimize = True)
    U22 += -1.0 * einsum('kc,bcji,ak->abij', g.ov[0], U21old, U12old, optimize = True)
    U22 += 1.0 * einsum('kc,acji,bk->abij', g.ov[0], U21old, U12old, optimize = True)
    U22 += -1.0 * einsum('kc,baki,cj->abij', g.ov[0], U21old, U12old, optimize = True)
    U22 += 1.0 * einsum('kc,bakj,ci->abij', g.ov[0], U21old, U12old, optimize = True)
    U22 += -1.0 * einsum('klic,cj,bk,al->abij', I.ooov, T1old, T1old, U12old, optimize = True)
    U22 += 1.0 * einsum('klic,cj,ak,bl->abij', I.ooov, T1old, T1old, U12old, optimize = True)
    U22 += -1.0 * einsum('klic,bk,al,cj->abij', I.ooov, T1old, T1old, U12old, optimize = True)
    U22 += 1.0 * einsum('kljc,ci,bk,al->abij', I.ooov, T1old, T1old, U12old, optimize = True)
    U22 += -1.0 * einsum('kljc,ci,ak,bl->abij', I.ooov, T1old, T1old, U12old, optimize = True)
    U22 += 1.0 * einsum('kljc,bk,al,ci->abij', I.ooov, T1old, T1old, U12old, optimize = True)
    U22 += 1.0 * einsum('kbcd,di,cj,ak->abij', I.ovvv, T1old, T1old, U12old, optimize = True)
    U22 += -1.0 * einsum('kacd,di,cj,bk->abij', I.ovvv, T1old, T1old, U12old, optimize = True)
    U22 += 1.0 * einsum('kbcd,di,ak,cj->abij', I.ovvv, T1old, T1old, U12old, optimize = True)
    U22 += -1.0 * einsum('kacd,di,bk,cj->abij', I.ovvv, T1old, T1old, U12old, optimize = True)
    U22 += -1.0 * einsum('kbcd,dj,ak,ci->abij', I.ovvv, T1old, T1old, U12old, optimize = True)
    U22 += 1.0 * einsum('kacd,dj,bk,ci->abij', I.ovvv, T1old, T1old, U12old, optimize = True)

    U22 += 1.0 * S1old * einsum('kc,ci,bakj->abij', g.ov[0], T1old, U22old, optimize = True)
    U22 += -1.0 * S1old * einsum('kc,cj,baki->abij', g.ov[0], T1old, U22old, optimize = True)
    U22 += 1.0 * S1old * einsum('kc,bk,acji->abij', g.ov[0], T1old, U22old, optimize = True)
    U22 += -1.0 * S1old * einsum('kc,ak,bcji->abij', g.ov[0], T1old, U22old, optimize = True)

    U22 += -1.0 * S1old * einsum('kc,bcji,ak->abij', g.ov[0], T2old, U12old, optimize = True)
    U22 += 1.0 * S1old * einsum('kc,acji,bk->abij', g.ov[0], T2old, U12old, optimize = True)
    U22 += -1.0 * S1old * einsum('kc,baki,cj->abij', g.ov[0], T2old, U12old, optimize = True)
    U22 += 1.0 * S1old * einsum('kc,bakj,ci->abij', g.ov[0], T2old, U12old, optimize = True)
    U22 += -1.0 * einsum('klcd,bdji,aclk->abij', I.oovv, U21old, U21old, optimize = True)
    U22 += 1.0 * einsum('klcd,adji,bclk->abij', I.oovv, U21old, U21old, optimize = True)
    U22 += 0.5 * einsum('klcd,dcji,balk->abij', I.oovv, U21old, U21old, optimize = True)
    U22 += -1.0 * einsum('klcd,baki,dclj->abij', I.oovv, U21old, U21old, optimize = True)
    U22 += 2.0 * einsum('klcd,bdki,aclj->abij', I.oovv, U21old, U21old, optimize = True)
    U22 += -2.0 * einsum('klcd,adki,bclj->abij', I.oovv, U21old, U21old, optimize = True)
    U22 += -1.0 * einsum('klcd,dcki,balj->abij', I.oovv, U21old, U21old, optimize = True)

    U22 += -0.5 * einsum('klcd,di,cj,balk->abij', I.oovv, T1old, T1old, U22old, optimize = True)
    U22 += 1.0 * einsum('klcd,di,bk,aclj->abij', I.oovv, T1old, T1old, U22old, optimize = True)
    U22 += -1.0 * einsum('klcd,di,ak,bclj->abij', I.oovv, T1old, T1old, U22old, optimize = True)
    U22 += 1.0 * einsum('klcd,di,ck,balj->abij', I.oovv, T1old, T1old, U22old, optimize = True)
    U22 += -1.0 * einsum('klcd,dj,bk,acli->abij', I.oovv, T1old, T1old, U22old, optimize = True)
    U22 += 1.0 * einsum('klcd,dj,ak,bcli->abij', I.oovv, T1old, T1old, U22old, optimize = True)
    U22 += -1.0 * einsum('klcd,dj,ck,bali->abij', I.oovv, T1old, T1old, U22old, optimize = True)
    U22 += -0.5 * einsum('klcd,bk,al,dcji->abij', I.oovv, T1old, T1old, U22old, optimize = True)
    U22 += 1.0 * einsum('klcd,bk,dl,acji->abij', I.oovv, T1old, T1old, U22old, optimize = True)
    U22 += -1.0 * einsum('klcd,ak,dl,bcji->abij', I.oovv, T1old, T1old, U22old, optimize = True)

    U22 += -1.0 * einsum('klcd,di,bakj,cl->abij', I.oovv, T1old, T2old, U12old, optimize = True)
    U22 += 1.0 * einsum('klcd,di,bckj,al->abij', I.oovv, T1old, T2old, U12old, optimize = True)
    U22 += -1.0 * einsum('klcd,di,ackj,bl->abij', I.oovv, T1old, T2old, U12old, optimize = True)
    U22 += -0.5 * einsum('klcd,di,balk,cj->abij', I.oovv, T1old, T2old, U12old, optimize = True)
    U22 += 1.0 * einsum('klcd,dj,baki,cl->abij', I.oovv, T1old, T2old, U12old, optimize = True)
    U22 += -1.0 * einsum('klcd,dj,bcki,al->abij', I.oovv, T1old, T2old, U12old, optimize = True)
    U22 += 1.0 * einsum('klcd,dj,acki,bl->abij', I.oovv, T1old, T2old, U12old, optimize = True)
    U22 += -1.0 * einsum('klcd,bk,adji,cl->abij', I.oovv, T1old, T2old, U12old, optimize = True)
    U22 += -0.5 * einsum('klcd,bk,dcji,al->abij', I.oovv, T1old, T2old, U12old, optimize = True)
    U22 += 1.0 * einsum('klcd,ak,bdji,cl->abij', I.oovv, T1old, T2old, U12old, optimize = True)
    U22 += 1.0 * einsum('klcd,dk,bcji,al->abij', I.oovv, T1old, T2old, U12old, optimize = True)
    U22 += 0.5 * einsum('klcd,ak,dcji,bl->abij', I.oovv, T1old, T2old, U12old, optimize = True)
    U22 += -1.0 * einsum('klcd,dk,acji,bl->abij', I.oovv, T1old, T2old, U12old, optimize = True)
    U22 += 1.0 * einsum('klcd,bk,adli,cj->abij', I.oovv, T1old, T2old, U12old, optimize = True)
    U22 += -1.0 * einsum('klcd,ak,bdli,cj->abij', I.oovv, T1old, T2old, U12old, optimize = True)
    U22 += 1.0 * einsum('klcd,dk,bali,cj->abij', I.oovv, T1old, T2old, U12old, optimize = True)
    U22 += 0.5 * einsum('klcd,dj,balk,ci->abij', I.oovv, T1old, T2old, U12old, optimize = True)
    U22 += -1.0 * einsum('klcd,bk,adlj,ci->abij', I.oovv, T1old, T2old, U12old, optimize = True)
    U22 += 1.0 * einsum('klcd,ak,bdlj,ci->abij', I.oovv, T1old, T2old, U12old, optimize = True)
    U22 += -1.0 * einsum('klcd,dk,balj,ci->abij', I.oovv, T1old, T2old, U12old, optimize = True)
    U22 += -2.0 * einsum('klic,cj,bk,al->abij', I.ooov, T1old, U11old, U11old, optimize = True)
    U22 += -2.0 * einsum('klic,bk,cj,al->abij', I.ooov, T1old, U11old, U11old, optimize = True)
    U22 += 2.0 * einsum('klic,ak,cj,bl->abij', I.ooov, T1old, U11old, U11old, optimize = True)
    U22 += 2.0 * einsum('kljc,ci,bk,al->abij', I.ooov, T1old, U11old, U11old, optimize = True)
    U22 += 2.0 * einsum('kljc,bk,ci,al->abij', I.ooov, T1old, U11old, U11old, optimize = True)
    U22 += -2.0 * einsum('kljc,ak,ci,bl->abij', I.ooov, T1old, U11old, U11old, optimize = True)
    U22 += 2.0 * einsum('kbcd,di,cj,ak->abij', I.ovvv, T1old, U11old, U11old, optimize = True)
    U22 += -2.0 * einsum('kacd,di,cj,bk->abij', I.ovvv, T1old, U11old, U11old, optimize = True)
    U22 += -2.0 * einsum('kbcd,dj,ci,ak->abij', I.ovvv, T1old, U11old, U11old, optimize = True)
    U22 += 2.0 * einsum('kacd,dj,ci,bk->abij', I.ovvv, T1old, U11old, U11old, optimize = True)
    U22 += 2.0 * einsum('kbcd,ak,di,cj->abij', I.ovvv, T1old, U11old, U11old, optimize = True)
    U22 += -2.0 * einsum('kacd,bk,di,cj->abij', I.ovvv, T1old, U11old, U11old, optimize = True)
    U22 += 1.0 * einsum('kc,ci,bk,aj->abij', g.ov[0], T1old, U11old, U12old, optimize = True)
    U22 += -1.0 * einsum('kc,ci,ak,bj->abij', g.ov[0], T1old, U11old, U12old, optimize = True)
    U22 += 1.0 * einsum('kc,bk,ci,aj->abij', g.ov[0], T1old, U11old, U12old, optimize = True)
    U22 += -1.0 * einsum('kc,ak,ci,bj->abij', g.ov[0], T1old, U11old, U12old, optimize = True)
    U22 += -1.0 * einsum('kc,cj,bk,ai->abij', g.ov[0], T1old, U11old, U12old, optimize = True)
    U22 += 1.0 * einsum('kc,cj,ak,bi->abij', g.ov[0], T1old, U11old, U12old, optimize = True)
    U22 += -1.0 * einsum('kc,bk,cj,ai->abij', g.ov[0], T1old, U11old, U12old, optimize = True)
    U22 += 1.0 * einsum('kc,ak,cj,bi->abij', g.ov[0], T1old, U11old, U12old, optimize = True)
    U22 += 1.0 * einsum('kc,ci,bk,aj->abij', g.ov[0], T1old, U11old, U12old, optimize = True)
    U22 += -1.0 * einsum('kc,ci,ak,bj->abij', g.ov[0], T1old, U11old, U12old, optimize = True)
    U22 += 1.0 * einsum('kc,bk,ci,aj->abij', g.ov[0], T1old, U11old, U12old, optimize = True)
    U22 += -1.0 * einsum('kc,ak,ci,bj->abij', g.ov[0], T1old, U11old, U12old, optimize = True)
    U22 += -1.0 * einsum('kc,cj,bk,ai->abij', g.ov[0], T1old, U11old, U12old, optimize = True)
    U22 += 1.0 * einsum('kc,cj,ak,bi->abij', g.ov[0], T1old, U11old, U12old, optimize = True)
    U22 += -1.0 * einsum('kc,bk,cj,ai->abij', g.ov[0], T1old, U11old, U12old, optimize = True)
    U22 += 1.0 * einsum('kc,ak,cj,bi->abij', g.ov[0], T1old, U11old, U12old, optimize = True)
    U22 += -1.0 * einsum('kc,ci,bj,ak->abij', g.ov[0], T1old, U11old, U12old, optimize = True)
    U22 += 1.0 * einsum('kc,ci,aj,bk->abij', g.ov[0], T1old, U11old, U12old, optimize = True)
    U22 += 1.0 * einsum('kc,cj,bi,ak->abij', g.ov[0], T1old, U11old, U12old, optimize = True)
    U22 += -1.0 * einsum('kc,cj,ai,bk->abij', g.ov[0], T1old, U11old, U12old, optimize = True)
    U22 += -1.0 * einsum('kc,bk,ai,cj->abij', g.ov[0], T1old, U11old, U12old, optimize = True)
    U22 += 1.0 * einsum('kc,ak,bi,cj->abij', g.ov[0], T1old, U11old, U12old, optimize = True)
    U22 += 1.0 * einsum('kc,bk,aj,ci->abij', g.ov[0], T1old, U11old, U12old, optimize = True)
    U22 += -1.0 * einsum('kc,ak,bj,ci->abij', g.ov[0], T1old, U11old, U12old, optimize = True)
    U22 += 2.0 * S1old * einsum('kc,ci,bakj->abij', g.ov[0], U11old, U21old, optimize = True)
    U22 += -2.0 * S1old * einsum('kc,cj,baki->abij', g.ov[0], U11old, U21old, optimize = True)
    U22 += 2.0 * S1old * einsum('kc,bk,acji->abij', g.ov[0], U11old, U21old, optimize = True)
    U22 += -2.0 * S1old * einsum('kc,ak,bcji->abij', g.ov[0], U11old, U21old, optimize = True)
    U22 += -1.0 * einsum('klcd,di,cj,balk->abij', I.oovv, T1old, U11old, U21old, optimize = True)
    U22 += 2.0 * einsum('klcd,di,bk,aclj->abij', I.oovv, T1old, U11old, U21old, optimize = True)
    U22 += -2.0 * einsum('klcd,di,ak,bclj->abij', I.oovv, T1old, U11old, U21old, optimize = True)
    U22 += 2.0 * einsum('klcd,di,ck,balj->abij', I.oovv, T1old, U11old, U21old, optimize = True)
    U22 += 1.0 * einsum('klcd,dj,ci,balk->abij', I.oovv, T1old, U11old, U21old, optimize = True)
    U22 += 2.0 * einsum('klcd,bk,di,aclj->abij', I.oovv, T1old, U11old, U21old, optimize = True)
    U22 += -2.0 * einsum('klcd,ak,di,bclj->abij', I.oovv, T1old, U11old, U21old, optimize = True)
    U22 += -2.0 * einsum('klcd,dk,ci,balj->abij', I.oovv, T1old, U11old, U21old, optimize = True)
    U22 += -2.0 * einsum('klcd,dj,bk,acli->abij', I.oovv, T1old, U11old, U21old, optimize = True)
    U22 += 2.0 * einsum('klcd,dj,ak,bcli->abij', I.oovv, T1old, U11old, U21old, optimize = True)
    U22 += -2.0 * einsum('klcd,dj,ck,bali->abij', I.oovv, T1old, U11old, U21old, optimize = True)
    U22 += -2.0 * einsum('klcd,bk,dj,acli->abij', I.oovv, T1old, U11old, U21old, optimize = True)
    U22 += 2.0 * einsum('klcd,ak,dj,bcli->abij', I.oovv, T1old, U11old, U21old, optimize = True)
    U22 += 2.0 * einsum('klcd,dk,cj,bali->abij', I.oovv, T1old, U11old, U21old, optimize = True)
    U22 += -1.0 * einsum('klcd,bk,al,dcji->abij', I.oovv, T1old, U11old, U21old, optimize = True)
    U22 += 2.0 * einsum('klcd,bk,dl,acji->abij', I.oovv, T1old, U11old, U21old, optimize = True)
    U22 += 1.0 * einsum('klcd,ak,bl,dcji->abij', I.oovv, T1old, U11old, U21old, optimize = True)
    U22 += -2.0 * einsum('klcd,dk,bl,acji->abij', I.oovv, T1old, U11old, U21old, optimize = True)
    U22 += -2.0 * einsum('klcd,ak,dl,bcji->abij', I.oovv, T1old, U11old, U21old, optimize = True)
    U22 += 2.0 * einsum('klcd,dk,al,bcji->abij', I.oovv, T1old, U11old, U21old, optimize = True)
    U22 += 2.0 * einsum('klcd,bdji,ak,cl->abij', I.oovv, T2old, U11old, U11old, optimize = True)
    U22 += -2.0 * einsum('klcd,adji,bk,cl->abij', I.oovv, T2old, U11old, U11old, optimize = True)
    U22 += -1.0 * einsum('klcd,dcji,bk,al->abij', I.oovv, T2old, U11old, U11old, optimize = True)
    U22 += 2.0 * einsum('klcd,baki,dj,cl->abij', I.oovv, T2old, U11old, U11old, optimize = True)
    U22 += 2.0 * einsum('klcd,bdki,cj,al->abij', I.oovv, T2old, U11old, U11old, optimize = True)
    U22 += -2.0 * einsum('klcd,adki,cj,bl->abij', I.oovv, T2old, U11old, U11old, optimize = True)
    U22 += -2.0 * einsum('klcd,bakj,di,cl->abij', I.oovv, T2old, U11old, U11old, optimize = True)
    U22 += -2.0 * einsum('klcd,bdkj,ci,al->abij', I.oovv, T2old, U11old, U11old, optimize = True)
    U22 += -1.0 * einsum('klcd,balk,di,cj->abij', I.oovv, T2old, U11old, U11old, optimize = True)
    U22 += 2.0 * einsum('klcd,adkj,ci,bl->abij', I.oovv, T2old, U11old, U11old, optimize = True)

    U22 += 2.0 * einsum('kc,bi,cj,ak->abij', g.ov[0], U11old, U11old, U11old, optimize = True)
    U22 += -2.0 * einsum('kc,ci,bj,ak->abij', g.ov[0], U11old, U11old, U11old, optimize = True)
    U22 += -2.0 * einsum('kc,ai,cj,bk->abij', g.ov[0], U11old, U11old, U11old, optimize = True)
    U22 += 2.0 * einsum('kc,ci,aj,bk->abij', g.ov[0], U11old, U11old, U11old, optimize = True)


    U2n[1][0,0] = U22

    return U2n


