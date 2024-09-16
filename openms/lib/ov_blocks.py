
# copy from Alec's cqcpy

import numpy

def block_diag(A, B):
    """Return a block diagonal matrix
       A 0
       0 B
    """
    ma = A.shape[0]
    mb = B.shape[0]
    na = A.shape[1]
    nb = B.shape[1]

    z1 = numpy.zeros((ma, nb))
    z2 = numpy.zeros((mb, na))
    M1 = numpy.hstack((A, z1))
    M2 = numpy.hstack((z2, B))
    return numpy.vstack((M1, M2))

class one_e_blocks(object):
    def __init__(self, oo, ov, vo, vv):
        self.oo = oo
        self.ov = ov
        self.vo = vo
        self.vv = vv


class two_e_blocks(object):
    def __init__(self,
                 vvvv=None, vvvo=None, vovv=None, vvoo=None,
                 vovo=None, oovv=None, vooo=None, ooov=None, oooo=None):
        self.vvvv = vvvv
        self.vvvo = vvvo
        self.vovv = vovv
        self.vvoo = vvoo
        self.vovo = vovo
        self.oovv = oovv
        self.vooo = vooo
        self.ooov = ooov
        self.oooo = oooo


def make_two_e_blocks(Itot, n1o, n1v, n2o, n2v, n3o, n3v, n4o, n4v):
    n1, n2, n3, n4 = Itot.shape
    assert(n1 == n1o + n1v)
    assert(n2 == n2o + n2v)
    assert(n3 == n3o + n3v)
    assert(n4 == n4o + n4v)
    Ivvvv = Itot[n1o:, n2o:, n3o:, n4o:]
    Ivvvo = Itot[n1o:, n2o:, n3o:, :n4o]
    Ivovv = Itot[n1o:, :n2o, n3o:, n4o:]
    Ivvoo = Itot[n1o:, n2o:, :n3o, :n4o]
    Ivovo = Itot[n1o:, :n2o, n3o:, :n4o]
    Ioovv = Itot[:n1o, :n2o, n3o:, n4o:]
    Ivooo = Itot[n1o:, :n2o, :n3o, :n4o]
    Iooov = Itot[:n1o, :n2o, :n3o, n4o:]
    Ioooo = Itot[:n1o, :n2o, :n3o, :n4o]
    return two_e_blocks_full(
        vvvv=Ivvvv, vvvo=Ivvvo, vovv=Ivovv, vvoo=Ivvoo,
        vovo=Ivovo, oovv=Ioovv, vooo=Ivooo, ooov=Iooov, oooo=Ioooo)


class two_e_blocks_full(object):
    def __init__(self,
                 vvvv=None, vvvo=None, vvov=None, vovv=None,
                 ovvv=None, vvoo=None, vovo=None, ovvo=None,
                 voov=None, ovov=None, oovv=None, vooo=None,
                 ovoo=None, oovo=None, ooov=None, oooo=None):
        self.vvvv = vvvv
        self.vvvo = vvvo
        self.vvov = vvov
        self.vovv = vovv
        self.ovvv = ovvv
        self.vvoo = vvoo
        self.vovo = vovo
        self.voov = voov
        self.ovvo = ovvo
        self.ovov = ovov
        self.oovv = oovv
        self.ooov = ooov
        self.oovo = oovo
        self.ovoo = ovoo
        self.vooo = vooo
        self.oooo = oooo


def make_two_e_blocks_full(Itot, n1o, n1v, n2o, n2v, n3o, n3v, n4o, n4v):
    n1, n2, n3, n4 = Itot.shape
    assert(n1 == n1o + n1v)
    assert(n2 == n2o + n2v)
    assert(n3 == n3o + n3v)
    assert(n4 == n4o + n4v)
    Ivvvv = Itot[n1o:, n2o:, n3o:, n4o:]
    Ivvvo = Itot[n1o:, n2o:, n3o:, :n4o]
    Ivvov = Itot[n1o:, n2o:, :n3o, n4o:]
    Ivovv = Itot[n1o:, :n2o, n3o:, n4o:]
    Iovvv = Itot[:n1o, n2o:, n3o:, n4o:]
    Ivvoo = Itot[n1o:, n2o:, :n3o, :n4o]
    Ivovo = Itot[n1o:, :n2o, n3o:, :n4o]
    Iovvo = Itot[:n1o, n2o:, n3o:, :n4o]
    Ivoov = Itot[n1o:, :n2o, :n3o, n4o:]
    Iovov = Itot[:n1o, n2o:, :n3o, n4o:]
    Ioovv = Itot[:n1o, :n2o, n3o:, n4o:]
    Ivooo = Itot[n1o:, :n2o, :n3o, :n4o]
    Iovoo = Itot[:n1o, n2o:, :n3o, :n4o]
    Ioovo = Itot[:n1o, :n2o, n3o:, :n4o]
    Iooov = Itot[:n1o, :n2o, :n3o, n4o:]
    Ioooo = Itot[:n1o, :n2o, :n3o, :n4o]
    return two_e_blocks_full(
        vvvv=Ivvvv, vvvo=Ivvvo, vvov=Ivvov, vovv=Ivovv,
        ovvv=Iovvv, vvoo=Ivvoo, vovo=Ivovo, ovov=Iovov,
        voov=Ivoov, ovvo=Iovvo, oovv=Ioovv, vooo=Ivooo,
        ovoo=Iovoo, oovo=Ioovo, ooov=Iooov, oooo=Ioooo)
