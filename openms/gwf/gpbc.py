
# GWF for periodic systems


def kernel(gobj):
    r"""Newton-udpate of the gwf cost function
    """
    # 1) rotate bare hamiltonian to symmetry-adapted basis
    gobj.rotate_bare_ham()

    # 2) correlated block energy window
    gobj.get_corr_ewindow()

    # 3) get fermi energy
    gobj.get_efermi()

    # 4) check band disperson
    gobj.check_band_disperison()

    # 5) remove local one-body part
    gobj.rm_h1e()

    # 6) init gkernel()
    gobj.init_kernel()

    # init guess for R and lambda
    gobj.init_R_lambda()

    #  newton solver



def g_cost_fcn(n, x, fvec, verbose=0):
    r""" cost function"""
    pass



class kpoints(object):
    r"""class for kpoints"""
    # TODO: kpoints class storing variables and function for generating k mesh
    def __init__(self, *args, **kwargs):
        self.nkx, self.nky, self.nkz = kwargs.get("kmesh", [10, 10, 1])

        self.ndim = self.nkx * self.nky * self.nkz



class gks(object):

    def __init__(self, *args, **kwargs):
        self.rtol = 1.e-6
        self.verbose = kwargs.get("verbose", 1)
        self.maxiter = kwargs.get("maxiter", 100)
        self.iembeddiag = kwargs.get("iembeddiag", 1)
        self.updaterho = kwargs.get("updaterho", 5)
        self.kpts = kwargs.get("kpts", 5)

    def build(self):
        r"""initialize the gks"""

        # 1) set impiurity index

        # 2) set band information

        # 3) and read bare Hamiltonian
        pass

    def rotate_bare_ham(self):
        pass

    def get_corr_ewindow(self):
        pass

    def get_efermi(self):
        pass

    def check_band_disperison(self):
        pass

    def rm_h1e(self):
        r"""remove the  the local one-body part"""
        pass

    def init_R_lambda(self):
        pass

    def init_kernel(self):
        pass

    kernel = kernel

    cost_fcn = g_cost_fcn
