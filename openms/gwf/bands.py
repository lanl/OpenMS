#


def g_fermi(ismear=0, verbose=0):
    r"""bare band fermi energy
    TODO:
    """
    if ismear == 0:
        fermi = 0.0
        # gutz_fermi_tetra_w2k()
    elif ismear == 1:
        fermi = 0.0
        # get_ef_fun()
    else:
        # give error
        raise ValueError("unsupported ismear type")
    # adjust_efermi_bandgap(verbose)


class band_structure(object):

    def __init__(self, *args, **kwargs):
        self.n_frozen = None  #  number of forzen orbitals
        self.nelec_frozen = None  #  number of forzen electrons
        self.ne = None #  
        self.ek = None # band eigen values (nband, nk, nspin)
        self.coeff = None # spin-up/down weights
        self.efermi = 0.0 # fermi energy
        self.hk0 = None
