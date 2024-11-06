#


r"""
This is a collection of analyticall solvable models

TBA.

"""



def Hubbard_Holstein_1D(n, t=1.0, filling=0.5, U=0.0, omega=1.0, g=1.0, PBC=True):

    H0 = H = numpy.diag([-t for i in range(n-1)], k = 1
    if PBC:
        H[0, n-1] = -t


