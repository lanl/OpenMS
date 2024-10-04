import unittest
import numpy
from pyscf import gto
from openms.mqed import qedhf

def run_qedhf(zshift, cs=True, nfock=1, nmode=1, gfac=0.1):

    atom = f"C   0.00000000   0.00000000    {zshift};\
             O   0.00000000   1.23456800    {zshift};\
             H   0.97075033  -0.54577032    {zshift};\
             C  -1.21509881  -0.80991169    {zshift};\
             H  -1.15288176  -1.89931439    {zshift};\
             C  -2.43440063  -0.19144555    {zshift};\
             H  -3.37262777  -0.75937214    {zshift};\
             O  -2.62194056   1.12501165    {zshift};\
             H  -1.71446384   1.51627790    {zshift}"

    mol = gto.M(
        atom = atom,
        basis="sto3g",
        #basis="cc-pvdz",
        unit="Angstrom",
        symmetry=True,
        verbose=1,
    )

    cavity_freq = numpy.zeros(nmode)
    cavity_mode = numpy.zeros((nmode, 3))
    cavity_freq[0] = 0.5
    cavity_mode[0, :] = gfac * numpy.asarray([0, 1, 0])
    zlambda = numpy.zeros(nmode)

    if cs:
        qedmf = qedhf.RHF(mol, xc=None, cavity_mode=cavity_mode,
                          cavity_freq=cavity_freq,
                          nboson_states = nfock)
    else:
        qedmf = qedhf.RHF(mol, xc=None, cavity_mode=cavity_mode,
                         cavity_freq=cavity_freq,
                         nboson_states = nfock,
                         z_alpha=zlambda)

    qedmf.max_cycle = 500
    qedmf.kernel() #dm0=dm)
    return qedmf.e_tot


class TestQEDHF(unittest.TestCase):

    def test_n_photon_convergence(self):

        nlist = range(1, 7, 1)
        # cs refs (no changes)
        refs2 = [-262.0528709277] * len(nlist)

        # fock (converges to cs)
        refs1 = [-262.050746414727, -262.052856900152,
                -262.052870889967, -262.052870927632,
                -262.052870927716, -262.052870927715]

        itest = 0
        zshift = itest * 2.0

        E_c = numpy.zeros(len(nlist))
        E_f = numpy.zeros(len(nlist))
        for i, n in enumerate(nlist):
            E_c[i] = run_qedhf(zshift, cs=True, nfock=n)
            E_f[i] = run_qedhf(zshift, cs=False, nfock=n)

        numpy.testing.assert_almost_equal(E_f, refs1, decimal=7, err_msg="Etot does not match the reference value.")
        numpy.testing.assert_almost_equal(E_c, refs2, decimal=7, err_msg="Etot does not match the reference value.")

if __name__ == '__main__':
    unittest.main()
