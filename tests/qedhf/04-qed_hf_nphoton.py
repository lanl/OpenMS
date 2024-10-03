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
        unit="Angstrom",
        symmetry=True,
        verbose=1,
    )

    cavity_freq = numpy.zeros(nmode)
    cavity_mode = numpy.zeros((nmode, 3))
    cavity_freq[0] = 0.5
    cavity_mode[0, :] = gfac * numpy.asarray([0, 1, 0])

    qedmf = qedhf.RHF(mol, omega=cavity_freq, vec=cavity_mode, nboson_states=nfock, use_cs=cs)

    qedmf.max_cycle = 500
    qedmf.kernel()

    return qedmf.e_tot


class TestQEDHF(unittest.TestCase):

    def test_n_photon_convergence(self):

        nlist = range(1,9,1)
        ref_qed_e_tots = [-262.050746414733, -262.052856900157,
                          -262.052870889973, -262.052870927649,
                          -262.052870927716, -262.052870927716,
                          -262.052870927716, -262.052870927716]

        zshift = 0.0
        for i, n in enumerate(nlist):
            e_tot = run_qedhf(zshift, nfock=n, cs=False)
            err_msg = f"FOCK STATE : E_tot does not match the reference value for n_photon = {n}."
            self.assertAlmostEqual(e_tot, ref_qed_e_tots[i], places=6, msg=err_msg)

            e_tot = run_qedhf(zshift, nfock=n, cs=True)
            err_msg = f"COHERENT STATE : E_tot does not match the reference value for n_photon = {n}."
            self.assertAlmostEqual(e_tot, ref_qed_e_tots[-1], places=6, msg=err_msg)

if __name__ == '__main__':
    unittest.main()
