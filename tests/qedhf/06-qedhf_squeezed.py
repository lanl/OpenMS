import unittest
import numpy
from pyscf import gto, scf
from openms.mqed import vtqedhf as qedhf


def vsq_qedhf(vsq=0.0, gfac=0.1, use_vsq=True, optimize_vsq=False):
    itest = -2
    zshift = itest * 2.5

    atom = f"H          0.86681        0.60144        {5.00000+zshift};\
             F         -0.86681        0.60144        {5.00000+zshift};\
             O          0.00000       -0.07579        {5.00000+zshift};\
             He         0.00000        0.00000        {7.50000+zshift}"

    mol = gto.M(atom=atom, basis="sto3g", unit="Angstrom", symmetry=True)

    nmode = 1
    cavity_freq = numpy.zeros(nmode)
    cavity_mode = numpy.zeros((nmode, 3))
    cavity_freq[0] = 0.05
    cavity_mode[0, :] = gfac * numpy.asarray([0, 1, 0])
    mol.verbose = 5

    qedmf = qedhf.RHF(
        mol,
        xc=None,
        vec=cavity_mode,
        omega=cavity_freq,
        add_nuc_dipole=True,
    )
    if optimize_vsq:
        qedmf.qed.squeezed_var = numpy.zeros(nmode)
        qedmf.qed.optimize_vsq = True
    elif use_vsq:
        qedmf.qed.squeezed_var = vsq * numpy.ones(nmode)
        qedmf.qed.optimize_vsq = False

    qedmf.precond = 5.e-3
    qedmf.max_cycle = 1000
    qedmf.init_guess = "hcore"
    qedmf.kernel()

    newvsq = qedmf.qed.squeezed_var
    e_tot = qedmf.e_tot
    eboson = qedmf.qed.e_boson
    dm = qedmf.make_rdm1()

    del qedmf
    return e_tot, eboson, newvsq


class TestVTQEDHF_f(unittest.TestCase):
    def test_vsqqed_f(self):
        energies_ref = [
            -174.8887604647,
            -174.8885555963,
            -174.8883017198,
        ]

        gfac = 0.5
        vt_qed, _, _ = vsq_qedhf(vsq=0.0, gfac=gfac, use_vsq=False)

        fbound = 0.02  # 0.06
        vsqs = numpy.arange(-fbound, fbound + 1.0e-6, fbound / 1)
        nvsq = len(vsqs)

        energies = numpy.zeros((nvsq, 2))
        for i in range(nvsq):
            vsq = vsqs[i]
            e_tot, e_photon, _ = vsq_qedhf(vsq=vsq, gfac=gfac)
            energies[i, 0] = e_tot
            energies[i, 1] = e_photon

        tot_energies = energies[:, 0]
        for i in range(nvsq):
            print(
                f"%9.4f   %15.10f   %15.10f   %15.10f"
                % (vsqs[i], energies[i, 0], energies[i, 1], tot_energies[i])
            )

        self.assertAlmostEqual(
            vt_qed,
            tot_energies[nvsq // 2],
            places=6,
            msg="Etot does not match the reference value.",
        )

        numpy.testing.assert_almost_equal(
            tot_energies,
            energies_ref,
            decimal=5,
            err_msg="Etot does not match the reference value.",
        )

    def test_vsq_grad(self):
        energy_ref = -174.889089263587

        gfac = 0.5
        e_tot, e_photon, vsq_opt = vsq_qedhf(gfac=gfac, optimize_vsq=True)

        print(f"%9.4f   %15.10f   %15.10f" % (vsq_opt[0], e_tot, e_photon))

        self.assertAlmostEqual(
            e_tot,
            energy_ref,
            places=6,
            msg="Etot does not match the reference value.",
        )

if __name__ == "__main__":
    unittest.main()
