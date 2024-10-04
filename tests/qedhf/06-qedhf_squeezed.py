import unittest
import numpy
from pyscf import gto, scf
from openms.mqed import vtqedhf, scqedhf


def get_mol():
    atom = f"C   0.00000000   0.00000000    0.0000;\
             O   0.00000000   1.23456800    0.0000;\
             H   0.97075033  -0.54577032    0.0000;\
             C  -1.21509881  -0.80991169    0.0000;\
             H  -1.15288176  -1.89931439    0.0000;\
             C  -2.43440063  -0.19144555    0.0000;\
             H  -3.37262777  -0.75937214    0.0000;\
             O  -2.62194056   1.12501165    0.0000;\
             H  -1.71446384   1.51627790    0.0000"

    return gto.M(atom=atom, basis="sto3g", unit="Angstrom", symmetry=True, verbose=1)



def run_vsq_qedhf(method="vtqedhf", vsq=None, gfac=0.1, optimize_vsq=False, falpha=None):

    mol = get_mol()

    nmode = 1
    cavity_freq = numpy.zeros(nmode)
    cavity_mode = numpy.zeros((nmode, 3))
    cavity_freq[0] = 0.5
    cavity_mode[0, :] = gfac * numpy.asarray([0, 1, 0])

    if method != "vtqedhf":
        qedmf = scqedhf.RHF(
            mol,
            xc=None,
            cavity_mode=cavity_mode,
            cavity_freq=cavity_freq,
            add_nuc_dipole=True,
        )
    else:
        # vt/vsq-qedhf
        qedmf = vtqedhf.RHF(
            mol,
            xc=None,
            cavity_mode=cavity_mode,
            cavity_freq=cavity_freq,
            add_nuc_dipole=True,
        )

        # optimize F or not
        if optimize_vsq:
            qedmf.qed.squeezed_var = numpy.zeros(nmode)
            qedmf.qed.optimize_vsq = True
        # introduce F or not
        elif vsq is not None:
            qedmf.qed.squeezed_var = vsq * numpy.ones(nmode)
            qedmf.qed.optimize_vsq = False

        # optimize f or not
        if falpha is not None:
            qedmf.qed.couplings_var = numpy.asarray([falpha])
            qedmf.qed.update_couplings()
            qedmf.qed.optimize_varf = False


    qedmf.precond = 5.0e-3
    qedmf.max_cycle = 500
    # qedmf.diis_space = 30
    qedmf.init_guess = "hcore"
    qedmf.kernel()

    newvsq = qedmf.qed.squeezed_var
    e_tot = qedmf.e_tot
    eboson = qedmf.qed.e_boson
    dm = qedmf.make_rdm1()

    del qedmf
    return e_tot, eboson, newvsq


class TestVTQEDHF_f(unittest.TestCase):

    def test_vsq_fixed_f(self):
        # falpha is fixed at 1.0 (scqedhf)
        print("\n ** scqedhf with parameterized SQ **")

        vsq_sc_ref = -262.142915

        energies_ref = [
            -262.1429140620,
            -262.1429160739,
            -262.1429117858,
            -262.1429012885,
            -262.1428845351,
        ]

        gfac = 5.e-2
        falpha = 1.0
        sc_qed, _, _ = run_vsq_qedhf(method="scqedhf", gfac=gfac)

        fbound = 0.01
        vsqs = numpy.arange(-fbound, 1.0e-6, fbound / 4)
        nvsq = len(vsqs)

        energies = numpy.zeros((nvsq, 2))
        for i in range(nvsq):
            vsq = vsqs[i]
            e_tot, e_photon, _ = run_vsq_qedhf(vsq=vsq, gfac=gfac, falpha=falpha)
            energies[i, 0] = e_tot
            energies[i, 1] = e_photon
            if abs(vsq) < 1.e-6: sc_qed2 = e_tot

        tot_energies = energies[:, 0]
        for i in range(nvsq):
            print(
                f"%9.4f   %15.10f   %15.10f   %15.10f"
                % (vsqs[i], energies[i, 0], energies[i, 1], tot_energies[i])
            )

        self.assertAlmostEqual(sc_qed, sc_qed2, places=6,
            msg="Etot does not match the reference value.",
        )

        numpy.testing.assert_almost_equal(
            tot_energies, energies_ref, decimal=5,
            err_msg="Etot does not match the reference value.",
        )

        # FIXME: optimize vsq in scqedhf not converged
        vsq_sc_eng, e_photon, vsq_opt = run_vsq_qedhf(gfac=gfac, optimize_vsq=True, falpha=1.0)
        print("optimized F=", vsq_opt, " optimized E", vsq_sc_eng, " e_photon=", e_photon)
        numpy.testing.assert_almost_equal(
            vsq_sc_eng, vsq_sc_ref, decimal=5,
            err_msg="Etot does not match the reference value.",
        )

    def test_vsqqed_f(self):
        print("\n ** vtqedhf with parameterized SQ **")
        energies_ref = [
            -262.1453664983,
            -262.1453785032,
            -262.1453837107,
            -262.1453825436,
            -262.1453750744,
        ]

        gfac = 5.e-2
        vt_qed, _, _ = run_vsq_qedhf(gfac=gfac)

        fbound = 0.01
        vsqs = numpy.arange(-fbound, 1.0e-6, fbound / 4)
        nvsq = len(vsqs)

        energies = numpy.zeros((nvsq, 2))
        for i in range(nvsq):
            vsq = vsqs[i]
            e_tot, e_photon, _ = run_vsq_qedhf(vsq=vsq, gfac=gfac)
            energies[i, 0] = e_tot
            energies[i, 1] = e_photon

        tot_energies = energies[:, 0]  # numpy.sum(energies, axis=1)
        for i in range(nvsq):
            print(
                f"%9.4f   %15.10f   %15.10f   %15.10f"
                % (vsqs[i], energies[i, 0], energies[i, 1], tot_energies[i])
            )

        self.assertAlmostEqual(
            vt_qed,
            tot_energies[-1],
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
        print("\n ** vtqedhf with VSQ **")
        energy_ref = -262.14538405860685

        gfac = 5.e-2
        e_tot, e_photon, vsq_opt = run_vsq_qedhf(gfac=gfac, optimize_vsq=True)
        # converged in 146 cycles

        print(f"%9.4f   %15.10f   %15.10f" % (vsq_opt[0], e_tot, e_photon))

        self.assertAlmostEqual(
            e_tot,
            energy_ref,
            places=6,
            msg="Etot does not match the reference value.",
        )

if __name__ == "__main__":
    unittest.main()
