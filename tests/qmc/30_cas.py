import numpy
import unittest
from pyscf import gto, scf, fci
from openms.qmc.afqmc import AFQMC
from openms.qmc import tools
from molecules import get_mol


class TestCAS(unittest.TestCase):

    # test the cas integrals
    def test1(self):
        basis = "ccpvdz"
        basis = "631g"
        verbose = 1

        # 1) RHF calculation
        mol = get_mol(basis=basis, verbose=verbose, name="C2N2H6")
        rmf = scf.HF(mol)
        rmf.kernel()
        rmo = rmf.mo_coeff

        # print("mol.nelec = ", mol.nelec)
        # print("mo.shape =  ", rmo.shape)

        # 2) UHF calculations
        mol = get_mol(basis=basis, verbose=verbose, name="C2N2H6")
        umf = scf.UHF(mol)
        umf.kernel()
        umo = umf.mo_coeff
        # print(f" ||C_RHF - C_UHF[0]  =", numpy.linalg.norm(rmo - umo[0]))
        # print(f" ||C_RHF - C_UHF[1]  =", numpy.linalg.norm(rmo - umo[1]))

        # check whether RHF/UHF energies are close
        self.assertLess(
            numpy.linalg.norm(rmo - umo[0]),
            1.0e-4,
            msg="E_RHF and E_UHF are not close",
        )
        self.assertLess(
            numpy.linalg.norm(rmo - umo[1]),
            1.0e-4,
            msg="E_RHF and E_UHF are not close",
        )

        # construct active space
        nao = rmo.shape[0]
        nocc_a, nocc_b = mol.nelec

        # core_idx = numpy.arange(nocc_a)
        core_idx = numpy.arange(nocc_a // 2)
        n_vir = nao - nocc_a
        act_idx = range(core_idx[-1], nocc_a + n_vir // 4)
        # print(" core indices =   ", core_idx)
        # print(" active indices = ", act_idx)

        h1_rhf, ltensor_rhf, e_nuc_rhf = tools.get_h1e_chols_cas(
            mol, rmo, core_idx, act_idx
        )
        h1_uhf, ltensor_uhf, e_nuc_uhf = tools.get_h1e_chols_cas(
            mol, umo, (core_idx, core_idx), (act_idx, act_idx)
        )

        # print("Nuc energy (no frozen core):                ", mol.energy_nuc())
        # print("Nuc energy with frozen core energies (RHF): ", e_nuc_rhf)
        # print("Nuc energy with frozen core energies (UHF): ", e_nuc_uhf)
        # print(f"RHF/UHF energy: {rmf.e_tot:.8f}  {umf.e_tot:.8f}")

        self.assertLess(
            abs(e_nuc_rhf - e_nuc_uhf),
            1.0e-4,
            msg="Nuc energies with frozen core does not match",
        )
        self.assertLess(
            numpy.linalg.norm(h1_rhf - h1_uhf[0]),
            1.0e-4,
            msg="OEIs in CAS do not match",
        )
        self.assertLess(
            numpy.linalg.norm(ltensor_rhf - ltensor_uhf[0]),
            1.0e-4,
            msg="ltensors do not match",
        )


    def test_cas_qmc(self):
        basis = "631g"
        verbose = 1

        mol = get_mol(basis=basis, verbose=verbose, name="C2N2H6")
        rmf = scf.HF(mol)
        rmf.kernel()
        Cmo = rmf.mo_coeff

        # define CAS
        nao = Cmo.shape[0]
        nocc_a, nocc_b = mol.nelec
        core_idx = numpy.arange(nocc_a // 2)
        n_vir = nao - nocc_a
        act_idx = range(core_idx[-1], nocc_a + n_vir // 4)

        h1e, ltensor, e_nuc = tools.get_h1e_chols_cas(
            mol, Cmo, core_idx, act_idx
        )

        #-------------------------------------
        # DO AFQMC calculation in CAS
        #-------------------------------------

        # 1) define trial in CAS
        C_core = Cmo[:, core_idx]      # (nao, ncore)
        C_act  = Cmo[:, act_idx]       # (nao, nact)
        C_cas = C_act

        # 2) overwrite integrals

        # 3) set up afqmc
        time = 10.0
        num_walkers = 500
        energy_scheme = "local"
        uhf = False

        afqmc = AFQMC(
            mol,
            dt=0.005,
            total_time=time,
            num_walkers=num_walkers,
            energy_scheme=energy_scheme,
            uhf=uhf,
            chol_thresh=1.0e-20,
            property_calc_freq=1,
            verbose=mol.verbose,
        )
        # get mean energy
        # times, energies = afqmc.kernel()



if __name__ == "__main__":
    unittest.main()
