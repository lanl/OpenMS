#
# @ 2023. Triad National Security, LLC. All rights reserved.
#
# This program was produced under U.S. Government contract 89233218CNA000001
# for Los Alamos National Laboratory (LANL), which is operated by Triad
# National Security, LLC for the U.S. Department of Energy/National Nuclear
# Security Administration. All rights in the program are reserved by Triad
# National Security, LLC, and the U.S. Department of Energy/National Nuclear
# Security Administration. The Government is granted for itself and others acting
# on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this
# material to reproduce, prepare derivative works, distribute copies to the
# public, perform publicly and display publicly, and to permit others to do so.
#
# Author: Yu Zhang <zhy@lanl.gov>
#

import sys
import copy
import numpy
from openms import __config__
from pyscf import lib

from openms.mqed import qedhf

from pyscf.scf import hf
from pyscf.dft import rks
# from mqed.lib      import logger

r"""
Theoretical background
^^^^^^^^^^^^^^^^^^^^^^

TBA.

"""

TIGHT_GRAD_CONV_TOL = getattr(__config__, "scf_hf_kernel_tight_grad_conv_tol", True)

# in the future, replace it with our own object?
class TDMixin(lib.StreamObject):
    conv_tol = getattr(__config__, "tdscf_rhf_TDA_conv_tol", 1e-9)
    nstates = getattr(__config__, "tdscf_rhf_TDA_nstates", 3)
    singlet = getattr(__config__, "tdscf_rhf_TDA_singlet", True)
    lindep = getattr(__config__, "tdscf_rhf_TDA_lindep", 1e-12)
    level_shift = getattr(__config__, "tdscf_rhf_TDA_level_shift", 0)
    max_space = getattr(__config__, "tdscf_rhf_TDA_max_space", 50)
    max_cycle = getattr(__config__, "tdscf_rhf_TDA_max_cycle", 100)


from openms.lib.boson import Photon

class RHF(qedhf.RHF):
    # class HF(lib.StreamObject):
    r"""
    QEDSCF base class.   non-relativistic RHF.

    """

    def __init__(self, mol, xc=None, **kwargs):
        qedhf.RHF.__init__(self, mol, xc, **kwargs)


    # functions specific to ms_qedhf to be implemented



if __name__ == "__main__":
    # will add a loop
    import numpy
    from pyscf import gto, scf

    itest = 1
    zshift = itest * 2.0

    atom = f"H          0.86681        0.60144        {5.00000+zshift};\
        F         -0.86681        0.60144        {5.00000+zshift};\
        O          0.00000       -0.07579        {5.00000+zshift};\
        He         0.00000        0.00000        {7.50000+zshift}"

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
        verbose=3,
    )
    print("mol coordinates=\n", mol.atom_coords())

    hf = scf.HF(mol)
    hf.max_cycle = 200
    hf.conv_tol = 1.0e-8
    hf.diis_space = 10
    hf.polariton = True
    mf = hf.run(verbose=4)

    print("electronic energies=", mf.energy_elec())
    print("nuclear energy=     ", mf.energy_nuc())
    dm = mf.make_rdm1()

    print("\n=========== QED-HF calculation  ======================\n")

    from openms.mqed import ms_qedhf as qedhf

    nmode = 2 # create a zero (second) mode to test the code works for multiple modes
    cavity_freq = numpy.zeros(nmode)
    cavity_mode = numpy.zeros((nmode, 3))
    cavity_freq[0] = 0.5
    cavity_mode[0, :] = 0.1 * numpy.asarray([0, 0, 1])

    qedmf = qedhf.RHF(mol, xc=None, cavity_mode=cavity_mode, cavity_freq=cavity_freq)
    qedmf.max_cycle = 500
    qedmf.kernel(dm0=dm)
