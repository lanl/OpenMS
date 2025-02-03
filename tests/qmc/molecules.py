import numpy
from pyscf import gto


# D = 179
c2n2h6isomer_coords = f"C  1.7913090545   -0.0745398644    0.0184596800;\
       N 0.4277379156    0.4416819705    0.0067776063;\
       N -0.4277379156   -0.4416819705    0.0067776063;\
       C -1.7913090545    0.0745398644    0.0184596800;\
       H 2.3109976118    0.3289846823   -0.8541305852;\
       H 1.8171632620   -1.1680305269    0.0140500788;\
       H 2.2884450671    0.3205358657    0.9079354075;\
       H -2.3109976118   -0.3289846823   -0.8541305852;\
       H -1.8171632620    1.1680305269    0.0140500788;\
       H -2.2884450671   -0.3205358657    0.9079354075"


def get_mol(natoms=2, bond=2.0, basis="sto6g", name="LiH", verbose=3):
    LiH_coords = f"Li 0.0    0.0     0.0; H 0.0  0.0 {bond}"
    LiF_coords = f"Li 0.0    0.0     0.0; F 0.0  0.0 {bond}"
    H2_coords = f"H 0.0    0.0     0.0; H 0.0  0.0 {bond}"
    HF_coords = f"H 0.0    0.0     0.0; F 0.0  0.0 {bond}"

    """
    if "chain" in name:
        element = name.split("chain")[0]
        if len(element) < 3:
            chain_coords = [(element, i * bond, 0, 0) for i in range(natoms)]
    """

    # Hydrogen chian
    Hchain_coords = [("H", i * bond, 0, 0) for i in range(natoms)]

    # here we use bond as the variable to shift the molecule from origin
    C3H4O2_coords = f"C   0.00000000   0.00000000    {bond};\
        O   0.00000000   1.23456800    {bond};\
        H   0.97075033  -0.54577032    {bond};\
        C  -1.21509881  -0.80991169    {bond};\
        H  -1.15288176  -1.89931439    {bond};\
        C  -2.43440063  -0.19144555    {bond};\
        H  -3.37262777  -0.75937214    {bond};\
        O  -2.62194056   1.12501165    {bond};\
        H  -1.71446384   1.51627790    {bond}"


    # isomer

    coords = {
      "LiH": LiH_coords,
      "LiF": LiF_coords,
      "H2": H2_coords,
      "HF": HF_coords,
      "C3H4O2" : C3H4O2_coords,
      "Hchain": Hchain_coords,
      "Hydrogen_chain": Hchain_coords,
      "C2N2H6": c2n2h6isomer_coords,
      # and more TBA
    }

    atom = coords[name]

    mol = gto.M(
        atom=atom,
        basis=basis,
        # basis="cc-pvdz",
        unit="Angstrom",
        symmetry=True,
        verbose=verbose,
    )
    return mol


def get_cavity(nmode, gfac, omega=0.5, pol_axis=2):
    r"""Return cavity properties"""

    cavity_freq = numpy.zeros(nmode)
    cavity_mode = numpy.zeros((nmode, 3))
    cavity_freq[0] = omega

    cavity_mode[:, pol_axis] = gfac
    return cavity_freq, cavity_mode
