import numpy
import math
from scipy.spatial.transform import Rotation as R
from scipy.stats import truncnorm
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


def get_mol(natoms=2, bond=2.0, basis="sto6g", name="LiH",
    verbose=3,
    return_atom=False
):
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

    if return_atom: return atom

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



# functions for creating molecule ensemble of certian molecule

# 1. Parse molecule coordinates
def parse_coordinates(coord_str):
    atoms = []
    coords = []
    for line in coord_str.split(";"):
        parts = line.split()
        atoms.append(parts[0])
        coords.append([float(x) for x in parts[1:]])
    return atoms, numpy.array(coords)



# Calculate center of mass
# (assumes all atoms have same mass for simplicity)
def center_of_mass(coords):
    return numpy.mean(coords, axis=0)


# Sample rotation angle (radians) from a truncated normal distribution
def sample_rotation_angle(center_deg=0, half_width_deg=30):
    stddev = half_width_deg / 2  # approximate std dev
    lower, upper = (center_deg - half_width_deg, center_deg + half_width_deg)
    lower_rad, upper_rad = numpy.radians([lower, upper])
    center_rad = numpy.radians(center_deg)
    stddev_rad = numpy.radians(stddev)
    return truncnorm.rvs((lower_rad - center_rad)/stddev_rad, (upper_rad - center_rad)/stddev_rad, loc=center_rad, scale=stddev_rad)


# Rotate molecule randomly around its center of mass with biased angle
def rotate_molecule(coords, center_deg=0, half_width_deg=50.0):
    com = center_of_mass(coords)
    shifted_coords = coords - com
    axis = numpy.random.normal(size=3)
    axis /= numpy.linalg.norm(axis)  # normalize to unit vector
    angle = sample_rotation_angle(center_deg=center_deg, half_width_deg=half_width_deg)
    rot = R.from_rotvec(angle * axis).as_matrix()
    rotated_coords = shifted_coords @ rot.T
    return rotated_coords + com

# Create a molecular ensemble
def create_ensemble(atoms, coords, dims=(1, 1, 1), lattice=(5.0, 5.0, 5.0),
    center_deg = 0.0,
    half_width_deg=50,
):
    ensemble_atoms = []
    ensemble_coords = []

    nx, ny, nz = dims
    a, b, c = lattice
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                rotated = rotate_molecule(coords, center_deg=0.0, half_width_deg=50.0)
                shift = numpy.array([i * a, j * b, k * c])
                new_coords = rotated + shift
                ensemble_atoms.extend(atoms)
                ensemble_coords.extend(new_coords)
    return ensemble_atoms, numpy.array(ensemble_coords)

# Output function in XYZ format
def to_xyz(atoms, coords, comment="Generated ensemble"):
    lines = [str(len(atoms)), comment]
    for atom, coord in zip(atoms, coords):
        lines.append(f"{atom} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}")
    return "\n".join(lines)



if __name__ == "__main__":
    # Example usage
    atoms, coords = parse_coordinates(c2n2h6isomer_coords)
    dims = (5, 5, 2)
    num_molecules = math.prod(dims)
    num_atoms = len(atoms) * num_molecules
    print(f"num_molecules = {num_molecules:d}")
    print(f"num_atoms     = {num_atoms:d}")
    ensemble_atoms, ensemble_coords = create_ensemble(atoms, coords, dims=dims, lattice=(8.0, 8.0, 5.0))
    xyz_output = to_xyz(ensemble_atoms, ensemble_coords)
    print(xyz_output)
    with open("ensemble.xyz", "w") as f:
        f.write(xyz_output)
