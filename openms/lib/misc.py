#
# @ 2023. Triad National Security, LLC. All rights reserved.
#
#This program was produced under U.S. Government contract 89233218CNA000001 
# for Los Alamos National Laboratory (LANL), which is operated by Triad 
#National Security, LLC for the U.S. Department of Energy/National Nuclear 
#Security Administration. All rights in the program are reserved by Triad 
#National Security, LLC, and the U.S. Department of Energy/National Nuclear 
#Security Administration. The Government is granted for itself and others acting 
#on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this 
#material to reproduce, prepare derivative works, distribute copies to the 
#public, perform publicly and display publicly, and to permit others to do so.
#
# Author: Yu Zhang <zhy@lanl.gov>
#

import os, sys
import ctypes
import numpy
import textwrap

from openms import __config__

c_double_p = ctypes.POINTER(ctypes.c_double)
c_int_p = ctypes.POINTER(ctypes.c_int)
c_null_ptr = ctypes.POINTER(ctypes.c_void_p)

def load_library(libname):
    try:
        _loaderpath = os.path.dirname(__file__)
        print('debug-zy: __file__=', __file__)
        print('debug-zy: _loaderpath=', _loaderpath)
        return numpy.ctypeslib.load_library(libname, _loaderpath)
    except OSError:
        from openms import __path__ as ext_modules
        for path in ext_modules:
            libpath = os.path.join(path, 'lib')
            if os.path.isdir(libpath):
                for files in os.listdir(libpath):
                    if files.startswith(libname):
                        return numpy.ctypeslib.load_library(libname, libpath)
        raise

# Atomic weight
periodictable = {
    "xx": {"z": 1, "m": 1.00794},
    "H": {"z": 1, "m": 1.00794},
    "He": {"z": 2, "m": 4.0026},
    "Li": {"z": 3, "m": 6.941},
    "Be": {"z": 4, "m": 9.012187},
    "B": {"z": 5, "m": 10.811},
    "C": {"z": 6, "m": 12.0107},
    "N": {"z": 7, "m": 14.00674},
    "O": {"z": 8, "m": 15.9994},
    "F": {"z": 9, "m": 18.9984},
    "Ne": {"z": 10, "m": 20.1797},
    "Na": {"z": 11, "m": 22.98977},
    "Mg": {"z": 12, "m": 24.305},
    "Al": {"z": 13, "m": 26.98152},
    "Si": {"z": 14, "m": 28.0855},
    "P": {"z": 15, "m": 30.97376},
    "S": {"z": 16, "m": 32.066},
    "Cl": {"z": 17, "m": 35.4527},
    "Ar": {"z": 18, "m": 39.948},
    "K": {"z": 19, "m": 39.0983},
    "Ca": {"z": 20, "m": 40.078},
    "Sc": {"z": 21, "m": 44.95591},
    "Ti": {"z": 22, "m": 47.867},
    "V": {"z": 23, "m": 50.9415},
    "Cr": {"z": 24, "m": 51.9961},
    "Mn": {"z": 25, "m": 54.93805},
    "Fe": {"z": 26, "m": 55.845},
    "Co": {"z": 27, "m": 58.9332},
    "Ni": {"z": 28, "m": 58.6934},
    "Cu": {"z": 29, "m": 63.546},
    "Zn": {"z": 30, "m": 65.39},
    "Ga": {"z": 31, "m": 69.723},
    "Ge": {"z": 32, "m": 72.61},
    "As": {"z": 33, "m": 74.9216},
    "Se": {"z": 34, "m": 78.96},
    "Br": {"z": 35, "m": 79.904},
    "Kr": {"z": 36, "m": 83.8},
    "Rb": {"z": 37, "m": 85.4678},
    "Sr": {"z": 38, "m": 87.62},
    "Y": {"z": 39, "m": 88.90585},
    "Zr": {"z": 40, "m": 91.224},
    "Nb": {"z": 41, "m": 92.90638},
    "Mo": {"z": 42, "m": 95.94},
    "Tc": {"z": 43, "m": 98.0},
    "Ru": {"z": 44, "m": 101.07},
    "Rh": {"z": 45, "m": 102.9055},
    "Pd": {"z": 46, "m": 106.42},
    "Ag": {"z": 47, "m": 107.8682},
    "Cd": {"z": 48, "m": 112.411},
    "In": {"z": 49, "m": 114.818},
    "Sn": {"z": 50, "m": 118.71},
    "Sb": {"z": 51, "m": 121.76},
    "Te": {"z": 52, "m": 127.6},
    "I": {"z": 53, "m": 126.90477},
    "Xe": {"z": 54, "m": 131.29},
    "Cs": {"z": 55, "m": 132.90545},
    "Ba": {"z": 56, "m": 137.327},
    "La": {"z": 57, "m": 138.9055},
    "Ce": {"z": 58, "m": 140.116},
    "Pr": {"z": 59, "m": 140.90765},
    "Nd": {"z": 60, "m": 144.24},
    "Pm": {"z": 61, "m": 145.0},
    "Sm": {"z": 62, "m": 150.36},
    "Eu": {"z": 63, "m": 151.964},
    "Gd": {"z": 64, "m": 157.24},
    "Tb": {"z": 65, "m": 158.92534},
    "Dy": {"z": 66, "m": 162.5},
    "Ho": {"z": 67, "m": 164.93032},
    "Er": {"z": 68, "m": 167.26},
    "Tm": {"z": 69, "m": 168.93421},
    "Yb": {"z": 70, "m": 173.04},
    "Lu": {"z": 71, "m": 174.967},
    "Hf": {"z": 72, "m": 178.49},
    "Ta": {"z": 73, "m": 180.9479},
    "W": {"z": 74, "m": 183.84},
    "Re": {"z": 75, "m": 186.207},
    "Os": {"z": 76, "m": 190.23},
    "Ir": {"z": 77, "m": 192.217},
    "Pt": {"z": 78, "m": 195.078},
    "Au": {"z": 79, "m": 196.96655},
    "Hg": {"z": 80, "m": 200.59},
    "Tl": {"z": 81, "m": 204.3833},
    "Pb": {"z": 82, "m": 207.2},
    "Bi": {"z": 83, "m": 208.98038},
    "Po": {"z": 84, "m": 209.0},
    "At": {"z": 85, "m": 210.0},
    "Rn": {"z": 86, "m": 222.0},
    "Fr": {"z": 87, "m": 223.0},
    "Ra": {"z": 88, "m": 226.0},
    "Ac": {"z": 89, "m": 227.0},
    "Th": {"z": 90, "m": 232.0381},
    "Pa": {"z": 91, "m": 231.03588},
    "U": {"z": 92, "m": 238.0289},
    "Np": {"z": 93, "m": 237.0},
    "Pu": {"z": 94, "m": 244.0},
    "Am": {"z": 95, "m": 243.0},
    "Cm": {"z": 96, "m": 247.0},
    "Bk": {"z": 97, "m": 247.0},
    "Cf": {"z": 98, "m": 251.0},
    "Es": {"z": 99, "m": 252.0},
    "Fm": {"z": 100, "m": 257.0},
    "Md": {"z": 101, "m": 258.0},
    "No": {"z": 102, "m": 259.0},
    "Lr": {"z": 103, "m": 262.0},
    "Rf": {"z": 104, "m": 261.0},
    "Db": {"z": 105, "m": 262.0},
    "Sg": {"z": 106, "m": 263.0},
    "Bh": {"z": 107, "m": 264.0},
    "Hs": {"z": 108, "m": 265.0},
    "Mt": {"z": 109, "m": 268.0},
    "Ds": {"z": 110, "m": 271.0},
    "Rg": {"z": 111, "m": 272.0},
    "Uub": {"z": 112, "m": 285.0},
    "Uut": {"z": 113, "m": 284.0},
    "Uuq": {"z": 114, "m": 289.0},
    "Uup": {"z": 115, "m": 288.0},
    "Uuh": {"z": 116, "m": 292.0},
}

# Conversion units
amu2au = 1822.888486192
au2A = 0.529177249
A2au = 1 / au2A
au2fs = 0.02418884344
fs2au = 1 / au2fs # = 41.34137304
au2K = 3.15774646E+5
au2eV = 27.2113961
eV2au = 1 / au2eV # = 0.03674931
au2kcalmol = 627.503
kcalmol2au = 1 / au2kcalmol # = 0.00159362

# Speed of light in atomic unit
c = 137.035999108108

# Frequency unit
cm2au = 1.0E-8 * au2A * c
eps = 1.0E-12

for k, v in periodictable.items():
    v["m"] *= amu2au

def wall_time(func):
    @wraps(func)
    def timer(*args, **kwargs):
        tbegin = time.time()
        func(*args, **kwargs)
        tend = time.time()
        print (f"{func.__name__} : Elapsed time = {tend - tbegin} seconds", flush=True)
    return timer

def gaussian1d(x, const, sigma, x0):
    if (sigma < 0.0):
        return -1
    else:
        res = const / (sigma * numpy.sqrt(2. * numpy.pi)) * numpy.exp(- (x - x0) ** 2 / (2. * sigma ** 2))
        return res

def Lorentz1d(x, const, sigma, x0):
    if (sigma < 0.0):
        return -1
    else:
        res = const * sigma / ((x-x0)**2 + sigma * sigma)
        return res

def call_name():
    return sys._getframe(1).f_code.co_name

def typewriter(string, dir_name, filename, mode):
    """ Function to open/write any string in dir_name/filename

        :param string string: Text string for output file
        :param string dir_name: Directory of output file
        :param string filename: Filename of output file
        :param string mode: Fileopen mode
    """
    tmp_name = os.path.join(dir_name, filename)
    with open(tmp_name, mode) as f:
        f.write(string + "\n")


# derived molecule class
from pyscf.gto import mole
from pyscf import data
class Molecule(mole.Mole):

    def __init__(self, **kwargs):
       super().__init__(**kwargs)
       self.mol_type = self.__class__.__name__
       lmodel  = False
       if "lmodel" in kwargs: lmodel = kwargs["lmodel"]
       self.ndim = kwargs["ndim"] if "ndim" in kwargs else 3
       self.lmodel = lmodel 
       if lmodel:
          print(f"this is model system in {self.ndim} dimension")

       self.veloc = None # velocity
       self.mass = None

       self.build(**kwargs)

       self.lqmmm = False
       self.coords = self.atom_coords()
       print(f"no. of atoms is {self.natm}")
#
    def build(self, **kwargs):
       #if not self.lmodel:
       print("using pyscf Mole.build()!")
       child_args = ["lmodel", "nstates", "ndim", "veloc"]
       parent_args = {k: v for k, v in kwargs.items() if k not in child_args}

       super().build(**parent_args)
       #else:
       #   print("own model build")

       nstates = kwargs["nstates"] if "nstates" in kwargs else 1
       self.nstates = nstates
       # self.states = [State(self.ndim, self.natm) for i in range(self.nstates)]

       #
       self.ndof = self.ndim * self.natm

       # Initialize other properties
       self.nac = numpy.zeros((self.nstates, self.nstates, self.natm, self.ndim))
       self.nac_old = numpy.zeros((self.nstates, self.nstates, self.natm, self.ndim))
       self.rho = numpy.zeros((self.nstates, self.nstates), dtype=numpy.complex128)

       self.nacme = numpy.zeros((self.nstates, self.nstates))
       self.nacme_old = numpy.zeros((self.nstates, self.nstates))

       # initialize velocities
       if self.veloc is None:
           self.veloc = numpy.full((self.natm, self.ndim), 0.0)

       self.mass = numpy.array([
           data.elements.COMMON_ISOTOPE_MASSES[m] * data.nist.AMU2AU
           for m in self.atom_charges()])

       self.ekin = 0.
       self.ekin_qm = 0.
       self.epot = 0.
       self.etot = 0.
       self.lnacme = False

    def get_ekin(self):
        """Compute kinetic energy
        """
        self.ekin = numpy.sum(0.5 * self.mass * numpy.sum(self.veloc ** 2, axis=1))

    def reset_bo(self, calc_coupling):
        """ Reset BO energies, forces and nonadiabatic couplings

            :param boolean calc_coupling: Check whether the dynamics includes coupling calculation
        """
        for states in self.states:
            states.energy = 0.
            states.force = numpy.zeros((self.natm, self.ndim))

        if (calc_coupling):
            if (self.lnacme):
                self.nacme = numpy.zeros((self.nstates, self.nstates))
            else:
                self.nac = numpy.zeros((self.nstates, self.nstates, self.natm, self.ndim))

    def __str__(self):
        """Print function for the molecular class (print the information about molecule)

        """

        header = textwrap.dedent(f"""\
        {"-" * 68}
        {"Initial Coordinate and Velocity (au)":>45s}
        {"-" * 68}\n""")
        header += " Atom "
        xyzlabel = ["X", "Y", "Z"]
        for i in range(self.ndim):
            if i == 0:
                header += f"{xyzlabel[i]:>6s}"
            else:
                header += f"{xyzlabel[i]:>14s}"
        xyzlabel = ["VX", "VY", "VZ"]
        for i in range(self.ndim):
            header += f"{xyzlabel[i]:>14s}"
        header += f"{'Mass':>16s}\n"
        geom_info = textwrap.dedent(header)

        for nth, atoms in enumerate(self._atom):
            #geom_info += f"  {atoms:3s}"
            geom_info += f" {atoms[0]}"
            for isp in range(self.ndim):
                geom_info += f"{self.pos[nth, isp]:14.6f}"
            for isp in range(self.ndim):
                geom_info += f"{self.veloc[nth, isp]:14.6f}"
            geom_info += f"{self.mass[nth]:15.5f}\n"
        #print (geom_info, flush=True)

        molecule_info = textwrap.dedent(f"""\
        {"-" * 68}
        {"Molecule Information":>43s}
        {"-" * 68}
          Number of Atoms (QM)     = {self.natm:>16d}
          multiplicity (QM)        = {self.spin:>16d}
        """)
        molecule_info += textwrap.indent(textwrap.dedent(f"""\
          Degrees of Freedom       = {int(self.ndof):>16d}
          Charge                   = {(self.charge)}
          Number of Electrons      = {(self.nelec)}
          Number of States         = {self.nstates}
        """), "  ")

        # Model case (todo)

        return geom_info + molecule_info

    def print_init(self):
        print(self)
