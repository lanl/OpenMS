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

#
# Atomic weight
periodictable = { "xx" : 1.00794, "H" : 1.00794, "He" : 4.00260, "Li" : 6.941, "Be" : 9.012187, "B" : 10.811,
        "C" : 12.0107, "N" : 14.00674, "O" : 15.9994, "F" : 18.99840, "Ne" : 20.1797, "Na" : 22.98977,
        "Mg" : 24.3050, "Al" : 26.98152, "Si" : 28.0855, "P" : 30.97376, "S" : 32.066, "Cl" : 35.4527,
        "Ar" : 39.948, "K" : 39.0983, "Ca" : 40.078, "Sc" : 44.95591, "Ti" : 47.867, "V" : 50.9415,
        "Cr" : 51.9961, "Mn" : 54.93805, "Fe" : 55.845, "Co" : 58.93320, "Ni" : 58.6934, "Cu" : 63.546,
        "Zn" : 65.39, "Ga" : 69.723, "Ge" : 72.61, "As" : 74.92160, "Se" : 78.96, "Br" : 79.904,
        "Kr" : 83.80, "Rb" : 85.4678, "Sr" : 87.62, "Y" : 88.90585, "Zr" : 91.224, "Nb" : 92.90638,
        "Mo" : 95.94, "Tc" : 98.0, "Ru" : 101.07, "Rh" : 102.90550, "Pd" : 106.42, "Ag" : 107.8682,
        "Cd" : 112.411, "In" : 114.818, "Sn" : 118.710, "Sb" : 121.760, "Te" : 127.60, "I" : 126.90477,
        "Xe" : 131.29, "Cs" : 132.90545, "Ba" : 137.327, "La" : 138.9055, "Ce" : 140.116, "Pr" : 140.90765,
        "Nd" : 144.24, "Pm" : 145.0, "Sm" : 150.36, "Eu" : 151.964, "Gd" : 157.24, "Tb" : 158.92534,
        "Dy" : 162.50, "Ho" : 164.93032, "Er" : 167.26, "Tm" : 168.93421, "Yb" : 173.04, "Lu" : 174.967,
        "Hf" : 178.49, "Ta" : 180.9479, "W" : 183.84, "Re" : 186.207, "Os" : 190.23, "Ir" : 192.217,
        "Pt" : 195.078, "Au" : 196.96655, "Hg" : 200.59, "Tl" : 204.3833, "Pb" : 207.2, "Bi" :208.98038,
        "Po" : 209.0, "At" : 210.0, "Rn" : 222.0, "Fr" :223.0, "Ra" : 226.0, "Ac" : 227.0,
        "Th" : 232.0381, "Pa" : 231.03588, "U" : 238.0289, "Np" : 237.0, "Pu" : 244.0, "Am" : 243.0,
        "Cm" : 247.0, "Bk" : 247.0, "Cf" : 251.0, "Es" : 252.0, "Fm" : 257.0, "Md" : 258.0,
        "No" : 259.0, "Lr" : 262.0, "Rf" : 261.0, "Db" : 262.0, "Sg" : 263.0, "Bh" : 264.0,
        "Hs" : 265.0, "Mt" : 268.0, "Ds" : 271.0, "Rg" : 272.0, "Uub" : 285.0, "Uut" : 284.0,
        "Uuq" : 289.0, "Uup" : 288.0, "Uuh" : 292.0}

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

periodictable.update({n : amu2au * periodictable[n] for n in periodictable.keys()})

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
       self.coord = self.atom_coords()
       self.pos = self.coord # todo: remove pos 
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
       self.states = [State(self.ndim, self.natm) for i in range(self.nstates)]

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
