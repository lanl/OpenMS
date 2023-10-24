import numpy

MHZ2KHZ = 1000

joule2khz = 1.50919E+30
hartree2khz = 6.57968E+12
rydberg2khz = 3.28984E+12
ev2khz = 2.41799E+11
kjpermole2khz = 2.50607E+09
kcalpermole2khz = 1.04854E+10
hz2khz = 1.00000E-03
khz2khz = 1.
mhz2khz = 1.00000E+03
ghz2khz = 1.00000E+06
thz2khz = 1.00000E+09
phz2khz = 1.00000E+12
wavenumber2khz = 2.99792E+07

m2angstrom = 1.00000E+10
bohr2angstrom = 5.29177E-01
a2angstrom = 1

TWOPI = numpy.pi * 2
HBAR_SI = 6.62607015e-34 / TWOPI
BOHR_MAGNETON = 9.274009994E-24
NUCLEAR_MAGNETON = 5.05078366E-27

HARTREE2MHZ = 6579680000.0
M2BOHR = 18897300000.0

HBAR_MU0_O4PI = 1.05457172  # When everything else in rad, kHz, ms, G, A

COMPLEX_DTYPE = numpy.complex128

BARN2BOHR2 = M2BOHR ** 2 * 1E-28
EFG_CONVERSION = BARN2BOHR2 * HARTREE2MHZ * MHZ2KHZ  # units to convert EFG


ELECTRON_GYRO = -17608.597050  # rad / (ms * Gauss) or rad * kHz / G
#electronic Gyromagnetic ratio \gamma_e
ELEC_GYRO = -1.76085963023e5  # rad * MHz / T
#\gamma_e/2PI =
ELEC_GFAC = -2.00231930426256 # unit less


