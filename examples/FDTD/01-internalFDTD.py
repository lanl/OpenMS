import inspect
import os
import sys

import numpy as np

# test configure file
from openms import __config__
from openms.lib import FDTD

# Get the path of the module
module_path = inspect.getfile(FDTD)
# Get the directory containing the module
module_dir = os.path.dirname(module_path)

"""
def far_field_param(OMEGA, DETECT):
    if not isinstance(OMEGA, np.ndarray) or OMEGA.dtype != np.float32 or OMEGA.ndim != 1:
        print(OMEGA.ndim, OMEGA.dtype)
        raise TypeError("OMEGA must be a 1-D np float32 array")
    OMEGA_ptr = OMEGA.__array_interface__['data'][0]
    OMEGA_carray = FDTD.float_array_from_pointer(OMEGA_ptr, OMEGA.size)
    DETECT_c = DETECT.astype(np.float32).ctypes.data_as(FDTD.floatp)
    FDTD.far_field_param(OMEGA_carray, DETECT_c)
"""


FDTD.cvar.shift = 10.0
print(FDTD.cvar.shift)


# Basic Parameters

N = 12
# bottom = 2.25
shift = 0.0

# dipole source parameter
DT = 50
DS = DT * 6
DD = 1500
WC = 0.33


# far-field calculations
specN = 2
FDTD.cvar.SpecN = specN

# OMEGA as a pointer in fdtd code, this pass this OMEGA directly C does not work
OMEGA = np.zeros(specN, dtype=np.float32)

# set frequencies
OMEGA[0] = 0.3179
OMEGA[1] = 0.3442


DETECT = 0.27
NROW = 512
NA = 0.85
Nfree = 1.0

# FDTD.structure_size(N,N,4+bottom);
FDTD.structure_size(N, N, 3)
FDTD.lattice_size(10, 10, 10)
FDTD.pml_size(10, 10, 10, 10, 10, 10)
FDTD.set_default_parameter(2)
FDTD.Hz_parity(1, 1, -1)
# // parities for Hz-field //
FDTD.memory()
FDTD.real_space_param(1, WC)


# structure

# slab structure
R = 0.35
T = 0.5
Rm = 0.25

EMP = "dummy"

FDTD.background(1.0)
# dielectric slab
FDTD.input_object("block", EMP, 0, 0, 0 + shift, N, N, T, 11.56)

# periodic lattice
for x in range(-N // 2, N // 2):
    for y in range(-N // 2, N // 2):
        FDTD.input_object("rod", EMP, x, np.sqrt(3) * y, 0 + shift, R, T, 0, 1)
        FDTD.input_object(
            "rod", EMP, x - 0.5, np.sqrt(3) * (y + 0.5), 0 + shift, R, T, 0, 1
        )

# fill
FDTD.input_object("rod", EMP, 0, 0, 0 + shift, (R + 0.01), T, 0, 11.56)
FDTD.input_object("rod", EMP, -1, 0, 0 + shift, (R + 0.01), T, 0, 11.56)
FDTD.input_object("rod", EMP, 1, 0, 0 + shift, (R + 0.01), T, 0, 11.56)
FDTD.input_object("rod", EMP, 0.5, np.sqrt(3) * 0.5, 0 + shift, (R + 0.01), T, 0, 11.56)
FDTD.input_object(
    "rod", EMP, -0.5, np.sqrt(3) * 0.5, 0 + shift, (R + 0.01), T, 0, 11.56
)
FDTD.input_object(
    "rod", EMP, 0.5, -np.sqrt(3) * 0.5, 0 + shift, (R + 0.01), T, 0, 11.56
)
FDTD.input_object(
    "rod", EMP, -0.5, -np.sqrt(3) * 0.5, 0 + shift, (R + 0.01), T, 0, 11.56
)

# dig
FDTD.input_object("rod", EMP, (1 + R - Rm) * -1, 0, 0 + shift, Rm, T, 0, 1)  # ; //1
FDTD.input_object("rod", EMP, (1 + R - Rm) * 1, 0, 0 + shift, Rm, T, 0, 1)  # ; //2
FDTD.input_object(
    "rod", EMP, (1 + R - Rm) * 0.5, 1.1 * np.sqrt(3) * 0.5, 0 + shift, Rm, T, 0, 1
)  # ; //3
FDTD.input_object(
    "rod", EMP, (1 + R - Rm) * -0.5, 1.1 * np.sqrt(3) * 0.5, 0 + shift, Rm, T, 0, 1
)  # ; //4
FDTD.input_object(
    "rod", EMP, (1 + R - Rm) * 0.5, 1.1 * -np.sqrt(3) * 0.5, 0 + shift, Rm, T, 0, 1
)  # ; //5
FDTD.input_object(
    "rod", EMP, (1 + R - Rm) * -0.5, 1.1 * -np.sqrt(3) * 0.5, 0 + shift, Rm, T, 0, 1
)  # ; //6

"""
# real structure
#input_object("contour","mono_new",231,189,0,60,0.551,(20/30.615),1.0);
#input_object("contour","post_matrix",224,203,-1.25,10,2.5,(20/26.6)*(32.37/285)*0.55,11.56);

# side blocks
FDTD.input_object("block",EMP,(N/2),0,0+shift,1.2,N+1,T,11.56);
FDTD.input_object("block",EMP,-(N/2),0,0+shift,1.2,N+1,T,11.56);
FDTD.input_object("block",EMP,0,(N/2),0+shift,N+1,1.2,T,11.56);
FDTD.input_object("block",EMP,0,-(N/2),0+shift,N+1,1.2,T,11.56);
"""

FDTD.make_epsilon()
# FDTD.make_metal_structure();

FDTD.out_epsilon("x", 0, "epsilon.x")
FDTD.out_epsilon("y", 0, "epsilon.y")
FDTD.out_epsilon("z", 0 + shift, "epsilon.z")

FDTD.coefficient()

# FDTD propagation

t = 0
while t < DD:
    # pass the var to c
    FDTD.cvar.t = t

    # add dipole source
    # FDTD.Gaussian_dipole_source("Hz",-0,1.0,0+shift,WC,0,3*DT,DT);
    FDTD.Gaussian_dipole_source("Hz", -0, -0.5, 0 + shift, WC, 0, 3 * DT, DT)
    FDTD.Gaussian_dipole_source("Hz", -0.2, -0.2, 0 + shift, WC, 0, 3 * DT, DT)
    FDTD.Gaussian_dipole_source("Hz", -0.4, -0.3, 0 + shift, WC, 0, 3 * DT, DT)

    FDTD.propagate()

    FDTD.out_point("Hz", -0, -0.5, 0 + shift, 0, DS, "source.dat")
    FDTD.out_point("Hz", -0, -0.5, 0 + shift, DS, DD, "mode1.dat")

    if (DS + 100) < t and (t < DD):
        FDTD.far_field_param(OMEGA, DETECT + shift)

    # if DD-300<t and t<DD:  # Qv and Qh calculation
    # 	FDTD.total_E_energy()
    # 	FDTD.total_E2()
    # 	FDTD.Poynting_total()
    # 	FDTD.Poynting_side(0.75,0)  #half width of the strip

    # if DD-10< t and t<DD and t%2==0:
    # 	FDTD.out_plane("Hz","z",0+shift,".Hz");

    t += 1


# tHx_real, tEx_real
FDTD.print_real_and_imag(0)
# Hx_real, Ex_real
FDTD.print_real_and_imag_2n_size(NROW, 0)
FDTD.far_field_FFT(NROW, NA, Nfree, OMEGA, 0)

# FFT
FDTD.transform_farfield(NROW, 101, "rad_tot_", 0)
FDTD.transform_farfield(NROW, 101, "rad_Ex_", 0)
FDTD.transform_farfield(NROW, 101, "rad_Ey_", 0)
FDTD.transform_farfield(NROW, 101, "rad_Et_", 0)
FDTD.transform_farfield(NROW, 101, "rad_Ep_", 0)

print("Calculation Complete!\n")
