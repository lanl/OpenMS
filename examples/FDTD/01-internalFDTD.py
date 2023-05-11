import numpy
import os, sys
import inspect

import numpy as np
import openms.lib.fdtd.fdtdc as pFDTD

# test configure file
from openms import __config__

# Get the path of the module
module_path = inspect.getfile(pFDTD)
# Get the directory containing the module
module_dir = os.path.dirname(module_path)


'''
def far_field_param(OMEGA, DETECT):
    if not isinstance(OMEGA, np.ndarray) or OMEGA.dtype != np.float32 or OMEGA.ndim != 1:
        print(OMEGA.ndim, OMEGA.dtype)
        raise TypeError("OMEGA must be a 1-D NumPy float32 array")
    OMEGA_ptr = OMEGA.__array_interface__['data'][0]
    OMEGA_carray = pFDTD.float_array_from_pointer(OMEGA_ptr, OMEGA.size)
    DETECT_c = DETECT.astype(np.float32).ctypes.data_as(pFDTD.floatp)
    pFDTD.far_field_param(OMEGA_carray, DETECT_c)
'''


pFDTD.cvar.shift = 10.0
print(pFDTD.cvar.shift)


# Basic Parameters 

N = 12
#bottom = 2.25
shift = 0.0

# dipole source parameter
DT=50
DS=DT*6
DD=1500
WC=0.33


# far-field calculations
specN = 2

# OMEGA as a pointer in fdtd code, this pass this OMEGA directly C does not work
OMEGA = np.zeros(specN, dtype=np.float32)

# set frequencies
OMEGA[0] = 0.3179
OMEGA[1] = 0.3442


DETECT = 0.27; 
NROW = 512;
NA = 0.85;
Nfree = 1.0;

#pFDTD.structure_size(N,N,4+bottom); 
pFDTD.structure_size(N,N,3); 
pFDTD.lattice_size(10,10,10);
pFDTD.pml_size(10,10,10,10,10,10);
pFDTD.set_default_parameter(2);
pFDTD.Hz_parity(1,1,-1);  #// parities for Hz-field //
pFDTD.memory();
pFDTD.real_space_param(1, WC);


#/////// Input Structure /////////

#////// slab structure //////////
R = 0.35
T = 0.5
Rm = 0.25
#////////////////////////////////

EMP = "dummy"

pFDTD.background(1.0);
#// dielectric slab
pFDTD.input_object("block", EMP ,0,0,0+shift,N,N,T,11.56);
#//input_object("block",EMP,0,0,1-(4+bottom)/2,N,N,2,11.56);

#// periodic lattice
for x in range(-N//2, N//2):
    for y in range(-N//2, N//2):
        pFDTD.input_object("rod",EMP,x,numpy.sqrt(3)*y,0+shift,R,T,0,1)
        pFDTD.input_object("rod",EMP,x-0.5,numpy.sqrt(3)*(y+0.5),0+shift,R,T,0,1)

# fill
pFDTD.input_object("rod",EMP,0,0,0+shift,(R+0.01),T,0,11.56)
pFDTD.input_object("rod",EMP,-1,0,0+shift,(R+0.01),T,0,11.56)
pFDTD.input_object("rod",EMP,1,0,0+shift,(R+0.01),T,0,11.56)
pFDTD.input_object("rod",EMP,0.5,numpy.sqrt(3)*0.5,0+shift,(R+0.01),T,0,11.56)
pFDTD.input_object("rod",EMP,-0.5,numpy.sqrt(3)*0.5,0+shift,(R+0.01),T,0,11.56)
pFDTD.input_object("rod",EMP,0.5,-numpy.sqrt(3)*0.5,0+shift,(R+0.01),T,0,11.56)
pFDTD.input_object("rod",EMP,-0.5,-numpy.sqrt(3)*0.5,0+shift,(R+0.01),T,0,11.56)

# dig
pFDTD.input_object("rod",EMP,(1+R-Rm)*-1,0,0+shift,Rm,T,0,1) #; //1
pFDTD.input_object("rod",EMP,(1+R-Rm)*1,0,0+shift,Rm,T,0,1) #; //2
pFDTD.input_object("rod",EMP,(1+R-Rm)*0.5,1.1*numpy.sqrt(3)*0.5,0+shift,Rm,T,0,1) #; //3
pFDTD.input_object("rod",EMP,(1+R-Rm)*-0.5,1.1*numpy.sqrt(3)*0.5,0+shift,Rm,T,0,1) #; //4
pFDTD.input_object("rod",EMP,(1+R-Rm)*0.5,1.1*-numpy.sqrt(3)*0.5,0+shift,Rm,T,0,1) #; //5
pFDTD.input_object("rod",EMP,(1+R-Rm)*-0.5,1.1*-numpy.sqrt(3)*0.5,0+shift,Rm,T,0,1) #; //6
	
'''
# real structure
#input_object("contour","mono_new",231,189,0,60,0.551,(20/30.615),1.0);
#input_object("contour","post_matrix",224,203,-1.25,10,2.5,(20/26.6)*(32.37/285)*0.55,11.56);

#// side blocks
pFDTD.input_object("block",EMP,(N/2),0,0+shift,1.2,N+1,T,11.56);
pFDTD.input_object("block",EMP,-(N/2),0,0+shift,1.2,N+1,T,11.56);
pFDTD.input_object("block",EMP,0,(N/2),0+shift,N+1,1.2,T,11.56);
pFDTD.input_object("block",EMP,0,-(N/2),0+shift,N+1,1.2,T,11.56);
'''

pFDTD.make_epsilon();
#pFDTD.make_metal_structure();
	
pFDTD.out_epsilon("x",0,"epsilon.x");
pFDTD.out_epsilon("y",0,"epsilon.y");
pFDTD.out_epsilon("z",0+shift,"epsilon.z");

pFDTD.coefficient(); 


# FDTD propagation

t = 0
while t<DD:
    # pass the var to c
    pFDTD.cvar.t = t

    # add dipole source
    pFDTD.Gaussian_dipole_source("Hz",-0,-0.5,0+shift,WC,0,3*DT,DT);
    pFDTD.Gaussian_dipole_source("Hz",-0.2,-0.2,0+shift,WC,0,3*DT,DT);
    pFDTD.Gaussian_dipole_source("Hz",-0.4,-0.3,0+shift,WC,0,3*DT,DT);
    
    pFDTD.propagate()

    pFDTD.out_point("Hz",-0,-0.5,0+shift,0,DS,"source.dat")
    pFDTD.out_point("Hz",-0,-0.5,0+shift,DS,DD,"mode1.dat")
    
    if (DS+100)<t and (t<DD):
        pFDTD.far_field_param(OMEGA, DETECT+shift);
    
    #if DD-300<t and t<DD:  # Qv and Qh calculation
    #	pFDTD.total_E_energy()
    #	pFDTD.total_E2()
    #	pFDTD.Poynting_total()
    #	pFDTD.Poynting_side(0.75,0)  #half width of the strip
    
    #if DD-10< t and t<DD and t%2==0:
    #	pFDTD.out_plane("Hz","z",0+shift,".Hz");
    
    t += 1


pFDTD.print_real_and_imag(0);
pFDTD.print_real_and_imag_2n_size(NROW,0);
pFDTD.far_field_FFT(NROW, NA, Nfree, OMEGA, 0);

print("Calculation Complete!\n");

