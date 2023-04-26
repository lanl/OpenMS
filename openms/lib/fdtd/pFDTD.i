/*
# Copyright 2022-2023 The OpenMS Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Yu Zhang <zhy@lanl.gov>
*/

%module fdtdc

/*
%import "config.h"
*/

#include "gsl/gsl_rng.h"
#include "gsl/gsl_randist.h"

%{
#define SWIG_FILE_WITH_INIT
#include "gsl/gsl_rng.h"
#include "gsl/gsl_randist.h"
#include "pFDTD.h"
#include "pFDTDvar.h"

/*
 * SWIG_PYTHON_THREAD_SCOPED_BLOCK is a macro that SWIG automatically generates
 * wrapping a class using an RAII pattern to automatically acquire/release
 * the GIL. See the generated pfdtd-python.c for details.
 *
 * We could instead just explicitly call SWIG_PYTHON_THREAD_BEGIN_BLOCK and
 * SWIG_PYTHON_THREAD_END_BLOCK everywhere - but this is error prone since we
 * have to ensure that SWIG_PYTHON_THREAD_END_BLOCK is called before every
 * return statement in a method.
 *
 * NOTE: This wont work with plain-old C.
 */
#define SWIG_PYTHON_THREAD_SCOPED_BLOCK   SWIG_PYTHON_THREAD_BEGIN_BLOCK

/*
namespace pfdtd {
  todo 
}
*/

%}

/* marcos */
%constant int EMP = (int)EMP;

/*  numpy array to c pointer */
%include "numpy.i"

%init %{
import_array();
%}

/* Apply float * type map to OMEGA argument */
%apply (float *IN_ARRAY1, int DIM1) {(float *OMEGA, int n)}
%typemap(in) float *OMEGA (int ndims, int *dims) {
  PyObject *obj = $input;
  PyArrayObject *arr = NULL;
  int i, n;

  /* Convert the input object to a NumPy array */
  arr = (PyArrayObject *) PyArray_FROM_OTF(obj, NPY_FLOAT, NPY_ARRAY_IN_ARRAY);

  /* Check for errors */
  if (arr == NULL) {
    SWIG_exception_fail(SWIG_RuntimeError, "Invalid input type for OMEGA");
    return NULL;
  }

  /* Check the number of dimensions */
  ndims = PyArray_NDIM(arr);
  if (ndims != 1) {
    SWIG_exception_fail(SWIG_RuntimeError, "Invalid number of dimensions for OMEGA");
    Py_DECREF(arr);
    return NULL;
  }

  /* Get the dimensions */
  dims = (int *) PyArray_DIMS(arr);
  n = dims[0];

  /* Allocate a C array for the data */
  $1 = (float *) malloc(n * sizeof(float));

  /* Copy the data from the NumPy array to the C array */
  memcpy($1, PyArray_DATA(arr), n * sizeof(float));

  /* Cleanup */
  Py_DECREF(arr);
}


/* global variables*/
%inline %{
#include "gsl/gsl_rng.h"
#include "gsl/gsl_randist.h"

/* # of multi-frequencies */
extern int SpecN;

/* vertical origin shift in the presence of the vertical asymmetry */
extern float shift;

/* FDTD update time variable */
extern long t;

/* variables used for separating the vertical loss and the horizontal loss */
extern float Sumx,Sumy,Sumz,SideSumx,SideSumy,SumUpper,SumLower;

/*variables for GSL random number generator*/
extern const gsl_rng_type * rng_type;
extern gsl_rng * rng_r;

/*//-----------------------------------------------------------------//
// extern variables from pFDTD.h
//-----------------------------------------------------------------//

TBA
*/
extern float shift;
extern int SpecN;
extern long t;
extern float Sumx,Sumy,Sumz,SideSumx,SideSumy;
extern float SumUpper, SumLower; 
extern const gsl_rng_type * rng_type;
extern gsl_rng * rng_r;

extern int isize, jsize, ksize;
extern int pmlil, pmlir, pmljl, pmljr, pmlkl, pmlkr;
extern int lattice_x, lattice_y, lattice_z;
extern int max_lattice_nz;
extern float xsize, ysize, zsize;
extern float xcenter, ycenter, zcenter;
extern float kx, ky, kz;
extern float orderxl, orderyl, orderzl;
extern float orderxr, orderyr, orderzr;
extern float sig_axl, sig_ayl, sig_azl;
extern float sig_axr, sig_ayr, sig_azr;
extern float ds_x, ds_y, ds_z, dt;
extern float *ds_nz;
extern float S_factor;
extern float pi, eo, uo, ups, light_speed;
extern int misize, mjsize, mksize;
extern int pisize, pjsize, pksize;
extern int cisize, cjsize, cksize;
extern int xparity, yparity, zparity;
extern float wave_vector_x, wave_vector_y;
extern int use_periodic_x, use_periodic_y;  



extern char ***position;
extern float ***epsilonx,***epsilony,***epsilonz;
extern float ***mepsilon,***momega,***mgamma;
extern float ***Ex,***Ey,***Ez;
extern float ***Jx,***Jy,***Jz;
extern float ***Hx,***Hy,***Hz;
extern float ***Dx,***Dy,***Dz;
extern float ***Bx,***By,***Bz;
extern float ***iEx,***iEy,***iEz;
extern float ***iJx,***iJy,***iJz;
extern float ***iHx,***iHy,***iHz;
extern float ***iDx,***iDy,***iDz;
extern float ***iBx,***iBy,***iBz;
extern float *aax,*aay,*aaz;
extern float *bbx,*bby,*bbz;
extern float *ccx,*ccy,*ccz;
extern float ***ddx,***ddy,***ddz;
extern float *eex,*eey,*eez;
extern float *ffx,*ffy,*ffz;
extern float *ggx,*ggy,*ggz;
extern float *hhx,*hhy,*hhz;
extern float *iix,*iiy,*iiz;
extern float *jjx,*jjy,*jjz;
extern float *kkx,*kky,*kkz;
extern float *llx,*lly,*llz;
extern float ***Ex_cos, ***Ex_sin;
extern float ***Ey_cos, ***Ey_sin;
extern float ***Hx_cos, ***Hx_sin;
extern float ***Hy_cos, ***Hy_sin;


extern float back_epsilon;
extern float global_W;

%}

/*
include the functions in headers
*/

/*
//-----------------------------------------------------------------//
// functions
//-----------------------------------------------------------------//
*/
void structure_size(float x,float y,float z);
void lattice_size(int lx, int ly, int lz);
void non_uniform_grid(char *component, float z_i, float z_f, int nlz);
int non_uniform_z_to_i(float z);
float non_uniform_i_to_z(int i);
int ngrid_lattice_nz_z(float z); 
int ngrid_lattice_nz_i(int i); 
int find_max_lattice_nz();
void pml_size(int il,int ir,int jl,int jr,int kl,int kr);
void set_default_parameter(float S);
void set_sigma_order(float oxl, float oxr, float oyl, float oyr, float ozl, float ozr);
void set_sigma_max(float axl, float axr, float ayl, float ayr, float azl, float azr);
void set_kappa(float kappa_x, float kappa_y, float kappa_z);
void Hz_parity(int x,int y,int z);
void periodic_boundary(int x_on, int y_on, float k_x, float k_y);
void memory();
void make_epsilon();
void make_metal_structure(); 
void background(float epsilon);
void input_object(char *shape,
                    char *matrix_file,
                    float centerx,
                    float centery,
                    float centerz,
                    float size1,	// discrination level
                    float size2,        // height
                    float size3,        // compression factor
                    float epsilon);
void input_object_Euler_rotation(char *shape, 
		char *matrix_file, 
		float centerx, 
		float centery, 
		float centerz, 
		float size1, 
		float size2, 
		float size3, 
		float alpha, 
		float beta, 
		float gamma, 
		float epsilon);
void input_Drude_medium(char *shape, float centerx, float centery, float centerz, float size1, float size2, float size3, float epsilon_b, float omega_p, float gamma_0, float lattice_n);
void input_Drude_medium2(char *shape, char *matrix_file, float centerx, float centery, float centerz, float size1, float size2, float size3, float epsilon_b, float omega_p, float gamma_0, float lattice_n);
void random_object(char *shape, float radius, float height, float epsilon, float x_min, float x_max, float y_min, float y_max, float z_min, float z_max, int gen_number, int seed);
void random_Gaussian_dipole(char *component, float frequency, float tdecay, float x_min, float x_max, float y_min, float y_max, float z_min, float z_max, int gen_number, int seed);
void far_field_param(float *OMEGA, float DETECT);
void make_2n_size(int NROW, int mm);
void far_field_FFT(int NROW, float NA, float Nfree, float *OMEGA, int mm);
void coefficient();
float sigmax(float a);
float sigmay(float a);
float sigmaz(float a);
void propagate();
void propagate_tri(); ///// for triangular lattice
void Gaussian_dipole_source(char *component,float x,float y,float z,float frequency,float phaes,long to,long tdecay);
void Gaussian_planewave(char *Ecomp, char *Hcomp, float position, float frequency, long to,long tdecay);
void Gaussian_beam_prop_Gauss(char *Ecomp, char *Hcomp, float x, float y, float z, float z_c, float wo, float n, float frequency, long to,long tdecay);
void Lorentzian_planewave(char *Ecomp, char *Hcomp, float position, float frequency, long to,long tdecay);
void Gaussian_beam_prop_Lorentz(char *Ecomp, char *Hcomp, float x, float y, float z, float z_c, float wo, float n, float frequency, long to,long tdecay);
void Gaussian_line_source(char *component, float position_x, float position_z, float frequency, float phase, long to,long tdecay);
void Lorentzian_line_source(char *component, float position_x, float position_z, float frequency, float phase, long to,long tdecay);
void Lorentzian_dipole_source(char *component,float x,float y,float z,float frequency,float phaes,long to,long tdecay);
float Lorentzian_phase(float Wn, long tdecay);
float Gaussian_phase(float Wn, long t_peak);
void incoherent_point_dipole(char *function, float x, float y, float z, float frequency, long to, long tdecay, float t_mu, float sd);
float eps(int i,int j,int k);
float eps_m(int i,int j,int k); // effective epsilon for Drude dispersive material 
float eps_m2(int i,int j,int k); // calculating kinetic energy of electrons
float eps_m3(int i,int j,int k); // calculating potential energy of electrons
float meps(int i,int j,int k); //For Hz_parity()
void out_epsilon(char *plane,float value,char *name);
void out_epsilon_periodic(char *plane,float value,char *name, int m_h, int m_v);
void out_epsilon_projection(char *dirc, char *name);
void out_plane(char *component,char *plane,float value,char *lastname);
void out_plane_projection(char *component,char *dirc,char *lastname, int k_shift);
void out_plane_periodic(char *component,char *plane,float value,char *lastname, int m_h, int m_v);
void out_plane_time_average(char *component,char *plane,float value, long start, long end, float ***field_avg, char *lastname);
void out_plane_time_average_projection(char *component,char *dirc, long start, long end, float ***field_avg, char *lastname, int k_shift);
void out_several_points(char *component, float zposition, float xside, float yside, int pNx, int pNy, long ti, long tf, char *name);
void out_point(char *component,float x,float y,float z,long ti,long tf,char *name);
float grid_value(char *component,int i,int j,int k);
void total_E_energy();
void total_EM_energy();
void total_E_energy_block(float centerx, float centery, float centerz, float size1, float size2, float size3);
void total_E_energy2_block(float centerx, float centery, float centerz, float size1, float size2, float size3);
void total_E_energy3_block(float centerx, float centery, float centerz, float size1, float size2, float size3);
void total_E_energy_thin_block_z(float centerx, float centery, float centerz, float size1, float size2, float size3, float eps_L, float eps_H, char *name);
void max_E_Energy_detector(float centerx, float centery, float centerz, float size1, float size2, float size3);
void total_EM_energy_block(float centerx, float centery, float centerz, float size1, float size2, float size3);
void total_E2();
void Drude_energy_loss_in_block(float centerx, float centery, float centerz, float size1, float size2, float size3);
void Drude_energy_loss_in_block2(float centerx, float centery, float centerz, float size1, float size2, float size3, float WC, float lattice_n);
void Poynting_total();
void Poynting_block(float centerx, float centery, float centerz, float size1, float size2, float size3);
float Poynting_half_sphere_point(char *component, float z0, float R, float theta, float phi);
void Poynting_side(float value, float zposition);
void print_energy();
void Poynting_UpDown(float value, float zposition);
void transform_farfield(int NROW, int tnum, char *name, int mm);
void add_farfield(int tnum, char *name);
void print_amp_and_phase(int mm);
void print_real_and_imag(int mm);
void print_real_and_imag_2n_size(int NROW, int mm);
void field_initialization();
void real_space_param(float a_nm, float w_n);
int get_period_in_update(float w_n);
float Gauss_amp(float frequency, float phase, long to, long tdecay);
float Lorentz_amp(float frequency, float phase, long to, long tdecay);
float iGauss_amp(float frequency, float phase, long to, long tdecay);
float iLorentz_amp(float frequency, float phase, long to, long tdecay);


