
/********************************************************************
           3D Finite Difference Time Domain (FDTD)
---------------------------------------------------------------------
---------------------------------------------------------------------
*/

//-----------------------------------------------------------------//
// Standard libraries in C
//-----------------------------------------------------------------//

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

//-----------------------------------------------------------------//
// Definitions of macros
//-----------------------------------------------------------------//
#define EMP 0  // for dummy input variables
#define CUTFACTOR 1  // matrix data cutting parameter
#define FALSE 0  // member variables for far_field_FFT(..)
#define TRUE 1  // member variables for far_field_FFT(..)
#define NR_END 1  // from numerical recipe, in memory.c
#define FREE_ARG char*  // from numerical recipe, in memory.c
#define KFDTDver 8.803

//-----------------------------------------------------------------//
// Definitions of structures
//-----------------------------------------------------------------//
struct obj 
{
    char shape[10];
    float centeri;
    float centerj;
    float centerk;
    float size1;
    float size2;
    float size3;
    float epsilon;
    float **matrix;  // contour matrix data
    int   col; // matrix col num
    int   row; // matrix row num
};
struct mobj 
{
    char shape[10];
    float centeri;
    float centerj;
    float centerk;
    float size1;     ///////////// Drude model /////////////////
    float size2;     //                                       //
    float size3;     //                    (omega_p)^2        //
    float epsilon_b; //  eps(w) = eps_b - ------------------  //
    float omega_p;   //                   w^2 + j w gamma_0   //
    float gamma_0;   ///////////////////////////////////////////
    float **matrix;  // contour matrix data
    int   col; // matrix col num
    int   row; // matrix row num
};
struct COMPLEX 
{
    float real;
    float imag;
};
struct ngrid_info // for non-uniform grid function
{
    int lattice_nz;
    float nz_start;
    float nz_end;
};

//-----------------------------------------------------------------//
// functions
//-----------------------------------------------------------------//
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
                    float size1,    // discrination level
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
void input_molecular(char *shape, float centerx, float centery, float centerz, float size1, float size2, float size3);
void input_Drude_medium(char *shape, float centerx, float centery, float centerz, float size1, float size2, float size3, float epsilon_b, float omega_p, float gamma_0, float lattice_n);
void input_Drude_medium2(char *shape, char *matrix_file, float centerx, float centery, float centerz, float size1, float size2, float size3, float epsilon_b, float omega_p, float gamma_0, float lattice_n);
void random_object(char *shape, float radius, float height, float epsilon, float x_min, float x_max, float y_min, float y_max, float z_min, float z_max, int gen_number, int seed);
void random_Gaussian_dipole(char *component, float frequency, float tdecay, float x_min, float x_max, float y_min, float y_max, float z_min, float z_max, int gen_number, int seed);
void far_field_param(float *OMEGA, float DETECT);
void make_2n_size(int NROW, int mm);
void far_field_FFT(int NROW, float NA, float Nfree, float *OMEGA, int mm);
void coefficient();
void coefficient_cpml();
float sigmax(float a);
float sigmay(float a);
float sigmaz(float a);
float cpmlax(float a, float pmlbl, float pmlbr);
float cpmlbx(float a, float pmlbl, float pmlbr);
float kappa_x(float a, float pmlbl, float pmlbr);

void propagate();
void propagate_tri(); ///// for triangular lattice
void Gaussian_dipole_source(char *component,float x,float y,float z,float frequency,float phaes,long to,long tdecay);
void external_source(char *component,float x,float y,float z,float Ext);
void external_planewave(char *Ecomp, char *Hcomp, float z,float Ext);
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
void free_3d_memory(float ***t, int nrh, int nch, int ndh);
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

//-----------------------------------------------------------------//
// extern variables
//-----------------------------------------------------------------//
/// in main.c ///
extern float shift;
extern int SpecN;
extern long t;
extern float Sumx,Sumy,Sumz,SideSumx,SideSumy;
extern float SumUpper, SumLower; 
extern const gsl_rng_type * rng_type;
extern gsl_rng * rng_r;

/// in parameter.c ///
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
extern float dt_nm;
extern float *ds_nz;
extern float S_factor;
extern float pi, eo, uo, ups, light_speed;
extern int misize, mjsize, mksize;
extern int pisize, pjsize, pksize;
extern int cisize, cjsize, cksize;
extern int xparity, yparity, zparity;
extern float wave_vector_x, wave_vector_y;
extern int use_periodic_x, use_periodic_y;  

extern int m_pml,ma_pml;
extern float sigmaCPML,alphaCPML,kappaCPML;

/// in memory.c ///
extern char ***position;
extern float ***epsilonx,***epsilony,***epsilonz;
extern float ***mepsilon,***momega,***mgamma;
extern float ***Ex,***Ey,***Ez;
extern float ***dPx,***dPy,***dPz;
extern float ***dPx_old,***dPy_old,***dPz_old;
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

// cpml
extern float ***psi_Exy, ***psi_Exz;
extern float ***psi_Eyx, ***psi_Eyz;
extern float ***psi_Ezy, ***psi_Ezx;
extern float ***psi_Hxy, ***psi_Hxz;
extern float ***psi_Hyx, ***psi_Hyz;
extern float ***psi_Hzy, ***psi_Hzx;


/// in input.c ///
extern float back_epsilon;

/// in output.c ///
extern float global_W;

