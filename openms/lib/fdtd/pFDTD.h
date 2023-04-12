/*    

       ######*        
             #*    #####*         *#######
            ##     ##   ##        ##        ###*   #####*   ###*  
          ##*      ##   ##  ***   #######   ##  #    ##     ##  #
            ##     ##   ##        ##        ##  #    ##     ##  #
             #*    #####*         #*        ###*     #*     ###* 
       ######*

                             Metal     < ver. 8.803 >
                                                                    */

/******************************************************************** 
           3D Finite Difference Time Domain (FDTD)
                      Version 8.803
---------------------------------------------------------------------
Se-Heon Kim
Department of Electrical Engineering, Caltech
Bug report to : seheon@caltech.edu 
---------------------------------------------------------------------
*********************************************************************/
/*
//---------------------------------------------------------------//
//--------------------- Update History --------------------------//
//---------------------------------------------------------------//
The Yee algorithm(Taflove I 59-) and UPML(Taflove II 275-) were used.
020904 grid_value function was added.
020904 dipole_source function was modified for grid value.
020908 Phase parameter was added in dipole_source function.
020909 Dipolesourde fuction was modified.
020918 Dielectric object input fuctions were modified.
020924 a bug was found in coefficient function. sigmay was subtituted by sigmaz in eez.
020925 Out functions were modified.
030113 Sigma function modified(not integrated form).
030114 PML range was modified.
030114 average_epsilon fuction was modified.
030116 Parity fuction for all axis was added.
030117 Out_epsilon function was modified.
030117 Out_plane fuction was modified.
030121 out_time function was added.
030121 Style of size input was modified.
030121 output fuction was modified. Log10E^2 is added.
030422 grid_value(Energy) was added.
----------------------------------------------------------
Original structures are coded by Kim Guk Hyun (tan@kaist.ac.kr)
Contour functions are added by Se-Heon Kim (seheon@kaist.ac.kr)
----------------------------------------------------------
040319 Contour input function was included. (by Se-Heon Kim)
040319 Struct obj was modified (to include matrix data). (by Se-Heon Kim)
040322 Struct obj was modified (bug fix) (by Se-Heon Kim)
040326 Coefficient() and propagate() was modified to preserve epsilon function (by Se-Heon Kim)
040510 Far-field functions were included. (by Se-Heon Kim)
040514 2n_size() bug was fixed. (by Se-Heon Kim)
040517 P_tot.txt, P_the.txt,and P_phi.txt (by Se-Heon Kim)
040618 multi-wavelength far-field (by Se-Heon Kim)
-----------------------------------------------------------
040711 Modular programming (by Se-Heon Kim)
040816 Change one parameter (0.1 --> 0.02) in transmap.c (by Se-Heon Kim)
040816 Modify shell program FDTDrun (by Se-Heon Kim)
040826 Bug fixing in Ex,Ey,Et,and Ep data printing (by Se-Heon Kim)
040827 Change name : farfield_param_test() --> print_real_and_imag() (by Se-Heon Kim)
050110 in energy.c; new global variables : SumUpper, SumLower
       change pml size (pmlir, pmlil , ... )
	include new function Poynting_UpDown(float value)
050112 Include structure shape "epllipse" (by Soon-Hong)
050112 Include version printing in memory.c 
050120 ver.6.11 : Change in sigma.c (orderx,ordery,orderz,sig_ax,sig_ay,sig_az..)
	Modify function parameter() --> parameter(ox,oy,oz,ax,ay,az)
050220 ver.6.30 : non-uniform space discretization ds_x, ds_y, ds_z
050226 ver.6.31 : Change in sigma.c (sig_axl, sig_axr. ....)
050228 ver.6.33 : parameter() --> set_default_parameter()
		add set_kappa(), set_sigma_max(), set_sigma_order()
050309 ver.6.35 : frequency bug in source.c and farfield.c by SoonHong Kwon
050326 ver.6.36 : Bug : lattice_x --> lattice_z in function farfield_param() 
050329 ver.6.37 : (int)(x) --> floor(0.5+x) 
050405 ver.6.40 : personal version : changing run program name --> FDTDhit
050907 ver.6.42 : in energe.c, add zposition in functions, Poynting_UpDown and Poynting_side 
060423 ver.6.44 : in source.c, change the dipole source condition 
060722 ver.6.50 : in input.c, add random_generate() 
			add object "sphere" 
060724 ver.6.51 : add random_dipole()
060820 ver.6.55 : in source.c, change the dipole source condition 
                  in source.c, change the name of source functions (Gaussian)
                  in memory.c, add release information (define macro : KFDTDver)
                  in source.c, add Lorentzian_dipole_source() 
                  in source.c, add Lorentzian_phase()
060821 ver.6.56   in source.c, add Gaussian_phase()
060821 ver.6.57   in source.c, bug correction in Lorentzian_dipole_source()
060901 ver.6.58   in output.c, x=x0 plane output 90deg. rotate 
061211 ver.6.60   in energy.c, corrections for nonuniform-grids 
061213 ver.6.61   in output.c, add LogH^2
061214 ver.6.65   in output.c, add Ex2Ey2 .. Hx2Hy2 ... LogEx2Ey2 .. LogHx2Hy2  
		  in energy.c, add Poynting_block()
070108 ver.6.70   in energy.c, add Poynting_half_sphere_point()
-------------------------------------------------------------------
070111 ver.7.00   metal FDTD! based on Drude model 
		  in memory.c, newly added arrays, mepsilon[i][j][k], momemga[i][j][k], mgamma[i][j][k], 
                                                   Jx[i][j][k], Jy[i][j][k], Jz[i][j][k]
		  in metalobj.c (new c file), newly added functions, 
                              input_Drude_medium(...) : create 1-D array of mobject[n] structure.
                              make_metal_structure() : create mepsilon[i][j][k], momemga[i][j][k], mgamma[i][j][k].
			      in_mobject(n, i,j,k) : judge if the position (i,j,k) lies within mobject[n].
	          in output.c, correct out_epsilon() : include output of metal structures. 
                               meps(i,j,k) : for Hz_parity().
                  in timeupdate.c, correct E-field update equations.
                       if (i,j,k) position lies 
                         1) in normal dielectrics, --> use normal FDTD update equations.
                         2) in metal, --> use Auxilary Differential Equations for Jx,Jy,Jz.   
070112 ver.7.50   Floquet periodic boundary condtion.  
070209 ver.7.60   Planewave source condition. 
		  Remove the function pGaussian_dipole_source(), phasor source condition is unified in 
                       the former Gaussian_dipole_source(). 
		  in parameter.c, in function periodic_boundary(), add pmlil,pmlir,pmljl,pmljr =0;
		  in output.c, in functions 'out_epsilon()' and 'out_plane()', in for-loop {isize-2 --> isize-1} now, the printed images are symmetric  
		  in output.c, add "&& (xparity==1 || xparity==-1)" 
	          in output.c, add "i_range, j_range, k_range" 
		  in input.c, in function "make_epsilon()", corrections for the periodic b.c.
		  in input.c, in function "in_object()", range change ">,<" --> ">=,<="
		  in source.c, error correction in Gaussian_planewave().... 
		  in parameter.c, see the range of pml layers. (-5) corrections. 
		  in sigma.c, found significant errors (by Jin-Kyu Yang)
		  in timeupdate.c, add a new function, field_initialization() (by Jin-Kyu Yang) 
070212 ver.7.62   in timeupdate.c, add Ez_parity_boundary_update(), Ez_parity_iboundary_update()
070214 ver.7.64   in timeupdate.c, parity boundary conditions become more stringent! (see addition of 0.0) 
070227 ver.7.70   add Perfect conducting medium. 
			See timeupdate.c, pcm_E_field_update(i,j,k) .....
			See also 'medium rules' in function make_metal_structure() in metalobj.c
                  memory saving in case of the Gamma point. 
                        See timeupdate.c, E_field_Gamma_boundary_update_x()....
070227 ver.7.75   print "real_space_param(float a_nm, float w_n)"
			See output.c 
		  in source.c, add "-Ex", "-Ey", "-Ez", "-Hx", "-Hy", "-Hz" for specifying out-phase oscillations
070228 ver.7.80   GNU Scientific Library compatible. 
070310 ver.7.85   add incoherent.c (for incoherent random dipole source) 
		  add envelope functions for dipole sources. 
                      float Gauss_amp(float frequency, float phase, long to, long tdecay);
		      float Lorentz_amp(float frequency, float phase, long to, long tdecay);
		      float iGauss_amp(float frequency, float phase, long to, long tdecay);
		      float iLorentz_amp(float frequency, float phase, long to, long tdecay);
		  and modify source.c to use the above 4 functions
070408 ver.7.90   in farfield.c, many changes have been done for memory savings. 
		  in output.c, many changes in farfield related functions. 
		  in trasmap.c, change argument in transform_farfield() 
		  in energy.c, add 'component' to Poynting_half_sphere_point()
070418 ver.7.92   in farfield.c, free() correction (adviced by Tailor) 
070507 ver.7.93   in farfield.c, in calculating_radiation() change the range in the for loop. 
070525 ver.7.94   in farfield.c, error correction in free variables, in functions : fourier_transform_..(), 
                             data_shifting_..().
070615 ver.7.95   add new input structure "ellipsoidal". See input.c 
070616 ver.7.97   add input_object_Euler_rotation(). See input.c 
070801 ver.7.98   add "ellipsoidal" in metal object
070912 ver.7.99   add Lorentzian_planewave() 
070914 ver.8.00   add out_several_points()
		  small corrections on real_space_param()
071023 ver.8.01   add new object "cone". See input.c
		  add "cone" in metal object
071128 ver.8.05   add out_epsilon_projection() and out_plane_projection().
080312 ver.8.07   add "rodX" and "rodY" 
080331 ver.8.08   add "donut" 
		  add "shell"
                  solve ">" "<" problem in "metal object" by replacing with ">=" "<"
080429 ver.8.10   add get_period_in_update() 
                  modify out_plane_projection() <-- add k_shift variable 
                  add out_plane_time_average() 
		  add out_plane_time_average_projection()
----------------------------------------
In Caltech......
080806 ver.8.11   small correction on 'time average' functions (change 'averaging interval')
081006 ver.8.20   small correction in GSL random number generator
			in FDTDvar.h, in function "set_default_parameter()"
		  in function "coherent_point_dipole" 
			replace sin(theta) --> sqrt(1-alpha^2) 
			by introducing the variable, p_theta_aux instead of p_theta
		  For Lorentizian envelope function, by setting tdecay zero, 
			one can use real continuous wave source. 
081008 ver.8.25   add Gaussian_line_source() and Lorentzian_line_source() 
081021 ver.8.30   add out_epsilon_periodic() and out_plane_periodic()
081029 ver.8.31   revival of FFT E^2 FFTH^2 print out 
081030 ver.8.33   metal region does not use PML condition 
081213 ver.8.41   non-uniform space grid 
		1) only for z direction 
		2) only for one region
		3) only for "rod" and "block" (both for metal and dielectric)
		4) only for Gaussian source 
		in parameter.c, pmlil = ... = 0
090106 ver.8.42   add Drude_energy_loss_in_block()
		very important error correction in Lorentzian_planewave() : missing '*sqrt(eo/uo)' term 
090713 ver.8.43 important corrections for non-uniform space grid (by Jingqing Huang)
 
		1) energy.c - Poynting_side(); Poynting_UpDown(); All other functions fine for non-uniform space grid calculation

		2) for "cone" - both dielectric and metal

090714 ver.8.45 add total_EM_energy() function
		add total_EM_energy_block() function
		add total_E_energy_block() function
		add "EM_Energy_m" option in function grid_value()
		add "E_Energy_m" option in functon grid_value()
		add eps_m() function in output.c. eps_m() uses global_W, which can be set through real_space_param() function
		add one line to set global normalized frequency in function real_space_param() function 
090714 ver.8.47 add max_E_Energy_detector() function
		error correction : mepsilon[i][j][k] --> meps(i,j,k) in total_E_energy(), total_EM_energy(),
 			total_E_energy_block.c, total_EM_energy_block() functions.
090724 ver.8.48 add void total_E_energy_thin_block_z() in energy.c (for the calculation of a QW confinement factor)
		eps(), eps_m(), and m_eps() are now declared as global functions
090814 ver.8.60 substantial modifications on non-uniform grid functions : 
		add generate_base_z()
		    generate_base_grid()
		modify set_default_parameter()
		add find_max_lattice_nz()
		    ngrid_lattice_nz_i()
		    ngrid_lattice_nz_z()
		    non_uniform_i_to_z() *set as public 
		    non_uniform_z_to_i() *set as public
		modify non_uniform_grid()
		add quick_sort_nz_start()
		define struct ngrid_info (in this header file) *set as public
		modify "cone" parts in input.c and metalobj.c 
090816 ver.8.70 True triangular lattice symmetry can be handled by using "propagate_tri()"
090824 ver.8.75 add Gaussain beam propagation. please open source.c to see the change. 
090928 ver.8.76 add corrections to out_epsilon, out_epsilon_periodic, out_plane, out_plane_periodic, out_plane_time_average,
		for correcting distorted images by the use of non_uniform_grid functions. 
091015 ver.8.77 non_uniform grid correction for contour FDTD
091121 ver.8.78 for calculating kinetic energy of electrons, add eps_m2(i,j,k) function
100324 ver.8.785 The meaning of eps_m(i,j,k) has been changed: the total electro-'magnetic' energy density that includes 'magnetic' energy density. For further details, refer to S. L. Shuang's paper on Opt. Lett. 
100424 ver.8.80 non_uniform grid correction for metallic-"donut" and metallic-"ellipse"
                add input_Drude_medium2(.... char *matrix_file ....), which can take "contour" input.
		add reading_matrix_m() in mobject.c
		add new elements in struct mobj: **matrix, col, and row. 
100428 ver.8.801 correction in eps_m(i,j,k).
100429 ver.8.802 From %f to %g in several functions in energy.c 
100508 ver.8.803 corrections to "contour", a certain gcc complier generate strange vertical cross sections 
                 (thanks to S. Y. Lee)
//----------------------------------------------------------------//
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
extern float *ds_nz;
extern float S_factor;
extern float pi, eo, uo, ups, light_speed;
extern int misize, mjsize, mksize;
extern int pisize, pjsize, pksize;
extern int cisize, cjsize, cksize;
extern int xparity, yparity, zparity;
extern float wave_vector_x, wave_vector_y;
extern int use_periodic_x, use_periodic_y;  

/// in memory.c ///
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

/// in input.c ///
extern float back_epsilon;

/// in output.c ///
extern float global_W;






