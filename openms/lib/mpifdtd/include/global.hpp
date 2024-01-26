
#ifndef GLOBAL_H
#define GLOBAL_H


// STL
// #include <complex>          // std::complex
// #include <fstream>          // std::ofstream
// #include <streambuf>        // std::streambuf
// #include <vector>           // std::vector

// MPI
#ifdef MPION
#include <mpi.h>
#endif

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

//  // jScience
//  #include "jScience/geometry/Vector.hpp"    // Vector
//  #include "jScience/geometry/Tensor.hpp"    // Tensor
//  #include "jScience/materials/Material.hpp" // Material
//  #include "jScience/meshing/Mesh.hpp"       // Mesh
//  #include "jScience/stl/MultiVector.hpp"    // Multivector
//  // FDTD
//  #include "bcs.hpp" // BoundaryConditions

// #include "precision.hpp"
// // variables used for separating the vertical loss and the horizontal loss
// precision Sumx,Sumy,Sumz,SideSumx,SideSumy,SumUpper,SumLower;
// extern precision shift; // vertical origin shift in the presence of the vertical asymmetry
// extern precision Sumx,Sumy,Sumz,SideSumx,SideSumy;
// extern precision SumUpper, SumLower;

//------------------------------
// new variables
//------------------------------



// time variable
extern long  time;
extern long  t;

extern int   Nfreq; // -> SpecN; // No. of multi-frequencies

// old variables to be modified



//   /// in parameter.c ///
//   extern int isize, jsize, ksize;
//   extern int pmlil, pmlir, pmljl, pmljr, pmlkl, pmlkr;
//   extern int lattice_x, lattice_y, lattice_z;
//   extern int max_lattice_nz;
//   extern precision xsize, ysize, zsize;
//   extern precision xcenter, ycenter, zcenter;
//   extern precision kx, ky, kz;
//   extern precision orderxl, orderyl, orderzl;
//   extern precision orderxr, orderyr, orderzr;
//   extern precision sig_axl, sig_ayl, sig_azl;
//   extern precision sig_axr, sig_ayr, sig_azr;
//   extern precision ds_x, ds_y, ds_z, dt;
//   extern precision dt_nm;
//   extern precision *ds_nz;
//   extern precision S_factor;
//   extern precision pi, eo, uo, ups, light_speed;
//   extern int misize, mjsize, mksize;
//   extern int pisize, pjsize, pksize;
//   extern int cisize, cjsize, cksize;
//   extern int xparity, yparity, zparity;
//   extern precision wave_vector_x, wave_vector_y;
//   extern int use_periodic_x, use_periodic_y;
//   
//   extern int m_pml,ma_pml;
//   extern precision sigmaCPML,alphaCPML,kappaCPML;
//   
//   
//   
//   // coordinates
//   extern char ***position;
//   
//   // dielectric functions
//   extern precision ***epsilonx,***epsilony,***epsilonz;
//   extern precision ***mepsilon,***momega,***mgamma;
//   
//   // EM field
//   extern precision ***Ex,***Ey,***Ez;
//   extern precision ***dPx,***dPy,***dPz;
//   extern precision ***dPx_old,***dPy_old,***dPz_old;
//   extern precision ***Jx,***Jy,***Jz;
//   extern precision ***Hx,***Hy,***Hz;
//   extern precision ***Dx,***Dy,***Dz;
//   extern precision ***Bx,***By,***Bz;
//   extern precision ***iEx,***iEy,***iEz;
//   extern precision ***iJx,***iJy,***iJz;
//   extern precision ***iHx,***iHy,***iHz;
//   extern precision ***iDx,***iDy,***iDz;
//   extern precision ***iBx,***iBy,***iBz;
//   extern precision *aax,*aay,*aaz;
//   extern precision *bbx,*bby,*bbz;
//   extern precision *ccx,*ccy,*ccz;
//   extern precision ***ddx,***ddy,***ddz;
//   extern precision *eex,*eey,*eez;
//   extern precision *ffx,*ffy,*ffz;
//   extern precision *ggx,*ggy,*ggz;
//   extern precision *hhx,*hhy,*hhz;
//   extern precision *iix,*iiy,*iiz;
//   extern precision *jjx,*jjy,*jjz;
//   extern precision *kkx,*kky,*kkz;
//   extern precision *llx,*lly,*llz;
//   extern precision ***Ex_cos, ***Ex_sin;
//   extern precision ***Ey_cos, ***Ey_sin;
//   extern precision ***Hx_cos, ***Hx_sin;
//   extern precision ***Hy_cos, ***Hy_sin;
//   
//   // cpml
//   extern precision ***psi_Exy, ***psi_Exz;
//   extern precision ***psi_Eyx, ***psi_Eyz;
//   extern precision ***psi_Ezy, ***psi_Ezx;
//   extern precision ***psi_Hxy, ***psi_Hxz;
//   extern precision ***psi_Hyx, ***psi_Hyz;
//   extern precision ***psi_Hzy, ***psi_Hzx;
//   
//   
//   // in input.c
//   extern precision back_epsilon;
//   
//   //
//   extern precision global_W;
//   
//   
//

//*********************
//  Functions
//*********************


#endif
