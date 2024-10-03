#include "./pFDTD.h"

void Gaussian_dipole_source_beam(char *component, int i, int j, float z,
                                 float frequency, float amp, float phase,
                                 long to, long tdecay);
void Lorentzian_dipole_source_beam(char *component, int i, int j, float z,
                                   float frequency, float amp, float phase,
                                   long to, long tdecay);
// since ver. 8.75, xxx_dipole_source_plane() were upgraded to
// xxx_dipole_source_beam()
void Gaussian_dipole_source_line(char *component, float x, int j, float z,
                                 float frequency, float phase, long to,
                                 long tdecay);
void Lorentzian_dipole_source_line(char *component, float x, int j, float z,
                                   float frequency, float phase, long to,
                                   long tdecay);

void random_Gaussian_dipole(char *component, float frequency, float tdecay,
                            float x_min, float x_max, float y_min, float y_max,
                            float z_min, float z_max, int gen_number,
                            int seed) {
  int i;
  float xr, yr, zr, phase;

  srand(seed); // initialization of random number generator

  for (i = 0; i < gen_number; i++) {
    xr = ((float)rand() / ((float)(RAND_MAX) + (float)(1))) * (x_max - x_min) +
         x_min;
    yr = ((float)rand() / ((float)(RAND_MAX) + (float)(1))) * (y_max - y_min) +
         y_min;
    zr = ((float)rand() / ((float)(RAND_MAX) + (float)(1))) * (z_max - z_min) +
         z_min;
    phase = ((float)rand() / ((float)(RAND_MAX) + (float)(1))) * 2 * pi;
    Gaussian_dipole_source(component, xr, yr, zr, frequency, phase, 3 * tdecay,
                           tdecay);
  }
}

void Gaussian_planewave(char *Ecomp, char *Hcomp, float position,
                        float frequency, long to, long tdecay) {
  int i, j;
  float phase;

  for (i = 0; i <= pisize; i++)
    for (j = 0; j <= pjsize; j++) {
      phase =
          2 * pi *
          (wave_vector_x * i / (lattice_x) + wave_vector_y * j / (lattice_y));
      Gaussian_dipole_source_beam(Ecomp, i, j, position, frequency, 1.0, phase,
                                  to, tdecay);
      Gaussian_dipole_source_beam(Hcomp, i, j, position, frequency, 1.0, phase,
                                  to, tdecay);
    }
}

void Gaussian_beam_prop_Gauss(char *Ecomp, char *Hcomp, float x, float y,
                              float z, float z_c, float wo, float n,
                              float frequency, long to, long tdecay)
/*////////////////////////////////////////////////////////////////////////////
x,y,z : center of the plane on which the Gaussian beam source is to be located
z_c : z-position at which the beam waist becomes the minimum (focal point)
wo : the minimum beam waist
The above parameters should be in the unit of 'a' (lattice constant)
n : refractive index of the medium where the Gaussian beam source is
Other paramters are the same as for Gaussian dipole soure()
//////////////////////////////////////////////////////////////////////////// */
{
  int i, j;
  float phase;
  float amp;
  float zo, eta_z, w_z, R_z;
  float r2;
  int ic, jc;
  // eta_z = atan(z/zo)
  // w_z = wo*sqrt(1+z^2/zo^2)
  // R_z = z*(1+z^2/zo^2)
  // zo = pi*wo^2/lambda
  // r2 : radius^2

  zo = pi * wo * wo * frequency * n;
  ic = floor(0.5 + ((x + xcenter) * lattice_x));
  jc = floor(0.5 + ((y + ycenter) * lattice_y));
  ////////////////////////////////////
  /// some parameters which depend only on 'z'
  eta_z = atan((z_c - z) / zo);
  w_z = wo * sqrt(1 + ((z_c - z) / zo) * ((z_c - z) / zo));
  R_z = (z_c - z) * (1 + (zo / (z_c - z)) * (zo / (z_c - z)));

  /*//////////////////////////////////
  amp = (wo/w_z)*exp[ -r^2 / w_z^2 ]
  phase = kz - eta_z + r^2 * k / (2 R_z)
  For further details, see Yariv's book.
  ////////////////////////////////// */

  for (i = 0; i <= pisize; i++)
    for (j = 0; j <= pjsize; j++) {
      r2 = (i - ic) * (i - ic) / ((float)(lattice_x * lattice_x)) +
           (j - jc) * (j - jc) / ((float)(lattice_y * lattice_y));
      amp = (wo / w_z) * exp(-r2 / (w_z * w_z));
      phase = -(2 * pi * (z_c - z) * frequency * n - eta_z +
                r2 * 2 * pi * frequency * n / (2 * R_z));
      Gaussian_dipole_source_beam(Ecomp, i, j, z, frequency, amp, phase, to,
                                  tdecay);
      Gaussian_dipole_source_beam(Hcomp, i, j, z, frequency, amp, phase, to,
                                  tdecay);
    }
}

void Gaussian_line_source(char *component, float position_x, float position_z,
                          float frequency, float phase, long to, long tdecay) {
  int j;

  for (j = 0; j <= pjsize; j++)
    Gaussian_dipole_source_line(component, position_x, j, position_z, frequency,
                                phase, to, tdecay);
}

void Lorentzian_planewave(char *Ecomp, char *Hcomp, float position,
                          float frequency, long to, long tdecay) {
  int i, j;
  float phase;

  for (i = 0; i <= pisize; i++)
    for (j = 0; j <= pjsize; j++) {
      phase =
          2 * pi *
          (wave_vector_x * i / (lattice_x) + wave_vector_y * j / (lattice_y));
      Lorentzian_dipole_source_beam(Ecomp, i, j, position, frequency, 1.0,
                                    phase, to, tdecay);
      Lorentzian_dipole_source_beam(Hcomp, i, j, position, frequency, 1.0,
                                    phase, to, tdecay);
    }
}

void Gaussian_beam_prop_Lorentz(char *Ecomp, char *Hcomp, float x, float y,
                                float z, float z_c, float wo, float n,
                                float frequency, long to, long tdecay)
/*////////////////////////////////////////////////////////////////////////////
x,y,z : center of the plane on which the Gaussian beam source is to be located
z_c : z-position at which the beam waist becomes the minimum (focal point)
wo : the minimum beam waist
The above parameters should be in the unit of 'a' (lattice constant)
n : refractive index of the medium where the Gaussian beam source is
Other paramters are the same as for Gaussian dipole soure()
//////////////////////////////////////////////////////////////////////////// */
{
  int i, j;
  float phase;
  float amp;
  float zo, eta_z, w_z, R_z;
  float r2;
  int ic, jc;
  // eta_z = atan(z/zo)
  // w_z = wo*sqrt(1+z^2/zo^2)
  // R_z = z*(1+z^2/zo^2)
  // zo = pi*wo^2/lambda
  // r2 : radius^2

  zo = pi * wo * wo * frequency * n;
  ic = floor(0.5 + ((x + xcenter) * lattice_x));
  jc = floor(0.5 + ((y + ycenter) * lattice_y));
  ////////////////////////////////////
  /// some parameters which depend only on 'z'
  eta_z = atan((z_c - z) / zo);
  w_z = wo * sqrt(1 + ((z_c - z) / zo) * ((z_c - z) / zo));
  R_z = (z_c - z) * (1 + (zo / (z_c - z)) * (zo / (z_c - z)));

  /*//////////////////////////////////
  amp = (wo/w_z)*exp[ -r^2 / w_z^2 ]
  phase = kz - eta_z + r^2 * k / (2 R_z)
  For further details, see Yariv's book.
  ////////////////////////////////// */

  for (i = 0; i <= pisize; i++)
    for (j = 0; j <= pjsize; j++) {
      r2 = (i - ic) * (i - ic) / ((float)(lattice_x * lattice_x)) +
           (j - jc) * (j - jc) / ((float)(lattice_y * lattice_y));
      amp = (wo / w_z) * exp(-r2 / (w_z * w_z));
      phase = -(2 * pi * (z_c - z) * frequency * n - eta_z +
                r2 * 2 * pi * frequency * n / (2 * R_z));
      Lorentzian_dipole_source_beam(Ecomp, i, j, z, frequency, amp, phase, to,
                                    tdecay);
      Lorentzian_dipole_source_beam(Hcomp, i, j, z, frequency, amp, phase, to,
                                    tdecay);
    }
}

void Lorentzian_line_source(char *component, float position_x, float position_z,
                            float frequency, float phase, long to,
                            long tdecay) {
  int j;

  for (j = 0; j <= pjsize; j++)
    Lorentzian_dipole_source_line(component, position_x, j, position_z,
                                  frequency, phase, to, tdecay);
}

void external_planewave(char *Ecomp, char *Hcomp, float z, float Ext) {
  int i, j, k;

  k = non_uniform_z_to_i(z);
  for (i = 0; i <= pisize; i++)
    for (j = 0; j <= pjsize; j++) {
      if (strcmp(Ecomp, "Ex") == 0) {
        Ex[i][j][k] = Ex[i][j][k] + Ext / 2;
        if (i > 0)
          Ex[i - 1][j][k] = Ex[i - 1][j][k] + Ext / 2;
      }
      if (strcmp(Ecomp, "Hy") == 0) {
        Hy[i][j][k] = Hy[i][j][k] + Ext / 2 * sqrt(eo / uo) / 4;
        if (i > 0)
          Hy[i - 1][j][k] = Hy[i - 1][j][k] + Ext / 2 * sqrt(eo / uo) / 4;
      }
    }
}

void external_source(char *component, float x, float y, float z, float Ext) {
  int i, j, k; // at 3*tdecay Gaussian=0.001 //

  i = floor(0.5 + ((x + xcenter) * lattice_x));
  j = floor(0.5 + ((y + ycenter) * lattice_y));
  k = non_uniform_z_to_i(z);

  // printf("Ext= %f \n",Ext);
  if (strcmp(component, "Ex") == 0) {
    Ex[i][j][k] = Ex[i][j][k] + Ext / 2;
    Ex[i - 1][j][k] = Ex[i - 1][j][k] + Ext / 2;
  }
}

void Gaussian_dipole_source(char *component, float x, float y, float z,
                            float frequency, float phase, long to,
                            long tdecay) {
  int i, j, k; // at 3*tdecay Gaussian=0.001 //

  i = floor(0.5 + ((x + xcenter) * lattice_x));
  j = floor(0.5 + ((y + ycenter) * lattice_y));
  k = non_uniform_z_to_i(z);

  if (strcmp(component, "Ex") == 0) {
    Ex[i][j][k] = Ex[i][j][k] + Gauss_amp(frequency, phase, to, tdecay) / 2;
    Ex[i - 1][j][k] =
        Ex[i - 1][j][k] + Gauss_amp(frequency, phase, to, tdecay) / 2;
    if ((use_periodic_x == 1 || use_periodic_y == 1) &&
        (wave_vector_x != 0.0 || wave_vector_y != 0.0)) {
      iEx[i][j][k] =
          iEx[i][j][k] + iGauss_amp(frequency, phase, to, tdecay) / 2;
      iEx[i - 1][j][k] =
          iEx[i - 1][j][k] + iGauss_amp(frequency, phase, to, tdecay) / 2;
    }
  }
  if (strcmp(component, "-Ex") == 0) {
    Ex[i][j][k] =
        Ex[i][j][k] + Gauss_amp(frequency, phase + pi, to, tdecay) / 2;
    Ex[i - 1][j][k] =
        Ex[i - 1][j][k] + Gauss_amp(frequency, phase + pi, to, tdecay) / 2;
    if ((use_periodic_x == 1 || use_periodic_y == 1) &&
        (wave_vector_x != 0.0 || wave_vector_y != 0.0)) {
      iEx[i][j][k] =
          iEx[i][j][k] + iGauss_amp(frequency, phase + pi, to, tdecay) / 2;
      iEx[i - 1][j][k] =
          iEx[i - 1][j][k] + iGauss_amp(frequency, phase + pi, to, tdecay) / 2;
    }
  }
  if (strcmp(component, "Ey") == 0) {
    Ey[i][j][k] = Ey[i][j][k] + Gauss_amp(frequency, phase, to, tdecay) / 2;
    Ey[i][j - 1][k] =
        Ey[i][j - 1][k] + Gauss_amp(frequency, phase, to, tdecay) / 2;
    if ((use_periodic_x == 1 || use_periodic_y == 1) &&
        (wave_vector_x != 0.0 || wave_vector_y != 0.0)) {
      iEy[i][j][k] =
          iEy[i][j][k] + iGauss_amp(frequency, phase, to, tdecay) / 2;
      iEy[i][j - 1][k] =
          iEy[i][j - 1][k] + iGauss_amp(frequency, phase, to, tdecay) / 2;
    }
  }
  if (strcmp(component, "-Ey") == 0) {
    Ey[i][j][k] =
        Ey[i][j][k] + Gauss_amp(frequency, phase + pi, to, tdecay) / 2;
    Ey[i][j - 1][k] =
        Ey[i][j - 1][k] + Gauss_amp(frequency, phase + pi, to, tdecay) / 2;
    if ((use_periodic_x == 1 || use_periodic_y == 1) &&
        (wave_vector_x != 0.0 || wave_vector_y != 0.0)) {
      iEy[i][j][k] =
          iEy[i][j][k] + iGauss_amp(frequency, phase + pi, to, tdecay) / 2;
      iEy[i][j - 1][k] =
          iEy[i][j - 1][k] + iGauss_amp(frequency, phase + pi, to, tdecay) / 2;
    }
  }
  if (strcmp(component, "Ez") == 0) {
    Ez[i][j][k] = Ez[i][j][k] + Gauss_amp(frequency, phase, to, tdecay) / 2;
    Ez[i][j][k - 1] =
        Ez[i][j][k - 1] + Gauss_amp(frequency, phase, to, tdecay) / 2;
    if ((use_periodic_x == 1 || use_periodic_y == 1) &&
        (wave_vector_x != 0.0 || wave_vector_y != 0.0)) {
      iEz[i][j][k] =
          iEz[i][j][k] + iGauss_amp(frequency, phase, to, tdecay) / 2;
      iEz[i][j][k - 1] =
          iEz[i][j][k - 1] + iGauss_amp(frequency, phase, to, tdecay) / 2;
    }
  }
  if (strcmp(component, "-Ez") == 0) {
    Ez[i][j][k] =
        Ez[i][j][k] + Gauss_amp(frequency, phase + pi, to, tdecay) / 2;
    Ez[i][j][k - 1] =
        Ez[i][j][k - 1] + Gauss_amp(frequency, phase + pi, to, tdecay) / 2;
    if ((use_periodic_x == 1 || use_periodic_y == 1) &&
        (wave_vector_x != 0.0 || wave_vector_y != 0.0)) {
      iEz[i][j][k] =
          iEz[i][j][k] + iGauss_amp(frequency, phase + pi, to, tdecay) / 2;
      iEz[i][j][k - 1] =
          iEz[i][j][k - 1] + iGauss_amp(frequency, phase + pi, to, tdecay) / 2;
    }
  }
  if (strcmp(component, "Hx") == 0) {
    Hx[i][j][k] = Hx[i][j][k] + Gauss_amp(frequency, phase, to, tdecay) / 4;
    Hx[i][j - 1][k] =
        Hx[i][j - 1][k] + Gauss_amp(frequency, phase, to, tdecay) / 4;
    Hx[i][j][k - 1] =
        Hx[i][j][k - 1] + Gauss_amp(frequency, phase, to, tdecay) / 4;
    Hx[i][j - 1][k - 1] =
        Hx[i][j - 1][k - 1] + Gauss_amp(frequency, phase, to, tdecay) / 4;
    if ((use_periodic_x == 1 || use_periodic_y == 1) &&
        (wave_vector_x != 0.0 || wave_vector_y != 0.0)) {
      iHx[i][j][k] =
          iHx[i][j][k] + iGauss_amp(frequency, phase, to, tdecay) / 4;
      iHx[i][j - 1][k] =
          iHx[i][j - 1][k] + iGauss_amp(frequency, phase, to, tdecay) / 4;
      iHx[i][j][k - 1] =
          iHx[i][j][k - 1] + iGauss_amp(frequency, phase, to, tdecay) / 4;
      iHx[i][j - 1][k - 1] =
          iHx[i][j - 1][k - 1] + iGauss_amp(frequency, phase, to, tdecay) / 4;
    }
  }
  if (strcmp(component, "-Hx") == 0) {
    Hx[i][j][k] =
        Hx[i][j][k] + Gauss_amp(frequency, phase + pi, to, tdecay) / 4;
    Hx[i][j - 1][k] =
        Hx[i][j - 1][k] + Gauss_amp(frequency, phase + pi, to, tdecay) / 4;
    Hx[i][j][k - 1] =
        Hx[i][j][k - 1] + Gauss_amp(frequency, phase + pi, to, tdecay) / 4;
    Hx[i][j - 1][k - 1] =
        Hx[i][j - 1][k - 1] + Gauss_amp(frequency, phase + pi, to, tdecay) / 4;
    if ((use_periodic_x == 1 || use_periodic_y == 1) &&
        (wave_vector_x != 0.0 || wave_vector_y != 0.0)) {
      iHx[i][j][k] =
          iHx[i][j][k] + iGauss_amp(frequency, phase + pi, to, tdecay) / 4;
      iHx[i][j - 1][k] =
          iHx[i][j - 1][k] + iGauss_amp(frequency, phase + pi, to, tdecay) / 4;
      iHx[i][j][k - 1] =
          iHx[i][j][k - 1] + iGauss_amp(frequency, phase + pi, to, tdecay) / 4;
      iHx[i][j - 1][k - 1] = iHx[i][j - 1][k - 1] +
                             iGauss_amp(frequency, phase + pi, to, tdecay) / 4;
    }
  }
  if (strcmp(component, "Hy") == 0) {
    Hy[i][j][k] = Hy[i][j][k] + Gauss_amp(frequency, phase, to, tdecay) / 4;
    Hy[i][j][k - 1] =
        Hy[i][j][k - 1] + Gauss_amp(frequency, phase, to, tdecay) / 4;
    Hy[i - 1][j][k] =
        Hy[i - 1][j][k] + Gauss_amp(frequency, phase, to, tdecay) / 4;
    Hy[i - 1][j][k - 1] =
        Hy[i - 1][j][k - 1] + Gauss_amp(frequency, phase, to, tdecay) / 4;
    if ((use_periodic_x == 1 || use_periodic_y == 1) &&
        (wave_vector_x != 0.0 || wave_vector_y != 0.0)) {
      iHy[i][j][k] =
          iHy[i][j][k] + iGauss_amp(frequency, phase, to, tdecay) / 4;
      iHy[i][j][k - 1] =
          iHy[i][j][k - 1] + iGauss_amp(frequency, phase, to, tdecay) / 4;
      iHy[i - 1][j][k] =
          iHy[i - 1][j][k] + iGauss_amp(frequency, phase, to, tdecay) / 4;
      iHy[i - 1][j][k - 1] =
          iHy[i - 1][j][k - 1] + iGauss_amp(frequency, phase, to, tdecay) / 4;
    }
  }
  if (strcmp(component, "-Hy") == 0) {
    Hy[i][j][k] =
        Hy[i][j][k] + Gauss_amp(frequency, phase + pi, to, tdecay) / 4;
    Hy[i][j][k - 1] =
        Hy[i][j][k - 1] + Gauss_amp(frequency, phase + pi, to, tdecay) / 4;
    Hy[i - 1][j][k] =
        Hy[i - 1][j][k] + Gauss_amp(frequency, phase + pi, to, tdecay) / 4;
    Hy[i - 1][j][k - 1] =
        Hy[i - 1][j][k - 1] + Gauss_amp(frequency, phase + pi, to, tdecay) / 4;
    if ((use_periodic_x == 1 || use_periodic_y == 1) &&
        (wave_vector_x != 0.0 || wave_vector_y != 0.0)) {
      iHy[i][j][k] =
          iHy[i][j][k] + iGauss_amp(frequency, phase + pi, to, tdecay) / 4;
      iHy[i][j][k - 1] =
          iHy[i][j][k - 1] + iGauss_amp(frequency, phase + pi, to, tdecay) / 4;
      iHy[i - 1][j][k] =
          iHy[i - 1][j][k] + iGauss_amp(frequency, phase + pi, to, tdecay) / 4;
      iHy[i - 1][j][k - 1] = iHy[i - 1][j][k - 1] +
                             iGauss_amp(frequency, phase + pi, to, tdecay) / 4;
    }
  }
  if (strcmp(component, "Hz") == 0) {
    Hz[i][j][k] = Hz[i][j][k] + Gauss_amp(frequency, phase, to, tdecay) / 4;
    Hz[i - 1][j][k] =
        Hz[i - 1][j][k] + Gauss_amp(frequency, phase, to, tdecay) / 4;
    Hz[i][j - 1][k] =
        Hz[i][j - 1][k] + Gauss_amp(frequency, phase, to, tdecay) / 4;
    Hz[i - 1][j - 1][k] =
        Hz[i - 1][j - 1][k] + Gauss_amp(frequency, phase, to, tdecay) / 4;
    if ((use_periodic_x == 1 || use_periodic_y == 1) &&
        (wave_vector_x != 0.0 || wave_vector_y != 0.0)) {
      iHz[i][j][k] =
          iHz[i][j][k] + iGauss_amp(frequency, phase, to, tdecay) / 4;
      iHz[i - 1][j][k] =
          iHz[i - 1][j][k] + iGauss_amp(frequency, phase, to, tdecay) / 4;
      iHz[i][j - 1][k] =
          iHz[i][j - 1][k] + iGauss_amp(frequency, phase, to, tdecay) / 4;
      iHz[i - 1][j - 1][k] =
          iHz[i - 1][j - 1][k] + iGauss_amp(frequency, phase, to, tdecay) / 4;
    }
  }
  if (strcmp(component, "-Hz") == 0) {
    Hz[i][j][k] =
        Hz[i][j][k] + Gauss_amp(frequency, phase + pi, to, tdecay) / 4;
    Hz[i - 1][j][k] =
        Hz[i - 1][j][k] + Gauss_amp(frequency, phase + pi, to, tdecay) / 4;
    Hz[i][j - 1][k] =
        Hz[i][j - 1][k] + Gauss_amp(frequency, phase + pi, to, tdecay) / 4;
    Hz[i - 1][j - 1][k] =
        Hz[i - 1][j - 1][k] + Gauss_amp(frequency, phase + pi, to, tdecay) / 4;
    if ((use_periodic_x == 1 || use_periodic_y == 1) &&
        (wave_vector_x != 0.0 || wave_vector_y != 0.0)) {
      iHz[i][j][k] =
          iHz[i][j][k] + iGauss_amp(frequency, phase + pi, to, tdecay) / 4;
      iHz[i - 1][j][k] =
          iHz[i - 1][j][k] + iGauss_amp(frequency, phase + pi, to, tdecay) / 4;
      iHz[i][j - 1][k] =
          iHz[i][j - 1][k] + iGauss_amp(frequency, phase + pi, to, tdecay) / 4;
      iHz[i - 1][j - 1][k] = iHz[i - 1][j - 1][k] +
                             iGauss_amp(frequency, phase + pi, to, tdecay) / 4;
    }
  }
}

/// modified since ver. 8.75
void Gaussian_dipole_source_beam(char *component, int i, int j, float z,
                                 float frequency, float amp, float phase,
                                 long to, long tdecay) {
  int k; // at 3*tdecay Gaussian=0.001 //

  k = non_uniform_z_to_i(z);

  if (strcmp(component, "Ex") == 0) {
    Ex[i][j][k] =
        Ex[i][j][k] + amp * Gauss_amp(frequency, phase, to, tdecay) / 2;
    if (i >= 1)
      Ex[i - 1][j][k] =
          Ex[i - 1][j][k] + amp * Gauss_amp(frequency, phase, to, tdecay) / 2;
    if ((use_periodic_x == 1 || use_periodic_y == 1) &&
        (wave_vector_x != 0.0 || wave_vector_y != 0.0)) {
      iEx[i][j][k] =
          iEx[i][j][k] + amp * iGauss_amp(frequency, phase, to, tdecay) / 2;
      if (i >= 1)
        iEx[i - 1][j][k] = iEx[i - 1][j][k] +
                           amp * iGauss_amp(frequency, phase, to, tdecay) / 2;
    }
  }
  if (strcmp(component, "-Ex") == 0) {
    Ex[i][j][k] =
        Ex[i][j][k] + amp * Gauss_amp(frequency, phase + pi, to, tdecay) / 2;
    if (i >= 1)
      Ex[i - 1][j][k] = Ex[i - 1][j][k] +
                        amp * Gauss_amp(frequency, phase + pi, to, tdecay) / 2;
    if ((use_periodic_x == 1 || use_periodic_y == 1) &&
        (wave_vector_x != 0.0 || wave_vector_y != 0.0)) {
      iEx[i][j][k] = iEx[i][j][k] +
                     amp * iGauss_amp(frequency, phase + pi, to, tdecay) / 2;
      if (i >= 1)
        iEx[i - 1][j][k] =
            iEx[i - 1][j][k] +
            amp * iGauss_amp(frequency, phase + pi, to, tdecay) / 2;
    }
  }
  if (strcmp(component, "Ey") == 0) {
    Ey[i][j][k] =
        Ey[i][j][k] + amp * Gauss_amp(frequency, phase, to, tdecay) / 2;
    if (j >= 1)
      Ey[i][j - 1][k] =
          Ey[i][j - 1][k] + amp * Gauss_amp(frequency, phase, to, tdecay) / 2;
    if ((use_periodic_x == 1 || use_periodic_y == 1) &&
        (wave_vector_x != 0.0 || wave_vector_y != 0.0)) {
      iEy[i][j][k] =
          iEy[i][j][k] + amp * iGauss_amp(frequency, phase, to, tdecay) / 2;
      if (j >= 1)
        iEy[i][j - 1][k] = iEy[i][j - 1][k] +
                           amp * iGauss_amp(frequency, phase, to, tdecay) / 2;
    }
  }
  if (strcmp(component, "-Ey") == 0) {
    Ey[i][j][k] =
        Ey[i][j][k] + amp * Gauss_amp(frequency, phase + pi, to, tdecay) / 2;
    if (j >= 1)
      Ey[i][j - 1][k] = Ey[i][j - 1][k] +
                        amp * Gauss_amp(frequency, phase + pi, to, tdecay) / 2;
    if ((use_periodic_x == 1 || use_periodic_y == 1) &&
        (wave_vector_x != 0.0 || wave_vector_y != 0.0)) {
      iEy[i][j][k] = iEy[i][j][k] +
                     amp * iGauss_amp(frequency, phase + pi, to, tdecay) / 2;
      if (j >= 1)
        iEy[i][j - 1][k] =
            iEy[i][j - 1][k] +
            amp * iGauss_amp(frequency, phase + pi, to, tdecay) / 2;
    }
  }
  if (strcmp(component, "Ez") == 0) {
    Ez[i][j][k] =
        Ez[i][j][k] + amp * Gauss_amp(frequency, phase, to, tdecay) / 2;
    if (k >= 1)
      Ez[i][j][k - 1] =
          Ez[i][j][k - 1] + amp * Gauss_amp(frequency, phase, to, tdecay) / 2;
    if ((use_periodic_x == 1 || use_periodic_y == 1) &&
        (wave_vector_x != 0.0 || wave_vector_y != 0.0)) {
      iEz[i][j][k] =
          iEz[i][j][k] + amp * iGauss_amp(frequency, phase, to, tdecay) / 2;
      if (k >= 1)
        iEz[i][j][k - 1] = iEz[i][j][k - 1] +
                           amp * iGauss_amp(frequency, phase, to, tdecay) / 2;
    }
  }
  if (strcmp(component, "-Ez") == 0) {
    Ez[i][j][k] =
        Ez[i][j][k] + amp * Gauss_amp(frequency, phase + pi, to, tdecay) / 2;
    if (k >= 1)
      Ez[i][j][k - 1] = Ez[i][j][k - 1] +
                        amp * Gauss_amp(frequency, phase + pi, to, tdecay) / 2;
    if ((use_periodic_x == 1 || use_periodic_y == 1) &&
        (wave_vector_x != 0.0 || wave_vector_y != 0.0)) {
      iEz[i][j][k] = iEz[i][j][k] +
                     amp * iGauss_amp(frequency, phase + pi, to, tdecay) / 2;
      if (k >= 1)
        iEz[i][j][k - 1] =
            iEz[i][j][k - 1] +
            amp * iGauss_amp(frequency, phase + pi, to, tdecay) / 2;
    }
  }
  if (strcmp(component, "Hx") == 0) {
    Hx[i][j][k] = Hx[i][j][k] + amp * Gauss_amp(frequency, phase, to, tdecay) *
                                    sqrt(eo / uo) / 4;
    if (j >= 1)
      Hx[i][j - 1][k] =
          Hx[i][j - 1][k] +
          amp * Gauss_amp(frequency, phase, to, tdecay) * sqrt(eo / uo) / 4;
    if (k >= 1)
      Hx[i][j][k - 1] =
          Hx[i][j][k - 1] +
          amp * Gauss_amp(frequency, phase, to, tdecay) * sqrt(eo / uo) / 4;
    if (j >= 1 && k >= 1)
      Hx[i][j - 1][k - 1] =
          Hx[i][j - 1][k - 1] +
          amp * Gauss_amp(frequency, phase, to, tdecay) * sqrt(eo / uo) / 4;
    if ((use_periodic_x == 1 || use_periodic_y == 1) &&
        (wave_vector_x != 0.0 || wave_vector_y != 0.0)) {
      iHx[i][j][k] =
          iHx[i][j][k] +
          amp * iGauss_amp(frequency, phase, to, tdecay) * sqrt(eo / uo) / 4;
      if (j >= 1)
        iHx[i][j - 1][k] =
            iHx[i][j - 1][k] +
            amp * iGauss_amp(frequency, phase, to, tdecay) * sqrt(eo / uo) / 4;
      if (k >= 1)
        iHx[i][j][k - 1] =
            iHx[i][j][k - 1] +
            amp * iGauss_amp(frequency, phase, to, tdecay) * sqrt(eo / uo) / 4;
      if (j >= 1 && k >= 1)
        iHx[i][j - 1][k - 1] =
            iHx[i][j - 1][k - 1] +
            amp * iGauss_amp(frequency, phase, to, tdecay) * sqrt(eo / uo) / 4;
    }
  }
  if (strcmp(component, "-Hx") == 0) {
    Hx[i][j][k] =
        Hx[i][j][k] +
        amp * Gauss_amp(frequency, phase + pi, to, tdecay) * sqrt(eo / uo) / 4;
    if (j >= 1)
      Hx[i][j - 1][k] =
          Hx[i][j - 1][k] + amp * Gauss_amp(frequency, phase + pi, to, tdecay) *
                                sqrt(eo / uo) / 4;
    if (k >= 1)
      Hx[i][j][k - 1] =
          Hx[i][j][k - 1] + amp * Gauss_amp(frequency, phase + pi, to, tdecay) *
                                sqrt(eo / uo) / 4;
    if (j >= 1 && k >= 1)
      Hx[i][j - 1][k - 1] = Hx[i][j - 1][k - 1] +
                            amp * Gauss_amp(frequency, phase + pi, to, tdecay) *
                                sqrt(eo / uo) / 4;
    if ((use_periodic_x == 1 || use_periodic_y == 1) &&
        (wave_vector_x != 0.0 || wave_vector_y != 0.0)) {
      iHx[i][j][k] =
          iHx[i][j][k] + amp * iGauss_amp(frequency, phase + pi, to, tdecay) *
                             sqrt(eo / uo) / 4;
      if (j >= 1)
        iHx[i][j - 1][k] = iHx[i][j - 1][k] +
                           amp * iGauss_amp(frequency, phase + pi, to, tdecay) *
                               sqrt(eo / uo) / 4;
      if (k >= 1)
        iHx[i][j][k - 1] = iHx[i][j][k - 1] +
                           amp * iGauss_amp(frequency, phase + pi, to, tdecay) *
                               sqrt(eo / uo) / 4;
      if (j >= 1 && k >= 1)
        iHx[i][j - 1][k - 1] =
            iHx[i][j - 1][k - 1] +
            amp * iGauss_amp(frequency, phase + pi, to, tdecay) *
                sqrt(eo / uo) / 4;
    }
  }
  if (strcmp(component, "Hy") == 0) {
    Hy[i][j][k] = Hy[i][j][k] + amp * Gauss_amp(frequency, phase, to, tdecay) *
                                    sqrt(eo / uo) / 4;
    if (k >= 1)
      Hy[i][j][k - 1] =
          Hy[i][j][k - 1] +
          amp * Gauss_amp(frequency, phase, to, tdecay) * sqrt(eo / uo) / 4;
    if (i >= 1)
      Hy[i - 1][j][k] =
          Hy[i - 1][j][k] +
          amp * Gauss_amp(frequency, phase, to, tdecay) * sqrt(eo / uo) / 4;
    if (i >= 1 && k >= 1)
      Hy[i - 1][j][k - 1] =
          Hy[i - 1][j][k - 1] +
          amp * Gauss_amp(frequency, phase, to, tdecay) * sqrt(eo / uo) / 4;
    if ((use_periodic_x == 1 || use_periodic_y == 1) &&
        (wave_vector_x != 0.0 || wave_vector_y != 0.0)) {
      iHy[i][j][k] =
          iHy[i][j][k] + amp * iGauss_amp(frequency, phase, to, tdecay) *
                             sqrt(eo / uo) * sqrt(eo / uo) / 4;
      if (k >= 1)
        iHy[i][j][k - 1] =
            iHy[i][j][k - 1] +
            amp * iGauss_amp(frequency, phase, to, tdecay) * sqrt(eo / uo) / 4;
      if (i >= 1)
        iHy[i - 1][j][k] =
            iHy[i - 1][j][k] +
            amp * iGauss_amp(frequency, phase, to, tdecay) * sqrt(eo / uo) / 4;
      if (i >= 1 && k >= 1)
        iHy[i - 1][j][k - 1] =
            iHy[i - 1][j][k - 1] +
            amp * iGauss_amp(frequency, phase, to, tdecay) * sqrt(eo / uo) / 4;
    }
  }
  if (strcmp(component, "-Hy") == 0) {
    Hy[i][j][k] =
        Hy[i][j][k] +
        amp * Gauss_amp(frequency, phase + pi, to, tdecay) * sqrt(eo / uo) / 4;
    if (k >= 1)
      Hy[i][j][k - 1] =
          Hy[i][j][k - 1] + amp * Gauss_amp(frequency, phase + pi, to, tdecay) *
                                sqrt(eo / uo) / 4;
    if (i >= 1)
      Hy[i - 1][j][k] =
          Hy[i - 1][j][k] + amp * Gauss_amp(frequency, phase + pi, to, tdecay) *
                                sqrt(eo / uo) / 4;
    if (i >= 1 && k >= 1)
      Hy[i - 1][j][k - 1] = Hy[i - 1][j][k - 1] +
                            amp * Gauss_amp(frequency, phase + pi, to, tdecay) *
                                sqrt(eo / uo) / 4;
    if ((use_periodic_x == 1 || use_periodic_y == 1) &&
        (wave_vector_x != 0.0 || wave_vector_y != 0.0)) {
      iHy[i][j][k] =
          iHy[i][j][k] + amp * iGauss_amp(frequency, phase + pi, to, tdecay) *
                             sqrt(eo / uo) / 4;
      if (k >= 1)
        iHy[i][j][k - 1] = iHy[i][j][k - 1] +
                           amp * iGauss_amp(frequency, phase + pi, to, tdecay) *
                               sqrt(eo / uo) / 4;
      if (i >= 1)
        iHy[i - 1][j][k] = iHy[i - 1][j][k] +
                           amp * iGauss_amp(frequency, phase + pi, to, tdecay) *
                               sqrt(eo / uo) / 4;
      if (i >= 1 && k >= 1)
        iHy[i - 1][j][k - 1] =
            iHy[i - 1][j][k - 1] +
            amp * iGauss_amp(frequency, phase + pi, to, tdecay) *
                sqrt(eo / uo) / 4;
    }
  }
  if (strcmp(component, "Hz") == 0) {
    Hz[i][j][k] = Hz[i][j][k] + amp * Gauss_amp(frequency, phase, to, tdecay) *
                                    sqrt(eo / uo) / 4;
    if (i >= 1)
      Hz[i - 1][j][k] =
          Hz[i - 1][j][k] +
          amp * Gauss_amp(frequency, phase, to, tdecay) * sqrt(eo / uo) / 4;
    if (j >= 1)
      Hz[i][j - 1][k] =
          Hz[i][j - 1][k] +
          amp * Gauss_amp(frequency, phase, to, tdecay) * sqrt(eo / uo) / 4;
    if (i >= 1 && j >= 1)
      Hz[i - 1][j - 1][k] =
          Hz[i - 1][j - 1][k] +
          amp * Gauss_amp(frequency, phase, to, tdecay) * sqrt(eo / uo) / 4;
    if ((use_periodic_x == 1 || use_periodic_y == 1) &&
        (wave_vector_x != 0.0 || wave_vector_y != 0.0)) {
      iHz[i][j][k] =
          iHz[i][j][k] +
          amp * iGauss_amp(frequency, phase, to, tdecay) * sqrt(eo / uo) / 4;
      if (i >= 1)
        iHz[i - 1][j][k] =
            iHz[i - 1][j][k] +
            amp * iGauss_amp(frequency, phase, to, tdecay) * sqrt(eo / uo) / 4;
      if (j >= 1)
        iHz[i][j - 1][k] =
            iHz[i][j - 1][k] +
            amp * iGauss_amp(frequency, phase, to, tdecay) * sqrt(eo / uo) / 4;
      if (i >= 1 && j >= 1)
        iHz[i - 1][j - 1][k] =
            iHz[i - 1][j - 1][k] +
            amp * iGauss_amp(frequency, phase, to, tdecay) * sqrt(eo / uo) / 4;
    }
  }
  if (strcmp(component, "-Hz") == 0) {
    Hz[i][j][k] =
        Hz[i][j][k] +
        amp * Gauss_amp(frequency, phase + pi, to, tdecay) * sqrt(eo / uo) / 4;
    if (i >= 1)
      Hz[i - 1][j][k] =
          Hz[i - 1][j][k] + amp * Gauss_amp(frequency, phase + pi, to, tdecay) *
                                sqrt(eo / uo) / 4;
    if (j >= 1)
      Hz[i][j - 1][k] =
          Hz[i][j - 1][k] + amp * Gauss_amp(frequency, phase + pi, to, tdecay) *
                                sqrt(eo / uo) / 4;
    if (i >= 1 && j >= 1)
      Hz[i - 1][j - 1][k] = Hz[i - 1][j - 1][k] +
                            amp * Gauss_amp(frequency, phase + pi, to, tdecay) *
                                sqrt(eo / uo) / 4;
    if ((use_periodic_x == 1 || use_periodic_y == 1) &&
        (wave_vector_x != 0.0 || wave_vector_y != 0.0)) {
      iHz[i][j][k] =
          iHz[i][j][k] + amp * iGauss_amp(frequency, phase + pi, to, tdecay) *
                             sqrt(eo / uo) / 4;
      if (i >= 1)
        iHz[i - 1][j][k] = iHz[i - 1][j][k] +
                           amp * iGauss_amp(frequency, phase + pi, to, tdecay) *
                               sqrt(eo / uo) / 4;
      if (j >= 1)
        iHz[i][j - 1][k] = iHz[i][j - 1][k] +
                           amp * iGauss_amp(frequency, phase + pi, to, tdecay) *
                               sqrt(eo / uo) / 4;
      if (i >= 1 && j >= 1)
        iHz[i - 1][j - 1][k] =
            iHz[i - 1][j - 1][k] +
            amp * iGauss_amp(frequency, phase + pi, to, tdecay) *
                sqrt(eo / uo) / 4;
    }
  }
}

void Gaussian_dipole_source_line(char *component, float x, int j, float z,
                                 float frequency, float phase, long to,
                                 long tdecay) {
  int i, k; // at 3*tdecay Gaussian=0.001 //

  i = floor(0.5 + ((x + xcenter) * lattice_x));
  k = non_uniform_z_to_i(z);

  if (strcmp(component, "Ex") == 0) {
    Ex[i][j][k] = Ex[i][j][k] + Gauss_amp(frequency, phase, to, tdecay) / 2;
    if (i >= 1)
      Ex[i - 1][j][k] =
          Ex[i - 1][j][k] + Gauss_amp(frequency, phase, to, tdecay) / 2;
    if ((use_periodic_x == 1 || use_periodic_y == 1) &&
        (wave_vector_x != 0.0 || wave_vector_y != 0.0)) {
      iEx[i][j][k] =
          iEx[i][j][k] + iGauss_amp(frequency, phase, to, tdecay) / 2;
      if (i >= 1)
        iEx[i - 1][j][k] =
            iEx[i - 1][j][k] + iGauss_amp(frequency, phase, to, tdecay) / 2;
    }
  }
  if (strcmp(component, "-Ex") == 0) {
    Ex[i][j][k] =
        Ex[i][j][k] + Gauss_amp(frequency, phase + pi, to, tdecay) / 2;
    if (i >= 1)
      Ex[i - 1][j][k] =
          Ex[i - 1][j][k] + Gauss_amp(frequency, phase + pi, to, tdecay) / 2;
    if ((use_periodic_x == 1 || use_periodic_y == 1) &&
        (wave_vector_x != 0.0 || wave_vector_y != 0.0)) {
      iEx[i][j][k] =
          iEx[i][j][k] + iGauss_amp(frequency, phase + pi, to, tdecay) / 2;
      if (i >= 1)
        iEx[i - 1][j][k] = iEx[i - 1][j][k] +
                           iGauss_amp(frequency, phase + pi, to, tdecay) / 2;
    }
  }
  if (strcmp(component, "Ey") == 0) {
    Ey[i][j][k] = Ey[i][j][k] + Gauss_amp(frequency, phase, to, tdecay) / 2;
    if (j >= 1)
      Ey[i][j - 1][k] =
          Ey[i][j - 1][k] + Gauss_amp(frequency, phase, to, tdecay) / 2;
    if ((use_periodic_x == 1 || use_periodic_y == 1) &&
        (wave_vector_x != 0.0 || wave_vector_y != 0.0)) {
      iEy[i][j][k] =
          iEy[i][j][k] + iGauss_amp(frequency, phase, to, tdecay) / 2;
      if (j >= 1)
        iEy[i][j - 1][k] =
            iEy[i][j - 1][k] + iGauss_amp(frequency, phase, to, tdecay) / 2;
    }
  }
  if (strcmp(component, "-Ey") == 0) {
    Ey[i][j][k] =
        Ey[i][j][k] + Gauss_amp(frequency, phase + pi, to, tdecay) / 2;
    if (j >= 1)
      Ey[i][j - 1][k] =
          Ey[i][j - 1][k] + Gauss_amp(frequency, phase + pi, to, tdecay) / 2;
    if ((use_periodic_x == 1 || use_periodic_y == 1) &&
        (wave_vector_x != 0.0 || wave_vector_y != 0.0)) {
      iEy[i][j][k] =
          iEy[i][j][k] + iGauss_amp(frequency, phase + pi, to, tdecay) / 2;
      if (j >= 1)
        iEy[i][j - 1][k] = iEy[i][j - 1][k] +
                           iGauss_amp(frequency, phase + pi, to, tdecay) / 2;
    }
  }
  if (strcmp(component, "Ez") == 0) {
    Ez[i][j][k] = Ez[i][j][k] + Gauss_amp(frequency, phase, to, tdecay) / 2;
    if (k >= 1)
      Ez[i][j][k - 1] =
          Ez[i][j][k - 1] + Gauss_amp(frequency, phase, to, tdecay) / 2;
    if ((use_periodic_x == 1 || use_periodic_y == 1) &&
        (wave_vector_x != 0.0 || wave_vector_y != 0.0)) {
      iEz[i][j][k] =
          iEz[i][j][k] + iGauss_amp(frequency, phase, to, tdecay) / 2;
      if (k >= 1)
        iEz[i][j][k - 1] =
            iEz[i][j][k - 1] + iGauss_amp(frequency, phase, to, tdecay) / 2;
    }
  }
  if (strcmp(component, "-Ez") == 0) {
    Ez[i][j][k] =
        Ez[i][j][k] + Gauss_amp(frequency, phase + pi, to, tdecay) / 2;
    if (k >= 1)
      Ez[i][j][k - 1] =
          Ez[i][j][k - 1] + Gauss_amp(frequency, phase + pi, to, tdecay) / 2;
    if ((use_periodic_x == 1 || use_periodic_y == 1) &&
        (wave_vector_x != 0.0 || wave_vector_y != 0.0)) {
      iEz[i][j][k] =
          iEz[i][j][k] + iGauss_amp(frequency, phase + pi, to, tdecay) / 2;
      if (k >= 1)
        iEz[i][j][k - 1] = iEz[i][j][k - 1] +
                           iGauss_amp(frequency, phase + pi, to, tdecay) / 2;
    }
  }
  if (strcmp(component, "Hx") == 0) {
    Hx[i][j][k] = Hx[i][j][k] +
                  Gauss_amp(frequency, phase, to, tdecay) * sqrt(eo / uo) / 4;
    if (j >= 1)
      Hx[i][j - 1][k] =
          Hx[i][j - 1][k] +
          Gauss_amp(frequency, phase, to, tdecay) * sqrt(eo / uo) / 4;
    if (k >= 1)
      Hx[i][j][k - 1] =
          Hx[i][j][k - 1] +
          Gauss_amp(frequency, phase, to, tdecay) * sqrt(eo / uo) / 4;
    if (j >= 1 && k >= 1)
      Hx[i][j - 1][k - 1] =
          Hx[i][j - 1][k - 1] +
          Gauss_amp(frequency, phase, to, tdecay) * sqrt(eo / uo) / 4;
    if ((use_periodic_x == 1 || use_periodic_y == 1) &&
        (wave_vector_x != 0.0 || wave_vector_y != 0.0)) {
      iHx[i][j][k] = iHx[i][j][k] + iGauss_amp(frequency, phase, to, tdecay) *
                                        sqrt(eo / uo) / 4;
      if (j >= 1)
        iHx[i][j - 1][k] =
            iHx[i][j - 1][k] +
            iGauss_amp(frequency, phase, to, tdecay) * sqrt(eo / uo) / 4;
      if (k >= 1)
        iHx[i][j][k - 1] =
            iHx[i][j][k - 1] +
            iGauss_amp(frequency, phase, to, tdecay) * sqrt(eo / uo) / 4;
      if (j >= 1 && k >= 1)
        iHx[i][j - 1][k - 1] =
            iHx[i][j - 1][k - 1] +
            iGauss_amp(frequency, phase, to, tdecay) * sqrt(eo / uo) / 4;
    }
  }
  if (strcmp(component, "-Hx") == 0) {
    Hx[i][j][k] = Hx[i][j][k] + Gauss_amp(frequency, phase + pi, to, tdecay) *
                                    sqrt(eo / uo) / 4;
    if (j >= 1)
      Hx[i][j - 1][k] =
          Hx[i][j - 1][k] +
          Gauss_amp(frequency, phase + pi, to, tdecay) * sqrt(eo / uo) / 4;
    if (k >= 1)
      Hx[i][j][k - 1] =
          Hx[i][j][k - 1] +
          Gauss_amp(frequency, phase + pi, to, tdecay) * sqrt(eo / uo) / 4;
    if (j >= 1 && k >= 1)
      Hx[i][j - 1][k - 1] =
          Hx[i][j - 1][k - 1] +
          Gauss_amp(frequency, phase + pi, to, tdecay) * sqrt(eo / uo) / 4;
    if ((use_periodic_x == 1 || use_periodic_y == 1) &&
        (wave_vector_x != 0.0 || wave_vector_y != 0.0)) {
      iHx[i][j][k] =
          iHx[i][j][k] +
          iGauss_amp(frequency, phase + pi, to, tdecay) * sqrt(eo / uo) / 4;
      if (j >= 1)
        iHx[i][j - 1][k] =
            iHx[i][j - 1][k] +
            iGauss_amp(frequency, phase + pi, to, tdecay) * sqrt(eo / uo) / 4;
      if (k >= 1)
        iHx[i][j][k - 1] =
            iHx[i][j][k - 1] +
            iGauss_amp(frequency, phase + pi, to, tdecay) * sqrt(eo / uo) / 4;
      if (j >= 1 && k >= 1)
        iHx[i][j - 1][k - 1] =
            iHx[i][j - 1][k - 1] +
            iGauss_amp(frequency, phase + pi, to, tdecay) * sqrt(eo / uo) / 4;
    }
  }
  if (strcmp(component, "Hy") == 0) {
    Hy[i][j][k] = Hy[i][j][k] +
                  Gauss_amp(frequency, phase, to, tdecay) * sqrt(eo / uo) / 4;
    if (k >= 1)
      Hy[i][j][k - 1] =
          Hy[i][j][k - 1] +
          Gauss_amp(frequency, phase, to, tdecay) * sqrt(eo / uo) / 4;
    if (i >= 1)
      Hy[i - 1][j][k] =
          Hy[i - 1][j][k] +
          Gauss_amp(frequency, phase, to, tdecay) * sqrt(eo / uo) / 4;
    if (i >= 1 && k >= 1)
      Hy[i - 1][j][k - 1] =
          Hy[i - 1][j][k - 1] +
          Gauss_amp(frequency, phase, to, tdecay) * sqrt(eo / uo) / 4;
    if ((use_periodic_x == 1 || use_periodic_y == 1) &&
        (wave_vector_x != 0.0 || wave_vector_y != 0.0)) {
      iHy[i][j][k] = iHy[i][j][k] + iGauss_amp(frequency, phase, to, tdecay) *
                                        sqrt(eo / uo) * sqrt(eo / uo) / 4;
      if (k >= 1)
        iHy[i][j][k - 1] =
            iHy[i][j][k - 1] +
            iGauss_amp(frequency, phase, to, tdecay) * sqrt(eo / uo) / 4;
      if (i >= 1)
        iHy[i - 1][j][k] =
            iHy[i - 1][j][k] +
            iGauss_amp(frequency, phase, to, tdecay) * sqrt(eo / uo) / 4;
      if (i >= 1 && k >= 1)
        iHy[i - 1][j][k - 1] =
            iHy[i - 1][j][k - 1] +
            iGauss_amp(frequency, phase, to, tdecay) * sqrt(eo / uo) / 4;
    }
  }
  if (strcmp(component, "-Hy") == 0) {
    Hy[i][j][k] = Hy[i][j][k] + Gauss_amp(frequency, phase + pi, to, tdecay) *
                                    sqrt(eo / uo) / 4;
    if (k >= 1)
      Hy[i][j][k - 1] =
          Hy[i][j][k - 1] +
          Gauss_amp(frequency, phase + pi, to, tdecay) * sqrt(eo / uo) / 4;
    if (i >= 1)
      Hy[i - 1][j][k] =
          Hy[i - 1][j][k] +
          Gauss_amp(frequency, phase + pi, to, tdecay) * sqrt(eo / uo) / 4;
    if (i >= 1 && k >= 1)
      Hy[i - 1][j][k - 1] =
          Hy[i - 1][j][k - 1] +
          Gauss_amp(frequency, phase + pi, to, tdecay) * sqrt(eo / uo) / 4;
    if ((use_periodic_x == 1 || use_periodic_y == 1) &&
        (wave_vector_x != 0.0 || wave_vector_y != 0.0)) {
      iHy[i][j][k] =
          iHy[i][j][k] +
          iGauss_amp(frequency, phase + pi, to, tdecay) * sqrt(eo / uo) / 4;
      if (k >= 1)
        iHy[i][j][k - 1] =
            iHy[i][j][k - 1] +
            iGauss_amp(frequency, phase + pi, to, tdecay) * sqrt(eo / uo) / 4;
      if (i >= 1)
        iHy[i - 1][j][k] =
            iHy[i - 1][j][k] +
            iGauss_amp(frequency, phase + pi, to, tdecay) * sqrt(eo / uo) / 4;
      if (i >= 1 && k >= 1)
        iHy[i - 1][j][k - 1] =
            iHy[i - 1][j][k - 1] +
            iGauss_amp(frequency, phase + pi, to, tdecay) * sqrt(eo / uo) / 4;
    }
  }
  if (strcmp(component, "Hz") == 0) {
    Hz[i][j][k] = Hz[i][j][k] +
                  Gauss_amp(frequency, phase, to, tdecay) * sqrt(eo / uo) / 4;
    if (i >= 1)
      Hz[i - 1][j][k] =
          Hz[i - 1][j][k] +
          Gauss_amp(frequency, phase, to, tdecay) * sqrt(eo / uo) / 4;
    if (j >= 1)
      Hz[i][j - 1][k] =
          Hz[i][j - 1][k] +
          Gauss_amp(frequency, phase, to, tdecay) * sqrt(eo / uo) / 4;
    if (i >= 1 && j >= 1)
      Hz[i - 1][j - 1][k] =
          Hz[i - 1][j - 1][k] +
          Gauss_amp(frequency, phase, to, tdecay) * sqrt(eo / uo) / 4;
    if ((use_periodic_x == 1 || use_periodic_y == 1) &&
        (wave_vector_x != 0.0 || wave_vector_y != 0.0)) {
      iHz[i][j][k] = iHz[i][j][k] + iGauss_amp(frequency, phase, to, tdecay) *
                                        sqrt(eo / uo) / 4;
      if (i >= 1)
        iHz[i - 1][j][k] =
            iHz[i - 1][j][k] +
            iGauss_amp(frequency, phase, to, tdecay) * sqrt(eo / uo) / 4;
      if (j >= 1)
        iHz[i][j - 1][k] =
            iHz[i][j - 1][k] +
            iGauss_amp(frequency, phase, to, tdecay) * sqrt(eo / uo) / 4;
      if (i >= 1 && j >= 1)
        iHz[i - 1][j - 1][k] =
            iHz[i - 1][j - 1][k] +
            iGauss_amp(frequency, phase, to, tdecay) * sqrt(eo / uo) / 4;
    }
  }
  if (strcmp(component, "-Hz") == 0) {
    Hz[i][j][k] = Hz[i][j][k] + Gauss_amp(frequency, phase + pi, to, tdecay) *
                                    sqrt(eo / uo) / 4;
    if (i >= 1)
      Hz[i - 1][j][k] =
          Hz[i - 1][j][k] +
          Gauss_amp(frequency, phase + pi, to, tdecay) * sqrt(eo / uo) / 4;
    if (j >= 1)
      Hz[i][j - 1][k] =
          Hz[i][j - 1][k] +
          Gauss_amp(frequency, phase + pi, to, tdecay) * sqrt(eo / uo) / 4;
    if (i >= 1 && j >= 1)
      Hz[i - 1][j - 1][k] =
          Hz[i - 1][j - 1][k] +
          Gauss_amp(frequency, phase + pi, to, tdecay) * sqrt(eo / uo) / 4;
    if ((use_periodic_x == 1 || use_periodic_y == 1) &&
        (wave_vector_x != 0.0 || wave_vector_y != 0.0)) {
      iHz[i][j][k] =
          iHz[i][j][k] +
          iGauss_amp(frequency, phase + pi, to, tdecay) * sqrt(eo / uo) / 4;
      if (i >= 1)
        iHz[i - 1][j][k] =
            iHz[i - 1][j][k] +
            iGauss_amp(frequency, phase + pi, to, tdecay) * sqrt(eo / uo) / 4;
      if (j >= 1)
        iHz[i][j - 1][k] =
            iHz[i][j - 1][k] +
            iGauss_amp(frequency, phase + pi, to, tdecay) * sqrt(eo / uo) / 4;
      if (i >= 1 && j >= 1)
        iHz[i - 1][j - 1][k] =
            iHz[i - 1][j - 1][k] +
            iGauss_amp(frequency, phase + pi, to, tdecay) * sqrt(eo / uo) / 4;
    }
  }
}

void Lorentzian_dipole_source(char *component, float x, float y, float z,
                              float frequency, float phase, long to,
                              long tdecay) {
  int i, j, k; // at 8*tdecay Lorentzian=0.0001 //

  i = floor(0.5 + ((x + xcenter) * lattice_x));
  j = floor(0.5 + ((y + ycenter) * lattice_y));
  k = non_uniform_z_to_i(z);

  if (strcmp(component, "Ex") == 0) {
    Ex[i][j][k] = Ex[i][j][k] + Lorentz_amp(frequency, phase, to, tdecay) / 2;
    Ex[i - 1][j][k] =
        Ex[i - 1][j][k] + Lorentz_amp(frequency, phase, to, tdecay) / 2;
    if ((use_periodic_x == 1 || use_periodic_y == 1) &&
        (wave_vector_x != 0.0 || wave_vector_y != 0.0)) {
      iEx[i][j][k] =
          iEx[i][j][k] + iLorentz_amp(frequency, phase, to, tdecay) / 2;
      iEx[i - 1][j][k] =
          iEx[i - 1][j][k] + iLorentz_amp(frequency, phase, to, tdecay) / 2;
    }
  }
  if (strcmp(component, "-Ex") == 0) {
    Ex[i][j][k] =
        Ex[i][j][k] + Lorentz_amp(frequency, phase + pi, to, tdecay) / 2;
    Ex[i - 1][j][k] =
        Ex[i - 1][j][k] + Lorentz_amp(frequency, phase + pi, to, tdecay) / 2;
    if ((use_periodic_x == 1 || use_periodic_y == 1) &&
        (wave_vector_x != 0.0 || wave_vector_y != 0.0)) {
      iEx[i][j][k] =
          iEx[i][j][k] + iLorentz_amp(frequency, phase + pi, to, tdecay) / 2;
      iEx[i - 1][j][k] = iEx[i - 1][j][k] +
                         iLorentz_amp(frequency, phase + pi, to, tdecay) / 2;
    }
  }
  if (strcmp(component, "Ey") == 0) {
    Ey[i][j][k] = Ey[i][j][k] + Lorentz_amp(frequency, phase, to, tdecay) / 2;
    Ey[i][j - 1][k] =
        Ey[i][j - 1][k] + Lorentz_amp(frequency, phase, to, tdecay) / 2;
    if ((use_periodic_x == 1 || use_periodic_y == 1) &&
        (wave_vector_x != 0.0 || wave_vector_y != 0.0)) {
      iEy[i][j][k] =
          iEy[i][j][k] + iLorentz_amp(frequency, phase, to, tdecay) / 2;
      iEy[i][j - 1][k] =
          iEy[i][j - 1][k] + iLorentz_amp(frequency, phase, to, tdecay) / 2;
    }
  }
  if (strcmp(component, "-Ey") == 0) {
    Ey[i][j][k] =
        Ey[i][j][k] + Lorentz_amp(frequency, phase + pi, to, tdecay) / 2;
    Ey[i][j - 1][k] =
        Ey[i][j - 1][k] + Lorentz_amp(frequency, phase + pi, to, tdecay) / 2;
    if ((use_periodic_x == 1 || use_periodic_y == 1) &&
        (wave_vector_x != 0.0 || wave_vector_y != 0.0)) {
      iEy[i][j][k] =
          iEy[i][j][k] + iLorentz_amp(frequency, phase + pi, to, tdecay) / 2;
      iEy[i][j - 1][k] = iEy[i][j - 1][k] +
                         iLorentz_amp(frequency, phase + pi, to, tdecay) / 2;
    }
  }
  if (strcmp(component, "Ez") == 0) {
    Ez[i][j][k] = Ez[i][j][k] + Lorentz_amp(frequency, phase, to, tdecay) / 2;
    Ez[i][j][k - 1] =
        Ez[i][j][k - 1] + Lorentz_amp(frequency, phase, to, tdecay) / 2;
    if ((use_periodic_x == 1 || use_periodic_y == 1) &&
        (wave_vector_x != 0.0 || wave_vector_y != 0.0)) {
      iEz[i][j][k] =
          iEz[i][j][k] + iLorentz_amp(frequency, phase, to, tdecay) / 2;
      iEz[i][j][k - 1] =
          iEz[i][j][k - 1] + iLorentz_amp(frequency, phase, to, tdecay) / 2;
    }
  }
  if (strcmp(component, "-Ez") == 0) {
    Ez[i][j][k] =
        Ez[i][j][k] + Lorentz_amp(frequency, phase + pi, to, tdecay) / 2;
    Ez[i][j][k - 1] =
        Ez[i][j][k - 1] + Lorentz_amp(frequency, phase + pi, to, tdecay) / 2;
    if ((use_periodic_x == 1 || use_periodic_y == 1) &&
        (wave_vector_x != 0.0 || wave_vector_y != 0.0)) {
      iEz[i][j][k] =
          iEz[i][j][k] + iLorentz_amp(frequency, phase + pi, to, tdecay) / 2;
      iEz[i][j][k - 1] = iEz[i][j][k - 1] +
                         iLorentz_amp(frequency, phase + pi, to, tdecay) / 2;
    }
  }
  if (strcmp(component, "Hx") == 0) {
    Hx[i][j][k] = Hx[i][j][k] + Lorentz_amp(frequency, phase, to, tdecay) / 4;
    Hx[i][j - 1][k] =
        Hx[i][j - 1][k] + Lorentz_amp(frequency, phase, to, tdecay) / 4;
    Hx[i][j][k - 1] =
        Hx[i][j][k - 1] + Lorentz_amp(frequency, phase, to, tdecay) / 4;
    Hx[i][j - 1][k - 1] =
        Hx[i][j - 1][k - 1] + Lorentz_amp(frequency, phase, to, tdecay) / 4;
    if ((use_periodic_x == 1 || use_periodic_y == 1) &&
        (wave_vector_x != 0.0 || wave_vector_y != 0.0)) {
      iHx[i][j][k] =
          iHx[i][j][k] + iLorentz_amp(frequency, phase, to, tdecay) / 4;
      iHx[i][j - 1][k] =
          iHx[i][j - 1][k] + iLorentz_amp(frequency, phase, to, tdecay) / 4;
      iHx[i][j][k - 1] =
          iHx[i][j][k - 1] + iLorentz_amp(frequency, phase, to, tdecay) / 4;
      iHx[i][j - 1][k - 1] =
          iHx[i][j - 1][k - 1] + iLorentz_amp(frequency, phase, to, tdecay) / 4;
    }
  }
  if (strcmp(component, "-Hx") == 0) {
    Hx[i][j][k] =
        Hx[i][j][k] + Lorentz_amp(frequency, phase + pi, to, tdecay) / 4;
    Hx[i][j - 1][k] =
        Hx[i][j - 1][k] + Lorentz_amp(frequency, phase + pi, to, tdecay) / 4;
    Hx[i][j][k - 1] =
        Hx[i][j][k - 1] + Lorentz_amp(frequency, phase + pi, to, tdecay) / 4;
    Hx[i][j - 1][k - 1] = Hx[i][j - 1][k - 1] +
                          Lorentz_amp(frequency, phase + pi, to, tdecay) / 4;
    if ((use_periodic_x == 1 || use_periodic_y == 1) &&
        (wave_vector_x != 0.0 || wave_vector_y != 0.0)) {
      iHx[i][j][k] =
          iHx[i][j][k] + iLorentz_amp(frequency, phase + pi, to, tdecay) / 4;
      iHx[i][j - 1][k] = iHx[i][j - 1][k] +
                         iLorentz_amp(frequency, phase + pi, to, tdecay) / 4;
      iHx[i][j][k - 1] = iHx[i][j][k - 1] +
                         iLorentz_amp(frequency, phase + pi, to, tdecay) / 4;
      iHx[i][j - 1][k - 1] =
          iHx[i][j - 1][k - 1] +
          iLorentz_amp(frequency, phase + pi, to, tdecay) / 4;
    }
  }
  if (strcmp(component, "Hy") == 0) {
    Hy[i][j][k] = Hy[i][j][k] + Lorentz_amp(frequency, phase, to, tdecay) / 4;
    Hy[i][j][k - 1] =
        Hy[i][j][k - 1] + Lorentz_amp(frequency, phase, to, tdecay) / 4;
    Hy[i - 1][j][k] =
        Hy[i - 1][j][k] + Lorentz_amp(frequency, phase, to, tdecay) / 4;
    Hy[i - 1][j][k - 1] =
        Hy[i - 1][j][k - 1] + Lorentz_amp(frequency, phase, to, tdecay) / 4;
    if ((use_periodic_x == 1 || use_periodic_y == 1) &&
        (wave_vector_x != 0.0 || wave_vector_y != 0.0)) {
      iHy[i][j][k] =
          iHy[i][j][k] + iLorentz_amp(frequency, phase, to, tdecay) / 4;
      iHy[i][j][k - 1] =
          iHy[i][j][k - 1] + iLorentz_amp(frequency, phase, to, tdecay) / 4;
      iHy[i - 1][j][k] =
          iHy[i - 1][j][k] + iLorentz_amp(frequency, phase, to, tdecay) / 4;
      iHy[i - 1][j][k - 1] =
          iHy[i - 1][j][k - 1] + iLorentz_amp(frequency, phase, to, tdecay) / 4;
    }
  }
  if (strcmp(component, "-Hy") == 0) {
    Hy[i][j][k] =
        Hy[i][j][k] + Lorentz_amp(frequency, phase + pi, to, tdecay) / 4;
    Hy[i][j][k - 1] =
        Hy[i][j][k - 1] + Lorentz_amp(frequency, phase + pi, to, tdecay) / 4;
    Hy[i - 1][j][k] =
        Hy[i - 1][j][k] + Lorentz_amp(frequency, phase + pi, to, tdecay) / 4;
    Hy[i - 1][j][k - 1] = Hy[i - 1][j][k - 1] +
                          Lorentz_amp(frequency, phase + pi, to, tdecay) / 4;
    if ((use_periodic_x == 1 || use_periodic_y == 1) &&
        (wave_vector_x != 0.0 || wave_vector_y != 0.0)) {
      iHy[i][j][k] =
          iHy[i][j][k] + iLorentz_amp(frequency, phase + pi, to, tdecay) / 4;
      iHy[i][j][k - 1] = iHy[i][j][k - 1] +
                         iLorentz_amp(frequency, phase + pi, to, tdecay) / 4;
      iHy[i - 1][j][k] = iHy[i - 1][j][k] +
                         iLorentz_amp(frequency, phase + pi, to, tdecay) / 4;
      iHy[i - 1][j][k - 1] =
          iHy[i - 1][j][k - 1] +
          iLorentz_amp(frequency, phase + pi, to, tdecay) / 4;
    }
  }
  if (strcmp(component, "Hz") == 0) {
    Hz[i][j][k] = Hz[i][j][k] + Lorentz_amp(frequency, phase, to, tdecay) / 4;
    Hz[i - 1][j][k] =
        Hz[i - 1][j][k] + Lorentz_amp(frequency, phase, to, tdecay) / 4;
    Hz[i][j - 1][k] =
        Hz[i][j - 1][k] + Lorentz_amp(frequency, phase, to, tdecay) / 4;
    Hz[i - 1][j - 1][k] =
        Hz[i - 1][j - 1][k] + Lorentz_amp(frequency, phase, to, tdecay) / 4;
    if ((use_periodic_x == 1 || use_periodic_y == 1) &&
        (wave_vector_x != 0.0 || wave_vector_y != 0.0)) {
      iHz[i][j][k] =
          iHz[i][j][k] + iLorentz_amp(frequency, phase, to, tdecay) / 4;
      iHz[i - 1][j][k] =
          iHz[i - 1][j][k] + iLorentz_amp(frequency, phase, to, tdecay) / 4;
      iHz[i][j - 1][k] =
          iHz[i][j - 1][k] + iLorentz_amp(frequency, phase, to, tdecay) / 4;
      iHz[i - 1][j - 1][k] =
          iHz[i - 1][j - 1][k] + iLorentz_amp(frequency, phase, to, tdecay) / 4;
    }
  }
  if (strcmp(component, "-Hz") == 0) {
    Hz[i][j][k] =
        Hz[i][j][k] + Lorentz_amp(frequency, phase + pi, to, tdecay) / 4;
    Hz[i - 1][j][k] =
        Hz[i - 1][j][k] + Lorentz_amp(frequency, phase + pi, to, tdecay) / 4;
    Hz[i][j - 1][k] =
        Hz[i][j - 1][k] + Lorentz_amp(frequency, phase + pi, to, tdecay) / 4;
    Hz[i - 1][j - 1][k] = Hz[i - 1][j - 1][k] +
                          Lorentz_amp(frequency, phase + pi, to, tdecay) / 4;
    if ((use_periodic_x == 1 || use_periodic_y == 1) &&
        (wave_vector_x != 0.0 || wave_vector_y != 0.0)) {
      iHz[i][j][k] =
          iHz[i][j][k] + iLorentz_amp(frequency, phase + pi, to, tdecay) / 4;
      iHz[i - 1][j][k] = iHz[i - 1][j][k] +
                         iLorentz_amp(frequency, phase + pi, to, tdecay) / 4;
      iHz[i][j - 1][k] = iHz[i][j - 1][k] +
                         iLorentz_amp(frequency, phase + pi, to, tdecay) / 4;
      iHz[i - 1][j - 1][k] =
          iHz[i - 1][j - 1][k] +
          iLorentz_amp(frequency, phase + pi, to, tdecay) / 4;
    }
  }
}

// modified since ver. 8.75
void Lorentzian_dipole_source_beam(char *component, int i, int j, float z,
                                   float frequency, float amp, float phase,
                                   long to, long tdecay) {
  int k; // at 8*tdecay Lorentzian=0.0001 //

  k = non_uniform_z_to_i(z);

  if (strcmp(component, "Ex") == 0) {
    Ex[i][j][k] =
        Ex[i][j][k] + amp * Lorentz_amp(frequency, phase, to, tdecay) / 2;
    if (i >= 1)
      Ex[i - 1][j][k] =
          Ex[i - 1][j][k] + amp * Lorentz_amp(frequency, phase, to, tdecay) / 2;
    if ((use_periodic_x == 1 || use_periodic_y == 1) &&
        (wave_vector_x != 0.0 || wave_vector_y != 0.0)) {
      iEx[i][j][k] =
          iEx[i][j][k] + amp * iLorentz_amp(frequency, phase, to, tdecay) / 2;
      if (i >= 1)
        iEx[i - 1][j][k] = iEx[i - 1][j][k] +
                           amp * iLorentz_amp(frequency, phase, to, tdecay) / 2;
    }
  }
  if (strcmp(component, "-Ex") == 0) {
    Ex[i][j][k] =
        Ex[i][j][k] + amp * Lorentz_amp(frequency, phase + pi, to, tdecay) / 2;
    if (i >= 1)
      Ex[i - 1][j][k] =
          Ex[i - 1][j][k] +
          amp * Lorentz_amp(frequency, phase + pi, to, tdecay) / 2;
    if ((use_periodic_x == 1 || use_periodic_y == 1) &&
        (wave_vector_x != 0.0 || wave_vector_y != 0.0)) {
      iEx[i][j][k] = iEx[i][j][k] +
                     amp * iLorentz_amp(frequency, phase + pi, to, tdecay) / 2;
      if (i >= 1)
        iEx[i - 1][j][k] =
            iEx[i - 1][j][k] +
            amp * iLorentz_amp(frequency, phase + pi, to, tdecay) / 2;
    }
  }
  if (strcmp(component, "Ey") == 0) {
    Ey[i][j][k] =
        Ey[i][j][k] + amp * Lorentz_amp(frequency, phase, to, tdecay) / 2;
    if (j >= 1)
      Ey[i][j - 1][k] =
          Ey[i][j - 1][k] + amp * Lorentz_amp(frequency, phase, to, tdecay) / 2;
    if ((use_periodic_x == 1 || use_periodic_y == 1) &&
        (wave_vector_x != 0.0 || wave_vector_y != 0.0)) {
      iEy[i][j][k] =
          iEy[i][j][k] + amp * iLorentz_amp(frequency, phase, to, tdecay) / 2;
      if (j >= 1)
        iEy[i][j - 1][k] = iEy[i][j - 1][k] +
                           amp * iLorentz_amp(frequency, phase, to, tdecay) / 2;
    }
  }
  if (strcmp(component, "-Ey") == 0) {
    Ey[i][j][k] =
        Ey[i][j][k] + amp * Lorentz_amp(frequency, phase + pi, to, tdecay) / 2;
    if (j >= 1)
      Ey[i][j - 1][k] =
          Ey[i][j - 1][k] +
          amp * Lorentz_amp(frequency, phase + pi, to, tdecay) / 2;
    if ((use_periodic_x == 1 || use_periodic_y == 1) &&
        (wave_vector_x != 0.0 || wave_vector_y != 0.0)) {
      iEy[i][j][k] = iEy[i][j][k] +
                     amp * iLorentz_amp(frequency, phase + pi, to, tdecay) / 2;
      if (j >= 1)
        iEy[i][j - 1][k] =
            iEy[i][j - 1][k] +
            amp * iLorentz_amp(frequency, phase + pi, to, tdecay) / 2;
    }
  }
  if (strcmp(component, "Ez") == 0) {
    Ez[i][j][k] =
        Ez[i][j][k] + amp * Lorentz_amp(frequency, phase, to, tdecay) / 2;
    if (k >= 1)
      Ez[i][j][k - 1] =
          Ez[i][j][k - 1] + amp * Lorentz_amp(frequency, phase, to, tdecay) / 2;
    if ((use_periodic_x == 1 || use_periodic_y == 1) &&
        (wave_vector_x != 0.0 || wave_vector_y != 0.0)) {
      iEz[i][j][k] =
          iEz[i][j][k] + amp * iLorentz_amp(frequency, phase, to, tdecay) / 2;
      if (k >= 1)
        iEz[i][j][k - 1] = iEz[i][j][k - 1] +
                           amp * iLorentz_amp(frequency, phase, to, tdecay) / 2;
    }
  }
  if (strcmp(component, "-Ez") == 0) {
    Ez[i][j][k] =
        Ez[i][j][k] + amp * Lorentz_amp(frequency, phase + pi, to, tdecay) / 2;
    if (k >= 1)
      Ez[i][j][k - 1] =
          Ez[i][j][k - 1] +
          amp * Lorentz_amp(frequency, phase + pi, to, tdecay) / 2;
    if ((use_periodic_x == 1 || use_periodic_y == 1) &&
        (wave_vector_x != 0.0 || wave_vector_y != 0.0)) {
      iEz[i][j][k] = iEz[i][j][k] +
                     amp * iLorentz_amp(frequency, phase + pi, to, tdecay) / 2;
      if (k >= 1)
        iEz[i][j][k - 1] =
            iEz[i][j][k - 1] +
            amp * iLorentz_amp(frequency, phase + pi, to, tdecay) / 2;
    }
  }
  if (strcmp(component, "Hx") == 0) {
    Hx[i][j][k] = Hx[i][j][k] + amp *
                                    Lorentz_amp(frequency, phase, to, tdecay) *
                                    sqrt(eo / uo) / 4;
    if (j >= 1)
      Hx[i][j - 1][k] =
          Hx[i][j - 1][k] +
          amp * Lorentz_amp(frequency, phase, to, tdecay) * sqrt(eo / uo) / 4;
    if (k >= 1)
      Hx[i][j][k - 1] =
          Hx[i][j][k - 1] +
          amp * Lorentz_amp(frequency, phase, to, tdecay) * sqrt(eo / uo) / 4;
    if (j >= 1 && k >= 1)
      Hx[i][j - 1][k - 1] =
          Hx[i][j - 1][k - 1] +
          amp * Lorentz_amp(frequency, phase, to, tdecay) * sqrt(eo / uo) / 4;
    if ((use_periodic_x == 1 || use_periodic_y == 1) &&
        (wave_vector_x != 0.0 || wave_vector_y != 0.0)) {
      iHx[i][j][k] =
          iHx[i][j][k] +
          amp * iLorentz_amp(frequency, phase, to, tdecay) * sqrt(eo / uo) / 4;
      if (j >= 1)
        iHx[i][j - 1][k] = iHx[i][j - 1][k] +
                           amp * iLorentz_amp(frequency, phase, to, tdecay) *
                               sqrt(eo / uo) / 4;
      if (k >= 1)
        iHx[i][j][k - 1] = iHx[i][j][k - 1] +
                           amp * iLorentz_amp(frequency, phase, to, tdecay) *
                               sqrt(eo / uo) / 4;
      if (j >= 1 && k >= 1)
        iHx[i][j - 1][k - 1] = iHx[i][j - 1][k - 1] +
                               amp *
                                   iLorentz_amp(frequency, phase, to, tdecay) *
                                   sqrt(eo / uo) / 4;
    }
  }
  if (strcmp(component, "-Hx") == 0) {
    Hx[i][j][k] =
        Hx[i][j][k] + amp * Lorentz_amp(frequency, phase + pi, to, tdecay) *
                          sqrt(eo / uo) / 4;
    if (j >= 1)
      Hx[i][j - 1][k] = Hx[i][j - 1][k] +
                        amp * Lorentz_amp(frequency, phase + pi, to, tdecay) *
                            sqrt(eo / uo) / 4;
    if (k >= 1)
      Hx[i][j][k - 1] = Hx[i][j][k - 1] +
                        amp * Lorentz_amp(frequency, phase + pi, to, tdecay) *
                            sqrt(eo / uo) / 4;
    if (j >= 1 && k >= 1)
      Hx[i][j - 1][k - 1] = Hx[i][j - 1][k - 1] +
                            amp *
                                Lorentz_amp(frequency, phase + pi, to, tdecay) *
                                sqrt(eo / uo) / 4;
    if ((use_periodic_x == 1 || use_periodic_y == 1) &&
        (wave_vector_x != 0.0 || wave_vector_y != 0.0)) {
      iHx[i][j][k] =
          iHx[i][j][k] + amp * iLorentz_amp(frequency, phase + pi, to, tdecay) *
                             sqrt(eo / uo) / 4;
      if (j >= 1)
        iHx[i][j - 1][k] = iHx[i][j - 1][k] +
                           amp *
                               iLorentz_amp(frequency, phase + pi, to, tdecay) *
                               sqrt(eo / uo) / 4;
      if (k >= 1)
        iHx[i][j][k - 1] = iHx[i][j][k - 1] +
                           amp *
                               iLorentz_amp(frequency, phase + pi, to, tdecay) *
                               sqrt(eo / uo) / 4;
      if (j >= 1 && k >= 1)
        iHx[i][j - 1][k - 1] =
            iHx[i][j - 1][k - 1] +
            amp * iLorentz_amp(frequency, phase + pi, to, tdecay) *
                sqrt(eo / uo) / 4;
    }
  }
  if (strcmp(component, "Hy") == 0) {
    Hy[i][j][k] = Hy[i][j][k] + amp *
                                    Lorentz_amp(frequency, phase, to, tdecay) *
                                    sqrt(eo / uo) / 4;
    if (k >= 1)
      Hy[i][j][k - 1] =
          Hy[i][j][k - 1] +
          amp * Lorentz_amp(frequency, phase, to, tdecay) * sqrt(eo / uo) / 4;
    if (i >= 1)
      Hy[i - 1][j][k] =
          Hy[i - 1][j][k] +
          amp * Lorentz_amp(frequency, phase, to, tdecay) * sqrt(eo / uo) / 4;
    if (i >= 1 && k >= 1)
      Hy[i - 1][j][k - 1] =
          Hy[i - 1][j][k - 1] +
          amp * Lorentz_amp(frequency, phase, to, tdecay) * sqrt(eo / uo) / 4;
    if ((use_periodic_x == 1 || use_periodic_y == 1) &&
        (wave_vector_x != 0.0 || wave_vector_y != 0.0)) {
      iHy[i][j][k] =
          iHy[i][j][k] +
          amp * iLorentz_amp(frequency, phase, to, tdecay) * sqrt(eo / uo) / 4;
      if (k >= 1)
        iHy[i][j][k - 1] = iHy[i][j][k - 1] +
                           amp * iLorentz_amp(frequency, phase, to, tdecay) *
                               sqrt(eo / uo) / 4;
      if (i >= 1)
        iHy[i - 1][j][k] = iHy[i - 1][j][k] +
                           amp * iLorentz_amp(frequency, phase, to, tdecay) *
                               sqrt(eo / uo) / 4;
      if (i >= 1 && k >= 1)
        iHy[i - 1][j][k - 1] = iHy[i - 1][j][k - 1] +
                               amp *
                                   iLorentz_amp(frequency, phase, to, tdecay) *
                                   sqrt(eo / uo) / 4;
    }
  }
  if (strcmp(component, "-Hy") == 0) {
    Hy[i][j][k] =
        Hy[i][j][k] + amp * Lorentz_amp(frequency, phase + pi, to, tdecay) *
                          sqrt(eo / uo) / 4;
    if (k >= 1)
      Hy[i][j][k - 1] = Hy[i][j][k - 1] +
                        amp * Lorentz_amp(frequency, phase + pi, to, tdecay) *
                            sqrt(eo / uo) / 4;
    if (i >= 1)
      Hy[i - 1][j][k] = Hy[i - 1][j][k] +
                        amp * Lorentz_amp(frequency, phase + pi, to, tdecay) *
                            sqrt(eo / uo) / 4;
    if (i >= 1 && k >= 1)
      Hy[i - 1][j][k - 1] = Hy[i - 1][j][k - 1] +
                            amp *
                                Lorentz_amp(frequency, phase + pi, to, tdecay) *
                                sqrt(eo / uo) / 4;
    if ((use_periodic_x == 1 || use_periodic_y == 1) &&
        (wave_vector_x != 0.0 || wave_vector_y != 0.0)) {
      iHy[i][j][k] =
          iHy[i][j][k] + amp * iLorentz_amp(frequency, phase + pi, to, tdecay) *
                             sqrt(eo / uo) / 4;
      if (k >= 1)
        iHy[i][j][k - 1] = iHy[i][j][k - 1] +
                           amp *
                               iLorentz_amp(frequency, phase + pi, to, tdecay) *
                               sqrt(eo / uo) / 4;
      if (i >= 1)
        iHy[i - 1][j][k] = iHy[i - 1][j][k] +
                           amp *
                               iLorentz_amp(frequency, phase + pi, to, tdecay) *
                               sqrt(eo / uo) / 4;
      if (i >= 1 && k >= 1)
        iHy[i - 1][j][k - 1] =
            iHy[i - 1][j][k - 1] +
            amp * iLorentz_amp(frequency, phase + pi, to, tdecay) *
                sqrt(eo / uo) / 4;
    }
  }
  if (strcmp(component, "Hz") == 0) {
    Hz[i][j][k] = Hz[i][j][k] + amp *
                                    Lorentz_amp(frequency, phase, to, tdecay) *
                                    sqrt(eo / uo) / 4;
    if (i >= 1)
      Hz[i - 1][j][k] =
          Hz[i - 1][j][k] +
          amp * Lorentz_amp(frequency, phase, to, tdecay) * sqrt(eo / uo) / 4;
    if (j >= 1)
      Hz[i][j - 1][k] =
          Hz[i][j - 1][k] +
          amp * Lorentz_amp(frequency, phase, to, tdecay) * sqrt(eo / uo) / 4;
    if (i >= 1 && j >= 1)
      Hz[i - 1][j - 1][k] =
          Hz[i - 1][j - 1][k] +
          amp * Lorentz_amp(frequency, phase, to, tdecay) * sqrt(eo / uo) / 4;
    if ((use_periodic_x == 1 || use_periodic_y == 1) &&
        (wave_vector_x != 0.0 || wave_vector_y != 0.0)) {
      iHz[i][j][k] =
          iHz[i][j][k] +
          amp * iLorentz_amp(frequency, phase, to, tdecay) * sqrt(eo / uo) / 4;
      if (i >= 1)
        iHz[i - 1][j][k] = iHz[i - 1][j][k] +
                           amp * iLorentz_amp(frequency, phase, to, tdecay) *
                               sqrt(eo / uo) / 4;
      if (j >= 1)
        iHz[i][j - 1][k] = iHz[i][j - 1][k] +
                           amp * iLorentz_amp(frequency, phase, to, tdecay) *
                               sqrt(eo / uo) / 4;
      if (i >= 1 && j >= 1)
        iHz[i - 1][j - 1][k] = iHz[i - 1][j - 1][k] +
                               amp *
                                   iLorentz_amp(frequency, phase, to, tdecay) *
                                   sqrt(eo / uo) / 4;
    }
  }
  if (strcmp(component, "-Hz") == 0) {
    Hz[i][j][k] =
        Hz[i][j][k] + amp * Lorentz_amp(frequency, phase + pi, to, tdecay) *
                          sqrt(eo / uo) / 4;
    if (i >= 1)
      Hz[i - 1][j][k] = Hz[i - 1][j][k] +
                        amp * Lorentz_amp(frequency, phase + pi, to, tdecay) *
                            sqrt(eo / uo) / 4;
    if (j >= 1)
      Hz[i][j - 1][k] = Hz[i][j - 1][k] +
                        amp * Lorentz_amp(frequency, phase + pi, to, tdecay) *
                            sqrt(eo / uo) / 4;
    if (i >= 1 && j >= 1)
      Hz[i - 1][j - 1][k] = Hz[i - 1][j - 1][k] +
                            amp *
                                Lorentz_amp(frequency, phase + pi, to, tdecay) *
                                sqrt(eo / uo) / 4;
    if ((use_periodic_x == 1 || use_periodic_y == 1) &&
        (wave_vector_x != 0.0 || wave_vector_y != 0.0)) {
      iHz[i][j][k] =
          iHz[i][j][k] + amp * iLorentz_amp(frequency, phase + pi, to, tdecay) *
                             sqrt(eo / uo) / 4;
      if (i >= 1)
        iHz[i - 1][j][k] = iHz[i - 1][j][k] +
                           amp *
                               iLorentz_amp(frequency, phase + pi, to, tdecay) *
                               sqrt(eo / uo) / 4;
      if (j >= 1)
        iHz[i][j - 1][k] = iHz[i][j - 1][k] +
                           amp *
                               iLorentz_amp(frequency, phase + pi, to, tdecay) *
                               sqrt(eo / uo) / 4;
      if (i >= 1 && j >= 1)
        iHz[i - 1][j - 1][k] =
            iHz[i - 1][j - 1][k] +
            amp * iLorentz_amp(frequency, phase + pi, to, tdecay) *
                sqrt(eo / uo) / 4;
    }
  }
}

void Lorentzian_dipole_source_line(char *component, float x, int j, float z,
                                   float frequency, float phase, long to,
                                   long tdecay) {
  int i, k; // at 8*tdecay Lorentzian=0.0001 //

  i = floor(0.5 + ((x + xcenter) * lattice_x));
  k = non_uniform_z_to_i(z);

  if (strcmp(component, "Ex") == 0) {
    Ex[i][j][k] = Ex[i][j][k] + Lorentz_amp(frequency, phase, to, tdecay) / 2;
    if (i >= 1)
      Ex[i - 1][j][k] =
          Ex[i - 1][j][k] + Lorentz_amp(frequency, phase, to, tdecay) / 2;
    if ((use_periodic_x == 1 || use_periodic_y == 1) &&
        (wave_vector_x != 0.0 || wave_vector_y != 0.0)) {
      iEx[i][j][k] =
          iEx[i][j][k] + iLorentz_amp(frequency, phase, to, tdecay) / 2;
      if (i >= 1)
        iEx[i - 1][j][k] =
            iEx[i - 1][j][k] + iLorentz_amp(frequency, phase, to, tdecay) / 2;
    }
  }
  if (strcmp(component, "-Ex") == 0) {
    Ex[i][j][k] =
        Ex[i][j][k] + Lorentz_amp(frequency, phase + pi, to, tdecay) / 2;
    if (i >= 1)
      Ex[i - 1][j][k] =
          Ex[i - 1][j][k] + Lorentz_amp(frequency, phase + pi, to, tdecay) / 2;
    if ((use_periodic_x == 1 || use_periodic_y == 1) &&
        (wave_vector_x != 0.0 || wave_vector_y != 0.0)) {
      iEx[i][j][k] =
          iEx[i][j][k] + iLorentz_amp(frequency, phase + pi, to, tdecay) / 2;
      if (i >= 1)
        iEx[i - 1][j][k] = iEx[i - 1][j][k] +
                           iLorentz_amp(frequency, phase + pi, to, tdecay) / 2;
    }
  }
  if (strcmp(component, "Ey") == 0) {
    Ey[i][j][k] = Ey[i][j][k] + Lorentz_amp(frequency, phase, to, tdecay) / 2;
    if (j >= 1)
      Ey[i][j - 1][k] =
          Ey[i][j - 1][k] + Lorentz_amp(frequency, phase, to, tdecay) / 2;
    if ((use_periodic_x == 1 || use_periodic_y == 1) &&
        (wave_vector_x != 0.0 || wave_vector_y != 0.0)) {
      iEy[i][j][k] =
          iEy[i][j][k] + iLorentz_amp(frequency, phase, to, tdecay) / 2;
      if (j >= 1)
        iEy[i][j - 1][k] =
            iEy[i][j - 1][k] + iLorentz_amp(frequency, phase, to, tdecay) / 2;
    }
  }
  if (strcmp(component, "-Ey") == 0) {
    Ey[i][j][k] =
        Ey[i][j][k] + Lorentz_amp(frequency, phase + pi, to, tdecay) / 2;
    if (j >= 1)
      Ey[i][j - 1][k] =
          Ey[i][j - 1][k] + Lorentz_amp(frequency, phase + pi, to, tdecay) / 2;
    if ((use_periodic_x == 1 || use_periodic_y == 1) &&
        (wave_vector_x != 0.0 || wave_vector_y != 0.0)) {
      iEy[i][j][k] =
          iEy[i][j][k] + iLorentz_amp(frequency, phase + pi, to, tdecay) / 2;
      if (j >= 1)
        iEy[i][j - 1][k] = iEy[i][j - 1][k] +
                           iLorentz_amp(frequency, phase + pi, to, tdecay) / 2;
    }
  }
  if (strcmp(component, "Ez") == 0) {
    Ez[i][j][k] = Ez[i][j][k] + Lorentz_amp(frequency, phase, to, tdecay) / 2;
    if (k >= 1)
      Ez[i][j][k - 1] =
          Ez[i][j][k - 1] + Lorentz_amp(frequency, phase, to, tdecay) / 2;
    if ((use_periodic_x == 1 || use_periodic_y == 1) &&
        (wave_vector_x != 0.0 || wave_vector_y != 0.0)) {
      iEz[i][j][k] =
          iEz[i][j][k] + iLorentz_amp(frequency, phase, to, tdecay) / 2;
      if (k >= 1)
        iEz[i][j][k - 1] =
            iEz[i][j][k - 1] + iLorentz_amp(frequency, phase, to, tdecay) / 2;
    }
  }
  if (strcmp(component, "-Ez") == 0) {
    Ez[i][j][k] =
        Ez[i][j][k] + Lorentz_amp(frequency, phase + pi, to, tdecay) / 2;
    if (k >= 1)
      Ez[i][j][k - 1] =
          Ez[i][j][k - 1] + Lorentz_amp(frequency, phase + pi, to, tdecay) / 2;
    if ((use_periodic_x == 1 || use_periodic_y == 1) &&
        (wave_vector_x != 0.0 || wave_vector_y != 0.0)) {
      iEz[i][j][k] =
          iEz[i][j][k] + iLorentz_amp(frequency, phase + pi, to, tdecay) / 2;
      if (k >= 1)
        iEz[i][j][k - 1] = iEz[i][j][k - 1] +
                           iLorentz_amp(frequency, phase + pi, to, tdecay) / 2;
    }
  }
  if (strcmp(component, "Hx") == 0) {
    Hx[i][j][k] = Hx[i][j][k] + Lorentz_amp(frequency, phase, to, tdecay) / 4;
    if (j >= 1)
      Hx[i][j - 1][k] =
          Hx[i][j - 1][k] + Lorentz_amp(frequency, phase, to, tdecay) / 4;
    if (k >= 1)
      Hx[i][j][k - 1] =
          Hx[i][j][k - 1] + Lorentz_amp(frequency, phase, to, tdecay) / 4;
    if (j >= 1 && k >= 1)
      Hx[i][j - 1][k - 1] =
          Hx[i][j - 1][k - 1] + Lorentz_amp(frequency, phase, to, tdecay) / 4;
    if ((use_periodic_x == 1 || use_periodic_y == 1) &&
        (wave_vector_x != 0.0 || wave_vector_y != 0.0)) {
      iHx[i][j][k] =
          iHx[i][j][k] + iLorentz_amp(frequency, phase, to, tdecay) / 4;
      if (j >= 1)
        iHx[i][j - 1][k] =
            iHx[i][j - 1][k] + iLorentz_amp(frequency, phase, to, tdecay) / 4;
      if (k >= 1)
        iHx[i][j][k - 1] =
            iHx[i][j][k - 1] + iLorentz_amp(frequency, phase, to, tdecay) / 4;
      if (j >= 1 && k >= 1)
        iHx[i][j - 1][k - 1] = iHx[i][j - 1][k - 1] +
                               iLorentz_amp(frequency, phase, to, tdecay) / 4;
    }
  }
  if (strcmp(component, "-Hx") == 0) {
    Hx[i][j][k] =
        Hx[i][j][k] + Lorentz_amp(frequency, phase + pi, to, tdecay) / 4;
    if (j >= 1)
      Hx[i][j - 1][k] =
          Hx[i][j - 1][k] + Lorentz_amp(frequency, phase + pi, to, tdecay) / 4;
    if (k >= 1)
      Hx[i][j][k - 1] =
          Hx[i][j][k - 1] + Lorentz_amp(frequency, phase + pi, to, tdecay) / 4;
    if (j >= 1 && k >= 1)
      Hx[i][j - 1][k - 1] = Hx[i][j - 1][k - 1] +
                            Lorentz_amp(frequency, phase + pi, to, tdecay) / 4;
    if ((use_periodic_x == 1 || use_periodic_y == 1) &&
        (wave_vector_x != 0.0 || wave_vector_y != 0.0)) {
      iHx[i][j][k] =
          iHx[i][j][k] + iLorentz_amp(frequency, phase + pi, to, tdecay) / 4;
      if (j >= 1)
        iHx[i][j - 1][k] = iHx[i][j - 1][k] +
                           iLorentz_amp(frequency, phase + pi, to, tdecay) / 4;
      if (k >= 1)
        iHx[i][j][k - 1] = iHx[i][j][k - 1] +
                           iLorentz_amp(frequency, phase + pi, to, tdecay) / 4;
      if (j >= 1 && k >= 1)
        iHx[i][j - 1][k - 1] =
            iHx[i][j - 1][k - 1] +
            iLorentz_amp(frequency, phase + pi, to, tdecay) / 4;
    }
  }
  if (strcmp(component, "Hy") == 0) {
    Hy[i][j][k] = Hy[i][j][k] + Lorentz_amp(frequency, phase, to, tdecay) / 4;
    if (k >= 1)
      Hy[i][j][k - 1] =
          Hy[i][j][k - 1] + Lorentz_amp(frequency, phase, to, tdecay) / 4;
    if (i >= 1)
      Hy[i - 1][j][k] =
          Hy[i - 1][j][k] + Lorentz_amp(frequency, phase, to, tdecay) / 4;
    if (i >= 1 && k >= 1)
      Hy[i - 1][j][k - 1] =
          Hy[i - 1][j][k - 1] + Lorentz_amp(frequency, phase, to, tdecay) / 4;
    if ((use_periodic_x == 1 || use_periodic_y == 1) &&
        (wave_vector_x != 0.0 || wave_vector_y != 0.0)) {
      iHy[i][j][k] =
          iHy[i][j][k] + iLorentz_amp(frequency, phase, to, tdecay) / 4;
      if (k >= 1)
        iHy[i][j][k - 1] =
            iHy[i][j][k - 1] + iLorentz_amp(frequency, phase, to, tdecay) / 4;
      if (i >= 1)
        iHy[i - 1][j][k] =
            iHy[i - 1][j][k] + iLorentz_amp(frequency, phase, to, tdecay) / 4;
      if (i >= 1 && k >= 1)
        iHy[i - 1][j][k - 1] = iHy[i - 1][j][k - 1] +
                               iLorentz_amp(frequency, phase, to, tdecay) / 4;
    }
  }
  if (strcmp(component, "-Hy") == 0) {
    Hy[i][j][k] =
        Hy[i][j][k] + Lorentz_amp(frequency, phase + pi, to, tdecay) / 4;
    if (k >= 1)
      Hy[i][j][k - 1] =
          Hy[i][j][k - 1] + Lorentz_amp(frequency, phase + pi, to, tdecay) / 4;
    if (i >= 1)
      Hy[i - 1][j][k] =
          Hy[i - 1][j][k] + Lorentz_amp(frequency, phase + pi, to, tdecay) / 4;
    if (i >= 1 && k >= 1)
      Hy[i - 1][j][k - 1] = Hy[i - 1][j][k - 1] +
                            Lorentz_amp(frequency, phase + pi, to, tdecay) / 4;
    if ((use_periodic_x == 1 || use_periodic_y == 1) &&
        (wave_vector_x != 0.0 || wave_vector_y != 0.0)) {
      iHy[i][j][k] =
          iHy[i][j][k] + iLorentz_amp(frequency, phase + pi, to, tdecay) / 4;
      if (k >= 1)
        iHy[i][j][k - 1] = iHy[i][j][k - 1] +
                           iLorentz_amp(frequency, phase + pi, to, tdecay) / 4;
      if (i >= 1)
        iHy[i - 1][j][k] = iHy[i - 1][j][k] +
                           iLorentz_amp(frequency, phase + pi, to, tdecay) / 4;
      if (i >= 1 && k >= 1)
        iHy[i - 1][j][k - 1] =
            iHy[i - 1][j][k - 1] +
            iLorentz_amp(frequency, phase + pi, to, tdecay) / 4;
    }
  }
  if (strcmp(component, "Hz") == 0) {
    Hz[i][j][k] = Hz[i][j][k] + Lorentz_amp(frequency, phase, to, tdecay) / 4;
    if (i >= 1)
      Hz[i - 1][j][k] =
          Hz[i - 1][j][k] + Lorentz_amp(frequency, phase, to, tdecay) / 4;
    if (j >= 1)
      Hz[i][j - 1][k] =
          Hz[i][j - 1][k] + Lorentz_amp(frequency, phase, to, tdecay) / 4;
    if (i >= 1 && j >= 1)
      Hz[i - 1][j - 1][k] =
          Hz[i - 1][j - 1][k] + Lorentz_amp(frequency, phase, to, tdecay) / 4;
    if ((use_periodic_x == 1 || use_periodic_y == 1) &&
        (wave_vector_x != 0.0 || wave_vector_y != 0.0)) {
      iHz[i][j][k] =
          iHz[i][j][k] + iLorentz_amp(frequency, phase, to, tdecay) / 4;
      if (i >= 1)
        iHz[i - 1][j][k] =
            iHz[i - 1][j][k] + iLorentz_amp(frequency, phase, to, tdecay) / 4;
      if (j >= 1)
        iHz[i][j - 1][k] =
            iHz[i][j - 1][k] + iLorentz_amp(frequency, phase, to, tdecay) / 4;
      if (i >= 1 && j >= 1)
        iHz[i - 1][j - 1][k] = iHz[i - 1][j - 1][k] +
                               iLorentz_amp(frequency, phase, to, tdecay) / 4;
    }
  }
  if (strcmp(component, "-Hz") == 0) {
    Hz[i][j][k] =
        Hz[i][j][k] + Lorentz_amp(frequency, phase + pi, to, tdecay) / 4;
    if (i >= 1)
      Hz[i - 1][j][k] =
          Hz[i - 1][j][k] + Lorentz_amp(frequency, phase + pi, to, tdecay) / 4;
    if (j >= 1)
      Hz[i][j - 1][k] =
          Hz[i][j - 1][k] + Lorentz_amp(frequency, phase + pi, to, tdecay) / 4;
    if (i >= 1 && j >= 1)
      Hz[i - 1][j - 1][k] = Hz[i - 1][j - 1][k] +
                            Lorentz_amp(frequency, phase + pi, to, tdecay) / 4;
    if ((use_periodic_x == 1 || use_periodic_y == 1) &&
        (wave_vector_x != 0.0 || wave_vector_y != 0.0)) {
      iHz[i][j][k] =
          iHz[i][j][k] + iLorentz_amp(frequency, phase + pi, to, tdecay) / 4;
      if (i >= 1)
        iHz[i - 1][j][k] = iHz[i - 1][j][k] +
                           iLorentz_amp(frequency, phase + pi, to, tdecay) / 4;
      if (j >= 1)
        iHz[i][j - 1][k] = iHz[i][j - 1][k] +
                           iLorentz_amp(frequency, phase + pi, to, tdecay) / 4;
      if (i >= 1 && j >= 1)
        iHz[i - 1][j - 1][k] =
            iHz[i - 1][j - 1][k] +
            iLorentz_amp(frequency, phase + pi, to, tdecay) / 4;
    }
  }
}

float Lorentzian_phase(float Wn, long tdecay) {
  /* Wn : normalized frequency */
  /* tdecay : decay constant of the exponential decay */
  /* gamma : inverse of the tdecay */

  float omega, gamma;

  omega = 2 * pi * Wn / S_factor / ds_x / lattice_x;
  printf("Lorentz omega=%f\n", omega);
  gamma = 1 / ((float)tdecay);

  return (asin(gamma / sqrt(omega * omega + gamma * gamma)));
}

float Gaussian_phase(float Wn, long t_peak) {
  /* Wn : normalized frequency */
  /* t_peak : the time at which the Gaussian envelop is maximized */

  float omega;

  omega = 2 * pi * Wn / S_factor / ds_x / lattice_x;
  printf("Gaussian omega=%f\n", omega);

  return (-omega * (float)t_peak);
}
