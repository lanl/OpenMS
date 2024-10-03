#include "./pFDTD.h"

void field_zero(float ***name);

static float tus;

void normal_E_field_update(int i, int j, int k);    // real field
void molecular_E_field_update(int i, int j, int k); // real field
void metal_E_field_update(int i, int j, int k);
void pml_E_field_update(int i, int j, int k);
void cpml_E_field_update(int i, int j, int k);
void pcm_E_field_update(int i, int j, int k);
void normal_H_field_update(int i, int j, int k);
void pml_H_field_update(int i, int j, int k);
void cpml_H_field_update(int i, int j, int k);
void pcm_H_field_update(int i, int j, int k);

void normal_iE_field_update(int i, int j, int k); // imaginary field
void metal_iE_field_update(int i, int j, int k);
void pml_iE_field_update(int i, int j, int k);
void pcm_iE_field_update(int i, int j, int k);
void normal_iH_field_update(int i, int j, int k);
void pml_iH_field_update(int i, int j, int k);
void pcm_iH_field_update(int i, int j, int k);

void E_field_periodic_boundary_update_x();
void E_field_periodic_boundary_update_y();
void H_field_periodic_boundary_update_x();
void H_field_periodic_boundary_update_y();
void E_field_Gamma_boundary_update_x();
void E_field_Gamma_boundary_update_y();
void H_field_Gamma_boundary_update_x();
void H_field_Gamma_boundary_update_y();

void E_field_periodic_boundary_update_Xwall();
void E_field_periodic_boundary_update_Ywall();
void H_field_periodic_boundary_update_Xwall();
void H_field_periodic_boundary_update_Ywall();
void E_field_Gamma_boundary_update_Xwall();
void E_field_Gamma_boundary_update_Ywall();
void H_field_Gamma_boundary_update_Xwall();
void H_field_Gamma_boundary_update_Ywall();

void Ez_parity_boundary_update();
void Ez_parity_iboundary_update();
void Hz_parity_boundary_update();
void Hz_parity_iboundary_update();

void coefficient() {
  int i, j, k;

  for (i = 0; i < misize; i++)
    for (j = 0; j < mjsize; j++)
      for (k = 0; k < mksize; k++) {
        if (position[i][j][k]) { // Do nothing!
        }

        else {
          aax[i] =
              (2 * eo * kx - dt * sigmax(i)) / (2 * eo * kx + dt * sigmax(i));
          aay[j] =
              (2 * eo * ky - dt * sigmay(j)) / (2 * eo * ky + dt * sigmay(j));
          aaz[k] =
              (2 * eo * kz - dt * sigmaz(k)) / (2 * eo * kz + dt * sigmaz(k));

          bbx[i] = (2 * eo * dt) / (2 * eo * kx + dt * sigmax(i));
          bby[j] = (2 * eo * dt) / (2 * eo * ky + dt * sigmay(j));
          bbz[k] = (2 * eo * dt) / (2 * eo * kz + dt * sigmaz(k));

          ccx[i] =
              (2 * eo * kx - dt * sigmax(i)) / (2 * eo * kx + dt * sigmax(i));
          ccy[j] =
              (2 * eo * ky - dt * sigmay(j)) / (2 * eo * ky + dt * sigmay(j));
          ccz[k] =
              (2 * eo * kz - dt * sigmaz(k)) / (2 * eo * kz + dt * sigmaz(k));

          ddx[i][j][k] = 2 / (2 * eo * kz + dt * sigmaz(k)) / epsilonx[i][j][k];
          ddy[i][j][k] = 2 / (2 * eo * kx + dt * sigmax(i)) / epsilony[i][j][k];
          ddz[i][j][k] = 2 / (2 * eo * ky + dt * sigmay(j)) / epsilonz[i][j][k];

          eex[i] = kx + dt * sigmax(i + 0.5) / 2 / eo;
          eey[j] = ky + dt * sigmay(j + 0.5) / 2 / eo;
          eez[k] = kz + dt * sigmaz(k + 0.5) / 2 / eo;

          ffx[i] = kx - dt * sigmax(i + 0.5) / 2 / eo;
          ffy[j] = ky - dt * sigmay(j + 0.5) / 2 / eo;
          ffz[k] = kz - dt * sigmaz(k + 0.5) / 2 / eo;

          ggx[i] = (2 * eo * kx - dt * sigmax(i + 0.5)) /
                   (2 * eo * kx + dt * sigmax(i + 0.5));
          ggy[j] = (2 * eo * ky - dt * sigmay(j + 0.5)) /
                   (2 * eo * ky + dt * sigmay(j + 0.5));
          ggz[k] = (2 * eo * kz - dt * sigmaz(k + 0.5)) /
                   (2 * eo * kz + dt * sigmaz(k + 0.5));

          hhx[i] = (2 * eo * dt) / (2 * eo * kx + dt * sigmax(i + 0.5));
          hhy[j] = (2 * eo * dt) / (2 * eo * ky + dt * sigmay(j + 0.5));
          hhz[k] = (2 * eo * dt) / (2 * eo * kz + dt * sigmaz(k + 0.5));

          iix[i] = (2 * eo * kx - dt * sigmax(i + 0.5)) /
                   (2 * eo * kx + dt * sigmax(i + 0.5));
          iiy[j] = (2 * eo * ky - dt * sigmay(j + 0.5)) /
                   (2 * eo * ky + dt * sigmay(j + 0.5));
          iiz[k] = (2 * eo * kz - dt * sigmaz(k + 0.5)) /
                   (2 * eo * kz + dt * sigmaz(k + 0.5));

          jjx[i] = (2 * eo) / (2 * eo * kx + dt * sigmax(i + 0.5)) / uo / ups;
          jjy[j] = (2 * eo) / (2 * eo * ky + dt * sigmay(j + 0.5)) / uo / ups;
          jjz[k] = (2 * eo) / (2 * eo * kz + dt * sigmaz(k + 0.5)) / uo / ups;

          kkx[i] = kx + dt * sigmax(i) / 2 / eo;
          kky[j] = ky + dt * sigmay(j) / 2 / eo;
          kkz[k] = kz + dt * sigmaz(k) / 2 / eo;

          llx[i] = kx - dt * sigmax(i) / 2 / eo;
          lly[j] = ky - dt * sigmay(j) / 2 / eo;
          llz[k] = kz - dt * sigmaz(k) / 2 / eo;
        }
      }
  tus = dt / uo / ups;

  printf("coefficient...ok\n");
}

void coefficient_cpml() {
  int i, j, k;
  float ax, bx;

  for (i = 0; i < misize; i++) {
    aax[i] = cpmlax(i, pmlil, pmlir);
    bbx[i] = cpmlbx(i, pmlil, pmlir);
    ccx[i] = kappa_x(i, pmlil, pmlir);

    ggx[i] = cpmlax(i + 0.5, pmlil, pmlir);
    hhx[i] = cpmlbx(i + 0.5, pmlil, pmlir);
    ffx[i] = kappa_x(i + 0.5, pmlil, pmlir);

    if (i > isize / 2)
      ggx[i] = cpmlax(i - 0.5, pmlil, pmlir);
    if (i > isize / 2)
      hhx[i] = cpmlbx(i - 0.5, pmlil, pmlir);
    if (i > isize / 2)
      ffx[i] = kappa_x(i - 0.5, pmlil, pmlir);
    printf("aax= %d %f %f %f %f %f %f \n", i, aax[i], bbx[i], ggx[i], hhx[i],
           ccx[i], ffx[i]);
  }

  for (j = 0; j < mjsize; j++) {
    aay[j] = cpmlax(j, pmljl, pmljr);
    bby[j] = cpmlbx(j, pmljl, pmljr);
    ccy[j] = kappa_x(j, pmljl, pmljr);

    ggy[j] = cpmlax(j + 0.5, pmljl, pmljr);
    hhy[j] = cpmlbx(j + 0.5, pmljl, pmljr);
    ffy[j] = kappa_x(j + 0.5, pmljl, pmljr);

    if (j > jsize / 2)
      ggy[j] = cpmlax(j - 0.5, pmljl, pmljr);
    if (j > jsize / 2)
      hhy[j] = cpmlbx(j - 0.5, pmljl, pmljr);
    if (j > jsize / 2)
      ffy[j] = kappa_x(j - 0.5, pmljl, pmljr);
    printf("aay= %d %f %f %f %f %f %f \n", j, aay[j], bby[j], ggy[j], hhy[j],
           ccy[j], ffy[j]);
  }

  for (k = 0; k < mksize; k++) {
    aaz[k] = cpmlax(k, pmlkl, pmlkr);
    bbz[k] = cpmlbx(k, pmlkl, pmlkr);
    ccz[k] = kappa_x(k, pmlkl, pmlkr);

    ggz[k] = cpmlax(k + 0.5, pmlkl, pmlkr);
    hhz[k] = cpmlbx(k + 0.5, pmlkl, pmlkr);
    ffz[k] = kappa_x(k + 0.5, pmlkl, pmlkr);

    if (k > ksize / 2)
      ggz[k] = cpmlax(k - 0.5, pmlkl, pmlkr);
    if (k > ksize / 2)
      hhz[k] = cpmlbx(k - 0.5, pmlkl, pmlkr);
    if (k > ksize / 2)
      ffz[k] = kappa_x(k - 0.5, pmlkl, pmlkr);
    printf("aaz= %d %f %f %f %f %f %f \n", k, aaz[k], bbz[k], ggz[k], hhz[k],
           ccz[k], ffz[k]);
  }

  /*
      for(i=0;i<misize;i++)
      for(j=0;j<mjsize;j++)
      for(k=0;k<mksize;k++)
      {
          if(position[i][j][k])
          {}  //Do nothing!
          else
          {
              // coefficient for Ex,Ey,Ez
              aax[i]=cpmlax(i,pmlil,pmlir);
              aay[j]=cpmlax(j,pmljl,pmljr);
              aaz[k]=cpmlax(k,pmlkl,pmlkr);

              bbx[i]=cpmlbx(i,pmlil,pmlir);
              bby[j]=cpmlbx(j,pmljl,pmljr);
              bbz[k]=cpmlbx(k,pmlkl,pmlkr);

              // coefficient (kappa) for Ex, Ey, Ez
              ccx[i]=kappa_x(i,pmlil,pmlir);
              ccy[j]=kappa_x(j,pmlil,pmlir);
              ccz[k]=kappa_x(k,pmlil,pmlir);

              // coefficient for Hx, Hy, Hz
              ggx[i]=cpmlax(i*1.0+0.5,pmlil-1,pmlir-1);
              ggy[j]=cpmlax(j+0.5,pmljl-1,pmljr-1);
              ggz[k]=cpmlax(k+0.5,pmlkl-1,pmlkr-1);

              hhx[i]=cpmlbx(i+0.5,pmlil-1,pmlir-1);
              hhy[j]=cpmlbx(j+0.5,pmljl-1,pmljr-1);
              hhz[k]=cpmlbx(k+0.5,pmlkl-1,pmlkr-1);

              // coefficient (kappa) for Hx, Hy, Hz
              ffx[i]=kappa_x(i,pmlil-1,pmlir-1);
              ffy[j]=kappa_x(j,pmlil-1,pmlir-1);
              ffz[k]=kappa_x(k,pmlil-1,pmlir-1);
          }
      }
  */

  tus = dt / uo / ups;

  printf("coefficient_cpml...ok\n");
}

void propagate() {
  int i, j, k;

  //******* real E-field ******//
  for (i = 1; i < pisize; i++)
    for (j = 1; j < pjsize; j++)
      for (k = 1; k < pksize; k++) {
        if (position[i][j][k] == 1 && mepsilon[i][j][k] == 0.0) // non-metal
          normal_E_field_update(i, j, k);
        else if (position[i][j][k] == 1 &&
                 mepsilon[i][j][k] ==
                     100) // molecular (include the feedbakc of molecular)
          molecular_E_field_update(i, j, k);
        else if (position[i][j][k] == 1 && (mepsilon[i][j][k] != 0.0 &&
                                            mepsilon[i][j][k] != 1000)) // metal
          metal_E_field_update(i, j, k);
        else if (position[i][j][k] == 0 &&
                 (mepsilon[i][j][k] != 0.0 &&
                  mepsilon[i][j][k] != 1000)) // metal & PML
          metal_E_field_update(i, j, k); // PML update not used for this region
        else if (position[i][j][k] == 1 && mepsilon[i][j][k] == 1000) // PCM
          pcm_E_field_update(i, j, k);
        else if (position[i][j][k] == 0 &&
                 mepsilon[i][j][k] == 0.0) // non-metal & PML region
          // cpml_E_field_update(i, j, k);
          pml_E_field_update(i, j, k);
        else {
        }
      }

  //******* imag E-field ******//
  if ((use_periodic_x == 1 || use_periodic_y == 1) &&
      (wave_vector_x != 0.0 || wave_vector_y != 0.0)) {
    for (i = 1; i < pisize; i++)
      for (j = 1; j < pjsize; j++)
        for (k = 1; k < pksize; k++) {
          if (position[i][j][k] == 1 && mepsilon[i][j][k] == 0.0) // non-metal
            normal_iE_field_update(i, j, k);
          else if (position[i][j][k] == 1 &&
                   (mepsilon[i][j][k] != 0.0 &&
                    mepsilon[i][j][k] != 1000)) // metal
            metal_iE_field_update(i, j, k);
          else if (position[i][j][k] == 0 &&
                   (mepsilon[i][j][k] != 0.0 &&
                    mepsilon[i][j][k] != 1000)) // metal & PML
            metal_iE_field_update(i, j,
                                  k); // PML update not used for this region
          else if (position[i][j][k] == 1 && mepsilon[i][j][k] == 1000) // PCM
            pcm_iE_field_update(i, j, k);
          else if (position[i][j][k] == 0 &&
                   mepsilon[i][j][k] == 0.0) // non-metal & PML region
            pml_iE_field_update(i, j, k);
          else {
          }
        }
  }

  //******* E-field periodic condition ******//
  if (use_periodic_x == 1 && (wave_vector_x != 0.0 || wave_vector_y != 0.0))
    E_field_periodic_boundary_update_x();
  if (use_periodic_y == 1 && (wave_vector_x != 0.0 || wave_vector_y != 0.0))
    E_field_periodic_boundary_update_y();
  if (use_periodic_x == 1 && (wave_vector_x == 0.0 && wave_vector_y == 0.0))
    E_field_Gamma_boundary_update_x();
  if (use_periodic_y == 1 && (wave_vector_x == 0.0 && wave_vector_y == 0.0))
    E_field_Gamma_boundary_update_y();

  //******* Ez_parity real-field ******//
  Ez_parity_boundary_update();

  //******* Ez_parity imag-field ******//
  if ((use_periodic_x == 1 || use_periodic_y == 1) &&
      (wave_vector_x != 0.0 || wave_vector_y != 0.0))
    Ez_parity_iboundary_update();

  //******* real H-field ******//
  for (i = 1; i < pisize; i++)
    for (j = 1; j < pjsize; j++)
      for (k = 1; k < pksize; k++) {
        if (position[i][j][k] == 1 && (mepsilon[i][j][k] != 1000))
          normal_H_field_update(i, j, k);
        else if (position[i][j][k] == 0 &&
                 (mepsilon[i][j][k] != 0.0 &&
                  mepsilon[i][j][k] != 1000)) // metal & PML region
          normal_H_field_update(i, j, k);
        else if (position[i][j][k] == 1 && mepsilon[i][j][k] == 1000) // PCM
          pcm_H_field_update(i, j, k);
        else if (position[i][j][k] == 0 &&
                 mepsilon[i][j][k] == 0.0) // non-metal & PML
          // cpml_H_field_update(i, j, k);  // only for non-periodic and
          // none-symmetry
          pml_H_field_update(i, j, k);
        else {
        }
      }

  //******* imag H-field ******//
  if ((use_periodic_x == 1 || use_periodic_y == 1) &&
      (wave_vector_x != 0.0 || wave_vector_y != 0.0)) {
    for (i = 1; i < pisize; i++)
      for (j = 1; j < pjsize; j++)
        for (k = 1; k < pksize; k++) {
          if (position[i][j][k] == 1 && (mepsilon[i][j][k] != 1000))
            normal_iH_field_update(i, j, k);
          else if (position[i][j][k] == 0 &&
                   (mepsilon[i][j][k] != 0.0 &&
                    mepsilon[i][j][k] != 1000)) // metal & PML region
            normal_iH_field_update(i, j, k);
          else if (position[i][j][k] == 1 && mepsilon[i][j][k] == 1000) // PCM
            pcm_iH_field_update(i, j, k);
          else if (position[i][j][k] == 0 &&
                   mepsilon[i][j][k] == 0.0) // non-metal & PML
            pml_iH_field_update(i, j, k);
          else {
          }
        }
  }

  //******* H-field periodic condition ******//
  if (use_periodic_x == 1 && (wave_vector_x != 0.0 || wave_vector_y != 0.0))
    H_field_periodic_boundary_update_x();
  if (use_periodic_y == 1 && (wave_vector_x != 0.0 || wave_vector_y != 0.0))
    H_field_periodic_boundary_update_y();
  if (use_periodic_x == 1 && (wave_vector_x == 0.0 && wave_vector_y == 0.0))
    H_field_Gamma_boundary_update_x();
  if (use_periodic_y == 1 && (wave_vector_x == 0.0 && wave_vector_y == 0.0))
    H_field_Gamma_boundary_update_y();

  //******* Hz_parity real-field ******//
  Hz_parity_boundary_update();

  //******* Hz_parity imag-field ******//
  if ((use_periodic_x == 1 || use_periodic_y == 1) &&
      (wave_vector_x != 0.0 || wave_vector_y != 0.0))
    Hz_parity_iboundary_update();

  printf("Current time: %d\n", t);
}

//////////////////////////////////////////////
/////// for triangular lattice ///////////////
//////////////////////////////////////////////
void propagate_tri() {
  int i, j, k;

  //******* real E-field ******//
  for (i = 1; i < pisize; i++)
    for (j = 1; j < pjsize; j++)
      for (k = 1; k < pksize; k++) {
        if (position[i][j][k] == 1 && mepsilon[i][j][k] == 0.0) // non-metal
          normal_E_field_update(i, j, k);
        else if (position[i][j][k] == 1 &&
                 mepsilon[i][j][k] ==
                     100) // molecular (include the feedbakc of molecular)
          molecular_E_field_update(i, j, k);
        else if (position[i][j][k] == 1 && (mepsilon[i][j][k] != 0.0 &&
                                            mepsilon[i][j][k] != 1000)) // metal
          metal_E_field_update(i, j, k);
        else if (position[i][j][k] == 0 &&
                 (mepsilon[i][j][k] != 0.0 &&
                  mepsilon[i][j][k] != 1000)) // metal & PML
          metal_E_field_update(i, j, k); // PML update not used for this region
        else if (position[i][j][k] == 1 && mepsilon[i][j][k] == 1000) // PCM
          pcm_E_field_update(i, j, k);
        else if (position[i][j][k] == 0 &&
                 mepsilon[i][j][k] == 0.0) // non-metal & PML region
          // cpml_E_field_update(i, j, k);
          pml_E_field_update(i, j, k);
        else {
        }
      }

  //******* imag E-field ******//
  if ((use_periodic_x == 1 || use_periodic_y == 1) &&
      (wave_vector_x != 0.0 || wave_vector_y != 0.0)) {
    for (i = 1; i < pisize; i++)
      for (j = 1; j < pjsize; j++)
        for (k = 1; k < pksize; k++) {
          if (position[i][j][k] == 1 && mepsilon[i][j][k] == 0.0) // non-metal
            normal_iE_field_update(i, j, k);
          else if (position[i][j][k] == 1 &&
                   (mepsilon[i][j][k] != 0.0 &&
                    mepsilon[i][j][k] != 1000)) // metal
            metal_iE_field_update(i, j, k);
          else if (position[i][j][k] == 0 &&
                   (mepsilon[i][j][k] != 0.0 &&
                    mepsilon[i][j][k] != 1000)) // metal & PML
            metal_iE_field_update(i, j,
                                  k); // PML update not used for this region
          else if (position[i][j][k] == 1 && mepsilon[i][j][k] == 1000) // PCM
            pcm_iE_field_update(i, j, k);
          else if (position[i][j][k] == 0 &&
                   mepsilon[i][j][k] == 0.0) // non-metal & PML region
            pml_iE_field_update(i, j, k);
          else {
          }
        }
  }

  //******* E-field periodic condition ******//
  if (use_periodic_x == 1 && (wave_vector_x != 0.0 || wave_vector_y != 0.0))
    E_field_periodic_boundary_update_Xwall();
  if (use_periodic_y == 1 && (wave_vector_x != 0.0 || wave_vector_y != 0.0))
    E_field_periodic_boundary_update_Ywall();
  if (use_periodic_x == 1 && (wave_vector_x == 0.0 && wave_vector_y == 0.0))
    E_field_Gamma_boundary_update_Xwall();
  if (use_periodic_y == 1 && (wave_vector_x == 0.0 && wave_vector_y == 0.0))
    E_field_Gamma_boundary_update_Ywall();

  //******* Ez_parity real-field ******//
  Ez_parity_boundary_update();

  //******* Ez_parity imag-field ******//
  if ((use_periodic_x == 1 || use_periodic_y == 1) &&
      (wave_vector_x != 0.0 || wave_vector_y != 0.0))
    Ez_parity_iboundary_update();

  //******* real H-field ******//
  for (i = 1; i < pisize; i++)
    for (j = 1; j < pjsize; j++)
      for (k = 1; k < pksize; k++) {
        if (position[i][j][k] == 1 && (mepsilon[i][j][k] != 1000))
          normal_H_field_update(i, j, k);
        else if (position[i][j][k] == 0 &&
                 (mepsilon[i][j][k] != 0.0 &&
                  mepsilon[i][j][k] != 1000)) // metal & PML region
          normal_H_field_update(i, j, k);
        else if (position[i][j][k] == 1 && mepsilon[i][j][k] == 1000) // PCM
          pcm_H_field_update(i, j, k);
        else if (position[i][j][k] == 0 &&
                 mepsilon[i][j][k] == 0.0) // non-metal & PML
          // cpml_H_field_update(i, j, k);
          pml_H_field_update(i, j, k);
        else {
        }
      }

  //******* imag H-field ******//
  if ((use_periodic_x == 1 || use_periodic_y == 1) &&
      (wave_vector_x != 0.0 || wave_vector_y != 0.0)) {
    for (i = 1; i < pisize; i++)
      for (j = 1; j < pjsize; j++)
        for (k = 1; k < pksize; k++) {
          if (position[i][j][k] == 1 && (mepsilon[i][j][k] != 1000))
            normal_iH_field_update(i, j, k);
          else if (position[i][j][k] == 0 &&
                   (mepsilon[i][j][k] != 0.0 &&
                    mepsilon[i][j][k] != 1000)) // metal & PML region
            normal_iH_field_update(i, j, k);
          else if (position[i][j][k] == 1 && mepsilon[i][j][k] == 1000) // PCM
            pcm_iH_field_update(i, j, k);
          else if (position[i][j][k] == 0 &&
                   mepsilon[i][j][k] == 0.0) // non-metal & PML
            pml_iH_field_update(i, j, k);
          else {
          }
        }
  }

  //******* H-field periodic condition ******//
  if (use_periodic_x == 1 && (wave_vector_x != 0.0 || wave_vector_y != 0.0))
    H_field_periodic_boundary_update_Xwall();
  if (use_periodic_y == 1 && (wave_vector_x != 0.0 || wave_vector_y != 0.0))
    H_field_periodic_boundary_update_Ywall();
  if (use_periodic_x == 1 && (wave_vector_x == 0.0 && wave_vector_y == 0.0))
    H_field_Gamma_boundary_update_Xwall();
  if (use_periodic_y == 1 && (wave_vector_x == 0.0 && wave_vector_y == 0.0))
    H_field_Gamma_boundary_update_Ywall();

  //******* Hz_parity real-field ******//
  Hz_parity_boundary_update();

  //******* Hz_parity imag-field ******//
  if ((use_periodic_x == 1 || use_periodic_y == 1) &&
      (wave_vector_x != 0.0 || wave_vector_y != 0.0))
    Hz_parity_iboundary_update();

  printf("%d\n", t);
}

void normal_E_field_update(int i, int j, int k) {
  Ex[i][j][k] = Ex[i][j][k] + (dt / eo / epsilonx[i][j][k]) *
                                  ((Hz[i][j][k] - Hz[i][j - 1][k]) / ds_y -
                                   (Hy[i][j][k] - Hy[i][j][k - 1]) *
                                       (2 / (ds_nz[k - 1] + ds_nz[k])));
  Ey[i][j][k] = Ey[i][j][k] + (dt / eo / epsilony[i][j][k]) *
                                  ((Hx[i][j][k] - Hx[i][j][k - 1]) *
                                       (2 / (ds_nz[k - 1] + ds_nz[k])) -
                                   (Hz[i][j][k] - Hz[i - 1][j][k]) / ds_x);
  Ez[i][j][k] = Ez[i][j][k] + (dt / eo / epsilonz[i][j][k]) *
                                  ((Hy[i][j][k] - Hy[i - 1][j][k]) / ds_x -
                                   (Hx[i][j][k] - Hx[i][j - 1][k]) / ds_y);
}

void molecular_E_field_update(int i, int j, int k) {
  // not finlished
  float Px, Py, Pz;
  Px = (dPx[i][j][k] + dPx_old[i][j][k]) / 2.0;
  Py = (dPy[i][j][k] + dPy_old[i][j][k]) / 2.0;
  Pz = (dPz[i][j][k] + dPz_old[i][j][k]) / 2.0;

  Ex[i][j][k] = Ex[i][j][k] + (dt / eo / epsilonx[i][j][k]) *
                                  ((Hz[i][j][k] - Hz[i][j - 1][k]) / ds_y -
                                   (Hy[i][j][k] - Hy[i][j][k - 1]) *
                                       (2 / (ds_nz[k - 1] + ds_nz[k])) -
                                   Px);
  Ey[i][j][k] = Ey[i][j][k] + (dt / eo / epsilony[i][j][k]) *
                                  ((Hx[i][j][k] - Hx[i][j][k - 1]) *
                                       (2 / (ds_nz[k - 1] + ds_nz[k])) -
                                   (Hz[i][j][k] - Hz[i - 1][j][k]) / ds_x - Py);
  Ez[i][j][k] = Ez[i][j][k] + (dt / eo / epsilonz[i][j][k]) *
                                  ((Hy[i][j][k] - Hy[i - 1][j][k]) / ds_x -
                                   (Hx[i][j][k] - Hx[i][j - 1][k]) / ds_y - Pz);
}

void metal_E_field_update(int i, int j, int k) {
  float km, bm;
  float Ex_temp, Ey_temp, Ez_temp;

  km = (2 - mgamma[i][j][k] * dt) / (2 + mgamma[i][j][k] * dt);
  bm = momega[i][j][k] * momega[i][j][k] * eo * dt / (2 + mgamma[i][j][k] * dt);

  Ex_temp = Ex[i][j][k]; // store E-field at FDTD time 'n'
  Ey_temp = Ey[i][j][k]; //          "
  Ez_temp = Ez[i][j][k]; //          "

  Ex[i][j][k] =
      ((2 * eo * mepsilon[i][j][k] - dt * bm) /
       (2 * eo * mepsilon[i][j][k] + dt * bm)) *
          Ex[i][j][k] +
      (2 * dt / (2 * eo * mepsilon[i][j][k] + dt * bm)) *
          ((Hz[i][j][k] - Hz[i][j - 1][k]) / ds_y -
           (Hy[i][j][k] - Hy[i][j][k - 1]) * (2 / (ds_nz[k - 1] + ds_nz[k])) -
           0.5 * (1 + km) * Jx[i][j][k]);
  Ey[i][j][k] =
      ((2 * eo * mepsilon[i][j][k] - dt * bm) /
       (2 * eo * mepsilon[i][j][k] + dt * bm)) *
          Ey[i][j][k] +
      (2 * dt / (2 * eo * mepsilon[i][j][k] + dt * bm)) *
          ((Hx[i][j][k] - Hx[i][j][k - 1]) * (2 / (ds_nz[k - 1] + ds_nz[k])) -
           (Hz[i][j][k] - Hz[i - 1][j][k]) / ds_x -
           0.5 * (1 + km) * Jy[i][j][k]);
  Ez[i][j][k] = ((2 * eo * mepsilon[i][j][k] - dt * bm) /
                 (2 * eo * mepsilon[i][j][k] + dt * bm)) *
                    Ez[i][j][k] +
                (2 * dt / (2 * eo * mepsilon[i][j][k] + dt * bm)) *
                    ((Hy[i][j][k] - Hy[i - 1][j][k]) / ds_x -
                     (Hx[i][j][k] - Hx[i][j - 1][k]) / ds_y -
                     0.5 * (1 + km) * Jz[i][j][k]);

  Jx[i][j][k] = km * Jx[i][j][k] + bm * (Ex[i][j][k] + Ex_temp);
  Jy[i][j][k] = km * Jy[i][j][k] + bm * (Ey[i][j][k] + Ey_temp);
  Jz[i][j][k] = km * Jz[i][j][k] + bm * (Ez[i][j][k] + Ez_temp);
}

void cpml_E_field_update(int i, int j, int k) {

  Ex[i][j][k] =
      Ex[i][j][k] + (dt / eo / epsilonx[i][j][k]) *
                        ((Hz[i][j][k] - Hz[i][j - 1][k]) / ds_y / ccy[j] -
                         (Hy[i][j][k] - Hy[i][j][k - 1]) * 2.0 /
                             (ds_nz[k - 1] + ds_nz[k]) / ccz[k]);
  Ey[i][j][k] =
      Ey[i][j][k] + (dt / eo / epsilony[i][j][k]) *
                        ((Hx[i][j][k] - Hx[i][j][k - 1]) * 2.0 /
                             (ds_nz[k - 1] + ds_nz[k]) / ccz[k] -
                         (Hz[i][j][k] - Hz[i - 1][j][k]) / ds_x / ccx[i]);
  Ez[i][j][k] =
      Ez[i][j][k] + (dt / eo / epsilonz[i][j][k]) *
                        ((Hy[i][j][k] - Hy[i - 1][j][k]) / ds_x / ccx[i] -
                         (Hx[i][j][k] - Hx[i][j - 1][k]) / ds_y / ccy[j]);

  // printf("testcpml_ %d %d %d %f %f %f \n", i, j, k, aax[i], aay[j], aaz[k]);

  psi_Exy[i][j][k] = bby[j] * psi_Exy[i][j][k] +
                     aay[j] * (Hz[i][j][k] - Hz[i][j - 1][k]) / ds_y;
  psi_Exz[i][j][k] =
      bbz[k] * psi_Exz[i][j][k] + aaz[k] * (Hy[i][j][k - 1] - Hy[i][j][k]) *
                                      2.0 / (ds_nz[k - 1] + ds_nz[k]);
  Ex[i][j][k] = Ex[i][j][k] + dt / eo * (psi_Exy[i][j][k] + psi_Exz[i][j][k]);

  psi_Eyz[i][j][k] =
      bbz[k] * psi_Eyz[i][j][k] + aaz[k] * (Hx[i][j][k] - Hx[i][j][k - 1]) *
                                      2.0 / (ds_nz[k - 1] + ds_nz[k]);
  psi_Eyx[i][j][k] = bbx[i] * psi_Eyx[i][j][k] +
                     aax[i] * (Hz[i - 1][j][k] - Hz[i][j][k]) / ds_x;
  Ey[i][j][k] = Ey[i][j][k] + dt / eo * (psi_Eyz[i][j][k] + psi_Eyx[i][j][k]);

  psi_Ezx[i][j][k] = bbx[i] * psi_Ezx[i][j][k] +
                     aax[i] * (Hy[i][j][k] - Hy[i - 1][j][k]) / ds_x;
  psi_Ezy[i][j][k] = bby[j] * psi_Ezy[i][j][k] +
                     aay[j] * (Hx[i][j - 1][k] - Hx[i][j][k]) / ds_y;
  Ez[i][j][k] = Ez[i][j][k] + dt / eo * (psi_Ezx[i][j][k] + psi_Ezy[i][j][k]);
}

void pml_E_field_update(int i, int j, int k) {
  float temp;

  temp = Dx[i][j][k];
  Dx[i][j][k] =
      aay[j] * Dx[i][j][k] + bby[j] * ((Hz[i][j][k] - Hz[i][j - 1][k]) / ds_y -
                                       (Hy[i][j][k] - Hy[i][j][k - 1]) *
                                           (2 / (ds_nz[k - 1] + ds_nz[k])));
  Ex[i][j][k] = ccz[k] * Ex[i][j][k] +
                ddx[i][j][k] * (eex[i] * Dx[i][j][k] - ffx[i] * temp);
  temp = Dy[i][j][k];
  Dy[i][j][k] =
      aaz[k] * Dy[i][j][k] + bbz[k] * ((Hx[i][j][k] - Hx[i][j][k - 1]) *
                                           (2 / (ds_nz[k - 1] + ds_nz[k])) -
                                       (Hz[i][j][k] - Hz[i - 1][j][k]) / ds_x);
  Ey[i][j][k] = ccx[i] * Ey[i][j][k] +
                ddy[i][j][k] * (eey[j] * Dy[i][j][k] - ffy[j] * temp);
  temp = Dz[i][j][k];
  Dz[i][j][k] =
      aax[i] * Dz[i][j][k] + bbx[i] * ((Hy[i][j][k] - Hy[i - 1][j][k]) / ds_x -
                                       (Hx[i][j][k] - Hx[i][j - 1][k]) / ds_y);
  Ez[i][j][k] = ccy[j] * Ez[i][j][k] +
                ddz[i][j][k] * (eez[k] * Dz[i][j][k] - ffz[k] * temp);
}

void pcm_E_field_update(int i, int j, int k) {
  Ex[i][j][k] = 0.0;
  Ey[i][j][k] = 0.0;
  Ez[i][j][k] = 0.0;
}

void normal_H_field_update(int i, int j, int k) {
  Hx[i][j][k] =
      Hx[i][j][k] - tus * ((Ez[i][j + 1][k] - Ez[i][j][k]) / ds_y -
                           (Ey[i][j][k + 1] - Ey[i][j][k]) / ds_nz[k]);
  Hy[i][j][k] =
      Hy[i][j][k] - tus * ((Ex[i][j][k + 1] - Ex[i][j][k]) / ds_nz[k] -
                           (Ez[i + 1][j][k] - Ez[i][j][k]) / ds_x);
  Hz[i][j][k] = Hz[i][j][k] - tus * ((Ey[i + 1][j][k] - Ey[i][j][k]) / ds_x -
                                     (Ex[i][j + 1][k] - Ex[i][j][k]) / ds_y);
}

void cpml_H_field_update(int i, int j, int k) {

  Hx[i][j][k] =
      Hx[i][j][k] - tus * ((Ez[i][j + 1][k] - Ez[i][j][k]) / ds_y / ffy[j] -
                           (Ey[i][j][k + 1] - Ey[i][j][k]) / ds_nz[k] / ffz[k]);
  Hy[i][j][k] =
      Hy[i][j][k] - tus * ((Ex[i][j][k + 1] - Ex[i][j][k]) / ds_nz[k] / ffz[k] -
                           (Ez[i + 1][j][k] - Ez[i][j][k]) / ds_x / ffx[i]);
  Hz[i][j][k] =
      Hz[i][j][k] - tus * ((Ey[i + 1][j][k] - Ey[i][j][k]) / ds_x / ffx[i] -
                           (Ex[i][j + 1][k] - Ex[i][j][k]) / ds_y / ffy[j]);

  psi_Hxy[i][j][k] = hhy[j] * psi_Hxy[i][j][k] +
                     ggy[j] * (Ez[i][j][k] - Ez[i][j + 1][k]) / ds_y;
  psi_Hxz[i][j][k] = hhz[k] * psi_Hxz[i][j][k] +
                     ggz[k] * (Ey[i][j][k + 1] - Ey[i][j][k]) / ds_nz[k];
  Hx[i][j][k] = Hx[i][j][k] + tus * (psi_Hxy[i][j][k] + psi_Hxz[i][j][k]);

  psi_Hyz[i][j][k] = hhz[k] * psi_Hyz[i][j][k] +
                     ggz[k] * (Ex[i][j][k] - Ex[i][j][k + 1]) / ds_nz[k];
  psi_Hyx[i][j][k] = hhx[i] * psi_Hyx[i][j][k] +
                     ggx[i] * (Ez[i + 1][j][k] - Ez[i][j][k]) / ds_x;
  Hy[i][j][k] = Hy[i][j][k] + tus * (psi_Hyx[i][j][k] + psi_Hyz[i][j][k]);

  psi_Hzx[i][j][k] = hhx[i] * psi_Hzx[i][j][k] +
                     ggx[i] * (Ey[i][j][k] - Ey[i + 1][j][k]) / ds_x;
  psi_Hzy[i][j][k] = hhy[j] * psi_Hzy[i][j][k] +
                     ggy[j] * (Ex[i][j + 1][k] - Ex[i][j][k]) / ds_y;
  Hz[i][j][k] = Hz[i][j][k] + tus * (psi_Hzx[i][j][k] + psi_Hzy[i][j][k]);
}

void pml_H_field_update(int i, int j, int k) {
  float temp;

  temp = Bx[i][j][k];
  Bx[i][j][k] = ggy[j] * Bx[i][j][k] -
                hhy[j] * ((Ez[i][j + 1][k] - Ez[i][j][k]) / ds_y -
                          (Ey[i][j][k + 1] - Ey[i][j][k]) / ds_nz[k]);
  Hx[i][j][k] =
      iiz[k] * Hx[i][j][k] + jjz[k] * (kkx[i] * Bx[i][j][k] - llx[i] * temp);
  temp = By[i][j][k];
  By[i][j][k] = ggz[k] * By[i][j][k] -
                hhz[k] * ((Ex[i][j][k + 1] - Ex[i][j][k]) / ds_nz[k] -
                          (Ez[i + 1][j][k] - Ez[i][j][k]) / ds_x);
  Hy[i][j][k] =
      iix[i] * Hy[i][j][k] + jjx[i] * (kky[j] * By[i][j][k] - lly[j] * temp);
  temp = Bz[i][j][k];
  Bz[i][j][k] =
      ggx[i] * Bz[i][j][k] - hhx[i] * ((Ey[i + 1][j][k] - Ey[i][j][k]) / ds_x -
                                       (Ex[i][j + 1][k] - Ex[i][j][k]) / ds_y);
  Hz[i][j][k] =
      iiy[j] * Hz[i][j][k] + jjy[j] * (kkz[k] * Bz[i][j][k] - llz[k] * temp);
}

void pcm_H_field_update(int i, int j, int k) {
  Hx[i][j][k] = 0.0;
  Hy[i][j][k] = 0.0;
  Hz[i][j][k] = 0.0;
}

void normal_iE_field_update(int i, int j, int k) {
  iEx[i][j][k] = iEx[i][j][k] + (dt / eo / epsilonx[i][j][k]) *
                                    ((iHz[i][j][k] - iHz[i][j - 1][k]) / ds_y -
                                     (iHy[i][j][k] - iHy[i][j][k - 1]) *
                                         (2 / (ds_nz[k - 1] + ds_nz[k])));
  iEy[i][j][k] = iEy[i][j][k] + (dt / eo / epsilony[i][j][k]) *
                                    ((iHx[i][j][k] - iHx[i][j][k - 1]) *
                                         (2 / (ds_nz[k - 1] + ds_nz[k])) -
                                     (iHz[i][j][k] - iHz[i - 1][j][k]) / ds_x);
  iEz[i][j][k] = iEz[i][j][k] + (dt / eo / epsilonz[i][j][k]) *
                                    ((iHy[i][j][k] - iHy[i - 1][j][k]) / ds_x -
                                     (iHx[i][j][k] - iHx[i][j - 1][k]) / ds_y);
}

void metal_iE_field_update(int i, int j, int k) {
  float km, bm;
  float iEx_temp, iEy_temp, iEz_temp;

  km = (2 - mgamma[i][j][k] * dt) / (2 + mgamma[i][j][k] * dt);
  bm = momega[i][j][k] * momega[i][j][k] * eo * dt / (2 + mgamma[i][j][k] * dt);

  iEx_temp = iEx[i][j][k]; // store E-field at FDTD time 'n'
  iEy_temp = iEy[i][j][k]; //          "
  iEz_temp = iEz[i][j][k]; //          "

  iEx[i][j][k] =
      ((2 * eo * mepsilon[i][j][k] - dt * bm) /
       (2 * eo * mepsilon[i][j][k] + dt * bm)) *
          iEx[i][j][k] +
      (2 * dt / (2 * eo * mepsilon[i][j][k] + dt * bm)) *
          ((iHz[i][j][k] - iHz[i][j - 1][k]) / ds_y -
           (iHy[i][j][k] - iHy[i][j][k - 1]) * (2 / (ds_nz[k - 1] + ds_nz[k])) -
           0.5 * (1 + km) * iJx[i][j][k]);
  iEy[i][j][k] =
      ((2 * eo * mepsilon[i][j][k] - dt * bm) /
       (2 * eo * mepsilon[i][j][k] + dt * bm)) *
          iEy[i][j][k] +
      (2 * dt / (2 * eo * mepsilon[i][j][k] + dt * bm)) *
          ((iHx[i][j][k] - iHx[i][j][k - 1]) * (2 / (ds_nz[k - 1] + ds_nz[k])) -
           (iHz[i][j][k] - iHz[i - 1][j][k]) / ds_x -
           0.5 * (1 + km) * iJy[i][j][k]);
  iEz[i][j][k] = ((2 * eo * mepsilon[i][j][k] - dt * bm) /
                  (2 * eo * mepsilon[i][j][k] + dt * bm)) *
                     iEz[i][j][k] +
                 (2 * dt / (2 * eo * mepsilon[i][j][k] + dt * bm)) *
                     ((iHy[i][j][k] - iHy[i - 1][j][k]) / ds_x -
                      (iHx[i][j][k] - iHx[i][j - 1][k]) / ds_y -
                      0.5 * (1 + km) * iJz[i][j][k]);

  iJx[i][j][k] = km * iJx[i][j][k] + bm * (iEx[i][j][k] + iEx_temp);
  iJy[i][j][k] = km * iJy[i][j][k] + bm * (iEy[i][j][k] + iEy_temp);
  iJz[i][j][k] = km * iJz[i][j][k] + bm * (iEz[i][j][k] + iEz_temp);
}

void pml_iE_field_update(int i, int j, int k) {
  float temp;

  temp = iDx[i][j][k];
  iDx[i][j][k] = aay[j] * iDx[i][j][k] +
                 bby[j] * ((iHz[i][j][k] - iHz[i][j - 1][k]) / ds_y -
                           (iHy[i][j][k] - iHy[i][j][k - 1]) *
                               (2 / (ds_nz[k - 1] + ds_nz[k])));
  iEx[i][j][k] = ccz[k] * iEx[i][j][k] +
                 ddx[i][j][k] * (eex[i] * iDx[i][j][k] - ffx[i] * temp);
  temp = iDy[i][j][k];
  iDy[i][j][k] = aaz[k] * iDy[i][j][k] +
                 bbz[k] * ((iHx[i][j][k] - iHx[i][j][k - 1]) *
                               (2 / (ds_nz[k - 1] + ds_nz[k])) -
                           (iHz[i][j][k] - iHz[i - 1][j][k]) / ds_x);
  iEy[i][j][k] = ccx[i] * iEy[i][j][k] +
                 ddy[i][j][k] * (eey[j] * iDy[i][j][k] - ffy[j] * temp);
  temp = iDz[i][j][k];
  iDz[i][j][k] = aax[i] * iDz[i][j][k] +
                 bbx[i] * ((iHy[i][j][k] - iHy[i - 1][j][k]) / ds_x -
                           (iHx[i][j][k] - iHx[i][j - 1][k]) / ds_y);
  iEz[i][j][k] = ccy[j] * iEz[i][j][k] +
                 ddz[i][j][k] * (eez[k] * iDz[i][j][k] - ffz[k] * temp);
}

void pcm_iE_field_update(int i, int j, int k) {
  iEx[i][j][k] = 0.0;
  iEy[i][j][k] = 0.0;
  iEz[i][j][k] = 0.0;
}

void normal_iH_field_update(int i, int j, int k) {
  iHx[i][j][k] =
      iHx[i][j][k] - tus * ((iEz[i][j + 1][k] - iEz[i][j][k]) / ds_y -
                            (iEy[i][j][k + 1] - iEy[i][j][k]) / ds_nz[k]);
  iHy[i][j][k] =
      iHy[i][j][k] - tus * ((iEx[i][j][k + 1] - iEx[i][j][k]) / ds_nz[k] -
                            (iEz[i + 1][j][k] - iEz[i][j][k]) / ds_x);
  iHz[i][j][k] =
      iHz[i][j][k] - tus * ((iEy[i + 1][j][k] - iEy[i][j][k]) / ds_x -
                            (iEx[i][j + 1][k] - iEx[i][j][k]) / ds_y);
}

void pml_iH_field_update(int i, int j, int k) {
  float temp;

  temp = iBx[i][j][k];
  iBx[i][j][k] = ggy[j] * iBx[i][j][k] -
                 hhy[j] * ((iEz[i][j + 1][k] - iEz[i][j][k]) / ds_y -
                           (iEy[i][j][k + 1] - iEy[i][j][k]) / ds_nz[k]);
  iHx[i][j][k] =
      iiz[k] * iHx[i][j][k] + jjz[k] * (kkx[i] * iBx[i][j][k] - llx[i] * temp);
  temp = iBy[i][j][k];
  iBy[i][j][k] = ggz[k] * iBy[i][j][k] -
                 hhz[k] * ((iEx[i][j][k + 1] - iEx[i][j][k]) / ds_nz[k] -
                           (iEz[i + 1][j][k] - iEz[i][j][k]) / ds_x);
  iHy[i][j][k] =
      iix[i] * iHy[i][j][k] + jjx[i] * (kky[j] * iBy[i][j][k] - lly[j] * temp);
  temp = iBz[i][j][k];
  iBz[i][j][k] = ggx[i] * iBz[i][j][k] -
                 hhx[i] * ((iEy[i + 1][j][k] - iEy[i][j][k]) / ds_x -
                           (iEx[i][j + 1][k] - iEx[i][j][k]) / ds_y);
  iHz[i][j][k] =
      iiy[j] * iHz[i][j][k] + jjy[j] * (kkz[k] * iBz[i][j][k] - llz[k] * temp);
}

void pcm_iH_field_update(int i, int j, int k) {
  iHx[i][j][k] = 0.0;
  iHy[i][j][k] = 0.0;
  iHz[i][j][k] = 0.0;
}

void Ez_parity_boundary_update() {
  int i, j, k;

  if (xparity == 1) // xparity is for Hz_parity
  {
    for (j = 1; j < pjsize; j++)
      for (k = 1; k < pksize; k++) {
        Ex[cisize][j][k] = Ex[cisize - 1][j][k];
        Ey[cisize][j][k] = 0.0;
        Ez[cisize][j][k] = 0.0;
      }
  }

  if (xparity == -1) {
    for (j = 1; j < pjsize; j++)
      for (k = 1; k < pksize; k++)
        Ex[cisize][j][k] = -Ex[cisize - 1][j][k];
  }

  if (yparity == 1) {
    for (k = 1; k < pksize; k++)
      for (i = 1; i < pisize; i++) {
        Ey[i][cjsize][k] = Ey[i][cjsize - 1][k];
        Ex[i][cjsize][k] = 0.0;
        Ez[i][cjsize][k] = 0.0;
      }
  }

  if (yparity == -1) {
    for (k = 1; k < pksize; k++)
      for (i = 1; i < pisize; i++)
        Ey[i][cjsize][k] = -Ey[i][cjsize - 1][k];
  }

  if (zparity == 1) {
    for (i = 1; i < pisize; i++)
      for (j = 1; j < pjsize; j++) {
        Ez[i][j][cksize] = Ez[i][j][cksize - 1];
        Ex[i][j][cksize] = 0.0;
        Ey[i][j][cksize] = 0.0;
      }
  }

  if (zparity == -1) {
    for (i = 1; i < pisize; i++)
      for (j = 1; j < pjsize; j++)
        Ez[i][j][cksize] = -Ez[i][j][cksize - 1];
  }
}

void Ez_parity_iboundary_update() {
  int i, j, k;

  if (xparity == 1) // xparity is for Hz_parity
  {
    for (j = 1; j < pjsize; j++)
      for (k = 1; k < pksize; k++) {
        iEx[cisize][j][k] = iEx[cisize - 1][j][k];
        iEy[cisize][j][k] = 0.0;
        iEz[cisize][j][k] = 0.0;
      }
  }

  if (xparity == -1) {
    for (j = 1; j < pjsize; j++)
      for (k = 1; k < pksize; k++)
        iEx[cisize][j][k] = -iEx[cisize - 1][j][k];
  }

  if (yparity == 1) {
    for (k = 1; k < pksize; k++)
      for (i = 1; i < pisize; i++) {
        iEy[i][cjsize][k] = iEy[i][cjsize - 1][k];
        iEx[i][cjsize][k] = 0.0;
        iEz[i][cjsize][k] = 0.0;
      }
  }

  if (yparity == -1) {
    for (k = 1; k < pksize; k++)
      for (i = 1; i < pisize; i++)
        iEy[i][cjsize][k] = -iEy[i][cjsize - 1][k];
  }

  if (zparity == 1) {
    for (i = 1; i < pisize; i++)
      for (j = 1; j < pjsize; j++) {
        iEz[i][j][cksize] = iEz[i][j][cksize - 1];
        iEx[i][j][cksize] = 0.0;
        iEy[i][j][cksize] = 0.0;
      }
  }

  if (zparity == -1) {
    for (i = 1; i < pisize; i++)
      for (j = 1; j < pjsize; j++)
        iEz[i][j][cksize] = -iEz[i][j][cksize - 1];
  }
}

void Hz_parity_boundary_update() {
  int i, j, k;

  if (xparity == 1) {
    for (j = 1; j < pjsize; j++)
      for (k = 1; k < pksize; k++) {
        Hy[cisize][j][k] = Hy[cisize - 1][j][k];
        Hz[cisize][j][k] = Hz[cisize - 1][j][k];
        Hx[cisize][j][k] = 0.0;
      }
  }
  if (xparity == -1) {
    for (j = 1; j < pjsize; j++)
      for (k = 1; k < pksize; k++) {
        Hy[cisize][j][k] = -Hy[cisize - 1][j][k];
        Hz[cisize][j][k] = -Hz[cisize - 1][j][k];
      }
  }

  if (yparity == 1) {
    for (k = 1; k < pksize; k++)
      for (i = 1; i < pisize; i++) {
        Hz[i][cjsize][k] = Hz[i][cjsize - 1][k];
        Hx[i][cjsize][k] = Hx[i][cjsize - 1][k];
        Hy[i][cjsize][k] = 0.0;
      }
  }
  if (yparity == -1) {
    for (k = 1; k < pksize; k++)
      for (i = 1; i < pisize; i++) {
        Hz[i][cjsize][k] = -Hz[i][cjsize - 1][k];
        Hx[i][cjsize][k] = -Hx[i][cjsize - 1][k];
      }
  }

  if (zparity == 1) {
    for (i = 1; i < pisize; i++)
      for (j = 1; j < pjsize; j++) {
        Hx[i][j][cksize] = Hx[i][j][cksize - 1];
        Hy[i][j][cksize] = Hy[i][j][cksize - 1];
        Hz[i][j][cksize] = 0.0;
      }
  }
  if (zparity == -1) {
    for (i = 1; i < pisize; i++)
      for (j = 1; j < pjsize; j++) {
        Hx[i][j][cksize] = -Hx[i][j][cksize - 1];
        Hy[i][j][cksize] = -Hy[i][j][cksize - 1];
      }
  }
}

void Hz_parity_iboundary_update() {
  int i, j, k;

  if (xparity == 1) {
    for (j = 1; j < pjsize; j++)
      for (k = 1; k < pksize; k++) {
        iHy[cisize][j][k] = iHy[cisize - 1][j][k];
        iHz[cisize][j][k] = iHz[cisize - 1][j][k];
        iHx[cisize][j][k] = 0.0;
      }
  }
  if (xparity == -1) {
    for (j = 1; j < pjsize; j++)
      for (k = 1; k < pksize; k++) {
        iHy[cisize][j][k] = -iHy[cisize - 1][j][k];
        iHz[cisize][j][k] = -iHz[cisize - 1][j][k];
      }
  }

  if (yparity == 1) {
    for (k = 1; k < pksize; k++)
      for (i = 1; i < pisize; i++) {
        iHz[i][cjsize][k] = iHz[i][cjsize - 1][k];
        iHx[i][cjsize][k] = iHx[i][cjsize - 1][k];
        iHy[i][cjsize][k] = 0.0;
      }
  }
  if (yparity == -1) {
    for (k = 1; k < pksize; k++)
      for (i = 1; i < pisize; i++) {
        iHz[i][cjsize][k] = -iHz[i][cjsize - 1][k];
        iHx[i][cjsize][k] = -iHx[i][cjsize - 1][k];
      }
  }

  if (zparity == 1) {
    for (i = 1; i < pisize; i++)
      for (j = 1; j < pjsize; j++) {
        iHx[i][j][cksize] = iHx[i][j][cksize - 1];
        iHy[i][j][cksize] = iHy[i][j][cksize - 1];
        iHz[i][j][cksize] = 0.0;
      }
  }
  if (zparity == -1) {
    for (i = 1; i < pisize; i++)
      for (j = 1; j < pjsize; j++) {
        iHx[i][j][cksize] = -iHx[i][j][cksize - 1];
        iHy[i][j][cksize] = -iHy[i][j][cksize - 1];
      }
  }
}

void E_field_periodic_boundary_update_x() {
  int j, k;
  float phase_ang;
  float cos_ph, sin_ph;

  ///// for speed up
  phase_ang = wave_vector_x * 2 * pi;
  cos_ph = cos(phase_ang);
  sin_ph = sin(phase_ang);

  for (j = 1; j < pjsize; j++)
    for (k = 1; k < pksize; k++) {
      Ex[0][j][k] = Ex[isize][j][k] * cos_ph + iEx[isize][j][k] * sin_ph;
      Ey[isize + 1][j][k] = Ey[1][j][k] * cos_ph - iEy[1][j][k] * sin_ph;
      Ez[isize + 1][j][k] = Ez[1][j][k] * cos_ph - iEz[1][j][k] * sin_ph;

      iEx[0][j][k] = -Ex[isize][j][k] * sin_ph + iEx[isize][j][k] * cos_ph;
      iEy[isize + 1][j][k] = Ey[1][j][k] * sin_ph + iEy[1][j][k] * cos_ph;
      iEz[isize + 1][j][k] = Ez[1][j][k] * sin_ph + iEz[1][j][k] * cos_ph;
    }
}

void E_field_periodic_boundary_update_Xwall() ////// for triangular lattice
{
  int j, k;
  float phase_ang;
  float cos_ph, sin_ph;

  ///// for speed up
  phase_ang = wave_vector_x * 2 * pi;
  cos_ph = cos(phase_ang);
  sin_ph = sin(phase_ang);

  for (j = 1; j < pjsize; j++)
    for (k = 1; k < pksize; k++) {
      Ex[0][j][k] = Ex[isize][j][k] * cos_ph + iEx[isize][j][k] * sin_ph;
      Ey[isize + 1][j][k] = Ey[1][j][k] * cos_ph - iEy[1][j][k] * sin_ph;
      Ez[isize + 1][j][k] = Ez[1][j][k] * cos_ph - iEz[1][j][k] * sin_ph;

      iEx[0][j][k] = -Ex[isize][j][k] * sin_ph + iEx[isize][j][k] * cos_ph;
      iEy[isize + 1][j][k] = Ey[1][j][k] * sin_ph + iEy[1][j][k] * cos_ph;
      iEz[isize + 1][j][k] = Ez[1][j][k] * sin_ph + iEz[1][j][k] * cos_ph;
    }
}

void E_field_periodic_boundary_update_y() {
  int i, k;
  float phase_ang;
  float cos_ph, sin_ph;

  ///// for speed up
  phase_ang = wave_vector_y * 2 * pi;
  cos_ph = cos(phase_ang);
  sin_ph = sin(phase_ang);

  for (i = 1; i < pisize; i++)
    for (k = 1; k < pksize; k++) {
      Ey[i][0][k] = Ey[i][jsize][k] * cos_ph + iEy[i][jsize][k] * sin_ph;
      Ex[i][jsize + 1][k] = Ex[i][1][k] * cos_ph - iEx[i][1][k] * sin_ph;
      Ez[i][jsize + 1][k] = Ez[i][1][k] * cos_ph - iEz[i][1][k] * sin_ph;

      iEy[i][0][k] = -Ey[i][jsize][k] * sin_ph + iEy[i][jsize][k] * cos_ph;
      iEx[i][jsize + 1][k] = Ex[i][1][k] * sin_ph + iEx[i][1][k] * cos_ph;
      iEz[i][jsize + 1][k] = Ez[i][1][k] * sin_ph + iEz[i][1][k] * cos_ph;
    }
}

void E_field_periodic_boundary_update_Ywall() ////// for triangular lattice
{
  int i, k;
  int Uxh; // a half lattice constant
  float phase_ang1, phase_ang2;
  float cos_ph1, sin_ph1;
  float cos_ph2, sin_ph2;

  ///// for speed up
  Uxh = 0.5 * isize;
  phase_ang1 = (0.5 * wave_vector_x + wave_vector_y) * 2 * pi;
  phase_ang2 = (-0.5 * wave_vector_x + wave_vector_y) * 2 * pi;
  cos_ph1 = cos(phase_ang1);
  sin_ph1 = sin(phase_ang1);
  cos_ph2 = cos(phase_ang2);
  sin_ph2 = sin(phase_ang2);

  for (i = 1; i < pisize - Uxh; i++)
    for (k = 1; k < pksize; k++) {
      Ey[i][0][k] =
          Ey[i + Uxh][jsize][k] * cos_ph1 + iEy[i + Uxh][jsize][k] * sin_ph1;
      Ex[i + Uxh][jsize + 1][k] =
          Ex[i][1][k] * cos_ph1 - iEx[i][1][k] * sin_ph1;
      Ez[i + Uxh][jsize + 1][k] =
          Ez[i][1][k] * cos_ph1 - iEz[i][1][k] * sin_ph1;

      iEy[i][0][k] =
          -Ey[i + Uxh][jsize][k] * sin_ph1 + iEy[i + Uxh][jsize][k] * cos_ph1;
      iEx[i + Uxh][jsize + 1][k] =
          Ex[i][1][k] * sin_ph1 + iEx[i][1][k] * cos_ph1;
      iEz[i + Uxh][jsize + 1][k] =
          Ez[i][1][k] * sin_ph1 + iEz[i][1][k] * cos_ph1;
    }

  for (i = pisize - Uxh; i < pisize; i++)
    for (k = 1; k < pksize; k++) {
      Ey[i][0][k] =
          Ey[i - Uxh][jsize][k] * cos_ph2 + iEy[i - Uxh][jsize][k] * sin_ph2;
      Ex[i - Uxh][jsize + 1][k] =
          Ex[i][1][k] * cos_ph2 - iEx[i][1][k] * sin_ph2;
      Ez[i - Uxh][jsize + 1][k] =
          Ez[i][1][k] * cos_ph2 - iEz[i][1][k] * sin_ph2;

      iEy[i][0][k] =
          -Ey[i - Uxh][jsize][k] * sin_ph2 + iEy[i - Uxh][jsize][k] * cos_ph2;
      iEx[i - Uxh][jsize + 1][k] =
          Ex[i][1][k] * sin_ph2 + iEx[i][1][k] * cos_ph2;
      iEz[i - Uxh][jsize + 1][k] =
          Ez[i][1][k] * sin_ph2 + iEz[i][1][k] * cos_ph2;
    }
}

void H_field_periodic_boundary_update_x() {
  int j, k;
  float phase_ang;
  float cos_ph, sin_ph;

  ///// for speed up
  phase_ang = wave_vector_x * 2 * pi;
  cos_ph = cos(phase_ang);
  sin_ph = sin(phase_ang);

  for (j = 1; j < pjsize; j++)
    for (k = 1; k < pksize; k++) {
      Hx[isize + 1][j][k] = Hx[1][j][k] * cos_ph - iHx[1][j][k] * sin_ph;
      Hy[0][j][k] = Hy[isize][j][k] * cos_ph + iHy[isize][j][k] * sin_ph;
      Hz[0][j][k] = Hz[isize][j][k] * cos_ph + iHz[isize][j][k] * sin_ph;

      iHx[isize + 1][j][k] = Hx[1][j][k] * sin_ph + iHx[1][j][k] * cos_ph;
      iHy[0][j][k] = -Hy[isize][j][k] * sin_ph + iHy[isize][j][k] * cos_ph;
      iHz[0][j][k] = -Hz[isize][j][k] * sin_ph + iHz[isize][j][k] * cos_ph;
    }
}

void H_field_periodic_boundary_update_Xwall() ///// for triangular lattice
{
  int i, j, k;
  float phase_ang;
  float cos_ph, sin_ph;

  ///// for speed up
  phase_ang = wave_vector_x * 2 * pi;
  cos_ph = cos(phase_ang);
  sin_ph = sin(phase_ang);

  for (j = 1; j < pjsize; j++)
    for (k = 1; k < pksize; k++) {
      Hx[isize + 1][j][k] = Hx[1][j][k] * cos_ph - iHx[1][j][k] * sin_ph;
      Hy[0][j][k] = Hy[isize][j][k] * cos_ph + iHy[isize][j][k] * sin_ph;
      Hz[0][j][k] = Hz[isize][j][k] * cos_ph + iHz[isize][j][k] * sin_ph;

      iHx[isize + 1][j][k] = Hx[1][j][k] * sin_ph + iHx[1][j][k] * cos_ph;
      iHy[0][j][k] = -Hy[isize][j][k] * sin_ph + iHy[isize][j][k] * cos_ph;
      iHz[0][j][k] = -Hz[isize][j][k] * sin_ph + iHz[isize][j][k] * cos_ph;
    }
}

void H_field_periodic_boundary_update_y() {
  int i, k;
  float phase_ang;
  float cos_ph, sin_ph;

  ///// for speed up
  phase_ang = wave_vector_y * 2 * pi;
  cos_ph = cos(phase_ang);
  sin_ph = sin(phase_ang);

  for (i = 1; i < pisize; i++)
    for (k = 1; k < pksize; k++) {
      Hy[i][jsize + 1][k] = Hy[i][1][k] * cos_ph - iHy[i][1][k] * sin_ph;
      Hx[i][0][k] = Hx[i][jsize][k] * cos_ph + iHx[i][jsize][k] * sin_ph;
      Hz[i][0][k] = Hz[i][jsize][k] * cos_ph + iHz[i][jsize][k] * sin_ph;

      iHy[i][jsize + 1][k] = Hy[i][1][k] * sin_ph + iHy[i][1][k] * cos_ph;
      iHx[i][0][k] = -Hx[i][jsize][k] * sin_ph + iHx[i][jsize][k] * cos_ph;
      iHz[i][0][k] = -Hz[i][jsize][k] * sin_ph + iHz[i][jsize][k] * cos_ph;
    }
}

void H_field_periodic_boundary_update_Ywall() ///// for triangular lattice
{
  int i, k;
  int Uxh; // a half lattice constant
  float phase_ang1, phase_ang2;
  float cos_ph1, sin_ph1;
  float cos_ph2, sin_ph2;

  ///// for speed up
  Uxh = 0.5 * isize;
  phase_ang1 = (0.5 * wave_vector_x + wave_vector_y) * 2 * pi;
  phase_ang2 = (-0.5 * wave_vector_x + wave_vector_y) * 2 * pi;
  cos_ph1 = cos(phase_ang1);
  sin_ph1 = sin(phase_ang1);
  cos_ph2 = cos(phase_ang2);
  sin_ph2 = sin(phase_ang2);

  for (i = 1; i < pisize - Uxh; i++)
    for (k = 1; k < pksize; k++) {
      Hy[i + Uxh][jsize + 1][k] =
          Hy[i][1][k] * cos_ph1 - iHy[i][1][k] * sin_ph1;
      Hx[i][0][k] =
          Hx[i + Uxh][jsize][k] * cos_ph1 + iHx[i + Uxh][jsize][k] * sin_ph1;
      Hz[i][0][k] =
          Hz[i + Uxh][jsize][k] * cos_ph1 + iHz[i + Uxh][jsize][k] * sin_ph1;

      iHy[i + Uxh][jsize + 1][k] =
          Hy[i][1][k] * sin_ph1 + iHy[i][1][k] * cos_ph1;
      iHx[i][0][k] =
          -Hx[i + Uxh][jsize][k] * sin_ph1 + iHx[i + Uxh][jsize][k] * cos_ph1;
      iHz[i][0][k] =
          -Hz[i + Uxh][jsize][k] * sin_ph1 + iHz[i + Uxh][jsize][k] * cos_ph1;
    }

  for (i = pisize - Uxh; i < pisize; i++)
    for (k = 1; k < pksize; k++) {
      Hy[i - Uxh][jsize + 1][k] =
          Hy[i][1][k] * cos_ph2 - iHy[i][1][k] * sin_ph2;
      Hx[i][0][k] =
          Hx[i - Uxh][jsize][k] * cos_ph2 + iHx[i - Uxh][jsize][k] * sin_ph2;
      Hz[i][0][k] =
          Hz[i - Uxh][jsize][k] * cos_ph2 + iHz[i - Uxh][jsize][k] * sin_ph2;

      iHy[i - Uxh][jsize + 1][k] =
          Hy[i][1][k] * sin_ph2 + iHy[i][1][k] * cos_ph2;
      iHx[i][0][k] =
          -Hx[i - Uxh][jsize][k] * sin_ph2 + iHx[i - Uxh][jsize][k] * cos_ph2;
      iHz[i][0][k] =
          -Hz[i - Uxh][jsize][k] * sin_ph2 + iHz[i - Uxh][jsize][k] * cos_ph2;
    }
}

void E_field_Gamma_boundary_update_x() {
  int j, k;

  for (j = 1; j < pjsize; j++)
    for (k = 1; k < pksize; k++) {
      Ex[0][j][k] = Ex[isize][j][k];
      Ey[isize + 1][j][k] = Ey[1][j][k];
      Ez[isize + 1][j][k] = Ez[1][j][k];
    }
}

void E_field_Gamma_boundary_update_Xwall() ///// for triangular lattice
{
  int i, j, k;

  for (j = 1; j < pjsize; j++)
    for (k = 1; k < pksize; k++) {
      Ex[0][j][k] = Ex[isize][j][k];
      Ey[isize + 1][j][k] = Ey[1][j][k];
      Ez[isize + 1][j][k] = Ez[1][j][k];
    }
}

void E_field_Gamma_boundary_update_y() {
  int i, k;

  for (i = 1; i < pisize; i++)
    for (k = 1; k < pksize; k++) {
      Ey[i][0][k] = Ey[i][jsize][k];
      Ex[i][jsize + 1][k] = Ex[i][1][k];
      Ez[i][jsize + 1][k] = Ez[i][1][k];
    }
}

void E_field_Gamma_boundary_update_Ywall() ///// for triangular lattice
{
  int i, k;
  int Uxh;

  ///// for speed up
  Uxh = 0.5 * isize;

  for (i = 1; i < pisize - Uxh; i++)
    for (k = 1; k < pksize; k++) {
      Ey[i][0][k] = Ey[i + Uxh][jsize][k];
      Ex[i + Uxh][jsize + 1][k] = Ex[i][1][k];
      Ez[i + Uxh][jsize + 1][k] = Ez[i][1][k];
    }

  for (i = pisize - Uxh; i < pisize; i++)
    for (k = 1; k < pksize; k++) {
      Ey[i][0][k] = Ey[i - Uxh][jsize][k];
      Ex[i - Uxh][jsize + 1][k] = Ex[i][1][k];
      Ez[i - Uxh][jsize + 1][k] = Ez[i][1][k];
    }
}

void H_field_Gamma_boundary_update_x() {
  int j, k;

  for (j = 1; j < pjsize; j++)
    for (k = 1; k < pksize; k++) {
      Hx[isize + 1][j][k] = Hx[1][j][k];
      Hy[0][j][k] = Hy[isize][j][k];
      Hz[0][j][k] = Hz[isize][j][k];
    }
}

void H_field_Gamma_boundary_update_Xwall() ///// for triangular lattice
{
  int i, j, k;

  for (j = 1; j < pjsize; j++)
    for (k = 1; k < pksize; k++) {
      Hx[isize + 1][j][k] = Hx[1][j][k];
      Hy[0][j][k] = Hy[isize][j][k];
      Hz[0][j][k] = Hz[isize][j][k];
    }
}

void H_field_Gamma_boundary_update_y() {
  int i, k;

  for (i = 1; i < pisize; i++)
    for (k = 1; k < pksize; k++) {
      Hy[i][jsize + 1][k] = Hy[i][1][k];
      Hx[i][0][k] = Hx[i][jsize][k];
      Hz[i][0][k] = Hz[i][jsize][k];
    }
}

void H_field_Gamma_boundary_update_Ywall() ///// for triangular lattice
{
  int i, k;
  int Uxh;

  ///// for speed up
  Uxh = 0.5 * isize;

  for (i = 1; i < pisize - Uxh; i++)
    for (k = 1; k < pksize; k++) {
      Hy[i + Uxh][jsize + 1][k] = Hy[i][1][k];
      Hx[i][0][k] = Hx[i + Uxh][jsize][k];
      Hz[i][0][k] = Hz[i + Uxh][jsize][k];
    }

  for (i = pisize - Uxh; i < pisize; i++)
    for (k = 1; k < pksize; k++) {
      Hy[i - Uxh][jsize + 1][k] = Hy[i][1][k];
      Hx[i][0][k] = Hx[i - Uxh][jsize][k];
      Hz[i][0][k] = Hz[i - Uxh][jsize][k];
    }
}

void field_initialization() {
  field_zero(Ex);
  field_zero(Ey);
  field_zero(Ez);
  field_zero(Jx);
  field_zero(Jy);
  field_zero(Jz);
  field_zero(Hx);
  field_zero(Hy);
  field_zero(Hz);
  field_zero(Dx);
  field_zero(Dy);
  field_zero(Dz);
  field_zero(Bx);
  field_zero(By);
  field_zero(Bz);

  if ((use_periodic_x == 1 || use_periodic_y == 1) &&
      (wave_vector_x != 0.0 || wave_vector_y != 0.0)) {
    field_zero(iEx);
    field_zero(iEy);
    field_zero(iEz);
    field_zero(iJx);
    field_zero(iJy);
    field_zero(iJz);
    field_zero(iHx);
    field_zero(iHy);
    field_zero(iHz);
    field_zero(iDx);
    field_zero(iDy);
    field_zero(iDz);
    field_zero(iBx);
    field_zero(iBy);
    field_zero(iBz);
  }
}

void field_zero(float ***name) {
  int i, j, k;

  for (i = 0; i < misize; i++)
    for (j = 0; j < mjsize; j++)
      for (k = 0; k < mksize; k++)
        name[i][j][k] = 0;
}
