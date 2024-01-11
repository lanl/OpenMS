#include "./pFDTD.h"

char ***char_3d_memory(int imax,int jmax,int kmax);
float *float_1d_memory(int imax);
float ***float_3d_memory(int imax,int jmax,int kmax);

/* input dielectric structure */
char ***position;
float ***epsilonx,***epsilony,***epsilonz;
float ***mepsilon,***momega,***mgamma;

/* field components */
float ***Ex,***Ey,***Ez;
float ***dPx,***dPy,***dPz;
float ***dPx_old,***dPy_old,***dPz_old;
float ***Jx,***Jy,***Jz;
float ***Hx,***Hy,***Hz;
float ***Dx,***Dy,***Dz;
float ***Bx,***By,***Bz;

/* imaginary field (periodic FDTD) */
float ***iEx,***iEy,***iEz;
float ***iJx,***iJy,***iJz;
float ***iHx,***iHy,***iHz;
float ***iDx,***iDy,***iDz;
float ***iBx,***iBy,***iBz;

/* FDTD update coefficients */
float *aax,*aay,*aaz;
float *bbx,*bby,*bbz;
float *ccx,*ccy,*ccz;
float ***ddx,***ddy,***ddz;
float *eex,*eey,*eez;
float *ffx,*ffy,*ffz;
float *ggx,*ggy,*ggz;
float *hhx,*hhy,*hhz;
float *iix,*iiy,*iiz;
float *jjx,*jjy,*jjz;
float *kkx,*kky,*kkz;
float *llx,*lly,*llz;

/* for farfield parameter calculation */
float ***Ex_cos, ***Ex_sin;
float ***Ey_cos, ***Ey_sin;
float ***Hx_cos, ***Hx_sin;
float ***Hy_cos, ***Hy_sin;
// for cpml
float ***psi_Exy, ***psi_Exz;
float ***psi_Eyx, ***psi_Eyz;
float ***psi_Ezy, ***psi_Ezx;

float ***psi_Hxy, ***psi_Hxz;
float ***psi_Hyx, ***psi_Hyz;
float ***psi_Hzy, ***psi_Hzx;

void memory()
/* Note: Use this function after defining the variable 'SpecN' in main.c */
{
    int i,j,k;

    epsilonx=float_3d_memory(misize,mjsize,mksize);
    epsilony=float_3d_memory(misize,mjsize,mksize);
    epsilonz=float_3d_memory(misize,mjsize,mksize);

    ddx = float_3d_memory(misize,mjsize,mksize);
    ddy = float_3d_memory(misize,mjsize,mksize);
    ddz = float_3d_memory(misize,mjsize,mksize);

    mepsilon=float_3d_memory(misize,mjsize,mksize);
    momega=float_3d_memory(misize,mjsize,mksize);
    mgamma=float_3d_memory(misize,mjsize,mksize);

    Ex=float_3d_memory(misize,mjsize,mksize);
    Ey=float_3d_memory(misize,mjsize,mksize);
    Ez=float_3d_memory(misize,mjsize,mksize);

    dPx=float_3d_memory(misize,mjsize,mksize);
    dPy=float_3d_memory(misize,mjsize,mksize);
    dPz=float_3d_memory(misize,mjsize,mksize);

    dPx_old=float_3d_memory(misize,mjsize,mksize);
    dPy_old=float_3d_memory(misize,mjsize,mksize);
    dPz_old=float_3d_memory(misize,mjsize,mksize);

    Jx=float_3d_memory(misize,mjsize,mksize);
    Jy=float_3d_memory(misize,mjsize,mksize);
    Jz=float_3d_memory(misize,mjsize,mksize);

    Hx=float_3d_memory(misize,mjsize,mksize);
    Hy=float_3d_memory(misize,mjsize,mksize);
    Hz=float_3d_memory(misize,mjsize,mksize);
    
    Dx=float_3d_memory(misize,mjsize,mksize);
    Dy=float_3d_memory(misize,mjsize,mksize);
    Dz=float_3d_memory(misize,mjsize,mksize);

    Bx=float_3d_memory(misize,mjsize,mksize);
    By=float_3d_memory(misize,mjsize,mksize);
    Bz=float_3d_memory(misize,mjsize,mksize);

    // for CPML
    psi_Exy=float_3d_memory(misize,mjsize,mksize);
    psi_Exz=float_3d_memory(misize,mjsize,mksize);
    psi_Eyx=float_3d_memory(misize,mjsize,mksize);
    psi_Eyz=float_3d_memory(misize,mjsize,mksize);
    psi_Ezx=float_3d_memory(misize,mjsize,mksize);
    psi_Ezy=float_3d_memory(misize,mjsize,mksize);

    psi_Hxy=float_3d_memory(misize,mjsize,mksize);
    psi_Hxz=float_3d_memory(misize,mjsize,mksize);
    psi_Hyx=float_3d_memory(misize,mjsize,mksize);
    psi_Hyz=float_3d_memory(misize,mjsize,mksize);
    psi_Hzx=float_3d_memory(misize,mjsize,mksize);
    psi_Hzy=float_3d_memory(misize,mjsize,mksize);

    // in case of periodic boundary condition
    if((use_periodic_x == 1 || use_periodic_y == 1) && (wave_vector_x!=0.0 || wave_vector_y!=0.0))
    // In case of Gamma-point, do not use the Imaginary fields
    {
        iEx=float_3d_memory(misize,mjsize,mksize);
        iEy=float_3d_memory(misize,mjsize,mksize);
        iEz=float_3d_memory(misize,mjsize,mksize);

        iJx=float_3d_memory(misize,mjsize,mksize);
        iJy=float_3d_memory(misize,mjsize,mksize);
        iJz=float_3d_memory(misize,mjsize,mksize);

        iHx=float_3d_memory(misize,mjsize,mksize);
        iHy=float_3d_memory(misize,mjsize,mksize);
        iHz=float_3d_memory(misize,mjsize,mksize);

        iDx=float_3d_memory(misize,mjsize,mksize);
        iDy=float_3d_memory(misize,mjsize,mksize);
        iDz=float_3d_memory(misize,mjsize,mksize);

        iBx=float_3d_memory(misize,mjsize,mksize);
        iBy=float_3d_memory(misize,mjsize,mksize);
        iBz=float_3d_memory(misize,mjsize,mksize);
    }

    position=char_3d_memory(misize,mjsize,mksize);

    for(i=0;i<misize;i++)
        for(j=0;j<mjsize;j++)
            for(k=0;k<mksize;k++)
            {
                if(pmlil+1<=i && i<=isize-pmlir-2 && pmljl+1<=j && j<=jsize-pmljr-2 && pmlkl+1<=k && k<=ksize-pmlkr-2)
                    position[i][j][k]=1;
                else position[i][j][k]=0;
            }

    aax=float_1d_memory(misize);
    aay=float_1d_memory(mjsize);
    aaz=float_1d_memory(mksize);
    bbx=float_1d_memory(misize);
    bby=float_1d_memory(mjsize);
    bbz=float_1d_memory(mksize);
    ccx=float_1d_memory(misize);
    ccy=float_1d_memory(mjsize);
    ccz=float_1d_memory(mksize);
    eex=float_1d_memory(misize);
    eey=float_1d_memory(mjsize);
    eez=float_1d_memory(mksize);
    ffx=float_1d_memory(misize);
    ffy=float_1d_memory(mjsize);
    ffz=float_1d_memory(mksize);
    ggx=float_1d_memory(misize);
    ggy=float_1d_memory(mjsize);
    ggz=float_1d_memory(mksize);
    hhx=float_1d_memory(misize);
    hhy=float_1d_memory(mjsize);
    hhz=float_1d_memory(mksize);
    iix=float_1d_memory(misize);
    iiy=float_1d_memory(mjsize);
    iiz=float_1d_memory(mksize);
    jjx=float_1d_memory(misize);
    jjy=float_1d_memory(mjsize);
    jjz=float_1d_memory(mksize);
    kkx=float_1d_memory(misize);
    kky=float_1d_memory(mjsize);
    kkz=float_1d_memory(mksize);
    llx=float_1d_memory(misize);
    lly=float_1d_memory(mjsize);
    llz=float_1d_memory(mksize);

    Ex_cos=float_3d_memory(isize,jsize,SpecN);
    Ex_sin=float_3d_memory(isize,jsize,SpecN);
    Ey_cos=float_3d_memory(isize,jsize,SpecN);
    Ey_sin=float_3d_memory(isize,jsize,SpecN);
    Hx_cos=float_3d_memory(isize,jsize,SpecN);
    Hx_sin=float_3d_memory(isize,jsize,SpecN);
    Hy_cos=float_3d_memory(isize,jsize,SpecN);
    Hy_sin=float_3d_memory(isize,jsize,SpecN);


    /* release information */
    printf("KAIST FDTD ver. %2.3f (Last modified by S.H.K)\n",KFDTDver);

    printf("memory...ok\n");
}

char ***char_3d_memory(int imax,int jmax,int kmax)
{
    int i,j,k;
    char ***memory,*cmemory;

    cmemory=(char *)malloc(sizeof(char)*imax*jmax*kmax);
    memory=(char ***)malloc(sizeof(char **)*imax);
    for(i=0;i<imax;i++)
    {
        memory[i]=(char **)malloc(sizeof(char *)*jmax);
        for(j=0;j<jmax;j++)
        {
            memory[i][j]=cmemory+i*jmax*kmax+j*kmax;
            for(k=0;k<kmax;k++)    memory[i][j][k]=0;
        }
    }

    return memory;
}

float *float_1d_memory(int imax)
{
    int i;
    float *memory;

    memory=(float *)malloc(sizeof(float)*imax);
    for(i=0;i<imax;i++) memory[i]=0.0;

    return memory;
}

float ***float_3d_memory(int imax,int jmax,int kmax)
{
    int i,j,k;
    float ***memory,*cmemory;

    cmemory=(float *)malloc(sizeof(float)*imax*jmax*kmax);
    memory=(float ***)malloc(sizeof(float **)*imax);
    for(i=0;i<imax;i++)
    {
        memory[i]=(float **)malloc(sizeof(float *)*jmax);
        for(j=0;j<jmax;j++)
        {
            memory[i][j]=cmemory+i*jmax*kmax+j*kmax;
            for(k=0;k<kmax;k++)    memory[i][j][k]=0.0;
        }
    }

    return memory;
}
