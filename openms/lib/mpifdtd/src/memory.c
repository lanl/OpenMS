#include "./fdtd.hpp"

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

	epsilonx=float_3d_memory(m_mygrid_nx,m_mygrid_ny,m_mygrid_nz);
	epsilony=float_3d_memory(m_mygrid_nx,m_mygrid_ny,m_mygrid_nz);
	epsilonz=float_3d_memory(m_mygrid_nx,m_mygrid_ny,m_mygrid_nz);

	ddx = float_3d_memory(m_mygrid_nx,m_mygrid_ny,m_mygrid_nz);
	ddy = float_3d_memory(m_mygrid_nx,m_mygrid_ny,m_mygrid_nz);
	ddz = float_3d_memory(m_mygrid_nx,m_mygrid_ny,m_mygrid_nz);

	mepsilon=float_3d_memory(m_mygrid_nx,m_mygrid_ny,m_mygrid_nz);
	momega=float_3d_memory(m_mygrid_nx,m_mygrid_ny,m_mygrid_nz);
	mgamma=float_3d_memory(m_mygrid_nx,m_mygrid_ny,m_mygrid_nz);

	Ex=float_3d_memory(m_mygrid_nx,m_mygrid_ny,m_mygrid_nz);
	Ey=float_3d_memory(m_mygrid_nx,m_mygrid_ny,m_mygrid_nz);
	Ez=float_3d_memory(m_mygrid_nx,m_mygrid_ny,m_mygrid_nz);

	dPx=float_3d_memory(m_mygrid_nx,m_mygrid_ny,m_mygrid_nz);
	dPy=float_3d_memory(m_mygrid_nx,m_mygrid_ny,m_mygrid_nz);
	dPz=float_3d_memory(m_mygrid_nx,m_mygrid_ny,m_mygrid_nz);

	dPx_old=float_3d_memory(m_mygrid_nx,m_mygrid_ny,m_mygrid_nz);
	dPy_old=float_3d_memory(m_mygrid_nx,m_mygrid_ny,m_mygrid_nz);
	dPz_old=float_3d_memory(m_mygrid_nx,m_mygrid_ny,m_mygrid_nz);

	Jx=float_3d_memory(m_mygrid_nx,m_mygrid_ny,m_mygrid_nz);
	Jy=float_3d_memory(m_mygrid_nx,m_mygrid_ny,m_mygrid_nz);
	Jz=float_3d_memory(m_mygrid_nx,m_mygrid_ny,m_mygrid_nz);

	Hx=float_3d_memory(m_mygrid_nx,m_mygrid_ny,m_mygrid_nz);
	Hy=float_3d_memory(m_mygrid_nx,m_mygrid_ny,m_mygrid_nz);
	Hz=float_3d_memory(m_mygrid_nx,m_mygrid_ny,m_mygrid_nz);

	Dx=float_3d_memory(m_mygrid_nx,m_mygrid_ny,m_mygrid_nz);
	Dy=float_3d_memory(m_mygrid_nx,m_mygrid_ny,m_mygrid_nz);
	Dz=float_3d_memory(m_mygrid_nx,m_mygrid_ny,m_mygrid_nz);

	Bx=float_3d_memory(m_mygrid_nx,m_mygrid_ny,m_mygrid_nz);
	By=float_3d_memory(m_mygrid_nx,m_mygrid_ny,m_mygrid_nz);
	Bz=float_3d_memory(m_mygrid_nx,m_mygrid_ny,m_mygrid_nz);

	// for CPML
	psi_Exy=float_3d_memory(m_mygrid_nx,m_mygrid_ny,m_mygrid_nz);
	psi_Exz=float_3d_memory(m_mygrid_nx,m_mygrid_ny,m_mygrid_nz);
	psi_Eyx=float_3d_memory(m_mygrid_nx,m_mygrid_ny,m_mygrid_nz);
	psi_Eyz=float_3d_memory(m_mygrid_nx,m_mygrid_ny,m_mygrid_nz);
	psi_Ezx=float_3d_memory(m_mygrid_nx,m_mygrid_ny,m_mygrid_nz);
	psi_Ezy=float_3d_memory(m_mygrid_nx,m_mygrid_ny,m_mygrid_nz);

	psi_Hxy=float_3d_memory(m_mygrid_nx,m_mygrid_ny,m_mygrid_nz);
	psi_Hxz=float_3d_memory(m_mygrid_nx,m_mygrid_ny,m_mygrid_nz);
	psi_Hyx=float_3d_memory(m_mygrid_nx,m_mygrid_ny,m_mygrid_nz);
	psi_Hyz=float_3d_memory(m_mygrid_nx,m_mygrid_ny,m_mygrid_nz);
	psi_Hzx=float_3d_memory(m_mygrid_nx,m_mygrid_ny,m_mygrid_nz);
	psi_Hzy=float_3d_memory(m_mygrid_nx,m_mygrid_ny,m_mygrid_nz);

	// in case of periodic boundary condition
	if((use_periodic_x == 1 || use_periodic_y == 1) && (wave_vector_x!=0.0 || wave_vector_y!=0.0))
	// In case of Gamma-point, do not use the Imaginary fields
	{
		iEx=float_3d_memory(m_mygrid_nx,m_mygrid_ny,m_mygrid_nz);
		iEy=float_3d_memory(m_mygrid_nx,m_mygrid_ny,m_mygrid_nz);
		iEz=float_3d_memory(m_mygrid_nx,m_mygrid_ny,m_mygrid_nz);

		iJx=float_3d_memory(m_mygrid_nx,m_mygrid_ny,m_mygrid_nz);
		iJy=float_3d_memory(m_mygrid_nx,m_mygrid_ny,m_mygrid_nz);
		iJz=float_3d_memory(m_mygrid_nx,m_mygrid_ny,m_mygrid_nz);

		iHx=float_3d_memory(m_mygrid_nx,m_mygrid_ny,m_mygrid_nz);
		iHy=float_3d_memory(m_mygrid_nx,m_mygrid_ny,m_mygrid_nz);
		iHz=float_3d_memory(m_mygrid_nx,m_mygrid_ny,m_mygrid_nz);

		iDx=float_3d_memory(m_mygrid_nx,m_mygrid_ny,m_mygrid_nz);
		iDy=float_3d_memory(m_mygrid_nx,m_mygrid_ny,m_mygrid_nz);
		iDz=float_3d_memory(m_mygrid_nx,m_mygrid_ny,m_mygrid_nz);

		iBx=float_3d_memory(m_mygrid_nx,m_mygrid_ny,m_mygrid_nz);
		iBy=float_3d_memory(m_mygrid_nx,m_mygrid_ny,m_mygrid_nz);
		iBz=float_3d_memory(m_mygrid_nx,m_mygrid_ny,m_mygrid_nz);
	}

	position=char_3d_memory(m_mygrid_nx,m_mygrid_ny,m_mygrid_nz);

	for(i=0;i<m_mygrid_nx;i++)
		for(j=0;j<m_mygrid_ny;j++)
			for(k=0;k<m_mygrid_nz;k++)
			{
				if(pmlil+1<=i && i<=mygrid_nx-pmlir-2 && pmljl+1<=j && j<=mygrid_ny-pmljr-2 && pmlkl+1<=k && k<=mygrid_nz-pmlkr-2)
					position[i][j][k]=1;
				else position[i][j][k]=0;
			}

	aax=float_1d_memory(m_mygrid_nx);
	aay=float_1d_memory(m_mygrid_ny);
	aaz=float_1d_memory(m_mygrid_nz);
	bbx=float_1d_memory(m_mygrid_nx);
	bby=float_1d_memory(m_mygrid_ny);
	bbz=float_1d_memory(m_mygrid_nz);
	ccx=float_1d_memory(m_mygrid_nx);
	ccy=float_1d_memory(m_mygrid_ny);
	ccz=float_1d_memory(m_mygrid_nz);
	eex=float_1d_memory(m_mygrid_nx);
	eey=float_1d_memory(m_mygrid_ny);
	eez=float_1d_memory(m_mygrid_nz);
	ffx=float_1d_memory(m_mygrid_nx);
	ffy=float_1d_memory(m_mygrid_ny);
	ffz=float_1d_memory(m_mygrid_nz);
	ggx=float_1d_memory(m_mygrid_nx);
	ggy=float_1d_memory(m_mygrid_ny);
	ggz=float_1d_memory(m_mygrid_nz);
	hhx=float_1d_memory(m_mygrid_nx);
	hhy=float_1d_memory(m_mygrid_ny);
	hhz=float_1d_memory(m_mygrid_nz);
	iix=float_1d_memory(m_mygrid_nx);
	iiy=float_1d_memory(m_mygrid_ny);
	iiz=float_1d_memory(m_mygrid_nz);
	jjx=float_1d_memory(m_mygrid_nx);
	jjy=float_1d_memory(m_mygrid_ny);
	jjz=float_1d_memory(m_mygrid_nz);
	kkx=float_1d_memory(m_mygrid_nx);
	kky=float_1d_memory(m_mygrid_ny);
	kkz=float_1d_memory(m_mygrid_nz);
	llx=float_1d_memory(m_mygrid_nx);
	lly=float_1d_memory(m_mygrid_ny);
	llz=float_1d_memory(m_mygrid_nz);

	Ex_cos=float_3d_memory(mygrid_nx,mygrid_ny,SpecN);
	Ex_sin=float_3d_memory(mygrid_nx,mygrid_ny,SpecN);
	Ey_cos=float_3d_memory(mygrid_nx,mygrid_ny,SpecN);
	Ey_sin=float_3d_memory(mygrid_nx,mygrid_ny,SpecN);
	Hx_cos=float_3d_memory(mygrid_nx,mygrid_ny,SpecN);
	Hx_sin=float_3d_memory(mygrid_nx,mygrid_ny,SpecN);
	Hy_cos=float_3d_memory(mygrid_nx,mygrid_ny,SpecN);
	Hy_sin=float_3d_memory(mygrid_nx,mygrid_ny,SpecN);

        printf("number of m_mygrid is %d %d %d\n", m_mygrid_nx, m_mygrid_ny, m_mygrid_nz);
	myprintf("memory...ok\n");
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
			for(k=0;k<kmax;k++)	memory[i][j][k]=0;
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
			for(k=0;k<kmax;k++)	memory[i][j][k]=0.0;
		}
	}

	return memory;
}
