#include "./pFDTD.h"

/////////////////////////////////////////////
///////// Define global variables ///////////
/////////////////////////////////////////////
static char string[80];  // global name

//static int l;

//static int m,n;
//static char ch;

//static float **Eu_real, **Eu_imag, **Eu_int;
//static float *polarization;
//static float sum_Eu;
//static float Ex_delta, Ey_delta, Hx_delta, Hy_delta;

static char name_freq[10];
static char name_Ex_real[20], name_Ex_imag[20], name_Ey_real[20], name_Ey_imag[20];
static char name_Hx_real[20], name_Hx_imag[20], name_Hy_real[20], name_Hy_imag[20];
static char FFT_E_int2[20], FFT_E_log[20], FFT_H_int2[20], FFT_H_log[20];
static char radiation_tot[20], radiation_Et[20], radiation_Ep[20], radiation_Ex[20], radiation_Ey[20];
static char polar[20], P_tot_name[20], P_the_name[20], P_phi_name[20];
//--------------------------------------------------------------------//

int FFT2D(struct COMPLEX **c,int nx,int ny,int dir);
int FFT1D(int dir,int m,float *x,float *y);
int pow_finder(int N);
void make_file_name(int mm);
void set_global_variable(int NROW, float OMEGA, float Nfree, int *POLROT, int *dir, float *Eta, float *k);

//-----------------------------------------------------------
// reading (NROW)x(NROW) source data 
// then, store them in efieldx_real[][], efieldx_imag[][] ... 
//-----------------------------------------------------------
void reading_source_Ex(int row, int col, float **efieldx_real, float **efieldx_imag);
void reading_source_Ey(int row, int col, float **efieldy_real, float **efieldy_imag);
void reading_source_Hx(int row, int col, float **hfieldx_real, float **hfieldx_imag);
void reading_source_Hy(int row, int col, float **hfieldy_real, float **hfieldy_imag);

//------------------------------------------------------------
// Fourier transforming Ex, Ey, Hx, Hy to get Lx, Ly, Nx, Ny
//------------------------------------------------------------
void fourier_transform_Ly_Ex(int row, int col, int dir, float **Ly_real, float **Ly_imag, float **efieldx_real, float **efieldx_imag);
void fourier_transform_Lx_Ey(int row, int col, int dir, float **Lx_real, float **Lx_imag, float **efieldy_real, float **efieldy_imag);
void fourier_transform_Ny_Hx(int row, int col, int dir, float **Ny_real, float **Ny_imag, float **hfieldx_real, float **hfieldx_imag);
void fourier_transform_Nx_Hy(int row, int col, int dir, float **Nx_real, float **Nx_imag, float **hfieldy_real, float **hfieldy_imag);

//------------------------------
// Data shifting Lx,Ly,Nx,Ny
//------------------------------
void data_shifting_Ly(int row, int col, float **Ly_real, float **Ly_imag);
void data_shifting_Lx(int row, int col, float **Lx_real, float **Lx_imag);
void data_shifting_Ny(int row, int col, float **Ny_real, float **Ny_imag);
void data_shifting_Nx(int row, int col, float **Nx_real, float **Nx_imag);

//------------------------------------------------------------------------
// Make small size data (-5*k ~ +5*k) ; k is the radius of the light-cone 
//------------------------------------------------------------------------ 
void Ly_data_shrink(int row, int col, float **Ly_real, float **Ly_imag, float **Ly_s_real, float **Ly_s_imag, float Nfree, float k);
void Lx_data_shrink(int row, int col, float **Lx_real, float **Lx_imag, float **Lx_s_real, float **Lx_s_imag, float Nfree, float k);
void Nx_data_shrink(int row, int col, float **Nx_real, float **Nx_imag, float **Nx_s_real, float **Nx_s_imag, float Nfree, float k);
void Ny_data_shrink(int row, int col, float **Ny_real, float **Ny_imag, float **Ny_s_real, float **Ny_s_imag, float Nfree, float k);

//----------------------------------------------------------
// Calculate Nt,Np,Lt,Lp, ex_real,ex_imag, ey_real,ey_imag
//----------------------------------------------------------
void set_radiation_variable(float Nfree, float k, float Eta, float **Nx_s_real, float **Nx_s_imag, float **Ny_s_real, float **Ny_s_imag, float **Lx_s_real, float **Lx_s_imag, float **Ly_s_real, float **Ly_s_imag, float **Nt_s_real, float **Nt_s_imag, float **Np_s_real, float **Np_s_imag, float **Lt_s_real, float **Lt_s_imag, float **Lp_s_real, float **Lp_s_imag, float **cosp, float **sinp, float **cost, float **ex_real, float **ex_imag, float **ey_real, float **ey_imag);

//------------------------------------------------
// Calculate radint, etheta, ephi, eintx, einty 
//------------------------------------------------
void calc_radiation(float Nfree, float k, float Eta, float **Nt_s_real, float **Nt_s_imag, float **Np_s_real, float **Np_s_imag, float **Lt_s_real, float **Lt_s_imag, float **Lp_s_real, float **Lp_s_imag, float **cosp, float **sinp, float **cost, float **ex_real, float **ex_imag, float **ey_real, float **ey_imag, float **radint, float **etheta, float **ephi, float **eintx, float **einty, float *P_tot, float *P_the, float *P_phi);

//------------------------------------------------
// Print radint, etheta, ephi, eintx, einty 
//------------------------------------------------
void print_radiation(float Nfree, float k, float Eta, float **radint, float **etheta, float **ephi, float **eintx, float **einty, float *P_tot, float *P_the, float *P_phi);

void FT_data_writing(float **Nx_s_real, float **Nx_s_imag, float **Ny_s_real, float **Ny_s_imag, float **Lx_s_real, float **Lx_s_imag, float **Ly_s_real, float **Ly_s_imag, int row_s, int col_s, float Nfree, float k);
//void calc_polar(float NA, float Nfree);

void far_field_param(float *OMEGA, float DETECT)
{
	int i, j, k;
	int mm;

	k = non_uniform_z_to_i(DETECT);

	for(j=1; j<=jsize-2; j++)
	{
		for(i=1; i<=isize-2; i++)
		{
			for(mm=0; mm<SpecN; mm++)
			{
				Ex_cos[i][j][mm]+=grid_value("Ex",i,j,k)*cos(2*pi*OMEGA[mm]*t/S_factor/ds_x/lattice_x);
				Ex_sin[i][j][mm]+=grid_value("Ex",i,j,k)*sin(2*pi*OMEGA[mm]*t/S_factor/ds_x/lattice_x);
				Ey_cos[i][j][mm]+=grid_value("Ey",i,j,k)*cos(2*pi*OMEGA[mm]*t/S_factor/ds_x/lattice_x);
				Ey_sin[i][j][mm]+=grid_value("Ey",i,j,k)*sin(2*pi*OMEGA[mm]*t/S_factor/ds_x/lattice_x);
				Hx_cos[i][j][mm]+=grid_value("Hx",i,j,k)*cos(2*pi*OMEGA[mm]*t/S_factor/ds_x/lattice_x);
				Hx_sin[i][j][mm]+=grid_value("Hx",i,j,k)*sin(2*pi*OMEGA[mm]*t/S_factor/ds_x/lattice_x);
				Hy_cos[i][j][mm]+=grid_value("Hy",i,j,k)*cos(2*pi*OMEGA[mm]*t/S_factor/ds_x/lattice_x);
				Hy_sin[i][j][mm]+=grid_value("Hy",i,j,k)*sin(2*pi*OMEGA[mm]*t/S_factor/ds_x/lattice_x);
			}
		}
	}	
}

void far_field_FFT(int NROW, float NA, float Nfree, float *OMEGA, int mm)
{
	int i;
	int row, col; // row num and column num
	int row_s, col_s; // shrinked row & col num

	int POLROT;
	int dir; // direction of the FFT
	float Eta;
	float k; // normalized k value, radius of the light-cone

	//// source matrix ////
	float **hfieldx_real, **hfieldx_imag;
	float **hfieldy_real, **hfieldy_imag;
	float **efieldx_real, **efieldx_imag;
	float **efieldy_real, **efieldy_imag;

	//// after FFT ////
	float **Nx_real, **Nx_imag, **Ny_real, **Ny_imag;
	float **Lx_real, **Lx_imag, **Ly_real, **Ly_imag;
	float **Nx_s_real, **Nx_s_imag, **Ny_s_real, **Ny_s_imag;   // s : shrinked size
	float **Lx_s_real, **Lx_s_imag, **Ly_s_real, **Ly_s_imag;

	float **Nt_s_real, **Nt_s_imag, **Np_s_real, **Np_s_imag;   // s : shrinked size
	float **Lt_s_real, **Lt_s_imag, **Lp_s_real, **Lp_s_imag;

	float **ex_real, **ex_imag, **ey_real, **ey_imag;  // shrinked size 

	float **cosp, **sinp, **cost;  // shrinked size

	float **radint; // radiation intensity
	float **etheta, **ephi;
	float **eintx, **einty;
	float P_tot, P_the, P_phi; // integrating the radiated power

	//------------------------------------------------------
	col = NROW, row = NROW;

	make_file_name(mm);
	set_global_variable(NROW, OMEGA[mm], Nfree, &POLROT, &dir, &Eta, &k);

	row_s = (int)(10*Nfree*k);
	col_s = (int)(10*Nfree*k);

	//////////////////////
	//---- Ex field ----//
	//////////////////////
	efieldx_real = (float **)malloc(sizeof(float *)*col);
	for(i=0; i<col; i++)
		efieldx_real[i] = (float *)malloc(sizeof(float)*row);
	efieldx_imag = (float **)malloc(sizeof(float *)*col);
	for(i=0; i<col; i++)
		efieldx_imag[i] = (float *)malloc(sizeof(float)*row);
	//------------------------------------------------------------
	reading_source_Ex(row, col, efieldx_real, efieldx_imag);
	//------------------------------------------------------------
	Ly_real = (float **)malloc(sizeof(float *)*(col+1));
	for(i=0; i<(col+1); i++)
		Ly_real[i] = (float *)malloc(sizeof(float)*(row+1));
	Ly_imag = (float **)malloc(sizeof(float *)*(col+1));
	for(i=0; i<(col+1); i++)
		Ly_imag[i] = (float *)malloc(sizeof(float)*(row+1));
	//-----------------------------------------------------------------------------------
	fourier_transform_Ly_Ex(row, col, dir, Ly_real, Ly_imag, efieldx_real, efieldx_imag);
	//-----------------------------------------------------------------------------------
	for(i=0; i<col; i++)
	{
		free(efieldx_real[i]); free(efieldx_imag[i]);
	}
		free(efieldx_real); free(efieldx_imag);
	//-------------------------------------------
	data_shifting_Ly(row, col, Ly_real, Ly_imag);
	//-------------------------------------------
	Ly_s_real = (float **)malloc(sizeof(float *)*(col_s+1));
	for(i=0; i<(col_s+1); i++)
		Ly_s_real[i] = (float *)malloc(sizeof(float)*(row_s+1));
	Ly_s_imag = (float **)malloc(sizeof(float *)*(col_s+1));
	for(i=0; i<(col_s+1); i++)
		Ly_s_imag[i] = (float *)malloc(sizeof(float)*(row_s+1));
	//-------------------------------------------------------------------------
	Ly_data_shrink(row, col, Ly_real, Ly_imag, Ly_s_real, Ly_s_imag, Nfree, k);
	//-------------------------------------------------------------------------
	for(i=0; i<col+1; i++)
	{
		free(Ly_real[i]); free(Ly_imag[i]);
	}
		free(Ly_real); free(Ly_imag);

	//////////////////////
	//---- Ey field ----//
	//////////////////////
	efieldy_real = (float **)malloc(sizeof(float *)*col);
	for(i=0; i<col; i++)
		efieldy_real[i] = (float *)malloc(sizeof(float)*row);
	efieldy_imag = (float **)malloc(sizeof(float *)*col);
	for(i=0; i<col; i++)
		efieldy_imag[i] = (float *)malloc(sizeof(float)*row);
	//------------------------------------------------------------
	reading_source_Ey(row, col, efieldy_real, efieldy_imag);
	//------------------------------------------------------------
	Lx_real = (float **)malloc(sizeof(float *)*(col+1));
	for(i=0; i<(col+1); i++)
		Lx_real[i] = (float *)malloc(sizeof(float)*(row+1));
	Lx_imag = (float **)malloc(sizeof(float *)*(col+1));
	for(i=0; i<(col+1); i++)
		Lx_imag[i] = (float *)malloc(sizeof(float)*(row+1));
	//-----------------------------------------------------------------------------------
	fourier_transform_Lx_Ey(row, col, dir, Lx_real, Lx_imag, efieldy_real, efieldy_imag);
	//-----------------------------------------------------------------------------------
	for(i=0; i<col; i++)
	{
		free(efieldy_real[i]); free(efieldy_imag[i]);
	}
		free(efieldy_real); free(efieldy_imag);
	//-------------------------------------------
	data_shifting_Lx(row, col, Lx_real, Lx_imag);
	//-------------------------------------------
	Lx_s_real = (float **)malloc(sizeof(float *)*(col_s+1));
	for(i=0; i<(col_s+1); i++)
		Lx_s_real[i] = (float *)malloc(sizeof(float)*(row_s+1));
	Lx_s_imag = (float **)malloc(sizeof(float *)*(col_s+1));
	for(i=0; i<(col_s+1); i++)
		Lx_s_imag[i] = (float *)malloc(sizeof(float)*(row_s+1));
	//-------------------------------------------------------------------------
	Lx_data_shrink(row, col, Lx_real, Lx_imag, Lx_s_real, Lx_s_imag, Nfree, k);
	//-------------------------------------------------------------------------
	for(i=0; i<col+1; i++)
	{
		free(Lx_real[i]); free(Lx_imag[i]);
	}
		free(Lx_real); free(Lx_imag);

	//////////////////////
	//---- Hx field ----//
	//////////////////////
	hfieldx_real = (float **)malloc(sizeof(float *)*col);
	for(i=0; i<col; i++)
		hfieldx_real[i] = (float *)malloc(sizeof(float)*row);
	hfieldx_imag = (float **)malloc(sizeof(float *)*col);
	for(i=0; i<col; i++)
		hfieldx_imag[i] = (float *)malloc(sizeof(float)*row);
	//------------------------------------------------------------
	reading_source_Hx(row, col, hfieldx_real, hfieldx_imag);
	//------------------------------------------------------------
	Ny_real = (float **)malloc(sizeof(float *)*(col+1));
	for(i=0; i<(col+1); i++)
		Ny_real[i] = (float *)malloc(sizeof(float)*(row+1));
	Ny_imag = (float **)malloc(sizeof(float *)*(col+1));
	for(i=0; i<(col+1); i++)
		Ny_imag[i] = (float *)malloc(sizeof(float)*(row+1));
	//-----------------------------------------------------------------------------------
	fourier_transform_Ny_Hx(row, col, dir, Ny_real, Ny_imag, hfieldx_real, hfieldx_imag);
	//-----------------------------------------------------------------------------------
	for(i=0; i<col; i++)
	{
		free(hfieldx_real[i]); free(hfieldx_imag[i]);
	}
		free(hfieldx_real); free(hfieldx_imag);
	//-------------------------------------------
	data_shifting_Ny(row, col, Ny_real, Ny_imag);
	//-------------------------------------------
	Ny_s_real = (float **)malloc(sizeof(float *)*(col_s+1));
	for(i=0; i<(col_s+1); i++)
		Ny_s_real[i] = (float *)malloc(sizeof(float)*(row_s+1));
	Ny_s_imag = (float **)malloc(sizeof(float *)*(col_s+1));
	for(i=0; i<(col_s+1); i++)
		Ny_s_imag[i] = (float *)malloc(sizeof(float)*(row_s+1));
	//-------------------------------------------------------------------------
	Ny_data_shrink(row, col, Ny_real, Ny_imag, Ny_s_real, Ny_s_imag, Nfree, k);
	//-------------------------------------------------------------------------
	for(i=0; i<col+1; i++)
	{
		free(Ny_real[i]); free(Ny_imag[i]);
	}
		free(Ny_real); free(Ny_imag);

	//////////////////////
	//---- Hy field ----//
	//////////////////////
	hfieldy_real = (float **)malloc(sizeof(float *)*col);
	for(i=0; i<col; i++)
		hfieldy_real[i] = (float *)malloc(sizeof(float)*row);
	hfieldy_imag = (float **)malloc(sizeof(float *)*col);
	for(i=0; i<col; i++)
		hfieldy_imag[i] = (float *)malloc(sizeof(float)*row);
	//------------------------------------------------------------
	reading_source_Hy(row, col, hfieldy_real, hfieldy_imag);
	//------------------------------------------------------------
	Nx_real = (float **)malloc(sizeof(float *)*(col+1));
	for(i=0; i<(col+1); i++)
		Nx_real[i] = (float *)malloc(sizeof(float)*(row+1));
	Nx_imag = (float **)malloc(sizeof(float *)*(col+1));
	for(i=0; i<(col+1); i++)
		Nx_imag[i] = (float *)malloc(sizeof(float)*(row+1));
	//-----------------------------------------------------------------------------------
	fourier_transform_Nx_Hy(row, col, dir, Nx_real, Nx_imag, hfieldy_real, hfieldy_imag);
	//-----------------------------------------------------------------------------------
	for(i=0; i<col; i++)
	{
		free(hfieldy_real[i]); free(hfieldy_imag[i]);
	}
		free(hfieldy_real); free(hfieldy_imag);
	//-------------------------------------------
	data_shifting_Nx(row, col, Nx_real, Nx_imag);
	//-------------------------------------------
	Nx_s_real = (float **)malloc(sizeof(float *)*(col_s+1));
	for(i=0; i<(col_s+1); i++)
		Nx_s_real[i] = (float *)malloc(sizeof(float)*(row_s+1));
	Nx_s_imag = (float **)malloc(sizeof(float *)*(col_s+1));
	for(i=0; i<(col_s+1); i++)
		Nx_s_imag[i] = (float *)malloc(sizeof(float)*(row_s+1));
	//-------------------------------------------------------------------------
	Nx_data_shrink(row, col, Nx_real, Nx_imag, Nx_s_real, Nx_s_imag, Nfree, k);
	//-------------------------------------------------------------------------
	for(i=0; i<col+1; i++)
	{
		free(Nx_real[i]); free(Nx_imag[i]);
	}
		free(Nx_real); free(Nx_imag);

	Nt_s_real = (float **)malloc(sizeof(float *)*(col_s+1));
	for(i=0; i<(col_s+1); i++)
		Nt_s_real[i] = (float *)malloc(sizeof(float)*(row_s+1));
	Nt_s_imag = (float **)malloc(sizeof(float *)*(col_s+1));
	for(i=0; i<(col_s+1); i++)
		Nt_s_imag[i] = (float *)malloc(sizeof(float)*(row_s+1)); 
	Np_s_real = (float **)malloc(sizeof(float *)*(col_s+1));
	for(i=0; i<(col_s+1); i++)
		Np_s_real[i] = (float *)malloc(sizeof(float)*(row_s+1));
	Np_s_imag = (float **)malloc(sizeof(float *)*(col_s+1));
	for(i=0; i<(col_s+1); i++)
		Np_s_imag[i] = (float *)malloc(sizeof(float)*(row_s+1));
	Lt_s_real = (float **)malloc(sizeof(float *)*(col_s+1));
	for(i=0; i<(col_s+1); i++)
		Lt_s_real[i] = (float *)malloc(sizeof(float)*(row_s+1));
	Lt_s_imag = (float **)malloc(sizeof(float *)*(col_s+1));
	for(i=0; i<(col_s+1); i++)
		Lt_s_imag[i] = (float *)malloc(sizeof(float)*(row_s+1));
	Lp_s_real = (float **)malloc(sizeof(float *)*(col_s+1));
	for(i=0; i<(col_s+1); i++)
		Lp_s_real[i] = (float *)malloc(sizeof(float)*(row_s+1));
	Lp_s_imag = (float **)malloc(sizeof(float *)*(col_s+1));
	for(i=0; i<(col_s+1); i++)
		Lp_s_imag[i] = (float *)malloc(sizeof(float)*(row_s+1));
	cosp = (float **)malloc(sizeof(float *)*(col_s+1));
	for(i=0; i<(col_s+1); i++)
		cosp[i] = (float *)malloc(sizeof(float)*(row_s+1));
	sinp = (float **)malloc(sizeof(float *)*(col_s+1));
	for(i=0; i<(col_s+1); i++)
		sinp[i] = (float *)malloc(sizeof(float)*(row_s+1));
	cost = (float **)malloc(sizeof(float *)*(col_s+1));
	for(i=0; i<(col_s+1); i++)
		cost[i] = (float *)malloc(sizeof(float)*(row_s+1));
    	ex_real = (float **)malloc(sizeof(float *)*(col_s+1));
	for(i=0; i<(col_s+1); i++)
		ex_real[i] = (float *)malloc(sizeof(float)*(row_s+1));
	ex_imag = (float **)malloc(sizeof(float *)*(col_s+1));
	for(i=0; i<(col_s+1); i++)
		ex_imag[i] = (float *)malloc(sizeof(float)*(row_s+1));
	ey_real = (float **)malloc(sizeof(float *)*(col_s+1));
	for(i=0; i<(col_s+1); i++)
		ey_real[i] = (float *)malloc(sizeof(float)*(row_s+1));
	ey_imag = (float **)malloc(sizeof(float *)*(col_s+1));
	for(i=0; i<(col_s+1); i++)
		ey_imag[i] = (float *)malloc(sizeof(float)*(row_s+1));
	printf("memory allocation ok! ...\n");

	///// old version of momentum space intensity distribution function 
	FT_data_writing(Nx_s_real, Nx_s_imag, Ny_s_real, Ny_s_imag, Lx_s_real, Lx_s_imag, Ly_s_real, Ly_s_imag,row_s, col_s,  Nfree, k);
	
	//--------------------------------------------------------------------------------------------------
	set_radiation_variable(Nfree, k, Eta, Nx_s_real, Nx_s_imag, Ny_s_real, Ny_s_imag, Lx_s_real, Lx_s_imag, Ly_s_real, Ly_s_imag, Nt_s_real, Nt_s_imag, Np_s_real, Np_s_imag, Lt_s_real, Lt_s_imag, Lp_s_real, Lp_s_imag, cosp, sinp, cost, ex_real, ex_imag, ey_real, ey_imag);
	//--------------------------------------------------------------------------------------------------
	for(i=0; i<col_s+1; i++)
	{
		free(Nx_s_real[i]); free(Nx_s_imag[i]); free(Ny_s_real[i]); free(Ny_s_imag[i]);
		free(Lx_s_real[i]); free(Lx_s_imag[i]); free(Ly_s_real[i]); free(Ly_s_imag[i]);
	}
		free(Nx_s_real); free(Nx_s_imag); free(Ny_s_real); free(Ny_s_imag);
		free(Lx_s_real); free(Lx_s_imag); free(Ly_s_real); free(Ly_s_imag);

	radint = (float **)malloc(sizeof(float *)*(col_s+1));
	for(i=0; i<(col_s+1); i++)
		radint[i] = (float *)malloc(sizeof(float)*(row_s+1));
	etheta = (float **)malloc(sizeof(float *)*(col_s+1));
	for(i=0; i<(col_s+1); i++)
		etheta[i] = (float *)malloc(sizeof(float)*(row_s+1));
	ephi = (float **)malloc(sizeof(float *)*(col_s+1));
	for(i=0; i<(col_s+1); i++)
		ephi[i] = (float *)malloc(sizeof(float)*(row_s+1));
	eintx = (float **)malloc(sizeof(float *)*(col_s+1));
	for(i=0; i<(col_s+1); i++)
		eintx[i] = (float *)malloc(sizeof(float)*(row_s+1));
	einty = (float **)malloc(sizeof(float *)*(col_s+1));
	for(i=0; i<(col_s+1); i++)
		einty[i] = (float *)malloc(sizeof(float)*(row_s+1));
	//--------------------------------------------------------------------------------------------------
	calc_radiation(Nfree, k, Eta, Nt_s_real, Nt_s_imag, Np_s_real, Np_s_imag, Lt_s_real, Lt_s_imag, Lp_s_real, Lp_s_imag, cosp, sinp, cost, ex_real, ex_imag, ey_real, ey_imag, radint, etheta, ephi, eintx, einty, &P_tot, &P_the, &P_phi);
	//--------------------------------------------------------------------------------------------------
	for(i=0; i<col_s+1; i++)
	{
		free(Nt_s_real[i]); free(Nt_s_imag[i]); free(Np_s_real[i]); free(Np_s_imag[i]);
		free(Lt_s_real[i]); free(Lt_s_imag[i]); free(Lp_s_real[i]); free(Lp_s_imag[i]);
	}
		free(Nt_s_real); free(Nt_s_imag); free(Np_s_real); free(Np_s_imag);
		free(Lt_s_real); free(Lt_s_imag); free(Lp_s_real); free(Lp_s_imag);

	//-----------------------------------------------------------------------------------------
	print_radiation(Nfree, k, Eta, radint, etheta, ephi, eintx, einty, &P_tot, &P_the, &P_phi);
	//-----------------------------------------------------------------------------------------

	//calc_polar(NA, Nfree);
	for(i=0; i<col_s+1; i++)
	{
		free(ex_real[i]); free(ey_real[i]); free(ex_imag[i]); free(ey_imag[i]);
	}
	free(ex_real); free(ey_real); free(ex_imag); free(ey_imag);
	free(cosp); free(sinp); free(cost);
	free(radint); free(etheta); free(ephi);
	free(eintx); free(einty);
	//free(Eu_real); free(Eu_imag); free(Eu_int);
	//free(polarization);
}

int FFT2D(struct COMPLEX **c,int nx,int ny,int dir)
{
   	int i,j;
	int m;
   	float *real,*imag;

   	/* Transform the rows */
   	real = (float *)malloc(nx * sizeof(float));
   	imag = (float *)malloc(nx * sizeof(float));

	m = pow_finder(nx);
   	for (j=0;j<ny;j++)
	{
      		for (i=0;i<nx;i++)
			{
         		real[i] = c[i][j].real;
         		imag[i] = c[i][j].imag;
      		}
			printf("i=%d,j=%d \r",i,j);
      		FFT1D(dir,m,real,imag);
      		for (i=0;i<nx;i++)
			{
         		c[i][j].real = real[i];
         		c[i][j].imag = imag[i];
      		}
   	}
	printf("\n");
   	free(real);
   	free(imag);

   	/* Transform the columns */
   	real = (float *)malloc(ny * sizeof(float));
   	imag = (float *)malloc(ny * sizeof(float));
   	
	m = pow_finder(ny);
   	for (i=0;i<nx;i++) 
	{
      		for (j=0;j<ny;j++)
			{
         		real[j] = c[i][j].real;
         		imag[j] = c[i][j].imag;
      		}
			printf("i=%d,j=%d \r",i,j);
      		FFT1D(dir,m,real,imag);
      		for (j=0;j<ny;j++) 
			{
         		c[i][j].real = real[j];
         		c[i][j].imag = imag[j];
      		}
   	}
	printf("\n");
   	free(real);
   	free(imag);

   	return(TRUE);
}

int FFT1D(int dir,int m,float *x,float *y)
{
   	long nn,i,i1,j,k,i2,l,l1,l2;
   	float c1,c2,tx,ty,t1,t2,u1,u2,z;

   	/* Calculate the number of points */
   	nn = 1;
   	for (i=0;i<m;i++)
      		nn *= 2;

   	/* Do the bit reversal */
   	i2 = nn >> 1;
   	j = 0;
   	for (i=0;i<nn-1;i++)
	{
      		if (i < j)
			{
         		tx = x[i];
         		ty = y[i];
         		x[i] = x[j];
         		y[i] = y[j];
         		x[j] = tx;
         		y[j] = ty;
      		}
      		k = i2;
      		while (k <= j)
			{
         		j -= k;
         		k >>= 1;
     	 	}
      		j += k;
   	}

   	/* Compute the FFT */
   	c1 = -1.0;
   	c2 = 0.0;
   	l2 = 1;
   	for (l=0;l<m;l++)
	{
      		l1 = l2;
      		l2 <<= 1;
      		u1 = 1.0;
      		u2 = 0.0;
      		for (j=0;j<l1;j++)
			{
         		for (i=j;i<nn;i+=l2)
				{
            			i1 = i + l1;
            			t1 = u1 * x[i1] - u2 * y[i1];
            			t2 = u1 * y[i1] + u2 * x[i1];
            			x[i1] = x[i] - t1;
            			y[i1] = y[i] - t2;
            			x[i] += t1;
           			y[i] += t2;
         		}
         		z =  u1 * c1 - u2 * c2;
         		u2 = u1 * c2 + u2 * c1;
         		u1 = z;
      		}
      		c2 = sqrt((1.0 - c1) / 2.0);
      		if (dir == 1)
         	c2 = -c2;
      		c1 = sqrt((1.0 + c1) / 2.0);
   	}

   	/* Scaling for forward transform */
   	if (dir == 1)
	{
      		for (i=0;i<nn;i++)
			{
         		x[i] /= (float)nn;
         		y[i] /= (float)nn;
      		}
   	}

   	return(TRUE);
}

int pow_finder(int N)
{
	int power=0;
	
	while( N != 1 )
	{
		N = (int)(N/2);
		power ++;
	}

	return(power);
}

void make_file_name(int mm)
{
	sprintf(name_freq,".ri%02d",mm);

	sprintf(name_Ex_real,"Ex_real");
	sprintf(name_Ex_imag,"Ex_imag");
	strcat(name_Ex_real,name_freq);
	strcat(name_Ex_imag,name_freq);

	sprintf(name_Ey_real,"Ey_real");
	sprintf(name_Ey_imag,"Ey_imag");
	strcat(name_Ey_real,name_freq);
	strcat(name_Ey_imag,name_freq);

	sprintf(name_Hx_real,"Hx_real");
	sprintf(name_Hx_imag,"Hx_imag");
	strcat(name_Hx_real,name_freq);
	strcat(name_Hx_imag,name_freq);

	sprintf(name_Hy_real,"Hy_real");
	sprintf(name_Hy_imag,"Hy_imag");
	strcat(name_Hy_real,name_freq);
	strcat(name_Hy_imag,name_freq);
	
	sprintf(FFT_E_int2,"FFT_E_int2_");
	sprintf(FFT_E_log,"FFT_E_log_");
	strcat(FFT_E_int2,name_freq);
	strcat(FFT_E_log,name_freq);
	sprintf(FFT_H_int2,"FFT_H_int2_");
	sprintf(FFT_H_log,"FFT_H_log_");
	strcat(FFT_H_int2,name_freq);
	strcat(FFT_H_log,name_freq);

	sprintf(radiation_tot,"rad_tot_");
	sprintf(radiation_Et,"rad_Et_");
	sprintf(radiation_Ep,"rad_Ep_");
	sprintf(radiation_Ex,"rad_Ex_");
	sprintf(radiation_Ey,"rad_Ey_");
	strcat(radiation_tot,name_freq);
	strcat(radiation_Et,name_freq);
	strcat(radiation_Ep,name_freq);
	strcat(radiation_Ex,name_freq);
	strcat(radiation_Ey,name_freq);

	sprintf(polar,"polarizaton_");
	sprintf(P_tot_name,"P_tot_");
	sprintf(P_the_name,"P_the_");
	sprintf(P_phi_name,"P_phi_");
	strcat(polar,name_freq);
	strcat(P_tot_name,name_freq);
	strcat(P_the_name,name_freq);
	strcat(P_phi_name,name_freq);
}

void set_global_variable(int NROW, float OMEGA, float Nfree, int *POLROT, int *dir, float *Eta, float *k)
{
	*POLROT = 36;
	*dir = 1; // forward transformation
	*Eta = 377/Nfree;
	*k = floor(NROW*OMEGA/lattice_x);
	printf("radius k=%4.1f\n",*k);

	printf("Now... Caculating far-field pattern......\n");
}

//////////////////////////////
/// Reading source         ///
//////////////////////////////

void reading_source_Ex(int row, int col, float **efieldx_real, float **efieldx_imag)
{
	int i, j; 
	FILE *stream;

	stream = fopen(name_Ex_real,"rt");
	for(j=row-1; j>=0; j--)
	{
		for(i=0; i<col; i++)
		{
			fscanf(stream, "%s", string);
			if( strcmp("nan",string) == 0 )
				efieldx_real[i][j] = 0.0;
			else
				efieldx_real[i][j] = atof(string);
		}
	}
	fclose(stream);

	stream = fopen(name_Ex_imag,"rt");
	for(j=row-1; j>=0; j--)
	{
		for(i=0; i<col; i++)
		{
			fscanf(stream, "%s", string);
			if( strcmp("nan",string) == 0 )
				efieldx_imag[i][j] = 0.0;
			else
				efieldx_imag[i][j] = atof(string);
		}
	}
	fclose(stream);

	printf("reading Ex check...\n");
}

void reading_source_Ey(int row, int col, float **efieldy_real, float **efieldy_imag)
{
	int i, j; 
	FILE *stream;

	stream = fopen(name_Ey_real,"rt");
	for(j=row-1; j>=0; j--)
	{
		for(i=0; i<col; i++)
		{
			fscanf(stream, "%s", string);
			if( strcmp("nan",string) == 0 )
				efieldy_real[i][j] = 0.0;
			else
				efieldy_real[i][j] = atof(string);
		}
	}
	fclose(stream);

	stream = fopen(name_Ey_imag,"rt");
	for(j=row-1; j>=0; j--)
	{
		for(i=0; i<col; i++)
		{
			fscanf(stream, "%s", string);
			if( strcmp("nan",string) == 0 )
				efieldy_imag[i][j] = 0.0;
			else
				efieldy_imag[i][j] = atof(string);
		}
	}
	fclose(stream);

	printf("reading Ey check...\n");
}

void reading_source_Hx(int row, int col, float **hfieldx_real, float **hfieldx_imag)
{
	int i, j; 
	FILE *stream;

	stream = fopen(name_Hx_real,"rt");
	for(j=row-1; j>=0; j--)
	{
		for(i=0; i<col; i++)
		{
			fscanf(stream, "%s", string);
			if( strcmp("nan",string) == 0 )
				hfieldx_real[i][j] = 0.0;
			else
				hfieldx_real[i][j] = atof(string);
		}
	}
	fclose(stream);

	stream = fopen(name_Hx_imag,"rt");
	for(j=row-1; j>=0; j--)
	{
		for(i=0; i<col; i++)
		{
			fscanf(stream, "%s", string);
			if( strcmp("nan",string) == 0 )
				hfieldx_imag[i][j] = 0.0;
			else
				hfieldx_imag[i][j] = atof(string);
		}
	}
	fclose(stream);

	printf("reading Hx check...\n");
}

void reading_source_Hy(int row, int col, float **hfieldy_real, float **hfieldy_imag)
{
	int i, j; 
	FILE *stream;

	stream = fopen(name_Hy_real,"rt");
	for(j=row-1; j>=0; j--)
	{
		for(i=0; i<col; i++)
		{
			fscanf(stream, "%s", string);
			if( strcmp("nan",string) == 0 )
				hfieldy_real[i][j] = 0.0;
			else
				hfieldy_real[i][j] = atof(string);
		}
	}
	fclose(stream);

	stream = fopen(name_Hy_imag,"rt");
	for(j=row-1; j>=0; j--)
	{
		for(i=0; i<col; i++)
		{
			fscanf(stream, "%s", string);
			if( strcmp("nan",string) == 0 )
				hfieldy_imag[i][j] = 0.0;
			else
				hfieldy_imag[i][j] = atof(string);
		}
	}
	fclose(stream);

	printf("reading Hy check...\n");
}

//////////////////////////////
/// Fourier Transformation ///
//////////////////////////////

void fourier_transform_Nx_Hy(int row, int col, int dir, float **Nx_real, float **Nx_imag, float **hfieldy_real, float **hfieldy_imag)
{
	int i, j; 
	struct COMPLEX **FFT; // FFT input and output
	int FFT_return;

	FFT = (struct COMPLEX **)malloc(sizeof(struct COMPLEX *)*col);
	for(i=0; i<col; i++)
		FFT[i] = (struct COMPLEX *)malloc(sizeof(struct COMPLEX)*row);
	printf("FFT allocation complete....\n");

	///------------( Nx calculation )------------
	for(j=row-1; j>=0; j--)
	{
		for(i=0; i<col; i++)
		{
			FFT[i][j].real = hfieldy_real[i][j];
			FFT[i][j].imag = hfieldy_imag[i][j];
		}
	}	
	printf("starting Nx caculation (FFT)...\n");
	FFT_return = FFT2D(FFT, col, row, dir);
	printf("FFT check... (FFT_return=%d)\n",FFT_return);

	for(j=row-1; j>=0; j--)
	{
		for(i=0; i<col; i++)
		{
			Nx_real[i][j] = -FFT[i][j].real;
			Nx_imag[i][j] = -FFT[i][j].imag;
		}
	}

	for(i=0; i<col; i++)
	{
		free(FFT[i]);
	}
	free(FFT);
}

void fourier_transform_Ny_Hx(int row, int col, int dir, float **Ny_real, float **Ny_imag, float **hfieldx_real, float **hfieldx_imag)
{
	int i, j; 
	struct COMPLEX **FFT; // FFT input and output
	int FFT_return;

	FFT = (struct COMPLEX **)malloc(sizeof(struct COMPLEX *)*col);
	for(i=0; i<col; i++)
		FFT[i] = (struct COMPLEX *)malloc(sizeof(struct COMPLEX)*row);
	printf("FFT allocation complete....\n");

	///------------( Ny calculation )------------
	for(j=row-1; j>=0; j--)
	{
		for(i=0; i<col; i++)
		{
			FFT[i][j].real = hfieldx_real[i][j];
			FFT[i][j].imag = hfieldx_imag[i][j];
		}
	}
	printf("starting Ny caculation (FFT)...\n");
	FFT_return = FFT2D(FFT, col, row, dir);
	printf("FFT check... (FFT_return=%d)\n",FFT_return);

	for(j=row-1; j>=0; j--)
	{
		for(i=0; i<col; i++)
		{
			Ny_real[i][j] = FFT[i][j].real;
			Ny_imag[i][j] = FFT[i][j].imag;
		}
	}

	for(i=0; i<col; i++)
	{
		free(FFT[i]);
	}
	free(FFT);
}

void fourier_transform_Lx_Ey(int row, int col, int dir, float **Lx_real, float **Lx_imag, float **efieldy_real, float **efieldy_imag)
{
	int i, j; 
	struct COMPLEX **FFT; // FFT input and output
	int FFT_return;

	FFT = (struct COMPLEX **)malloc(sizeof(struct COMPLEX *)*col);
	for(i=0; i<col; i++)
		FFT[i] = (struct COMPLEX *)malloc(sizeof(struct COMPLEX)*row);
	printf("FFT allocation complete....\n");

	///------------( Lx calculation )------------
	for(j=row-1; j>=0; j--)
	{
		for(i=0; i<col; i++)
		{
			FFT[i][j].real = efieldy_real[i][j];
			FFT[i][j].imag = efieldy_imag[i][j];
		}
	}
	printf("starting Lx caculation (FFT)...\n");
	FFT_return = FFT2D(FFT, col, row, dir);
	printf("FFT check... (FFT_return=%d)\n",FFT_return);

	for(j=row-1; j>=0; j--)
	{
		for(i=0; i<col; i++)
		{
			Lx_real[i][j] = FFT[i][j].real;
			Lx_imag[i][j] = FFT[i][j].imag;
		}
	}

	for(i=0; i<col; i++)
	{
		free(FFT[i]);
	}
	free(FFT);
}

void fourier_transform_Ly_Ex(int row, int col, int dir, float **Ly_real, float **Ly_imag, float **efieldx_real, float **efieldx_imag)
{
	int i, j; 
	struct COMPLEX **FFT; // FFT input and output
	int FFT_return;

	FFT = (struct COMPLEX **)malloc(sizeof(struct COMPLEX *)*col);
	for(i=0; i<col; i++)
		FFT[i] = (struct COMPLEX *)malloc(sizeof(struct COMPLEX)*row);
	printf("FFT allocation complete....\n");

	///------------( Ly calculation )------------
	for(j=row-1; j>=0; j--)
	{
		for(i=0; i<col; i++)
		{
			FFT[i][j].real = efieldx_real[i][j];
			FFT[i][j].imag = efieldx_imag[i][j];
		}
	}
	printf("starting Ly caculation (FFT)...\n");
	FFT_return = FFT2D(FFT, col, row, dir);
	printf("FFT check... (FFT_return=%d)\n",FFT_return);

	for(j=row-1; j>=0; j--)
	{
		for(i=0; i<col; i++)
		{
			Ly_real[i][j] = -FFT[i][j].real;
			Ly_imag[i][j] = -FFT[i][j].imag;
		}
	}

	for(i=0; i<col; i++)
	{
		free(FFT[i]);
	}
	free(FFT);
}

//////////////////////////////
/// Data shifting          ///
//////////////////////////////

void data_shifting_Nx(int row, int col, float **Nx_real, float **Nx_imag)
{
	int i,j;
	int m,n;
	float **temp_real, **temp_imag;
	int half_row, half_col;

	half_col = (int)(col/2);
	half_row = (int)(row/2);

	temp_real = (float **)malloc(sizeof(float *)*(col+1));
	for(i=0; i<(col+1); i++)
		temp_real[i] = (float *)malloc(sizeof(float)*(row+1));
	temp_imag = (float **)malloc(sizeof(float *)*(col+1));
	for(i=0; i<(col+1); i++)
		temp_imag[i] = (float *)malloc(sizeof(float)*(row+1));

	printf("Nx shifting\n");

	///------------( Nx shifting )------------
	for(j=row-1; j>=0; j--)
	{
		for(i=0; i<col; i++)
		{
			temp_real[i][j] = Nx_real[i][j];
			temp_imag[i][j] = Nx_imag[i][j];
		}
	}
	for(j=row; j>=0; j--)
	{
		for(i=0; i<(col+1); i++)
		{
			m = i-half_col;
			n = j-half_row;
			if(m>=0 && n>=0)
			{
				Nx_real[i][j] = temp_real[m][n];
				Nx_imag[i][j] = temp_imag[m][n];
			}
			if(m>=0 && n<0)
			{
				Nx_real[i][j] = temp_real[m][row+n];
				Nx_imag[i][j] = temp_imag[m][row+n];
			}
			if(m<0 && n>=0)
			{
				Nx_real[i][j] = temp_real[col+m][n];
				Nx_imag[i][j] = temp_imag[col+m][n];
			}
			if(m<0 && n<0)
			{
				Nx_real[i][j] = temp_real[col+m][row+n];
				Nx_imag[i][j] = temp_imag[col+m][row+n];
			}
		}
	}

	for(i=0; i<(col+1); i++)
	{
		free(temp_real[i]); free(temp_imag[i]);
	}
	free(temp_real); free(temp_imag);
}

void data_shifting_Ny(int row, int col, float **Ny_real, float **Ny_imag)
{
	int i,j;
	int m,n;
	float **temp_real, **temp_imag;
	int half_row, half_col;

	half_col = (int)(col/2);
	half_row = (int)(row/2);

	temp_real = (float **)malloc(sizeof(float *)*(col+1));
	for(i=0; i<(col+1); i++)
		temp_real[i] = (float *)malloc(sizeof(float)*(row+1));
	temp_imag = (float **)malloc(sizeof(float *)*(col+1));
	for(i=0; i<(col+1); i++)
		temp_imag[i] = (float *)malloc(sizeof(float)*(row+1));

	printf("Ny shifting\n");

	///------------( Ny shifting )------------
	for(j=row-1; j>=0; j--)
	{
		for(i=0; i<col; i++)
		{
			temp_real[i][j] = Ny_real[i][j];
			temp_imag[i][j] = Ny_imag[i][j];
		}
	}
	for(j=row; j>=0; j--)
	{
		for(i=0; i<(col+1); i++)
		{
			m = i-half_col;
			n = j-half_row;
			if(m>=0 && n>=0)
			{
				Ny_real[i][j] = temp_real[m][n];
				Ny_imag[i][j] = temp_imag[m][n];
			}
			if(m>=0 && n<0)
			{
				Ny_real[i][j] = temp_real[m][row+n];
				Ny_imag[i][j] = temp_imag[m][row+n];
			}
			if(m<0 && n>=0)
			{
				Ny_real[i][j] = temp_real[col+m][n];
				Ny_imag[i][j] = temp_imag[col+m][n];
			}
			if(m<0 && n<0)
			{
				Ny_real[i][j] = temp_real[col+m][row+n];
				Ny_imag[i][j] = temp_imag[col+m][row+n];
			}
		}
	}

	for(i=0; i<(col+1); i++)
	{
		free(temp_real[i]); free(temp_imag[i]);
	}
	free(temp_real); free(temp_imag);
}

void data_shifting_Lx(int row, int col, float **Lx_real, float **Lx_imag)
{
	int i,j;
	int m,n;
	float **temp_real, **temp_imag;
	int half_row, half_col;

	half_col = (int)(col/2);
	half_row = (int)(row/2);

	temp_real = (float **)malloc(sizeof(float *)*(col+1));
	for(i=0; i<(col+1); i++)
		temp_real[i] = (float *)malloc(sizeof(float)*(row+1));
	temp_imag = (float **)malloc(sizeof(float *)*(col+1));
	for(i=0; i<(col+1); i++)
		temp_imag[i] = (float *)malloc(sizeof(float)*(row+1));

	printf("Lx shifting\n");

	///------------( Lx shifting )------------
	for(j=row-1; j>=0; j--)
	{
		for(i=0; i<col; i++)
		{
			temp_real[i][j] = Lx_real[i][j];
			temp_imag[i][j] = Lx_imag[i][j];
		}
	}
	for(j=row; j>=0; j--)
	{
		for(i=0; i<(col+1); i++)
		{
			m = i-half_col;
			n = j-half_row;
			if(m>=0 && n>=0)
			{
				Lx_real[i][j] = temp_real[m][n];
				Lx_imag[i][j] = temp_imag[m][n];
			}
			if(m>=0 && n<0)
			{
				Lx_real[i][j] = temp_real[m][row+n];
				Lx_imag[i][j] = temp_imag[m][row+n];
			}
			if(m<0 && n>=0)
			{
				Lx_real[i][j] = temp_real[col+m][n];
				Lx_imag[i][j] = temp_imag[col+m][n];
			}
			if(m<0 && n<0)
			{
				Lx_real[i][j] = temp_real[col+m][row+n];
				Lx_imag[i][j] = temp_imag[col+m][row+n];
			}
		}
	}

	for(i=0; i<(col+1); i++)
	{
		free(temp_real[i]); free(temp_imag[i]);
	}
	free(temp_real); free(temp_imag);
}

void data_shifting_Ly(int row, int col, float **Ly_real, float **Ly_imag)
{
	int i,j;
	int m,n;
	float **temp_real, **temp_imag;
	int half_row, half_col;

	half_col = (int)(col/2);
	half_row = (int)(row/2);

	temp_real = (float **)malloc(sizeof(float *)*(col+1));
	for(i=0; i<(col+1); i++)
		temp_real[i] = (float *)malloc(sizeof(float)*(row+1));
	temp_imag = (float **)malloc(sizeof(float *)*(col+1));
	for(i=0; i<(col+1); i++)
		temp_imag[i] = (float *)malloc(sizeof(float)*(row+1));

	printf("Ly shifting\n");

	///------------( Ly shifting )------------
	for(j=row-1; j>=0; j--)
	{
		for(i=0; i<col; i++)
		{
			temp_real[i][j] = Ly_real[i][j];
			temp_imag[i][j] = Ly_imag[i][j];
		}
	}
	for(j=row; j>=0; j--)
	{
		for(i=0; i<(col+1); i++)
		{
			m = i-half_col;
			n = j-half_row;
			if(m>=0 && n>=0)
			{
				Ly_real[i][j] = temp_real[m][n];
				Ly_imag[i][j] = temp_imag[m][n];
			}
			if(m>=0 && n<0)
			{
				Ly_real[i][j] = temp_real[m][row+n];
				Ly_imag[i][j] = temp_imag[m][row+n];
			}
			if(m<0 && n>=0)
			{
				Ly_real[i][j] = temp_real[col+m][n];
				Ly_imag[i][j] = temp_imag[col+m][n];
			}
			if(m<0 && n<0)
			{
				Ly_real[i][j] = temp_real[col+m][row+n];
				Ly_imag[i][j] = temp_imag[col+m][row+n];
			}
		}
	}

	for(i=0; i<(col+1); i++)
	{
		free(temp_real[i]); free(temp_imag[i]);
	}
	free(temp_real); free(temp_imag);
}

//////////////////////////////
/// Data shrink            ///
//////////////////////////////

void Nx_data_shrink(int row, int col, float **Nx_real, float **Nx_imag, float **Nx_s_real, float **Nx_s_imag, float Nfree, float k)
{
	int i,j;
	int half_row, half_col;
	int row_s, col_s;           //shrinked row & col
	int half_row_s, half_col_s;

	half_col = (int)(col/2);
	half_row = (int)(row/2);

	row_s = (int)(10*Nfree*k);
	col_s = (int)(10*Nfree*k);

	half_col_s = (int)(col_s/2);
	half_row_s = (int)(row_s/2);

	for(j=(half_row)+(half_row_s); j>=(half_row)-(half_row_s); j--)
	{
		for(i=(half_col)-(half_col_s); i<=(half_col)+(half_col_s); i++)
		{
			Nx_s_real[i-(half_col)+(half_col_s)][j-(half_row)+(half_row_s)] = Nx_real[i][j];
			Nx_s_imag[i-(half_col)+(half_col_s)][j-(half_row)+(half_row_s)] = Nx_imag[i][j];
		}
	}
}

void Ny_data_shrink(int row, int col, float **Ny_real, float **Ny_imag, float **Ny_s_real, float **Ny_s_imag, float Nfree, float k)
{
	int i,j;
	int half_row, half_col;
	int row_s, col_s;           //shrinked row & col
	int half_row_s, half_col_s;

	half_col = (int)(col/2);
	half_row = (int)(row/2);

	row_s = (int)(10*Nfree*k);
	col_s = (int)(10*Nfree*k);

	half_col_s = (int)(col_s/2);
	half_row_s = (int)(row_s/2);

	for(j=(half_row)+(half_row_s); j>=(half_row)-(half_row_s); j--)
	{
		for(i=(half_col)-(half_col_s); i<=(half_col)+(half_col_s); i++)
		{
			Ny_s_real[i-(half_col)+(half_col_s)][j-(half_row)+(half_row_s)] = Ny_real[i][j];
			Ny_s_imag[i-(half_col)+(half_col_s)][j-(half_row)+(half_row_s)] = Ny_imag[i][j];
		}
	}
}

void Lx_data_shrink(int row, int col, float **Lx_real, float **Lx_imag, float **Lx_s_real, float **Lx_s_imag, float Nfree, float k)
{
	int i,j;
	int half_row, half_col;
	int row_s, col_s;           //shrinked row & col
	int half_row_s, half_col_s;

	half_col = (int)(col/2);
	half_row = (int)(row/2);

	row_s = (int)(10*Nfree*k);
	col_s = (int)(10*Nfree*k);

	half_col_s = (int)(col_s/2);
	half_row_s = (int)(row_s/2);

	for(j=(half_row)+(half_row_s); j>=(half_row)-(half_row_s); j--)
	{
		for(i=(half_col)-(half_col_s); i<=(half_col)+(half_col_s); i++)
		{
			Lx_s_real[i-(half_col)+(half_col_s)][j-(half_row)+(half_row_s)] = Lx_real[i][j];
			Lx_s_imag[i-(half_col)+(half_col_s)][j-(half_row)+(half_row_s)] = Lx_imag[i][j];
		}
	}
}

void Ly_data_shrink(int row, int col, float **Ly_real, float **Ly_imag, float **Ly_s_real, float **Ly_s_imag, float Nfree, float k)
{
	int i,j;
	int half_row, half_col;
	int row_s, col_s;           //shrinked row & col
	int half_row_s, half_col_s;

	half_col = (int)(col/2);
	half_row = (int)(row/2);

	row_s = (int)(10*Nfree*k);
	col_s = (int)(10*Nfree*k);

	half_col_s = (int)(col_s/2);
	half_row_s = (int)(row_s/2);

	for(j=(half_row)+(half_row_s); j>=(half_row)-(half_row_s); j--)
	{
		for(i=(half_col)-(half_col_s); i<=(half_col)+(half_col_s); i++)
		{
			Ly_s_real[i-(half_col)+(half_col_s)][j-(half_row)+(half_row_s)] = Ly_real[i][j];
			Ly_s_imag[i-(half_col)+(half_col_s)][j-(half_row)+(half_row_s)] = Ly_imag[i][j];
		}
	}
}

//////////////////////////////
/// FFT E & FFT H          ///
//////////////////////////////

void FT_data_writing(float **Nx_s_real, float **Nx_s_imag, float **Ny_s_real, float **Ny_s_imag, float **Lx_s_real, float **Lx_s_imag, float **Ly_s_real, float **Ly_s_imag, int row_s, int col_s, float Nfree, float k)
{
	int half_row_s, half_col_s;
	float FFT_Emax, FFT_Hmax;  // FFT normalization
	float temp;
	FILE *stream;
	int i,j;

	half_col_s = (int)(col_s/2);
	half_row_s = (int)(row_s/2);

	FFT_Emax = 0.0;
	FFT_Hmax = 0.0;

	//////////[ FFT(Ex)^2 + FFT(Ey)^2 ]///////////
	for(j=row_s; j>=0; j--)
	{
		for(i=0; i<=(col_s); i++)
		{
				temp = Lx_s_real[i][j]*Lx_s_real[i][j]+Ly_s_real[i][j]*Ly_s_real[i][j]+Lx_s_imag[i][j]*Lx_s_imag[i][j]+Ly_s_imag[i][j]*Ly_s_imag[i][j];
				if( temp > FFT_Emax)
					FFT_Emax = temp;
		}
	}

	stream = fopen(FFT_E_int2, "wt");
	for(j=row_s; j>=0; j--)
	{
		for(i=0; i<=(col_s); i++)
		{
			fprintf(stream,"%g\t",(Lx_s_real[i][j]*Lx_s_real[i][j]+Ly_s_real[i][j]*Ly_s_real[i][j]+Lx_s_imag[i][j]*Lx_s_imag[i][j]+Ly_s_imag[i][j]*Ly_s_imag[i][j])/FFT_Emax);
		}
		fprintf(stream,"\n");
	}
	fclose(stream);

	stream = fopen(FFT_E_log, "wt");
	for(j=row_s; j>=0; j--)
	{
		for(i=0; i<=(col_s); i++)
		{
			fprintf(stream,"%g\t",log10((Lx_s_real[i][j]*Lx_s_real[i][j]+Ly_s_real[i][j]*Ly_s_real[i][j]+Lx_s_imag[i][j]*Lx_s_imag[i][j]+Ly_s_imag[i][j]*Ly_s_imag[i][j])/FFT_Emax));
		}
		fprintf(stream,"\n");
	}
	fclose(stream);

	//////////[ FFT(Hx)^2 + FFT(Hy)^2 ]///////////
	for(j=row_s; j>=0; j--)
	{
		for(i=0; i<=(col_s); i++)
		{
				temp = Nx_s_real[i][j]*Nx_s_real[i][j]+Ny_s_real[i][j]*Ny_s_real[i][j]+Nx_s_imag[i][j]*Nx_s_imag[i][j]+Ny_s_imag[i][j]*Ny_s_imag[i][j];
				if( temp > FFT_Hmax)
					FFT_Hmax = temp;
		}
	}

	stream = fopen(FFT_H_int2, "wt");
	for(j=row_s; j>=0; j--)
	{
		for(i=0; i<=(col_s); i++)
		{
			fprintf(stream,"%g\t",(Nx_s_real[i][j]*Nx_s_real[i][j]+Ny_s_real[i][j]*Ny_s_real[i][j]+Nx_s_imag[i][j]*Nx_s_imag[i][j]+Ny_s_imag[i][j]*Ny_s_imag[i][j])/FFT_Hmax);
		}
		fprintf(stream,"\n");
	}
	fclose(stream);

	stream = fopen(FFT_H_log, "wt");
	for(j=row_s; j>=0; j--)
	{
		for(i=0; i<=(col_s); i++)
		{
			fprintf(stream,"%g\t",log10((Nx_s_real[i][j]*Nx_s_real[i][j]+Ny_s_real[i][j]*Ny_s_real[i][j]+Nx_s_imag[i][j]*Nx_s_imag[i][j]+Ny_s_imag[i][j]*Ny_s_imag[i][j])/FFT_Hmax));
		}
		fprintf(stream,"\n");
	}
	fclose(stream);
}	

void set_radiation_variable(float Nfree, float k, float Eta, float **Nx_s_real, float **Nx_s_imag, float **Ny_s_real, float **Ny_s_imag, float **Lx_s_real, float **Lx_s_imag, float **Ly_s_real, float **Ly_s_imag, float **Nt_s_real, float **Nt_s_imag, float **Np_s_real, float **Np_s_imag, float **Lt_s_real, float **Lt_s_imag, float **Lp_s_real, float **Lp_s_imag, float **cosp, float **sinp, float **cost, float **ex_real, float **ex_imag, float **ey_real, float **ey_imag)
{
	int i,j;
	int row_s, col_s;           //shrinked row & col
	int half_row_s, half_col_s;

	row_s = (int)(10*Nfree*k);
	col_s = (int)(10*Nfree*k);

	half_col_s = (int)(col_s/2);
	half_row_s = (int)(row_s/2);

	printf("calculating N(theta), N(phi).....\n");

	for(j=row_s; j>=0; j--)
	{
		for(i=0; i<(col_s+1); i++)
		{
			if( (i-half_col_s)*(i-half_col_s)+(j-half_row_s)*(j-half_row_s) < Nfree*Nfree*k*k )
			{
				cosp[i][j] = (i-half_col_s)/sqrt( (i-half_col_s)*(i-half_col_s) + (j-half_row_s)*(j-half_row_s) );
				sinp[i][j] = (j-half_row_s)/sqrt( (i-half_col_s)*(i-half_col_s) + (j-half_row_s)*(j-half_row_s) );
				cost[i][j] = sqrt( 1 - ((i-half_col_s)*(i-half_col_s) + (j-half_row_s)*(j-half_row_s))/(Nfree*Nfree*k*k) );
			}
			else
			{
				cosp[i][j] = 0.0;
				sinp[i][j] = 0.0;
				cost[i][j] = 0.0;
			}
		}
	}
	cosp[half_col_s][half_row_s] = 1/sqrt(2);
	sinp[half_col_s][half_row_s] = 1/sqrt(2);
	cost[half_col_s][half_row_s] = 1;

	printf("trigonometric fucntion define...\n");

	for(j=row_s; j>=0; j--)
	{
		for(i=0; i<(col_s+1); i++)
		{
			Nt_s_real[i][j] = (Nx_s_real[i][j]*cosp[i][j]+Ny_s_real[i][j]*sinp[i][j])*cost[i][j];
			Nt_s_imag[i][j] = (Nx_s_imag[i][j]*cosp[i][j]+Ny_s_imag[i][j]*sinp[i][j])*cost[i][j];
			Np_s_real[i][j] = (-Nx_s_real[i][j]*sinp[i][j]+Ny_s_real[i][j]*cosp[i][j]);
			Np_s_imag[i][j] = (-Nx_s_imag[i][j]*sinp[i][j]+Ny_s_imag[i][j]*cosp[i][j]);
			Lt_s_real[i][j] = (Lx_s_real[i][j]*cosp[i][j]+Ly_s_real[i][j]*sinp[i][j])*cost[i][j];
			Lt_s_imag[i][j] = (Lx_s_imag[i][j]*cosp[i][j]+Ly_s_imag[i][j]*sinp[i][j])*cost[i][j];
			Lp_s_real[i][j] = (-Lx_s_real[i][j]*sinp[i][j]+Ly_s_real[i][j]*cosp[i][j]);
			Lp_s_imag[i][j] = (-Lx_s_imag[i][j]*sinp[i][j]+Ly_s_imag[i][j]*cosp[i][j]);
		}
	}
	printf("Nt, Np, Lt, Lp ok!.....\n");

	for(j=row_s; j>=0; j--)
	{
		for(i=0; i<(col_s+1); i++)
		{
			ex_real[i][j] = -(Np_s_real[i][j] - Lt_s_real[i][j]/Eta)*sinp[i][j] + (Nt_s_real[i][j] + Lp_s_real[i][j]/377)*cost[i][j]*cosp[i][j];
			ex_imag[i][j] = -(Np_s_imag[i][j] - Lt_s_imag[i][j]/Eta)*sinp[i][j] + (Nt_s_imag[i][j] + Lp_s_imag[i][j]/377)*cost[i][j]*cosp[i][j];
			ey_real[i][j] = (Np_s_real[i][j] - Lt_s_real[i][j]/Eta)*cosp[i][j] + (Nt_s_real[i][j] + Lp_s_real[i][j]/Eta)*cost[i][j]*sinp[i][j];
			ey_imag[i][j] = (Np_s_imag[i][j] - Lt_s_imag[i][j]/Eta)*cosp[i][j] + (Nt_s_imag[i][j] + Lp_s_imag[i][j]/Eta)*cost[i][j]*sinp[i][j];
		}
	}
	printf("Ex, Ey ok!......\n");
}

void calc_radiation(float Nfree, float k, float Eta, float **Nt_s_real, float **Nt_s_imag, float **Np_s_real, float **Np_s_imag, float **Lt_s_real, float **Lt_s_imag, float **Lp_s_real, float **Lp_s_imag, float **cosp, float **sinp, float **cost, float **ex_real, float **ex_imag, float **ey_real, float **ey_imag, float **radint, float **etheta, float **ephi, float **eintx, float **einty, float *P_tot, float *P_the, float *P_phi)
{
	int i,j;
	int row_s, col_s;           //shrinked row & col
	int half_row_s, half_col_s;

	row_s = (int)(10*Nfree*k);
	col_s = (int)(10*Nfree*k);

	half_col_s = (int)(col_s/2);
	half_row_s = (int)(row_s/2);

	///------------( total intensity )------------
	printf("calculating radiation intensity.....\n");

	for(j=row_s; j>=0; j--)
	{
		for(i=0; i<(col_s+1); i++)
		{
			if( (i-half_col_s)*(i-half_col_s)+(j-half_row_s)*(j-half_row_s) < Nfree*Nfree*k*k) 
				radint[i][j] = ((Nt_s_real[i][j]+Lp_s_real[i][j]/Eta)*(Nt_s_real[i][j]+Lp_s_real[i][j]/Eta)+ (Nt_s_imag[i][j]+Lp_s_imag[i][j]/Eta)*(Nt_s_imag[i][j]+Lp_s_imag[i][j]/Eta)+(Np_s_real[i][j]-Lt_s_real[i][j]/Eta)*(Np_s_real[i][j]-Lt_s_real[i][j]/Eta)+(Np_s_imag[i][j]-Lt_s_imag[i][j]/Eta)*(Np_s_imag[i][j]-Lt_s_imag[i][j]/Eta));
			else
				radint[i][j] = 0.0;
		}
	}
	printf("radint ok!....\n");

	///------------( Total Radiated Power )------------
	*P_tot = 0.0; //initialization
	for(j=row_s; j>=0; j--)
	{
		for(i=0; i<(col_s+1); i++)
		{
			if( (i-half_col_s)*(i-half_col_s)+(j-half_row_s)*(j-half_row_s) < Nfree*Nfree*k*k) 
				*P_tot = *P_tot + radint[i][j]/cost[i][j];
		}
	}


	///------------( Etheta intensity )------------
	for(j=row_s; j>=0; j--)
	{
		for(i=0; i<(col_s+1); i++)
		{
			if( (i-half_col_s)*(i-half_col_s)+(j-half_row_s)*(j-half_row_s) < Nfree*Nfree*k*k) 
				etheta[i][j] = ((Nt_s_real[i][j]+Lp_s_real[i][j]/Eta)*(Nt_s_real[i][j]+Lp_s_real[i][j]/Eta)
					+ (Nt_s_imag[i][j]+Lp_s_imag[i][j]/Eta)*(Nt_s_imag[i][j]+Lp_s_imag[i][j]/Eta));
			else
				etheta[i][j] = 0;
		}
	}

	///------------( Etheta Radiated Power )------------
	*P_the = 0.0; //initialization
	for(j=row_s; j>=0; j--)
	{
		for(i=0; i<(col_s+1); i++)
		{
			if( (i-half_col_s)*(i-half_col_s)+(j-half_row_s)*(j-half_row_s) < Nfree*Nfree*k*k) 
				*P_the = *P_the + etheta[i][j]/cost[i][j];
		}
	}

	///------------( Ephi intensity )------------
	for(j=row_s; j>=0; j--)
	{
		for(i=0; i<(col_s+1); i++)
		{
			if( (i-half_col_s)*(i-half_col_s)+(j-half_row_s)*(j-half_row_s) < Nfree*Nfree*k*k) 
				ephi[i][j] = ((Np_s_real[i][j]-Lt_s_real[i][j]/Eta)*(Np_s_real[i][j]-Lt_s_real[i][j]/Eta)
					+ (Np_s_imag[i][j]-Lt_s_imag[i][j]/Eta)*(Np_s_imag[i][j]-Lt_s_imag[i][j]/Eta));
			else
				ephi[i][j] = 0;
		}
	}

	///------------( Ephi Radiated Power )------------
	*P_phi = 0.0; //initialization
	for(j=row_s; j>=0; j--)
	{
		for(i=0; i<(col_s+1); i++)
		{
			if( (i-half_col_s)*(i-half_col_s)+(j-half_row_s)*(j-half_row_s) < Nfree*Nfree*k*k) 
				*P_phi = *P_phi + ephi[i][j]/cost[i][j];
		}
	}
	printf("etheta, ephi  ok!...\n");

	///------------( Ex intensity )------------
	for(j=row_s; j>=0; j--)
	{
		for(i=0; i<(col_s+1); i++)
		{
			if( (i-half_col_s)*(i-half_col_s)+(j-half_row_s)*(j-half_row_s) < Nfree*Nfree*k*k) 
				eintx[i][j] = (ex_real[i][j]*ex_real[i][j] + ex_imag[i][j]*ex_imag[i][j]);
			else
				eintx[i][j] = 0;
		}
	}

	///------------( Ey intensity )------------
	for(j=row_s; j>=0; j--)
	{
		for(i=0; i<(col_s+1); i++)
		{
			if( (i-half_col_s)*(i-half_col_s)+(j-half_row_s)*(j-half_row_s) < Nfree*Nfree*k*k) 
				einty[i][j] = (ey_real[i][j]*ey_real[i][j] + ey_imag[i][j]*ey_imag[i][j]);
			else
				einty[i][j] = 0;
		}
	}
	printf("eintx, einty ok!...\n");
}

void print_radiation(float Nfree, float k, float Eta, float **radint, float **etheta, float **ephi, float **eintx, float **einty, float *P_tot, float *P_the, float *P_phi)
{
	int i,j;
	int row_s, col_s;           //shrinked row & col
	int half_row_s, half_col_s;
	FILE *stream;

	row_s = (int)(10*Nfree*k);
	col_s = (int)(10*Nfree*k);

	half_col_s = (int)(col_s/2);
	half_row_s = (int)(row_s/2);

	///------------( total intensity )------------
	printf("radiation result writing......\n");
	stream = fopen(radiation_tot, "wt");
	for(j=half_row_s+(int)(Nfree*k)-1; j>=half_row_s-(int)(Nfree*k)+1; j--)
	{
		for(i=half_col_s-(int)(Nfree*k)+1; i<=half_col_s+(int)(Nfree*k)-1; i++)
		{
				fprintf(stream,"%g\t",radint[i][j]);
		}
		fprintf(stream,"\n");
	}
	fclose(stream);

	///------------( Total Radiated Power )------------
	printf("radiated power writing.......\n");
	stream = fopen(P_tot_name,"wt");
	fprintf(stream,"%g",P_tot);
	fclose(stream);

	///------------( Etheta Radiated Power )------------
	printf("radiated power writing.......\n");
	stream = fopen(P_the_name,"wt");
	fprintf(stream,"%g",P_the);
	fclose(stream);

	///------------( Ephi Radiated Power )------------
	printf("radiated power writing.......\n");
	stream = fopen(P_phi_name,"wt");
	fprintf(stream,"%g",P_phi);
	fclose(stream);

	///------------( Etheta intensity )------------
	stream = fopen(radiation_Et, "wt");
	for(j=half_row_s+(int)(Nfree*k)-1; j>=half_row_s-(int)(Nfree*k)+1; j--)
	{
		for(i=half_col_s-(int)(Nfree*k)+1; i<=half_col_s+(int)(Nfree*k)-1; i++)
		{
			fprintf(stream,"%g\t",etheta[i][j]);
		}
		fprintf(stream,"\n");
	}
	fclose(stream);

	///------------( Ephi intensity )------------
	stream = fopen(radiation_Ep, "wt");
	for(j=half_row_s+(int)(Nfree*k)-1; j>=half_row_s-(int)(Nfree*k)+1; j--)
	{
		for(i=half_col_s-(int)(Nfree*k)+1; i<=half_col_s+(int)(Nfree*k)-1; i++)
		{
			fprintf(stream,"%g\t",ephi[i][j]);
		}
		fprintf(stream,"\n");
	}
	fclose(stream);

	///------------( Ex intensity )------------
	stream = fopen(radiation_Ex, "wt");
	for(j=half_row_s+(int)(Nfree*k)-1; j>=half_row_s-(int)(Nfree*k)+1; j--)
	{
		for(i=half_col_s-(int)(Nfree*k)+1; i<=half_col_s+(int)(Nfree*k)-1; i++)
		{
			fprintf(stream,"%g\t",eintx[i][j]);
		}
		fprintf(stream,"\n");
	}
	fclose(stream);

	///------------( Ey intensity )------------
	stream = fopen(radiation_Ey, "wt");
	for(j=half_row_s+(int)(Nfree*k)-1; j>=half_row_s-(int)(Nfree*k)+1; j--)
	{
		for(i=half_col_s-(int)(Nfree*k)+1; i<=half_col_s+(int)(Nfree*k)-1; i++)
		{
			fprintf(stream,"%g\t",einty[i][j]);
		}
		fprintf(stream,"\n");
	}
	fclose(stream);
}

/*
void calc_polar(float NA, float Nfree)
{
	Eu_real = (float **)malloc(sizeof(float *)*(col+1));
	for(i=0; i<(col+1); i++)
		Eu_real[i] = (float *)malloc(sizeof(float)*(row+1));
	Eu_imag = (float **)malloc(sizeof(float *)*(col+1));
	for(i=0; i<(col+1); i++)
		Eu_imag[i] = (float *)malloc(sizeof(float)*(row+1));
	Eu_int = (float **)malloc(sizeof(float *)*(col+1));
	for(i=0; i<(col+1); i++)
		Eu_int[i] = (float *)malloc(sizeof(float)*(row+1));
	polarization = (float *)malloc(sizeof(float)*POLROT);

	for(l=0; l<POLROT; l++)
	{
		sum_Eu = 0.0;
		////// projection into u(phi) //////
		for(j=row; j>=0; j--)
		{
			for(i=0; i<(col+1); i++)
			{
				if( (i-half_col)*(i-half_col)+(j-half_row)*(j-half_row) < NA*NA*Nfree*Nfree*k*k)
				{
					Eu_real[i][j] = ex_real[i][j]*cos(2*pi*l/POLROT) + ey_real[i][j]*sin(2*pi*l/POLROT);
					Eu_imag[i][j] = ex_imag[i][j]*cos(2*pi*l/POLROT) + ey_imag[i][j]*sin(2*pi*l/POLROT);
					Eu_int[i][j] = Eu_real[i][j]*Eu_real[i][j] + Eu_imag[i][j]*Eu_imag[i][j];
					sum_Eu = sum_Eu + Eu_int[i][j]*sqrt(1-cost[i][j]*cost[i][j])/FFT_Emax;
				}
			}
		}
		printf("polarization [%d] = %g\n", l,sum_Eu);
		polarization[l] = sum_Eu;
	}

	///// writing file ///////
	stream = fopen(polar,"wt");
	for(l=0; l<POLROT; l++)
	{
		fprintf(stream,"%d\t%g\t\n",l*10,polarization[l]);
	}
	fprintf(stream,"%d\t%g\t\n",l*10,polarization[0]);
	fclose(stream);

	free(ex_real); free(ey_real); free(ex_imag); free(ey_imag);
	free(cosp); free(sinp); free(cost);
	free(radint); free(etheta); free(ephi);
	free(eintx); free(einty);
	free(Eu_real); free(Eu_imag); free(Eu_int);
	free(polarization);
}
*/

