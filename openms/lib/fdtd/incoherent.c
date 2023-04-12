#include "./pFDTD.h"

int random_on(int p_tau);
float get_random_float(gsl_rng *r, float range);
long get_Gaussian_dist(gsl_rng *r, float mu, float sigma);

int inc_dipole_num = 0; 
//////// Random variables ////////////
float *p_theta_aux;
float *p_phi;
float *p_eps;
long *p_tau;
//////////////////////////////////////
long store_t=-100; // temp 't' store variable
int one_round_turn =0; // 0:no, 1:yes

void incoherent_point_dipole(char *function, float x, float y, float z, float frequency, long to, long tdecay, float t_mu, float sd)
{
	int i,j,k;
	float Jx, Jy, Jz;
	float iJx, iJy, iJz;

	if( (store_t==-100) || (store_t==t) )
	{
		/// dipole number increase
		inc_dipole_num++; 
		if( one_round_turn == 0 )
		{
			/// re-allocation of 1-D arrays 
			p_theta_aux=(float *)realloc(p_theta_aux,sizeof(float)*inc_dipole_num);
			p_phi=(float *)realloc(p_phi,sizeof(float)*inc_dipole_num);
			p_eps=(float *)realloc(p_eps,sizeof(float)*inc_dipole_num);
			p_tau=(long *)realloc(p_tau,sizeof(long)*inc_dipole_num);
			//initialization
			p_tau[inc_dipole_num-1] = t;
		} 
	}
	else
	{
		inc_dipole_num = 1;  
		one_round_turn = 1; 
	}

	/// save the present time 't'
	store_t = t; 

	// dipole position
	i=floor(0.5+((x+xcenter)*lattice_x));
	j=floor(0.5+((y+ycenter)*lattice_y));
	k=floor(0.5+((z+zcenter)*lattice_z));

	if(random_on(p_tau[inc_dipole_num-1])==1)
	{
		p_theta_aux[inc_dipole_num-1] = get_random_float(rng_r, 1.0);  
			// to make uniform random generation over the spherical surface
			// instead of using 'theta' directly, we use 0< 'theta_aux' <1 
			// in such a way sin(theta) = sqrt( 1 - theta_aux^2)
			// modified after ver. 8.2
		p_phi[inc_dipole_num-1] = get_random_float(rng_r, 2*pi);
		p_eps[inc_dipole_num-1] = get_random_float(rng_r, 2*pi);
		p_tau[inc_dipole_num-1] = t + get_Gaussian_dist(rng_r, t_mu, sd);
	}
	else //no change
	{}

	if(strcmp(function,"Gauss")==0) //Gaussian envelope
	{
		Jx = Gauss_amp(frequency, p_eps[inc_dipole_num-1], to, tdecay)*sqrt(1-p_theta_aux[inc_dipole_num-1]*p_theta_aux[inc_dipole_num-1])*cos(p_phi[inc_dipole_num-1]);
		Jy = Gauss_amp(frequency, p_eps[inc_dipole_num-1], to, tdecay)*sqrt(1-p_theta_aux[inc_dipole_num-1]*p_theta_aux[inc_dipole_num-1])*sin(p_phi[inc_dipole_num-1]);
		Jz = Gauss_amp(frequency, p_eps[inc_dipole_num-1], to, tdecay)*p_theta_aux[inc_dipole_num-1];
		if((use_periodic_x == 1 || use_periodic_y == 1) && (wave_vector_x!=0.0 || wave_vector_y!=0.0))
		{
			iJx = iGauss_amp(frequency, p_eps[inc_dipole_num-1], to, tdecay)*sqrt(1-p_theta_aux[inc_dipole_num-1]*p_theta_aux[inc_dipole_num-1])*cos(p_phi[inc_dipole_num-1]);
			iJy = iGauss_amp(frequency, p_eps[inc_dipole_num-1], to, tdecay)*sqrt(1-p_theta_aux[inc_dipole_num-1]*p_theta_aux[inc_dipole_num-1])*sin(p_phi[inc_dipole_num-1]);
			iJz = iGauss_amp(frequency, p_eps[inc_dipole_num-1], to, tdecay)*p_theta_aux[inc_dipole_num-1];
		}
	}
	if(strcmp(function,"Lorentz")==0) //Lorentzian envelope
	{
		Jx = Lorentz_amp(frequency, p_eps[inc_dipole_num-1], to, tdecay)*sqrt(1-p_theta_aux[inc_dipole_num-1]*p_theta_aux[inc_dipole_num-1])*cos(p_phi[inc_dipole_num-1]);
		Jy = Lorentz_amp(frequency, p_eps[inc_dipole_num-1], to, tdecay)*sqrt(1-p_theta_aux[inc_dipole_num-1]*p_theta_aux[inc_dipole_num-1])*sin(p_phi[inc_dipole_num-1]);
		Jz = Lorentz_amp(frequency, p_eps[inc_dipole_num-1], to, tdecay)*p_theta_aux[inc_dipole_num-1];
		if((use_periodic_x == 1 || use_periodic_y == 1) && (wave_vector_x!=0.0 || wave_vector_y!=0.0))
		{
			iJx = iLorentz_amp(frequency, p_eps[inc_dipole_num-1], to, tdecay)*sqrt(1-p_theta_aux[inc_dipole_num-1]*p_theta_aux[inc_dipole_num-1])*cos(p_phi[inc_dipole_num-1]);
			iJy = iLorentz_amp(frequency, p_eps[inc_dipole_num-1], to, tdecay)*sqrt(1-p_theta_aux[inc_dipole_num-1]*p_theta_aux[inc_dipole_num-1])*sin(p_phi[inc_dipole_num-1]);
			iJz = iLorentz_amp(frequency, p_eps[inc_dipole_num-1], to, tdecay)*p_theta_aux[inc_dipole_num-1];
		}
	}

	Ex[i][j][k]=Ex[i][j][k]+Jx/2;
	Ex[i-1][j][k]=Ex[i-1][j][k]+Jx/2;
	Ey[i][j][k]=Ey[i][j][k]+Jy/2;
	Ey[i][j-1][k]=Ey[i][j-1][k]+Jy/2;
	Ez[i][j][k]=Ez[i][j][k]+Jz/2;
	Ez[i][j][k-1]=Ez[i][j][k-1]+Jz/2;

	if((use_periodic_x == 1 || use_periodic_y == 1) && (wave_vector_x!=0.0 || wave_vector_y!=0.0))
	{
		iEx[i][j][k]=iEx[i][j][k]+iJx/2;
		iEx[i-1][j][k]=iEx[i-1][j][k]+iJx/2;
		iEy[i][j][k]=iEy[i][j][k]+iJy/2;
		iEy[i][j-1][k]=iEy[i][j-1][k]+iJy/2;
		iEz[i][j][k]=iEz[i][j][k]+iJz/2;
		iEz[i][j][k-1]=iEz[i][j][k-1]+iJz/2;
	}
}

int random_on(p_tau)
{
	if(t == p_tau)
		return(1);
	else
		return(0);
}

float get_random_float(gsl_rng *r, float range)
{
	return(range*gsl_rng_uniform(r));
}

long get_Gaussian_dist(gsl_rng *r, float mu, float sigma)
{
	return(fabs(mu + gsl_ran_gaussian(r, sigma)));
}

//////////////////////////////////////////////////
//    Rotating phasor in case of periodic BC    //
//  --------------------------------------------//
//        Use the relation :                    //
//     cos(wt-90) + i sin(wt-90)                //
//        = sin(wt) - i cos(wt)                 //
//     Then, we can preserve the same "sin()"   //
//        form for the real field source        //
////////////////////////////////////////////////// 

float Gauss_amp(float frequency, float phase, long to, long tdecay)
{
	if(to-3*tdecay<t && t<to+3*tdecay)
		return( sin(2*pi*frequency*t/S_factor/ds_x/lattice_x+phase)*exp(-1.0*pow((float)(t-to)/tdecay,2.0)) );
	else
		return(0.0);
}

float Lorentz_amp(float frequency, float phase, long to, long tdecay)
{
	if(tdecay == 0)
		return( sin(2*pi*frequency*(float)(t-to)/S_factor/ds_x/lattice_x+phase) );
	else if(to<=t && t<to+8*tdecay)	
		return( sin(2*pi*frequency*(float)(t-to)/S_factor/ds_x/lattice_x+phase)*exp(-(float)(t-to)/tdecay) );
	else
		return(0.0);
}

float iGauss_amp(float frequency, float phase, long to, long tdecay)
{
	if(to-3*tdecay<t && t<to+3*tdecay)
		return( -cos(2*pi*frequency*t/S_factor/ds_x/lattice_x+phase)*exp(-1.0*pow((float)(t-to)/tdecay,2.0)) );
	else
		return(0.0);
}

float iLorentz_amp(float frequency, float phase, long to, long tdecay)
{
	if(to<=t && t<to+8*tdecay)
	{
		if(tdecay == 0)
			return( -cos(2*pi*frequency*(float)(t-to)/S_factor/ds_x/lattice_x+phase) );
		else
			return( -cos(2*pi*frequency*(float)(t-to)/S_factor/ds_x/lattice_x+phase)*exp(-(float)(t-to)/tdecay) );
	}
	else
		return(0.0);
}
