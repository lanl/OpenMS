#include "./fdtd.hpp"

float sigmax(float a)
{
	float sigma_maxl, sigma_maxr;

	if(use_periodic_x == 0) // no use of periodic boundary condition 
	{
		sigma_maxl=(sig_axl)*(orderxl+1)/(150*pi*sqrt(back_epsilon));
		sigma_maxr=(sig_axr)*(orderxr+1)/(150*pi*sqrt(back_epsilon));

		if(a<pmlil+0.5) return sigma_maxl*pow(-(a-(pmlil+0.5))/pmlil,orderxl);
		else if(a<mygrid_nx-pmlir-1) return 0.0;
		else return sigma_maxr*pow((a-(mygrid_nx-pmlir-1))/pmlir,orderxr);
	}
	else // use of periodic boundary condition
		return 0.0;
}

float sigmay(float a)
{
	float sigma_maxl, sigma_maxr;

	if(use_periodic_y == 0) // no use of periodic boundary condition 
	{
		sigma_maxl=(sig_ayl)*(orderyl+1)/(150*pi*sqrt(back_epsilon));
		sigma_maxr=(sig_ayr)*(orderyr+1)/(150*pi*sqrt(back_epsilon));

		if(a<pmljl+0.5) return sigma_maxl*pow(-(a-(pmljl+0.5))/pmljl,orderyl);
		else if(a<mygrid_ny-pmljr-1) return 0.0;
		else return sigma_maxr*pow((a-(mygrid_ny-pmljr-1))/pmljr,orderyr);
	}
	else // use of periodic boundary condition
		return 0.0;
}

float sigmaz(float a)
{
	float sigma_maxl, sigma_maxr;

	sigma_maxl=(sig_azl)*(orderzl+1)/(150*pi*sqrt(back_epsilon));
	sigma_maxr=(sig_azr)*(orderzr+1)/(150*pi*sqrt(back_epsilon));

	if(a<pmlkl+0.5) return sigma_maxl*pow(-(a-(pmlkl+0.5))/pmlkl,orderzl);
	else if(a<mygrid_nz-pmlkr-1) return 0.0;
	else return sigma_maxr*pow((a-(mygrid_nz-pmlkr-1))/pmlkr,orderzr);

}

float cpmlbx(float a, float pmlbl, float pmlbr)
{
	float sigmae, alphae,kappae;
	float bx;

	if(use_periodic_x==0) // no use of periodic boundary condition
	{
		if(a<pmlbl+0.5)
		{
			sigmae=sigmaCPML*pow((pmlbl-a)/pmlbl,m_pml);
			alphae=alphaCPML*pow(a/pmlbl, ma_pml);
			kappae=1.0+(kappaCPML-1.0)*pow((pmlbl-a)/pmlbl,m_pml);
			return bx=exp(-(sigmae/kappae+alphae)*dt_nm/eo);
		}
		else if(a<mygrid_nx-pmlbr-1) return 1.0;
		else 
		{
			sigmae=sigmaCPML*pow((a-(mygrid_nx-pmlbr-1.0))/pmlbr, m_pml);
			alphae=alphaCPML*pow((mygrid_nx-1.0-a)/pmlbr,ma_pml);
			kappae=1.0+(kappaCPML-1.0)*pow((a-(mygrid_nx-pmlbr-1.0))/pmlbr,m_pml);
			return bx=exp(-(sigmae/kappae+alphae)*dt_nm/eo);
			
		}
	}
	else
		return 1.0;
}

float cpmlax(float a, float pmlbl, float pmlbr)
{
	float sigmae, alphae,kappae;
	float bx,ax;

	if(use_periodic_x==0) // no use of periodic boundary condition
	{
		if(a<pmlbl+0.5)
		{
			sigmae=sigmaCPML*pow((pmlbl-a)/pmlbl,m_pml);
			alphae=alphaCPML*pow(a/pmlbl, ma_pml);
			kappae=1.0+(kappaCPML-1.0)*pow((pmlbl-a)/pmlbl,m_pml);
			//printf("sigmae= %f %f %f \n", a, sigmaCPML, sigmae);
			//printf("sigmae= %f %f %f \n", a, alphaCPML, alphae);
			//printf("sigmae= %f %f %f \n", a, kappaCPML, kappae);

			bx=exp(-(sigmae/kappae+alphae)*dt_nm/eo);
		}
		else if( a<mygrid_nx-pmlbr-1) return 0.0;
		else
		{
			sigmae=sigmaCPML*pow((a-(mygrid_nx-pmlbr-1.0))/pmlbr, m_pml);
			alphae=alphaCPML*pow((mygrid_nx-1.0-a)/pmlbr,ma_pml);
			kappae=1.0+(kappaCPML-1.0)*pow((a-(mygrid_nx-pmlbr-1.0))/pmlbr,m_pml);
			
			bx=exp(-(sigmae/kappae+alphae)*dt_nm/eo);
		}

		ax=sigmae*(bx-1.0)/(sigmae+kappae*alphae)/kappae;
		//printf("bx ax= %f %f %f \n", a, bx, ax);
		return ax;
	}
	else
		return 0.0;
}

float kappa_x(float a, float pmlbl, float pmlbr)
{
	float kappae;
	if(use_periodic_x==0) // no use of periodic boundary condition
	{
		if(a<pmlbl+0.5) kappae=1.0+(kappaCPML-1.0)*pow((pmlbl-a)/pmlbl,m_pml);
		else if(a<mygrid_nx-pmlbr-1) return 1.0;
		else kappae=1.0+(kappaCPML-1.0)*pow((a-(mygrid_nx-pmlbr-1.0))/pmlbr,m_pml);

		return kappae;

	}
	else
		return 1.0;
}

