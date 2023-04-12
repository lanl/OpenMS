#include "./pFDTD.h"

float sigmax(float a)
{
	float sigma_maxl, sigma_maxr;

	if(use_periodic_x == 0) // no use of periodic boundary condition 
	{
		sigma_maxl=(sig_axl)*(orderxl+1)/(150*pi*sqrt(back_epsilon));
		sigma_maxr=(sig_axr)*(orderxr+1)/(150*pi*sqrt(back_epsilon));

		if(a<pmlil+0.5) return sigma_maxl*pow(-(a-(pmlil+0.5))/pmlil,orderxl);
		else if(a<isize-pmlir-1) return 0.0;
		else return sigma_maxr*pow((a-(isize-pmlir-1))/pmlir,orderxr);
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
		else if(a<jsize-pmljr-1) return 0.0;
		else return sigma_maxr*pow((a-(jsize-pmljr-1))/pmljr,orderyr);
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
	else if(a<ksize-pmlkr-1) return 0.0;
	else return sigma_maxr*pow((a-(ksize-pmlkr-1))/pmlkr,orderzr);

}
