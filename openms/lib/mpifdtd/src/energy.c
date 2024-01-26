#include "./fdtd.hpp"

/* corrections for nonuniform-grids */

void total_E_energy()
{
	FILE *stream_eEE;
	int i,j,k;
	float eEE=0;

	for(i=1+pmlil+0.2*lattice_x;i<=mygrid_nx-2-pmlir-0.2*lattice_x;i++)
		for(k=mygrid_nz-2-pmlkr-0.2*lattice_z;k>=1+pmlkl+0.2*lattice_z;k--)
			for(j=1+pmljl+0.2*lattice_y;j<=mygrid_ny-2-pmljr-0.2*lattice_y;j++) 
				eEE = eEE + ds_x*ds_y*ds_nz[k]*grid_value("E_Energy",i,j,k);

	stream_eEE=fopen("total_eEE.en","a");
	fprintf(stream_eEE,"%d\t %g\n",t,eEE); // from %f to %g ver. 8.802
	fclose(stream_eEE);
}

void total_E_energy_block(float centerx, float centery, float centerz, float size1, float size2, float size3)
{
	FILE *stream_Eb;
	int i,j,k;
	float Eb=0;
	int x_n,x_p, y_n,y_p, z_n,z_p;  //n:- p:+, position of the sides of the block

	x_n=(centerx + xcenter - 0.5*size1)*lattice_x;
	x_p=(centerx + xcenter + 0.5*size1)*lattice_x;
	y_n=(centery + ycenter - 0.5*size2)*lattice_y;
	y_p=(centery + ycenter + 0.5*size2)*lattice_y;
	z_n=non_uniform_z_to_i(centerz-0.5*size3);
	z_p=non_uniform_z_to_i(centerz+0.5*size3);
	
	for(i=x_n;i<=x_p;i++)
		for(k=z_p;k>=z_n;k--)
			for(j=y_n;j<=y_p;j++) 
				Eb = Eb + ds_x*ds_y*ds_nz[k]*grid_value("E_Energy",i,j,k);

	stream_Eb=fopen("total_eEE_b.en","a");
	fprintf(stream_Eb,"%d\t %g\n",t,Eb); // from %f to %g ver. 8.802
	fclose(stream_Eb);
}

void total_E_energy2_block(float centerx, float centery, float centerz, float size1, float size2, float size3)
{ // for calculating kinetic energy of electrons (from ver. 8.78)
	FILE *stream_E2b;
	int i,j,k;
	float Eb=0;
	int x_n,x_p, y_n,y_p, z_n,z_p;  //n:- p:+, position of the sides of the block

	x_n=(centerx + xcenter - 0.5*size1)*lattice_x;
	x_p=(centerx + xcenter + 0.5*size1)*lattice_x;
	y_n=(centery + ycenter - 0.5*size2)*lattice_y;
	y_p=(centery + ycenter + 0.5*size2)*lattice_y;
	z_n=non_uniform_z_to_i(centerz-0.5*size3);
	z_p=non_uniform_z_to_i(centerz+0.5*size3);
	
	for(i=x_n;i<=x_p;i++)
		for(k=z_p;k>=z_n;k--)
			for(j=y_n;j<=y_p;j++) 
				Eb = Eb + ds_x*ds_y*ds_nz[k]*grid_value("E_Energy2",i,j,k);

	stream_E2b=fopen("total_eEE2_b.en","a");
	fprintf(stream_E2b,"%d\t %g\n",t,Eb); // from %f to %g ver. 8.802
	fclose(stream_E2b);
}

void total_E_energy3_block(float centerx, float centery, float centerz, float size1, float size2, float size3)
{ // for calculating potential energy of electrons (from ver. 8.78)
	FILE *stream_E3b;
	int i,j,k;
	float Eb=0;
	int x_n,x_p, y_n,y_p, z_n,z_p;  //n:- p:+, position of the sides of the block

	x_n=(centerx + xcenter - 0.5*size1)*lattice_x;
	x_p=(centerx + xcenter + 0.5*size1)*lattice_x;
	y_n=(centery + ycenter - 0.5*size2)*lattice_y;
	y_p=(centery + ycenter + 0.5*size2)*lattice_y;
	z_n=non_uniform_z_to_i(centerz-0.5*size3);
	z_p=non_uniform_z_to_i(centerz+0.5*size3);
	
	for(i=x_n;i<=x_p;i++)
		for(k=z_p;k>=z_n;k--)
			for(j=y_n;j<=y_p;j++) 
				Eb = Eb + ds_x*ds_y*ds_nz[k]*grid_value("E_Energy3",i,j,k);

	stream_E3b=fopen("total_eEE3_b.en","a");
	fprintf(stream_E3b,"%d\t %g\n",t,Eb);  // from %f to %g ver. 8.802
	fclose(stream_E3b);
}

void total_E_energy_thin_block_z(float centerx, float centery, float centerz, float size1, float size2, float size3, float eps_L, float eps_H, char *name)
{
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/// This calculation assumes very very thin z-size (=size3), so we takes an arithmatic average of U(z_n) and U(z_p)
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	FILE *file;
	int i,j,k;
	float Ebt=0;
	int x_n,x_p, y_n,y_p, z_n,z_p;  //n:- p:+, position of the sides of the block

	x_n=(centerx + xcenter - 0.5*size1)*lattice_x;
	x_p=(centerx + xcenter + 0.5*size1)*lattice_x;
	y_n=(centery + ycenter - 0.5*size2)*lattice_y;
	y_p=(centery + ycenter + 0.5*size2)*lattice_y;
	z_n=non_uniform_z_to_i(centerz-0.5*size3);
	z_p=non_uniform_z_to_i(centerz+0.5*size3);
	
	for(i=x_n;i<=x_p;i++)
		for(j=y_n;j<=y_p;j++)
			if(eps(i,j,z_n)>=eps_L && eps(i,j,z_n)<=eps_H && eps(i,j,z_p)>=eps_L && eps(i,j,z_p)<=eps_H)
				Ebt = Ebt + 0.5*ds_x*ds_y*(size3*ds_x*lattice_x)*(grid_value("E_Energy",i,j,z_n)+grid_value("E_Energy",i,j,z_p));

	file=fopen(name,"a");
	fprintf(file,"%d\t %g\n",t,Ebt);
	fclose(file);
}

void max_E_Energy_detector(float centerx, float centery, float centerz, float size1, float size2, float size3)
{
	FILE *stream_max;
	int i,j,k;
	float max_val=-12345;
	int max_i, max_j, max_k;
	int x_n,x_p, y_n,y_p, z_n,z_p;  //n:- p:+, position of the sides of the block

	x_n=(centerx + xcenter - 0.5*size1)*lattice_x;
	x_p=(centerx + xcenter + 0.5*size1)*lattice_x;
	y_n=(centery + ycenter - 0.5*size2)*lattice_y;
	y_p=(centery + ycenter + 0.5*size2)*lattice_y;
	z_n=non_uniform_z_to_i(centerz-0.5*size3);
	z_p=non_uniform_z_to_i(centerz+0.5*size3);
	
	for(i=x_n;i<=x_p;i++)
		for(k=z_p;k>=z_n;k--)
			for(j=y_n;j<=y_p;j++) 
			{
				if(grid_value("E_Energy",i,j,k) > max_val)
				{
					max_val = grid_value("E_Energy", i,j,k);
					max_i = i; max_j = j; max_k = k;
				}
			}
	stream_max=fopen("max_detect.dat","a");
	fprintf(stream_max,"%d\t%d\t%d\t%d\t%g\n",t, max_i, max_j, max_k, max_val);
	fclose(stream_max);
}

void total_EM_energy()
{
	FILE *stream_eEE_uHH;
	int i,j,k;
	float eEE_uHH=0;
	
	for(i=1+pmlil+0.2*lattice_x;i<=mygrid_nx-2-pmlir-0.2*lattice_x;i++)
		for(k=mygrid_nz-2-pmlkr-0.2*lattice_z;k>=1+pmlkl+0.2*lattice_z;k--)
			for(j=1+pmljl+0.2*lattice_y;j<=mygrid_ny-2-pmljr-0.2*lattice_y;j++) 
				eEE_uHH = eEE_uHH + ds_x*ds_y*ds_nz[k]*grid_value("EM_Energy",i,j,k);

	stream_eEE_uHH=fopen("total_eEE_uHH.en","a");
	fprintf(stream_eEE_uHH,"%d\t %g\n",t,eEE_uHH); // from %f to %g ver. 8.802
	fclose(stream_eEE_uHH);
}

void total_EM_energy_block(float centerx, float centery, float centerz, float size1, float size2, float size3)
{
	FILE *stream_EMb;
	int i,j,k;
	float EMb=0;
	int x_n,x_p, y_n,y_p, z_n,z_p;  //n:- p:+, position of the sides of the block

	x_n=(centerx + xcenter - 0.5*size1)*lattice_x;
	x_p=(centerx + xcenter + 0.5*size1)*lattice_x;
	y_n=(centery + ycenter - 0.5*size2)*lattice_y;
	y_p=(centery + ycenter + 0.5*size2)*lattice_y;
	z_n=non_uniform_z_to_i(centerz-0.5*size3);
	z_p=non_uniform_z_to_i(centerz+0.5*size3);
	
	for(i=x_n;i<=x_p;i++)
		for(k=z_p;k>=z_n;k--)
			for(j=y_n;j<=y_p;j++) 
				EMb = EMb + ds_x*ds_y*ds_nz[k]*grid_value("EM_Energy",i,j,k);

	stream_EMb=fopen("total_eEE_uHH_b.en","a");
	fprintf(stream_EMb,"%d\t %g\n",t,EMb);  // from %f to %g ver. 8.802
	fclose(stream_EMb);
}

void total_E2()
{
	FILE *stream_EE;
	int i,j,k;
	double EE=0;	

	for(i=1+pmlil+0.2*lattice_x;i<=mygrid_nx-2-pmlir-0.2*lattice_x;i++)
		for(k=mygrid_nz-2-pmlkr-0.2*lattice_z;k>=1+pmlkl+0.2*lattice_z;k--)
			for(j=1+pmljl+0.2*lattice_y;j<=mygrid_ny-2-pmljr-0.2*lattice_y;j++)  
				EE = EE + ds_x*ds_y*ds_nz[k]*grid_value("E^2",i,j,k);

	stream_EE=fopen("total_EE.en","a");
	fprintf(stream_EE,"%d\t %g\n",t,EE);  // from %f to %g ver. 8.802
	fclose(stream_EE);
}

// The below function is for testing the code..... , which is not open to public
void Drude_energy_loss_in_block2(float centerx, float centery, float centerz, float size1, float size2, float size3, float WC, float lattice_n)
{
	FILE *stream_Eloss;
	int i,j,k;
	double Eloss=0.0;
	double Im_eps;
	double w, wr;
	double temp_fac;
	int x_n,x_p, y_n,y_p, z_n,z_p;  //n:- p:+, position of the sides of the block
	///////////////////////////////////////////////////////
	/// WC : loss calculated at this normalized frequency
	///////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////
	//	power loss per unit volume = 2 WC Im{epsilon(WC)} E(x,t)*E(x,t) 
	//	(See Jackson, Eq.6.127, no explicit ohmic loss is assumed, i.e. J.E = 0)
	//
	//	Im{epsilon(w)} = momega[i][j][k]^2 * mgamma[i][j][k] / ( w^3 + w*mgamma[i][j][k]^2 )
	/////////////////////////////////////////////////////////////////////////////////////////////

	x_n=(centerx + xcenter - 0.5*size1)*lattice_x;
	x_p=(centerx + xcenter + 0.5*size1)*lattice_x;
	y_n=(centery + ycenter - 0.5*size2)*lattice_y;
	y_p=(centery + ycenter + 0.5*size2)*lattice_y;
	z_n=non_uniform_z_to_i(centerz-0.5*size3);
	z_p=non_uniform_z_to_i(centerz+0.5*size3);

	/////////////////////////////////////////////
	/// WC --> to fit into the FDTD time unit 
	/////////////////////////////////////////////
	w = WC*2*3.1415926*light_speed/(ds_x*lattice_x);
	wr = WC*2*3.1415926*light_speed/lattice_n;// into actual angular frequency in the unit of Hz 

	temp_fac = (ds_x*lattice_x)/lattice_n; // to be used to reconvert the reduced wp and gamma
	
	for(i=x_n; i<=x_p; i++)
		for(j=y_n; j<=y_p; j++)
			for(k=z_n; k<=z_p; k++)
			{ 
				Im_eps = temp_fac*temp_fac*temp_fac*momega[i][j][k]*momega[i][j][k]*mgamma[i][j][k]/(wr*wr*wr + wr*mgamma[i][j][k]*mgamma[i][j][k]*temp_fac*temp_fac);
				Eloss = Eloss + 2*w*eo*Im_eps*ds_x*ds_y*ds_nz[k]*grid_value("E^2",i,j,k);
			}

	stream_Eloss=fopen("Drude_E_loss.en","a");
	fprintf(stream_Eloss,"%d\t%lf\n",t,Eloss);
	fclose(stream_Eloss);
}

void Drude_energy_loss_in_block(float centerx, float centery, float centerz, float size1, float size2, float size3)
{
	FILE *stream_Eloss;
	int i,j,k;
	double Eloss=0.0;
	int x_n,x_p, y_n,y_p, z_n,z_p;  //n:- p:+, position of the sides of the block

	x_n=(centerx + xcenter - 0.5*size1)*lattice_x;
	x_p=(centerx + xcenter + 0.5*size1)*lattice_x;
	y_n=(centery + ycenter - 0.5*size2)*lattice_y;
	y_p=(centery + ycenter + 0.5*size2)*lattice_y;
	z_n=non_uniform_z_to_i(centerz-0.5*size3);
	z_p=non_uniform_z_to_i(centerz+0.5*size3);
	
	for(i=x_n; i<=x_p; i++)
		for(j=y_n; j<=y_p; j++)
			for(k=z_n; k<=z_p; k++)
			{ 
				Eloss = Eloss + ds_x*ds_y*ds_nz[k]*grid_value("J.E",i,j,k);
			}

	stream_Eloss=fopen("Drude_E_loss.en","a");
	fprintf(stream_Eloss,"%d\t%g\n",t,Eloss);  // from %f to %g ver. 8.802
	fclose(stream_Eloss);
}

void Poynting_total()
{
	int i,j,k;
	float sumxp=0,sumxn=0,sumyp=0,sumyn=0,sumzp=0,sumzn=0;

	for(k=mygrid_nz-2-pmlkr-0.2*lattice_z;k>=1+pmlkl+0.2*lattice_z;k--)
		for(j=1+pmljl+0.2*lattice_y;j<=mygrid_ny-2-pmljr-0.2*lattice_y;j++) 
		{	
			sumxp+=(ds_y*ds_nz[k])*grid_value("Sx",mygrid_nx-2-pmlir-0.2*lattice_x,j,k);
			sumxn+=(ds_y*ds_nz[k])*grid_value("Sx",pmlil+0.2*lattice_x,j,k);
		}
	for(i=1+pmlil+0.2*lattice_x;i<=mygrid_nx-2-pmlir-0.2*lattice_x;i++)
		for(k=mygrid_nz-2-pmlkr-0.2*lattice_z;k>=1+pmlkl+0.2*lattice_z;k--)
		{
			sumyp+=(ds_nz[k]*ds_x)*grid_value("Sy",i,mygrid_ny-2-pmljl-0.2*lattice_y,k);
			sumyn+=(ds_nz[k]*ds_x)*grid_value("Sy",i,pmljl+0.2*lattice_y,k);
		}
	for(i=1+pmlil+0.2*lattice_x;i<=mygrid_nx-2-pmlir-0.2*lattice_x;i++)
		for(j=1+pmljl+0.2*lattice_y;j<=mygrid_ny-2-pmljr-0.2*lattice_y;j++) 
		{
			sumzp+=(ds_x*ds_y)*grid_value("Sz",i,j,mygrid_nz-2-pmlkr-0.2*lattice_z);
			sumzn+=(ds_x*ds_y)*grid_value("Sz",i,j,pmlkl+0.2*lattice_z);
		}
	Sumx=Sumx+(sumxp-sumxn);  // Sumx,Sumy,Sumz --> global variable
	Sumy=Sumy+(sumyp-sumyn);
	Sumz=Sumz+(sumzp-sumzn);
}


float Poynting_half_sphere_point(char *component, float z0, float R, float theta, float phi)
{
	int xg,yg,zg;

	int k; // for non_uniform grid

	float x, y, z;
	float Pi; //'i'-component of Poynting vector

	////// unit = a //////
	x = R*sin(theta)*cos(phi);
	y = R*sin(theta)*sin(phi);
	z = R*cos(theta)+z0;
	k = non_uniform_z_to_i(z);
	
	////// unit = grid //////
	xg = floor(0.5+((x+xcenter)*lattice_x));
	yg = floor(0.5+((y+xcenter)*lattice_y));
	zg = non_uniform_z_to_i(z);

	if(strcmp("r",component)==0)
	{
		Pi = (ds_y*ds_nz[k])*grid_value("Sx", xg,yg,zg)*sin(theta)*cos(phi) 
             	 	+ (ds_nz[k]*ds_x)*grid_value("Sy" ,xg,yg,zg)*sin(theta)*sin(phi) 
              		+ (ds_x*ds_y)*grid_value("Sz", xg,yg,zg)*cos(theta);
	}
	else if(strcmp("theta",component)==0)
	{
		Pi = (ds_y*ds_nz[k])*grid_value("Sx", xg,yg,zg)*cos(theta)*cos(phi) 
             	 	+ (ds_nz[k]*ds_x)*grid_value("Sy" ,xg,yg,zg)*cos(theta)*sin(phi) 
              		- (ds_x*ds_y)*grid_value("Sz", xg,yg,zg)*sin(theta);
	}
	else if(strcmp("phi",component)==0)
	{
		Pi = -(ds_y*ds_nz[k])*grid_value("Sx", xg,yg,zg)*sin(phi) 
             	 	+ (ds_nz[k]*ds_x)*grid_value("Sy" ,xg,yg,zg)*cos(phi); 
	}
	else
	{}

	return(Pi);
}

void Poynting_block(float centerx, float centery, float centerz, float size1, float size2, float size3)
{
	int i,j,k;
	float sumxp=0,sumxn=0,sumyp=0,sumyn=0,sumzp=0,sumzn=0;
	int x_n,x_p, y_n,y_p, z_n,z_p;  //n:- p:+, position of the sides of the block

	x_n=(centerx + xcenter - 0.5*size1)*lattice_x;
	x_p=(centerx + xcenter + 0.5*size1)*lattice_x;
	y_n=(centery + ycenter - 0.5*size2)*lattice_y;
	y_p=(centery + ycenter + 0.5*size2)*lattice_y;
	z_n=non_uniform_z_to_i(centerz-0.5*size3);
	z_p=non_uniform_z_to_i(centerz+0.5*size3);

	for(k=z_p; k>=z_n; k--)
		for(j=y_n; j<=y_p; j++) 
		{	
			sumxp+=(ds_y*ds_nz[k])*grid_value("Sx",x_p,j,k);
			sumxn+=(ds_y*ds_nz[k])*grid_value("Sx",x_n,j,k);
		}
	for(i=x_n; i<=x_p; i++)
		for(k=z_p; k>=z_n; k--)
		{
			sumyp+=(ds_nz[k]*ds_x)*grid_value("Sy",i,y_p,k);
			sumyn+=(ds_nz[k]*ds_x)*grid_value("Sy",i,y_n,k);
		}
	for(i=x_n; i<=x_p; i++)
		for(j=y_n; j<=y_p; j++) 
		{
			sumzp+=(ds_x*ds_y)*grid_value("Sz",i,j,z_p);
			sumzn+=(ds_x*ds_y)*grid_value("Sz",i,j,z_n);
		}
	Sumx=Sumx+(sumxp-sumxn);  // Sumx,Sumy,Sumz --> global variable
	Sumy=Sumy+(sumyp-sumyn);
	Sumz=Sumz+(sumzp-sumzn);
}

void Poynting_side(float value, float zposition)
{
	int i,j,k;
	float sumxp=0,sumxn=0,sumyp=0,sumyn=0;
	
	for(k=non_uniform_z_to_i(value+zposition);k>=non_uniform_z_to_i(-value+zposition);k--)
		for(j=1+pmljl+0.2*lattice_y;j<=mygrid_ny-2-pmljr-0.2*lattice_y;j++) 
		{	
			sumxp+=(ds_y*ds_nz[k])*grid_value("Sx",mygrid_nx-2-pmlir-0.2*lattice_x,j,k);
			sumxn+=(ds_y*ds_nz[k])*grid_value("Sx",pmlil+0.2*lattice_x,j,k);
		}
	for(i=1+pmlil+0.2*lattice_x;i<=mygrid_nx-2-pmlir-0.2*lattice_x;i++)
		for(k=non_uniform_z_to_i(value+zposition);k>=non_uniform_z_to_i(-value+zposition);k--)
		{
			sumyp+=(ds_nz[k]*ds_x)*grid_value("Sy",i,mygrid_ny-2-pmljr-0.2*lattice_y,k);
			sumyn+=(ds_nz[k]*ds_x)*grid_value("Sy",i,pmljl+0.2*lattice_y,k);
		}

	SideSumx=SideSumx+(sumxp-sumxn);
	SideSumy=SideSumy+(sumyp-sumyn);
}

void print_energy()
{
	FILE *VQSQ;
	float Et, Eh, Ev;

	Et = Sumx+Sumy+Sumz;  //total loss
	Eh = SideSumx+SideSumy;  //horizontal loss
	Ev = Et-Eh;  //vertical loss

	VQSQ=fopen("VQSQ.en","wt");
	fprintf(VQSQ,"VerticalLoss=%g\t HorizontalLoss=%g\t VerticalLoss/HorizontalLoss(K)=%g\n",Ev,Eh,Ev/Eh);
	fprintf(VQSQ,"Sumx=%g\t Sumy=%g\t Sumz=%g\n",Sumx,Sumy,Sumz);
	fprintf(VQSQ,"SideSumx=%g\t SideSumy=%g\n",SideSumx,SideSumy);
	fprintf(VQSQ,"SumUpper=%g\t SumLower=%g\n",SumUpper,SumLower);
	fclose(VQSQ);
}

void Poynting_UpDown(float value, float zposition)
{
	int i,j,k;
	float UpStripXp=0, UpStripXn=0, UpStripYp=0, UpStripYn=0;
	float DnStripXp=0, DnStripXn=0, DnStripYp=0, DnStripYn=0;
	float PlanZp=0, PlanZn=0;

	//////// Upper plane & Bottom plane
	for(i=1+pmlil+0.2*lattice_x;i<=mygrid_nx-2-pmlir-0.2*lattice_x;i++)
		for(j=1+pmljl+0.2*lattice_y;j<=mygrid_ny-2-pmljr-0.2*lattice_y;j++) 
		{
			PlanZp+=(ds_x*ds_y)*grid_value("Sz",i,j,mygrid_nz-2-pmlkr-0.2*lattice_z);
			PlanZn+=(ds_x*ds_y)*grid_value("Sz",i,j,pmlkl+0.2*lattice_z);
		}
	
	///////// Upper Strip 
	for(k=mygrid_nz-2-pmlkr-0.2*lattice_z; k>=non_uniform_z_to_i(value+zposition); k--)
		for(j=1+pmljl+0.2*lattice_y;j<=mygrid_ny-2-pmljr-0.2*lattice_y;j++) 
		{	
			UpStripXp+=(ds_y*ds_nz[k])*grid_value("Sx",mygrid_nx-2-pmlir-0.2*lattice_x,j,k);
			UpStripXn+=(ds_y*ds_nz[k])*grid_value("Sx",pmlil+0.2*lattice_x,j,k);
		}
	for(k=mygrid_nz-2-pmlkr-0.2*lattice_z; k>=non_uniform_z_to_i(value+zposition); k--)
		for(i=1+pmlil+0.2*lattice_x;i<=mygrid_nx-2-pmlir-0.2*lattice_x;i++)
		{
			UpStripYp+=(ds_nz[k]*ds_x)*grid_value("Sy",i,mygrid_ny-2-pmljr-0.2*lattice_y,k);
			UpStripYn+=(ds_nz[k]*ds_x)*grid_value("Sy",i,pmljl+0.2*lattice_y,k);
		}

	///////// Lower Strip
	for(k=1+pmlkl+0.2*lattice_z; k<=non_uniform_z_to_i(-value+zposition); k++)
		for(j=1+pmljl+0.2*lattice_y;j<=mygrid_ny-2-pmljr-0.2*lattice_y;j++) 
		{	
			DnStripXp+=(ds_y*ds_nz[k])*grid_value("Sx",mygrid_nx-2-pmlir-0.2*lattice_x,j,k);
			DnStripXn+=(ds_y*ds_nz[k])*grid_value("Sx",pmlil+0.2*lattice_x,j,k);
		}
	for(k=1+pmlkl+0.2*lattice_z; k<=non_uniform_z_to_i(-value+zposition); k++)
		for(i=1+pmlil+0.2*lattice_x;i<=mygrid_nx-2-pmlir-0.2*lattice_x;i++)
		{
			DnStripYp+=(ds_nz[k]*ds_x)*grid_value("Sy",i,mygrid_ny-2-pmljr-0.2*lattice_y,k);
			DnStripYn+=(ds_nz[k]*ds_x)*grid_value("Sy",i,pmljl+0.2*lattice_y,k);
		}

	SumUpper = SumUpper + PlanZp + (UpStripXp-UpStripXn) + (UpStripYp-UpStripYn);
	SumLower = SumLower + (-PlanZn) + (DnStripXp-DnStripXn) + (DnStripYp-DnStripYn);
}

