#include "./pFDTD.h"

float efieldx(int i,int j,int k);
float efieldy(int i,int j,int k);
float efieldz(int i,int j,int k);
float jfieldx(int i,int j,int k);
float jfieldy(int i,int j,int k);
float jfieldz(int i,int j,int k);
float hfieldx(int i,int j,int k);
float hfieldy(int i,int j,int k);
float hfieldz(int i,int j,int k);
float iefieldx(int i,int j,int k);
float iefieldy(int i,int j,int k);
float iefieldz(int i,int j,int k);
float ijfieldx(int i,int j,int k);
float ijfieldy(int i,int j,int k);
float ijfieldz(int i,int j,int k);
float ihfieldx(int i,int j,int k);
float ihfieldy(int i,int j,int k);
float ihfieldz(int i,int j,int k);

float grid_value_periodic_x(char *component,int i,int j,int k, int h);
float grid_value_periodic_y(char *component,int i,int j,int k, int h);
float grid_value_periodic_xy(char *component,int i,int j,int k, int h, int v);

int find_k_projection(int i, int j, int k_range);

int nz_multiple(int k); //for non_uniform_grid multiplication, used in various out_put image functions

float global_W;

void real_space_param(float a_nm, float w_n)
{
	// a_nm is in the unit of 'nanometer'
	// w_n is the normalized frequency
	FILE *PARAM;

	/// Set global normalized frequency////////////// 
	global_W = w_n;
	//////////////////////////////////////////// (ver. 8.45)

	PARAM=fopen("Real_Space_Param.dat","wt");

	fprintf(PARAM,"=================================\n");	
	fprintf(PARAM,"The FDTD calculation domain \n");
	fprintf(PARAM,"=================================\n");	
	fprintf(PARAM,"Lx =%lf (a) \n",xsize);
	fprintf(PARAM,"Ly =%lf (a) \n",ysize);
	fprintf(PARAM,"Lz =%lf (a) \n",zsize);
	fprintf(PARAM,"---------------------------------\n");
	fprintf(PARAM,"Lx =%lf (nm) \n",xsize*a_nm);
	fprintf(PARAM,"Ly =%lf (nm) \n",ysize*a_nm);
	fprintf(PARAM,"Lz =%lf (nm) \n",zsize*a_nm);
	fprintf(PARAM,"\n");

	fprintf(PARAM,"=================================\n");	
	fprintf(PARAM,"The PML size \n");
	fprintf(PARAM,"=================================\n");	
	fprintf(PARAM,"pmlil =%d (u), pmlir =%d (u) \n",pmlil,pmlir);
	fprintf(PARAM,"pmljl =%d (u), pmljr =%d (u) \n",pmljl,pmljr);
	fprintf(PARAM,"pmlkl =%d (u), pmlkr =%d (u) \n",pmlkl,pmlkr);
	fprintf(PARAM,"\n");

	fprintf(PARAM,"=================================\n");	
	fprintf(PARAM,"The speed of light \n");
	fprintf(PARAM,"=================================\n");	
	fprintf(PARAM,"c =%E (m/s) \n",light_speed);
	fprintf(PARAM,"\n");

	fprintf(PARAM,"=================================\n");	
	fprintf(PARAM,"The Stability parameter 'S' \n");
	fprintf(PARAM,"=================================\n");	
	fprintf(PARAM,"S =%lf (1/q) \n",S_factor);
	fprintf(PARAM,"S =%E (1/m) \n",S_factor*ds_x*lattice_x/(a_nm*1E-9));
	fprintf(PARAM,"\n");

	fprintf(PARAM,"=================================\n");	
	fprintf(PARAM,"The size of the FDTD grid \n");
	fprintf(PARAM,"=================================\n");	
	fprintf(PARAM,"dx =%lf (q) \n",ds_x);
	fprintf(PARAM,"dy =%lf (q) \n",ds_y);
	fprintf(PARAM,"dz =%lf (q) \n",ds_z);
	fprintf(PARAM,"---------------------------------\n");
	fprintf(PARAM,"dx =%lf (nm) \n",a_nm/lattice_x);
	fprintf(PARAM,"dy =%lf (nm) \n",a_nm/lattice_y);
	fprintf(PARAM,"dz =%lf (nm) \n",a_nm/max_lattice_nz);
	fprintf(PARAM,"\n");

	fprintf(PARAM,"=================================\n");	
	fprintf(PARAM,"The size of the FDTD time step \n");
	fprintf(PARAM,"=================================\n");	
	fprintf(PARAM,"dt =%g (sec) \n",(a_nm*1E-9)/(light_speed*S_factor*ds_x*lattice_x));
	fprintf(PARAM,"\n");

	fprintf(PARAM,"=================================\n");	
	fprintf(PARAM,"The dipole wavelength in vacuum (in FDTD)\n");
	fprintf(PARAM,"=================================\n");	
	fprintf(PARAM,"lambda =%lf (u) \n",lattice_x/w_n);
	fprintf(PARAM,"\n");

	fprintf(PARAM,"=================================\n");	
	fprintf(PARAM,"The dipole frequency (in FDTD)\n");
	fprintf(PARAM,"=================================\n");	
	fprintf(PARAM,"f =%lf (1/update) \n",w_n/(S_factor*ds_x*lattice_x));
	fprintf(PARAM,"T =%lf (update) \n",(S_factor*ds_x*lattice_x)/w_n);
	fprintf(PARAM,"\n");

	fprintf(PARAM,"=================================\n");	
	fprintf(PARAM,"The Origin FFT correction factor \n");
	fprintf(PARAM,"=================================\n");	
	fprintf(PARAM,"x%lf\n",S_factor*ds_x*lattice_x);
	fprintf(PARAM,"\n");

	fclose(PARAM);
}

int get_period_in_update(float w_n)
{
	return((int)(S_factor*ds_x*lattice_x)/w_n);	
}

void out_epsilon(char *plane,float value,char *name)
{
	FILE *file;
	int i,j,k;
	int i_range, j_range, k_range; 
	int mz; // index for non_uniform_grid multiplication 

	// for Hz_parity, normal cases
	i_range = isize-1;
	j_range = jsize-1;
	k_range = ksize-1; 
	
	if(use_periodic_x == 1)
		i_range = isize;
	if(use_periodic_y == 1)
		j_range = jsize; 

	file=fopen(name,"w");

	if(strcmp("x",plane)==0)
	{
		i=floor(0.5+((value+xcenter)*lattice_x));
		for(k=k_range;k>=1;k--)
		{
			for(mz=0; mz<nz_multiple(k); mz++) // for non_uniform_grid multiplication
			{
				for(j=1;j<=j_range;j++)
				{
					if(meps(i,j,k)==0.0)
						fprintf(file,"%g ",eps(i,j,k));
					else
						fprintf(file,"%g ",meps(i,j,k));
				}
				fprintf(file,"\n");
			}
		}
	}
	if(strcmp("y",plane)==0)
	{
		j=floor(0.5+((value+ycenter)*lattice_y));
		for(k=k_range;k>=1;k--)
		{
			for(mz=0; mz<nz_multiple(k); mz++) // for non_uniform_grid multiplication
			{
				for(i=1;i<=i_range;i++)
				{
					if(meps(i,j,k)==0.0)
						fprintf(file,"%g ",eps(i,j,k));
					else
						fprintf(file,"%g ",meps(i,j,k));
				}
				fprintf(file,"\n");
			}
		}
	}

	if(strcmp("z",plane)==0)
	{
		k=non_uniform_z_to_i(value);
		for(j=j_range;j>=1;j--)
		{
			for(i=1;i<=i_range;i++)
			{
				if(meps(i,j,k)==0.0)
					fprintf(file,"%g ",eps(i,j,k));
				else
					fprintf(file,"%g ",meps(i,j,k));
			}
			fprintf(file,"\n");
		}
	}
	fclose(file);
	printf("out_epsilon...ok\n");
}

void out_epsilon_periodic(char *plane,float value,char *name, int m_h, int m_v)
{
	FILE *file;
	int i,j,k;
	int v,h;
	int i_range, j_range, k_range; 
	int mz; // index for non_uniform_grid multiplication 

	// for Hz_parity, normal cases
	i_range = isize-1;
	j_range = jsize-1;
	k_range = ksize-1; 
	
	file=fopen(name,"w");

	if(strcmp("x",plane)==0)
	{
		i=floor(0.5+((value+xcenter)*lattice_x));
		for(v=0; v<m_v; v++)
		{
			for(k=k_range;k>=1;k--)
			{
				for(mz=0; mz<nz_multiple(k); mz++) // for non_uniform_grid multiplication
				{
					for(h=0; h<m_h; h++)
					{
						for(j=1;j<=j_range;j++)
						{
							if(meps(i,j,k)==0.0)
								fprintf(file,"%g ",eps(i,j,k));
							else
								fprintf(file,"%g ",meps(i,j,k));
						}
					}
					fprintf(file,"\n");
				}
			}
		}
	}

	if(strcmp("y",plane)==0)
	{
		j=floor(0.5+((value+ycenter)*lattice_y));
		for(v=0; v<m_v; v++)
		{
			for(k=k_range;k>=1;k--)
			{
				for(mz=0; mz<nz_multiple(k); mz++) // for non_uniform_grid multiplication
				{
					for(h=0; h<m_h; h++)
					{
						for(i=1;i<=i_range;i++)
						{
							if(meps(i,j,k)==0.0)
								fprintf(file,"%g ",eps(i,j,k));
							else
								fprintf(file,"%g ",meps(i,j,k));
						}
					}
					fprintf(file,"\n");
				}
			}
		}
	}

	if(strcmp("z",plane)==0)
	{
		k=non_uniform_z_to_i(value);
		for(v=0; v<m_v; v++)
		{
			for(j=j_range;j>=1;j--)
			{
				for(h=0; h<m_h; h++)
				{
					for(i=1;i<=i_range;i++)
					{
						if(meps(i,j,k)==0.0)
							fprintf(file,"%g ",eps(i,j,k));
						else
							fprintf(file,"%g ",meps(i,j,k));
					}
				}
				fprintf(file,"\n");
			}
		}
	}
	fclose(file);
	printf("out_epsilon...ok\n");
}

void out_epsilon_projection(char *dirc, char *name)
{
	FILE *file;
	int i,j,k;
	int i_range, j_range, k_range; 

	// for Hz_parity, normal cases
	i_range = isize-1;
	j_range = jsize-1;
	k_range = ksize-1; 
	
	if(use_periodic_x == 1)
		i_range = isize;
	if(use_periodic_y == 1)
		j_range = jsize; 

	file=fopen(name,"w");

	if(strcmp("+z",dirc)==0)
	{
		for(j=j_range;j>=1;j--)
		{
			for(i=1;i<=i_range;i++)
			{
				k= find_k_projection(i,j,k_range);
				if(meps(i,j,k)==0.0)
					fprintf(file,"%g ",eps(i,j,k));
				else
					fprintf(file,"%g ",meps(i,j,k));
			}
			fprintf(file,"\n");
		}
	}
	fclose(file);
	printf("out_epsilon...ok\n");
}

void out_plane(char *component,char *plane,float value,char *lastname)
{
	FILE *file;
	char name[20];
	int i,j,k;
	int i_range, j_range, k_range; 
	int mz; //for non_uniform_grid multiplication

	// for Hz_parity, normal cases
	i_range = isize-1;
	j_range = jsize-1;
	k_range = ksize-1; 
	
	if(use_periodic_x == 1)
		i_range = isize;
	if(use_periodic_y == 1)
		j_range = jsize; 
	
	sprintf(name,"%07d",t);
	strcat(name,lastname);
	file=fopen(name,"w");

	if(strcmp("x",plane)==0)
	{
		i=floor(0.5+((value+xcenter)*lattice_x));
		for(k=k_range;k>=1;k--)
		{
			for(mz=0; mz<nz_multiple(k); mz++) // for non_uniform_grid multiplication
			{
				for(j=1;j<=j_range;j++)	fprintf(file,"%g ",grid_value(component,i,j,k));
				fprintf(file,"\n");
			}
		}
	}

	if(strcmp("y",plane)==0)
	{
		j=floor(0.5+((value+ycenter)*lattice_y));
		for(k=k_range;k>=1;k--)
		{
			for(mz=0; mz<nz_multiple(k); mz++) // for non_uniform_grid multiplication
			{
				for(i=1;i<=i_range;i++)	fprintf(file,"%g ",grid_value(component,i,j,k));
				fprintf(file,"\n");
			}
		}
	}

	if(strcmp("z",plane)==0)
	{
		k=non_uniform_z_to_i(value);
		for(j=j_range;j>=1;j--)
		{
			for(i=1;i<=i_range;i++) fprintf(file,"%g ",grid_value(component,i,j,k));	
			fprintf(file,"\n");
		}
	}
	fclose(file);
	printf("out %s...ok\n",component);
}

void out_plane_periodic(char *component,char *plane,float value,char *lastname, int m_h, int m_v)
{
	FILE *file;
	char name[20];
	int i,j,k;
	int h, v;
	int i_range, j_range, k_range; 
	int mz; //for non_uniform_grid mulitplication

	// for Hz_parity, normal cases
	i_range = isize-1;
	j_range = jsize-1;
	k_range = ksize-1; 
	
	sprintf(name,"%07d",t);
	strcat(name,lastname);
	file=fopen(name,"w");

	if(strcmp("x",plane)==0)
	{
		i=floor(0.5+((value+xcenter)*lattice_x));
		for(v=0; v<1; v++)  // no periodic for z-direction
		{
			for(k=k_range;k>=1;k--)
			{
				for(mz=0; mz<nz_multiple(k); mz++) // for non_uniform_grid multiplication
				{
					for(h=0; h<m_h; h++)
					{
						for(j=1;j<=j_range;j++)
							fprintf(file,"%g ",grid_value_periodic_y(component,i,j,k,h));	
					}
					fprintf(file,"\n");
				}
			}
		}
	}

	if(strcmp("y",plane)==0)
	{
		j=floor(0.5+((value+ycenter)*lattice_y));
		for(v=0; v<1; v++)  // no periodic for z-direction
		{
			for(k=k_range;k>=1;k--)
			{
				for(mz=0; mz<nz_multiple(k); mz++) // for non_uniform_grid multiplication
				{
					for(h=0; h<m_h; h++)
					{
						for(i=1;i<=i_range;i++)
							fprintf(file,"%g ",grid_value_periodic_x(component,i,j,k,h));	
					}
					fprintf(file,"\n");
				}
			}
		}
	}

	if(strcmp("z",plane)==0)
	{
		k=non_uniform_z_to_i(value);
		for(v=0; v<m_v; v++)
		{
			for(j=j_range;j>=1;j--)
			{
				for(h=0; h<m_h; h++)
				{
					for(i=1;i<=i_range;i++)
						fprintf(file,"%g ",grid_value_periodic_xy(component,i,j,k,h,v));	
				}
				fprintf(file,"\n");
			}
		}
	}
	fclose(file);
	printf("out %s...ok\n",component);
}

void out_plane_time_average(char *component,char *plane,float value, long start, long end, float ***field_avg, char *lastname)
{
	FILE *file;
	char name[20];
	int i,j,k;
	int i_range, j_range, k_range; 
	int mz; //for non_uniform_grid multiplication

	// for Hz_parity, normal cases
	i_range = isize-1;
	j_range = jsize-1;
	k_range = ksize-1; 
	
	if(use_periodic_x == 1)
		i_range = isize;
	if(use_periodic_y == 1)
		j_range = jsize; 

	if(strcmp("x",plane)==0)
	{
		if(t == start)
		{
			///// *field_avg[j][k] /////
			(*field_avg) = (float **)malloc(sizeof(float *)*(j_range+1));
			for(j=0; j<j_range+1; j++)
				(*field_avg)[j] = (float *)malloc(sizeof(float)*(k_range+1));

			for(j=0; j<j_range+1; j++) 
				for(k=0; k<k_range+1; k++)
					(*field_avg)[j][k] = 0.0;
		}

		if(t>start && t<=end)
		{
			i=floor(0.5+((value+xcenter)*lattice_x));
			for(k=k_range;k>=1;k--)
			{
				for(j=1;j<=j_range;j++)  
					(*field_avg)[j][k] = (*field_avg)[j][k] + grid_value(component,i,j,k); 
			}
		}

		if(t == end)
		{
			sprintf(name,"%07d",start);
			strcat(name,lastname);
			file=fopen(name,"w");

			for(k=k_range;k>=1;k--)
			{
				for(mz=0; mz<nz_multiple(k); mz++) // for non_uniform_grid multiplication
				{
					for(j=1;j<=j_range;j++)	fprintf(file,"%g ",(*field_avg)[j][k]/(float)(end-start));
					fprintf(file,"\n");
				}
			}

			fclose(file);
		}
	}

	if(strcmp("y",plane)==0)
	{
		if(t == start)
		{
			///// *field_avg[k][i] /////
			(*field_avg) = (float **)malloc(sizeof(float *)*(k_range+1));
			for(k=0; k<k_range+1; k++)
				(*field_avg)[k] = (float *)malloc(sizeof(float)*(i_range+1));

			for(k=0; k<k_range+1; k++)
				for(i=0; i<i_range+1; i++)
					(*field_avg)[k][i] = 0.0;
		}

		if(t>start && t<=end)
		{
			j=floor(0.5+((value+ycenter)*lattice_y));
			for(k=k_range;k>=1;k--)
			{
				for(i=1;i<=i_range;i++)  
					(*field_avg)[k][i] = (*field_avg)[k][i] + grid_value(component,i,j,k); 
			}
		}

		if(t == end)
		{
			sprintf(name,"%07d",start);
			strcat(name,lastname);
			file=fopen(name,"w");

			for(k=k_range;k>=1;k--)
			{
				for(mz=0; mz<nz_multiple(k); mz++) // for non_uniform_grid multiplication
				{
					for(i=1;i<=i_range;i++)	fprintf(file,"%g ",(*field_avg)[k][i]/(float)(end-start));
					fprintf(file,"\n");
				}
			}

			fclose(file);
		}
	}

	if(strcmp("z",plane)==0)
	{
		if(t == start)
		{
			///// *field_avg[i][j] /////
			(*field_avg) = (float **)malloc(sizeof(float *)*(i_range+1));
			for(i=0; i<i_range+1; i++)
				(*field_avg)[i] = (float *)malloc(sizeof(float)*(j_range+1));

			for(i=0; i<i_range+1; i++)
				for(j=0; j<j_range+1; j++)
					(*field_avg)[i][j] = 0.0;
		}

		if(t>start && t<=end)
		{
			k=non_uniform_z_to_i(value);
			for(j=j_range;j>=1;j--)
			{
				for(i=1;i<=i_range;i++)  
					(*field_avg)[i][j] = (*field_avg)[i][j] + grid_value(component,i,j,k); 
			}
		}

		if(t == end)
		{
			sprintf(name,"%07d",start);
			strcat(name,lastname);
			file=fopen(name,"w");

			for(j=j_range;j>=1;j--)
			{
			
				for(i=1;i<=i_range;i++)	fprintf(file,"%g ",(*field_avg)[i][j]/(float)(end-start));
				fprintf(file,"\n");
			}

			fclose(file);
		}
	}
}

void out_plane_time_average_projection(char *component,char *dirc, long start, long end, float ***field_avg, char *lastname, int k_shift)
{
	FILE *file;
	char name[20];
	int i,j,k;
	int i_range, j_range, k_range; 

	// for Hz_parity, normal cases
	i_range = isize-1;
	j_range = jsize-1;
	k_range = ksize-1; 
	
	if(use_periodic_x == 1)
		i_range = isize;
	if(use_periodic_y == 1)
		j_range = jsize; 

	if(strcmp("+z",dirc)==0)
	{
		if(t == start)
		{
			///// *field_avg[i][j] /////
			(*field_avg) = (float **)malloc(sizeof(float *)*(i_range+1));
			for(i=0; i<i_range+1; i++)
				(*field_avg)[i] = (float *)malloc(sizeof(float)*(j_range+1));

			for(i=0; i<i_range+1; i++)
				for(j=0; j<j_range+1; j++)
					(*field_avg)[i][j] = 0.0;
		}

		if(t>start && t<=end)
		{
			for(j=j_range;j>=1;j--)
			{
				for(i=1;i<=i_range;i++)  
				{
					k= find_k_projection(i,j,k_range) + k_shift;
					(*field_avg)[i][j] = (*field_avg)[i][j] + grid_value(component,i,j,k); 
				}
			}
		}

		if(t == end)
		{
			sprintf(name,"%07d",start);
			strcat(name,lastname);
			file=fopen(name,"w");

			for(j=j_range;j>=1;j--)
			{
				for(i=1;i<=i_range;i++)
					fprintf(file,"%g ",(*field_avg)[i][j]/(float)(end-start));

				fprintf(file,"\n");
			}

			fclose(file);
		}
	}
}

void out_plane_projection(char *component,char *dirc,char *lastname, int k_shift)
{
	FILE *file;
	char name[20];
	int i,j,k;
	int i_range, j_range, k_range; 

	// for Hz_parity, normal cases
	i_range = isize-1;
	j_range = jsize-1;
	k_range = ksize-1; 
	
	if(use_periodic_x == 1)
		i_range = isize;
	if(use_periodic_y == 1)
		j_range = jsize; 
	
	sprintf(name,"%07d",t);
	strcat(name,lastname);
	file=fopen(name,"w");

	if(strcmp("+z",dirc)==0)
	{
		for(j=j_range;j>=1;j--)
		{
			for(i=1;i<=i_range;i++)
			{
				k= find_k_projection(i,j,k_range) + k_shift;
				fprintf(file,"%g ",grid_value(component,i,j,k));
			}	
			fprintf(file,"\n");
		}
	}
	fclose(file);
	printf("out %s...ok\n",component);
}

void out_several_points(char *component, float zposition, float xside, float yside, int pNx, int pNy, long ti, long tf, char *name)
{
	FILE *file;
	int i,j,k;
	int m,n; //for loop index

	//// xside, yside --> i_range, j_range
	
	if(ti<=t && t<tf)
	{
		k=non_uniform_z_to_i(zposition);  // fixed 

		file=fopen(name,"a");

		for(m=0; m<=pNx; m++)
		{
			for(n=0; n<=pNy; n++)
			{
				i=floor(0.5+((-0.5*xside+m*(xside/pNx)+xcenter)*lattice_x)); 
				j=floor(0.5+((-0.5*yside+n*(yside/pNx)+ycenter)*lattice_y));
				fprintf(file,"%g\t",grid_value(component,i,j,k));
			}
		}	
		fprintf(file,"\n");
		fclose(file);
	}

}

void out_point(char *component,float x,float y,float z,long ti,long tf,char *name)
{
	FILE *file;
	int i,j,k;

	if(ti<=t && t<tf)
	{
		i=floor(0.5+((x+xcenter)*lattice_x));
		j=floor(0.5+((y+ycenter)*lattice_y));
		k=non_uniform_z_to_i(z);

		if(t==ti) remove(name);
		file=fopen(name,"a");
		fprintf(file,"%g\n",grid_value(component,i,j,k));
		fclose(file);
	}
}

float grid_value(char *component,int i,int j,int k)
{
	if(strcmp(component,"Ex")==0) return efieldx(i,j,k);
	if(strcmp(component,"Ey")==0) return efieldy(i,j,k);
	if(strcmp(component,"Ez")==0) return efieldz(i,j,k);
	if(strcmp(component,"Hx")==0) return hfieldx(i,j,k);
	if(strcmp(component,"Hy")==0) return hfieldy(i,j,k);
	if(strcmp(component,"Hz")==0) return hfieldz(i,j,k);
	if(strcmp(component,"Jx")==0) return jfieldx(i,j,k);
	if(strcmp(component,"Jy")==0) return jfieldy(i,j,k);
	if(strcmp(component,"Jz")==0) return jfieldz(i,j,k);	
	if(strcmp(component,"J.E")==0) return efieldx(i,j,k)*jfieldx(i,j,k)+efieldy(i,j,k)*jfieldy(i,j,k)+efieldz(i,j,k)*jfieldz(i,j,k);
	if(strcmp(component,"E^2")==0) return pow(efieldx(i,j,k),2)+pow(efieldy(i,j,k),2)+pow(efieldz(i,j,k),2);
	if(strcmp(component,"H^2")==0) return pow(hfieldx(i,j,k),2)+pow(hfieldy(i,j,k),2)+pow(hfieldz(i,j,k),2);
	if(strcmp(component,"Sx")==0) return efieldy(i,j,k)*hfieldz(i,j,k)-efieldz(i,j,k)*hfieldy(i,j,k);
	if(strcmp(component,"Sy")==0) return efieldz(i,j,k)*hfieldx(i,j,k)-efieldx(i,j,k)*hfieldz(i,j,k);
	if(strcmp(component,"Sz")==0) return efieldx(i,j,k)*hfieldy(i,j,k)-efieldy(i,j,k)*hfieldx(i,j,k);
	if(strcmp(component,"LogE^2")==0) return log10(pow(efieldx(i,j,k),2)+pow(efieldy(i,j,k),2)+pow(efieldz(i,j,k),2));
	if(strcmp(component,"LogH^2")==0) return log10(pow(hfieldx(i,j,k),2)+pow(hfieldy(i,j,k),2)+pow(hfieldz(i,j,k),2));

	if(strcmp(component,"EM_Energy")==0 && meps(i,j,k)==0.0) return 0.5*eo*eps(i,j,k)*(pow(efieldx(i,j,k),2)+pow(efieldy(i,j,k),2)+pow(efieldz(i,j,k),2))+0.5*uo*ups*(pow(hfieldx(i,j,k),2)+pow(hfieldy(i,j,k),2)+pow(hfieldz(i,j,k),2));
	if(strcmp(component,"EM_Energy")==0 && meps(i,j,k)!=0.0) return 0.5*eo*eps_m(i,j,k)*(pow(efieldx(i,j,k),2)+pow(efieldy(i,j,k),2)+pow(efieldz(i,j,k),2))+0.5*uo*ups*(pow(hfieldx(i,j,k),2)+pow(hfieldy(i,j,k),2)+pow(hfieldz(i,j,k),2));
	if(strcmp(component,"EM_Energy2")==0 && meps(i,j,k)==0.0) return 0.0;
	if(strcmp(component,"EM_Energy2")==0 && meps(i,j,k)!=0.0) return 0.5*eo*eps_m2(i,j,k)*(pow(efieldx(i,j,k),2)+pow(efieldy(i,j,k),2)+pow(efieldz(i,j,k),2))+0.5*uo*ups*(pow(hfieldx(i,j,k),2)+pow(hfieldy(i,j,k),2)+pow(hfieldz(i,j,k),2));
	if(strcmp(component,"EM_Energy3")==0 && meps(i,j,k)==0.0) return 0.0;
	if(strcmp(component,"EM_Energy3")==0 && meps(i,j,k)!=0.0) return 0.5*eo*eps_m3(i,j,k)*(pow(efieldx(i,j,k),2)+pow(efieldy(i,j,k),2)+pow(efieldz(i,j,k),2))+0.5*uo*ups*(pow(hfieldx(i,j,k),2)+pow(hfieldy(i,j,k),2)+pow(hfieldz(i,j,k),2));

	if(strcmp(component,"E_Energy")==0 && meps(i,j,k)==0.0) return 0.5*eo*eps(i,j,k)*(pow(efieldx(i,j,k),2)+pow(efieldy(i,j,k),2)+pow(efieldz(i,j,k),2));
	if(strcmp(component,"E_Energy")==0 && meps(i,j,k)!=0.0) return 0.5*eo*eps_m(i,j,k)*(pow(efieldx(i,j,k),2)+pow(efieldy(i,j,k),2)+pow(efieldz(i,j,k),2));
	if(strcmp(component,"E_Energy2")==0 && meps(i,j,k)==0.0) return 0.0;
	if(strcmp(component,"E_Energy2")==0 && meps(i,j,k)!=0.0) return 0.5*eo*eps_m2(i,j,k)*(pow(efieldx(i,j,k),2)+pow(efieldy(i,j,k),2)+pow(efieldz(i,j,k),2));
	if(strcmp(component,"E_Energy3")==0 && meps(i,j,k)==0.0) return 0.0;
	if(strcmp(component,"E_Energy3")==0 && meps(i,j,k)!=0.0) return 0.5*eo*eps_m3(i,j,k)*(pow(efieldx(i,j,k),2)+pow(efieldy(i,j,k),2)+pow(efieldz(i,j,k),2));

	if(strcmp(component,"Ex2Ey2")==0) return pow(efieldx(i,j,k),2)+pow(efieldy(i,j,k),2);
	if(strcmp(component,"Ey2Ez2")==0) return pow(efieldy(i,j,k),2)+pow(efieldz(i,j,k),2);
	if(strcmp(component,"Ez2Ex2")==0) return pow(efieldz(i,j,k),2)+pow(efieldx(i,j,k),2);	
	if(strcmp(component,"LogEx2Ey2")==0) return log10(pow(efieldx(i,j,k),2)+pow(efieldy(i,j,k),2));
	if(strcmp(component,"LogEy2Ez2")==0) return log10(pow(efieldy(i,j,k),2)+pow(efieldz(i,j,k),2));
	if(strcmp(component,"LogEz2Ex2")==0) return log10(pow(efieldz(i,j,k),2)+pow(efieldx(i,j,k),2));
	if(strcmp(component,"Hx2Hy2")==0) return pow(hfieldx(i,j,k),2)+pow(hfieldy(i,j,k),2);
	if(strcmp(component,"Hy2Hz2")==0) return pow(hfieldy(i,j,k),2)+pow(hfieldz(i,j,k),2);
	if(strcmp(component,"Hz2Hx2")==0) return pow(hfieldz(i,j,k),2)+pow(hfieldx(i,j,k),2);	
	if(strcmp(component,"LogHx2Hy2")==0) return log10(pow(hfieldx(i,j,k),2)+pow(hfieldy(i,j,k),2));
	if(strcmp(component,"LogHy2Hz2")==0) return log10(pow(hfieldy(i,j,k),2)+pow(hfieldz(i,j,k),2));
	if(strcmp(component,"LogHz2Hx2")==0) return log10(pow(hfieldz(i,j,k),2)+pow(hfieldx(i,j,k),2));	
}

float grid_value_periodic_x(char *component,int i,int j,int k, int h)
{
	float E_x, E_y, E_z, J_x, J_y, J_z, H_x, H_y, H_z;

	E_x = efieldx(i,j,k)*cos(2*pi*h*wave_vector_x)-iefieldx(i,j,k)*sin(2*pi*h*wave_vector_x);
	E_y = efieldy(i,j,k)*cos(2*pi*h*wave_vector_x)-iefieldy(i,j,k)*sin(2*pi*h*wave_vector_x);
	E_z = efieldz(i,j,k)*cos(2*pi*h*wave_vector_x)-iefieldz(i,j,k)*sin(2*pi*h*wave_vector_x);
	J_x = jfieldx(i,j,k)*cos(2*pi*h*wave_vector_x)-ijfieldx(i,j,k)*sin(2*pi*h*wave_vector_x);
	J_y = jfieldy(i,j,k)*cos(2*pi*h*wave_vector_x)-ijfieldy(i,j,k)*sin(2*pi*h*wave_vector_x);
	J_z = jfieldz(i,j,k)*cos(2*pi*h*wave_vector_x)-ijfieldz(i,j,k)*sin(2*pi*h*wave_vector_x);
	H_x = hfieldx(i,j,k)*cos(2*pi*h*wave_vector_x)-ihfieldx(i,j,k)*sin(2*pi*h*wave_vector_x);
	H_y = hfieldy(i,j,k)*cos(2*pi*h*wave_vector_x)-ihfieldy(i,j,k)*sin(2*pi*h*wave_vector_x);
	H_z = hfieldz(i,j,k)*cos(2*pi*h*wave_vector_x)-ihfieldz(i,j,k)*sin(2*pi*h*wave_vector_x);

	if(strcmp(component,"Ex")==0) return (E_x);
	if(strcmp(component,"Ey")==0) return (E_y);
	if(strcmp(component,"Ez")==0) return (E_z);
	if(strcmp(component,"Hx")==0) return (H_x);
	if(strcmp(component,"Hy")==0) return (H_y);
	if(strcmp(component,"Hz")==0) return (H_z);
	if(strcmp(component,"Jx")==0) return (J_x);
	if(strcmp(component,"Jy")==0) return (J_y);
	if(strcmp(component,"Jz")==0) return (J_z);
	if(strcmp(component,"J.E")==0) return E_x*J_x+E_y*J_y+E_z*J_z;
	if(strcmp(component,"E^2")==0) return pow(E_x,2)+pow(E_y,2)+pow(E_z,2);
	if(strcmp(component,"H^2")==0) return pow(H_x,2)+pow(H_y,2)+pow(H_z,2);
	if(strcmp(component,"Sx")==0) return E_y*H_z-E_z*H_y;
	if(strcmp(component,"Sy")==0) return E_z*H_x-E_x*H_z;
	if(strcmp(component,"Sz")==0) return E_x*H_y-E_y*H_x;
	if(strcmp(component,"LogE^2")==0) return log10(pow(E_x,2)+pow(E_y,2)+pow(E_z,2));
	if(strcmp(component,"LogH^2")==0) return log10(pow(H_x,2)+pow(H_y,2)+pow(H_z,2));

	if(strcmp(component,"EM_Energy")==0 && meps(i,j,k)==0.0) return 0.5*eo*eps(i,j,k)*(pow(E_x,2)+pow(E_y,2)+pow(E_z,2))+0.5*uo*ups*(pow(H_x,2)+pow(H_y,2)+pow(H_z,2));
	if(strcmp(component,"EM_Energy")==0 && meps(i,j,k)!=0.0) return 0.5*eo*eps_m(i,j,k)*(pow(E_x,2)+pow(E_y,2)+pow(E_z,2))+0.5*uo*ups*(pow(H_x,2)+pow(H_y,2)+pow(H_z,2));
	if(strcmp(component,"EM_Energy2")==0 && meps(i,j,k)==0.0) return 0.0;
	if(strcmp(component,"EM_Energy2")==0 && meps(i,j,k)!=0.0) return 0.5*eo*eps_m2(i,j,k)*(pow(E_x,2)+pow(E_y,2)+pow(E_z,2))+0.5*uo*ups*(pow(H_x,2)+pow(H_y,2)+pow(H_z,2));
	if(strcmp(component,"EM_Energy3")==0 && meps(i,j,k)==0.0) return 0.0;
	if(strcmp(component,"EM_Energy3")==0 && meps(i,j,k)!=0.0) return 0.5*eo*eps_m3(i,j,k)*(pow(E_x,2)+pow(E_y,2)+pow(E_z,2))+0.5*uo*ups*(pow(H_x,2)+pow(H_y,2)+pow(H_z,2));

	if(strcmp(component,"E_Energy")==0 && meps(i,j,k)==0.0) return 0.5*eo*eps(i,j,k)*(pow(E_x,2)+pow(E_y,2)+pow(E_z,2));
	if(strcmp(component,"E_Energy")==0 && meps(i,j,k)!=0.0) return 0.5*eo*eps_m(i,j,k)*(pow(E_x,2)+pow(E_y,2)+pow(E_z,2));
	if(strcmp(component,"E_Energy2")==0 && meps(i,j,k)==0.0) return 0.0;
	if(strcmp(component,"E_Energy2")==0 && meps(i,j,k)!=0.0) return 0.5*eo*eps_m2(i,j,k)*(pow(E_x,2)+pow(E_y,2)+pow(E_z,2));
	if(strcmp(component,"E_Energy3")==0 && meps(i,j,k)==0.0) return 0.0;
	if(strcmp(component,"E_Energy3")==0 && meps(i,j,k)!=0.0) return 0.5*eo*eps_m3(i,j,k)*(pow(E_x,2)+pow(E_y,2)+pow(E_z,2));

	if(strcmp(component,"Ex2Ey2")==0) return pow(E_x,2)+pow(E_y,2);
	if(strcmp(component,"Ey2Ez2")==0) return pow(E_y,2)+pow(E_z,2);
	if(strcmp(component,"Ez2Ex2")==0) return pow(E_z,2)+pow(E_x,2);	
	if(strcmp(component,"LogEx2Ey2")==0) return log10(pow(E_x,2)+pow(E_y,2));
	if(strcmp(component,"LogEy2Ez2")==0) return log10(pow(E_y,2)+pow(E_z,2));
	if(strcmp(component,"LogEz2Ex2")==0) return log10(pow(E_z,2)+pow(E_x,2));
	if(strcmp(component,"Hx2Hy2")==0) return pow(H_x,2)+pow(H_y,2);
	if(strcmp(component,"Hy2Hz2")==0) return pow(H_y,2)+pow(H_z,2);
	if(strcmp(component,"Hz2Hx2")==0) return pow(H_z,2)+pow(H_x,2);	
	if(strcmp(component,"LogHx2Hy2")==0) return log10(pow(H_x,2)+pow(H_y,2));
	if(strcmp(component,"LogHy2Hz2")==0) return log10(pow(H_y,2)+pow(H_z,2));
	if(strcmp(component,"LogHz2Hx2")==0) return log10(pow(H_z,2)+pow(H_x,2));	
}

float grid_value_periodic_y(char *component,int i,int j,int k, int h)
{
	float E_x, E_y, E_z, J_x, J_y, J_z, H_x, H_y, H_z;

	E_x = efieldx(i,j,k)*cos(2*pi*h*wave_vector_y)-iefieldx(i,j,k)*sin(2*pi*h*wave_vector_y);
	E_y = efieldy(i,j,k)*cos(2*pi*h*wave_vector_y)-iefieldy(i,j,k)*sin(2*pi*h*wave_vector_y);
	E_z = efieldz(i,j,k)*cos(2*pi*h*wave_vector_y)-iefieldz(i,j,k)*sin(2*pi*h*wave_vector_y);
	J_x = jfieldx(i,j,k)*cos(2*pi*h*wave_vector_y)-ijfieldx(i,j,k)*sin(2*pi*h*wave_vector_y);
	J_y = jfieldy(i,j,k)*cos(2*pi*h*wave_vector_y)-ijfieldy(i,j,k)*sin(2*pi*h*wave_vector_y);
	J_z = jfieldz(i,j,k)*cos(2*pi*h*wave_vector_y)-ijfieldz(i,j,k)*sin(2*pi*h*wave_vector_y);
	H_x = hfieldx(i,j,k)*cos(2*pi*h*wave_vector_y)-ihfieldx(i,j,k)*sin(2*pi*h*wave_vector_y);
	H_y = hfieldy(i,j,k)*cos(2*pi*h*wave_vector_y)-ihfieldy(i,j,k)*sin(2*pi*h*wave_vector_y);
	H_z = hfieldz(i,j,k)*cos(2*pi*h*wave_vector_y)-ihfieldz(i,j,k)*sin(2*pi*h*wave_vector_y);

	if(strcmp(component,"Ex")==0) return (E_x);
	if(strcmp(component,"Ey")==0) return (E_y);
	if(strcmp(component,"Ez")==0) return (E_z);
	if(strcmp(component,"Hx")==0) return (H_x);
	if(strcmp(component,"Hy")==0) return (H_y);
	if(strcmp(component,"Hz")==0) return (H_z);
	if(strcmp(component,"Jx")==0) return (J_x);
	if(strcmp(component,"Jy")==0) return (J_y);
	if(strcmp(component,"Jz")==0) return (J_z);
	if(strcmp(component,"J.E")==0) return E_x*J_x+E_y*J_y+E_z*J_z;
	if(strcmp(component,"E^2")==0) return pow(E_x,2)+pow(E_y,2)+pow(E_z,2);
	if(strcmp(component,"H^2")==0) return pow(H_x,2)+pow(H_y,2)+pow(H_z,2);
	if(strcmp(component,"Sx")==0) return E_y*H_z-E_z*H_y;
	if(strcmp(component,"Sy")==0) return E_z*H_x-E_x*H_z;
	if(strcmp(component,"Sz")==0) return E_x*H_y-E_y*H_x;
	if(strcmp(component,"LogE^2")==0) return log10(pow(E_x,2)+pow(E_y,2)+pow(E_z,2));
	if(strcmp(component,"LogH^2")==0) return log10(pow(H_x,2)+pow(H_y,2)+pow(H_z,2));

	if(strcmp(component,"EM_Energy")==0 && meps(i,j,k)==0.0) return 0.5*eo*eps(i,j,k)*(pow(E_x,2)+pow(E_y,2)+pow(E_z,2))+0.5*uo*ups*(pow(H_x,2)+pow(H_y,2)+pow(H_z,2));
	if(strcmp(component,"EM_Energy")==0 && meps(i,j,k)!=0.0) return 0.5*eo*eps_m(i,j,k)*(pow(E_x,2)+pow(E_y,2)+pow(E_z,2))+0.5*uo*ups*(pow(H_x,2)+pow(H_y,2)+pow(H_z,2));
	if(strcmp(component,"EM_Energy2")==0 && meps(i,j,k)==0.0) return 0.0;
	if(strcmp(component,"EM_Energy2")==0 && meps(i,j,k)!=0.0) return 0.5*eo*eps_m2(i,j,k)*(pow(E_x,2)+pow(E_y,2)+pow(E_z,2))+0.5*uo*ups*(pow(H_x,2)+pow(H_y,2)+pow(H_z,2));
	if(strcmp(component,"EM_Energy3")==0 && meps(i,j,k)==0.0) return 0.0;
	if(strcmp(component,"EM_Energy3")==0 && meps(i,j,k)!=0.0) return 0.5*eo*eps_m3(i,j,k)*(pow(E_x,2)+pow(E_y,2)+pow(E_z,2))+0.5*uo*ups*(pow(H_x,2)+pow(H_y,2)+pow(H_z,2));

	if(strcmp(component,"E_Energy")==0 && meps(i,j,k)==0.0) return 0.5*eo*eps(i,j,k)*(pow(E_x,2)+pow(E_y,2)+pow(E_z,2));
	if(strcmp(component,"E_Energy")==0 && meps(i,j,k)!=0.0) return 0.5*eo*eps_m(i,j,k)*(pow(E_x,2)+pow(E_y,2)+pow(E_z,2));
	if(strcmp(component,"E_Energy2")==0 && meps(i,j,k)==0.0) return 0.0;
	if(strcmp(component,"E_Energy2")==0 && meps(i,j,k)!=0.0) return 0.5*eo*eps_m2(i,j,k)*(pow(E_x,2)+pow(E_y,2)+pow(E_z,2));
	if(strcmp(component,"E_Energy3")==0 && meps(i,j,k)==0.0) return 0.0;
	if(strcmp(component,"E_Energy3")==0 && meps(i,j,k)!=0.0) return 0.5*eo*eps_m3(i,j,k)*(pow(E_x,2)+pow(E_y,2)+pow(E_z,2));

	if(strcmp(component,"Ex2Ey2")==0) return pow(E_x,2)+pow(E_y,2);
	if(strcmp(component,"Ey2Ez2")==0) return pow(E_y,2)+pow(E_z,2);
	if(strcmp(component,"Ez2Ex2")==0) return pow(E_z,2)+pow(E_x,2);	
	if(strcmp(component,"LogEx2Ey2")==0) return log10(pow(E_x,2)+pow(E_y,2));
	if(strcmp(component,"LogEy2Ez2")==0) return log10(pow(E_y,2)+pow(E_z,2));
	if(strcmp(component,"LogEz2Ex2")==0) return log10(pow(E_z,2)+pow(E_x,2));
	if(strcmp(component,"Hx2Hy2")==0) return pow(H_x,2)+pow(H_y,2);
	if(strcmp(component,"Hy2Hz2")==0) return pow(H_y,2)+pow(H_z,2);
	if(strcmp(component,"Hz2Hx2")==0) return pow(H_z,2)+pow(H_x,2);	
	if(strcmp(component,"LogHx2Hy2")==0) return log10(pow(H_x,2)+pow(H_y,2));
	if(strcmp(component,"LogHy2Hz2")==0) return log10(pow(H_y,2)+pow(H_z,2));
	if(strcmp(component,"LogHz2Hx2")==0) return log10(pow(H_z,2)+pow(H_x,2));	
}

float grid_value_periodic_xy(char *component,int i,int j,int k, int h, int v)
{
	float E_x, E_y, E_z, J_x, J_y, J_z, H_x, H_y, H_z;

	E_x = efieldx(i,j,k)*cos(2*pi*h*wave_vector_x+v*wave_vector_y)-iefieldx(i,j,k)*sin(2*pi*h*wave_vector_x+v*wave_vector_y);
	E_y = efieldy(i,j,k)*cos(2*pi*h*wave_vector_x+v*wave_vector_y)-iefieldy(i,j,k)*sin(2*pi*h*wave_vector_x+v*wave_vector_y);
	E_z = efieldz(i,j,k)*cos(2*pi*h*wave_vector_x+v*wave_vector_y)-iefieldz(i,j,k)*sin(2*pi*h*wave_vector_x+v*wave_vector_y);
	J_x = jfieldx(i,j,k)*cos(2*pi*h*wave_vector_x+v*wave_vector_y)-ijfieldx(i,j,k)*sin(2*pi*h*wave_vector_x+v*wave_vector_y);
	J_y = jfieldy(i,j,k)*cos(2*pi*h*wave_vector_x+v*wave_vector_y)-ijfieldy(i,j,k)*sin(2*pi*h*wave_vector_x+v*wave_vector_y);
	J_z = jfieldz(i,j,k)*cos(2*pi*h*wave_vector_x+v*wave_vector_y)-ijfieldz(i,j,k)*sin(2*pi*h*wave_vector_x+v*wave_vector_y);
	H_x = hfieldx(i,j,k)*cos(2*pi*h*wave_vector_x+v*wave_vector_y)-ihfieldx(i,j,k)*sin(2*pi*h*wave_vector_x+v*wave_vector_y);
	H_y = hfieldy(i,j,k)*cos(2*pi*h*wave_vector_x+v*wave_vector_y)-ihfieldy(i,j,k)*sin(2*pi*h*wave_vector_x+v*wave_vector_y);
	H_z = hfieldz(i,j,k)*cos(2*pi*h*wave_vector_x+v*wave_vector_y)-ihfieldz(i,j,k)*sin(2*pi*h*wave_vector_x+v*wave_vector_y);

	if(strcmp(component,"Ex")==0) return (E_x);
	if(strcmp(component,"Ey")==0) return (E_y);
	if(strcmp(component,"Ez")==0) return (E_z);
	if(strcmp(component,"Hx")==0) return (H_x);
	if(strcmp(component,"Hy")==0) return (H_y);
	if(strcmp(component,"Hz")==0) return (H_z);
	if(strcmp(component,"Jx")==0) return (J_x);
	if(strcmp(component,"Jy")==0) return (J_y);
	if(strcmp(component,"Jz")==0) return (J_z);
	if(strcmp(component,"J.E")==0) return E_x*J_x+E_y*J_y+E_z*J_z;
	if(strcmp(component,"E^2")==0) return pow(E_x,2)+pow(E_y,2)+pow(E_z,2);
	if(strcmp(component,"H^2")==0) return pow(H_x,2)+pow(H_y,2)+pow(H_z,2);
	if(strcmp(component,"Sx")==0) return E_y*H_z-E_z*H_y;
	if(strcmp(component,"Sy")==0) return E_z*H_x-E_x*H_z;
	if(strcmp(component,"Sz")==0) return E_x*H_y-E_y*H_x;
	if(strcmp(component,"LogE^2")==0) return log10(pow(E_x,2)+pow(E_y,2)+pow(E_z,2));
	if(strcmp(component,"LogH^2")==0) return log10(pow(H_x,2)+pow(H_y,2)+pow(H_z,2));

	if(strcmp(component,"EM_Energy")==0 && meps(i,j,k)==0.0) return 0.5*eo*eps(i,j,k)*(pow(E_x,2)+pow(E_y,2)+pow(E_z,2))+0.5*uo*ups*(pow(H_x,2)+pow(H_y,2)+pow(H_z,2));
	if(strcmp(component,"EM_Energy")==0 && meps(i,j,k)!=0.0) return 0.5*eo*eps_m(i,j,k)*(pow(E_x,2)+pow(E_y,2)+pow(E_z,2))+0.5*uo*ups*(pow(H_x,2)+pow(H_y,2)+pow(H_z,2));
	if(strcmp(component,"EM_Energy2")==0 && meps(i,j,k)==0.0) return 0.0;
	if(strcmp(component,"EM_Energy2")==0 && meps(i,j,k)!=0.0) return 0.5*eo*eps_m2(i,j,k)*(pow(E_x,2)+pow(E_y,2)+pow(E_z,2))+0.5*uo*ups*(pow(H_x,2)+pow(H_y,2)+pow(H_z,2));
	if(strcmp(component,"EM_Energy3")==0 && meps(i,j,k)==0.0) return 0.0;
	if(strcmp(component,"EM_Energy3")==0 && meps(i,j,k)!=0.0) return 0.5*eo*eps_m3(i,j,k)*(pow(E_x,2)+pow(E_y,2)+pow(E_z,2))+0.5*uo*ups*(pow(H_x,2)+pow(H_y,2)+pow(H_z,2));

	if(strcmp(component,"E_Energy")==0 && meps(i,j,k)==0.0) return 0.5*eo*eps(i,j,k)*(pow(E_x,2)+pow(E_y,2)+pow(E_z,2));
	if(strcmp(component,"E_Energy")==0 && meps(i,j,k)!=0.0) return 0.5*eo*eps_m(i,j,k)*(pow(E_x,2)+pow(E_y,2)+pow(E_z,2));
	if(strcmp(component,"E_Energy2")==0 && meps(i,j,k)==0.0) return 0.0;
	if(strcmp(component,"E_Energy2")==0 && meps(i,j,k)!=0.0) return 0.5*eo*eps_m2(i,j,k)*(pow(E_x,2)+pow(E_y,2)+pow(E_z,2));
	if(strcmp(component,"E_Energy3")==0 && meps(i,j,k)==0.0) return 0.0;
	if(strcmp(component,"E_Energy3")==0 && meps(i,j,k)!=0.0) return 0.5*eo*eps_m3(i,j,k)*(pow(E_x,2)+pow(E_y,2)+pow(E_z,2));

	if(strcmp(component,"Ex2Ey2")==0) return pow(E_x,2)+pow(E_y,2);
	if(strcmp(component,"Ey2Ez2")==0) return pow(E_y,2)+pow(E_z,2);
	if(strcmp(component,"Ez2Ex2")==0) return pow(E_z,2)+pow(E_x,2);	
	if(strcmp(component,"LogEx2Ey2")==0) return log10(pow(E_x,2)+pow(E_y,2));
	if(strcmp(component,"LogEy2Ez2")==0) return log10(pow(E_y,2)+pow(E_z,2));
	if(strcmp(component,"LogEz2Ex2")==0) return log10(pow(E_z,2)+pow(E_x,2));
	if(strcmp(component,"Hx2Hy2")==0) return pow(H_x,2)+pow(H_y,2);
	if(strcmp(component,"Hy2Hz2")==0) return pow(H_y,2)+pow(H_z,2);
	if(strcmp(component,"Hz2Hx2")==0) return pow(H_z,2)+pow(H_x,2);	
	if(strcmp(component,"LogHx2Hy2")==0) return log10(pow(H_x,2)+pow(H_y,2));
	if(strcmp(component,"LogHy2Hz2")==0) return log10(pow(H_y,2)+pow(H_z,2));
	if(strcmp(component,"LogHz2Hx2")==0) return log10(pow(H_z,2)+pow(H_x,2));	
}

float efieldx(int i,int j,int k)
{
	int p=1;

	if(pisize<=i && (xparity==1 || xparity==-1)) {i=isize-i;p=p*xparity;}
	if(pjsize<=j && (yparity==1 || yparity==-1)) {j=jsize-j;p=p*(-yparity);}
	if(pksize<=k && (zparity==1 || zparity==-1)) {k=ksize-k;p=p*(-zparity);}

	return p*(Ex[i][j][k]+Ex[i-1][j][k])/2;
}

float efieldy(int i,int j,int k)
{
	int p=1;

	if(pisize<=i && (xparity==1 || xparity==-1)) {i=isize-i;p=p*(-xparity);}
	if(pjsize<=j && (yparity==1 || yparity==-1)) {j=jsize-j;p=p*yparity;}
	if(pksize<=k && (zparity==1 || zparity==-1)) {k=ksize-k;p=p*(-zparity);}

	return p*(Ey[i][j][k]+Ey[i][j-1][k])/2;
}

float efieldz(int i,int j,int k)
{
	int p=1;

	if(pisize<=i && (xparity==1 || xparity==-1)) {i=isize-i;p=p*(-xparity);}
	if(pjsize<=j && (yparity==1 || yparity==-1)) {j=jsize-j;p=p*(-yparity);}
	if(pksize<=k && (zparity==1 || zparity==-1)) {k=ksize-k;p=p*zparity;}

	return p*(Ez[i][j][k]+Ez[i][j][k-1])/2;
}

float jfieldx(int i,int j,int k)
{
	int p=1;

	if(pisize<=i && (xparity==1 || xparity==-1)) {i=isize-i;p=p*xparity;}
	if(pjsize<=j && (yparity==1 || yparity==-1)) {j=jsize-j;p=p*(-yparity);}
	if(pksize<=k && (zparity==1 || zparity==-1)) {k=ksize-k;p=p*(-zparity);}

	return p*(Jx[i][j][k]+Jx[i-1][j][k])/2;
}

float jfieldy(int i,int j,int k)
{
	int p=1;

	if(pisize<=i && (xparity==1 || xparity==-1)) {i=isize-i;p=p*(-xparity);}
	if(pjsize<=j && (yparity==1 || yparity==-1)) {j=jsize-j;p=p*yparity;}
	if(pksize<=k && (zparity==1 || zparity==-1)) {k=ksize-k;p=p*(-zparity);}

	return p*(Jy[i][j][k]+Jy[i][j-1][k])/2;
}

float jfieldz(int i,int j,int k)
{
	int p=1;

	if(pisize<=i && (xparity==1 || xparity==-1)) {i=isize-i;p=p*(-xparity);}
	if(pjsize<=j && (yparity==1 || yparity==-1)) {j=jsize-j;p=p*(-yparity);}
	if(pksize<=k && (zparity==1 || zparity==-1)) {k=ksize-k;p=p*zparity;}

	return p*(Jz[i][j][k]+Jz[i][j][k-1])/2;
}

float hfieldx(int i,int j,int k)
{
	int p=1;

	if(pisize<=i && (xparity==1 || xparity==-1)) {i=isize-i;p=p*(-xparity);}
	if(pjsize<=j && (yparity==1 || yparity==-1)) {j=jsize-j;p=p*yparity;}
	if(pksize<=k && (zparity==1 || zparity==-1)) {k=ksize-k;p=p*zparity;}

	return p*(Hx[i][j][k]+Hx[i][j-1][k]+Hx[i][j][k-1]+Hx[i][j-1][k-1])/4;
}

float hfieldy(int i,int j,int k)
{
	int p=1;

	if(pisize<=i && (xparity==1 || xparity==-1)) {i=isize-i;p=p*xparity;}
	if(pjsize<=j && (yparity==1 || yparity==-1)) {j=jsize-j;p=p*(-yparity);}
	if(pksize<=k && (zparity==1 || zparity==-1)) {k=ksize-k;p=p*zparity;}

	return p*(Hy[i][j][k]+Hy[i][j][k-1]+Hy[i-1][j][k]+Hy[i-1][j][k-1])/4;
}

float hfieldz(int i,int j,int k)
{
	int p=1;

	if(pisize<=i && (xparity==1 || xparity==-1)) {i=isize-i;p=p*xparity;}
	if(pjsize<=j && (yparity==1 || yparity==-1)) {j=jsize-j;p=p*yparity;}
	if(pksize<=k && (zparity==1 || zparity==-1)) {k=ksize-k;p=p*(-zparity);}

	return p*(Hz[i][j][k]+Hz[i-1][j][k]+Hz[i][j-1][k]+Hz[i-1][j-1][k])/4;
}

float iefieldx(int i,int j,int k)
{
	int p=1;

	if(pisize<=i && (xparity==1 || xparity==-1)) {i=isize-i;p=p*xparity;}
	if(pjsize<=j && (yparity==1 || yparity==-1)) {j=jsize-j;p=p*(-yparity);}
	if(pksize<=k && (zparity==1 || zparity==-1)) {k=ksize-k;p=p*(-zparity);}

	return p*(iEx[i][j][k]+iEx[i-1][j][k])/2;
}

float iefieldy(int i,int j,int k)
{
	int p=1;

	if(pisize<=i && (xparity==1 || xparity==-1)) {i=isize-i;p=p*(-xparity);}
	if(pjsize<=j && (yparity==1 || yparity==-1)) {j=jsize-j;p=p*yparity;}
	if(pksize<=k && (zparity==1 || zparity==-1)) {k=ksize-k;p=p*(-zparity);}

	return p*(iEy[i][j][k]+iEy[i][j-1][k])/2;
}

float iefieldz(int i,int j,int k)
{
	int p=1;

	if(pisize<=i && (xparity==1 || xparity==-1)) {i=isize-i;p=p*(-xparity);}
	if(pjsize<=j && (yparity==1 || yparity==-1)) {j=jsize-j;p=p*(-yparity);}
	if(pksize<=k && (zparity==1 || zparity==-1)) {k=ksize-k;p=p*zparity;}

	return p*(iEz[i][j][k]+iEz[i][j][k-1])/2;
}

float ijfieldx(int i,int j,int k)
{
	int p=1;

	if(pisize<=i && (xparity==1 || xparity==-1)) {i=isize-i;p=p*xparity;}
	if(pjsize<=j && (yparity==1 || yparity==-1)) {j=jsize-j;p=p*(-yparity);}
	if(pksize<=k && (zparity==1 || zparity==-1)) {k=ksize-k;p=p*(-zparity);}

	return p*(iJx[i][j][k]+iJx[i-1][j][k])/2;
}

float ijfieldy(int i,int j,int k)
{
	int p=1;

	if(pisize<=i && (xparity==1 || xparity==-1)) {i=isize-i;p=p*(-xparity);}
	if(pjsize<=j && (yparity==1 || yparity==-1)) {j=jsize-j;p=p*yparity;}
	if(pksize<=k && (zparity==1 || zparity==-1)) {k=ksize-k;p=p*(-zparity);}

	return p*(iJy[i][j][k]+iJy[i][j-1][k])/2;
}

float ijfieldz(int i,int j,int k)
{
	int p=1;

	if(pisize<=i && (xparity==1 || xparity==-1)) {i=isize-i;p=p*(-xparity);}
	if(pjsize<=j && (yparity==1 || yparity==-1)) {j=jsize-j;p=p*(-yparity);}
	if(pksize<=k && (zparity==1 || zparity==-1)) {k=ksize-k;p=p*zparity;}

	return p*(iJz[i][j][k]+iJz[i][j][k-1])/2;
}

float ihfieldx(int i,int j,int k)
{
	int p=1;

	if(pisize<=i && (xparity==1 || xparity==-1)) {i=isize-i;p=p*(-xparity);}
	if(pjsize<=j && (yparity==1 || yparity==-1)) {j=jsize-j;p=p*yparity;}
	if(pksize<=k && (zparity==1 || zparity==-1)) {k=ksize-k;p=p*zparity;}

	return p*(iHx[i][j][k]+iHx[i][j-1][k]+iHx[i][j][k-1]+iHx[i][j-1][k-1])/4;
}

float ihfieldy(int i,int j,int k)
{
	int p=1;

	if(pisize<=i && (xparity==1 || xparity==-1)) {i=isize-i;p=p*xparity;}
	if(pjsize<=j && (yparity==1 || yparity==-1)) {j=jsize-j;p=p*(-yparity);}
	if(pksize<=k && (zparity==1 || zparity==-1)) {k=ksize-k;p=p*zparity;}

	return p*(iHy[i][j][k]+iHy[i][j][k-1]+iHy[i-1][j][k]+iHy[i-1][j][k-1])/4;
}

float ihfieldz(int i,int j,int k)
{
	int p=1;

	if(pisize<=i && (xparity==1 || xparity==-1)) {i=isize-i;p=p*xparity;}
	if(pjsize<=j && (yparity==1 || yparity==-1)) {j=jsize-j;p=p*yparity;}
	if(pksize<=k && (zparity==1 || zparity==-1)) {k=ksize-k;p=p*(-zparity);}

	return p*(iHz[i][j][k]+iHz[i-1][j][k]+iHz[i][j-1][k]+iHz[i-1][j-1][k])/4;
}

float eps(int i,int j,int k)
{
	if(pisize<=i && (xparity==1 || xparity==-1)) i=isize-i;  
	if(pjsize<=j && (yparity==1 || yparity==-1)) j=jsize-j; 
	if(pksize<=k && (zparity==1 || zparity==-1)) k=ksize-k;

	return (epsilonx[i][j][k]+epsilonx[i-1][j][k]+epsilony[i][j][k]+epsilony[i][j-1][k]+epsilonz[i][j][k]+epsilonz[i][j][k-1])/6;
}

int nz_multiple(int k)
{
	if(pksize<=k && (zparity==1 || zparity==-1)) k=ksize-k;

	return(ds_nz[k]);
}

float meps(int i,int j,int k)
{
	if(pisize<=i && (xparity==1 || xparity==-1)) i=isize-i;  
	if(pjsize<=j && (yparity==1 || yparity==-1)) j=jsize-j; 
	if(pksize<=k && (zparity==1 || zparity==-1)) k=ksize-k;

	return mepsilon[i][j][k];
}

float eps_m(int i,int j,int k)
{
	float wp, w, go;

	if(pisize<=i && (xparity==1 || xparity==-1)) i=isize-i;  
	if(pjsize<=j && (yparity==1 || yparity==-1)) j=jsize-j; 
	if(pksize<=k && (zparity==1 || zparity==-1)) k=ksize-k;

	wp = momega[i][j][k];
	go = mgamma[i][j][k];
	w = global_W*2*3.1415926*light_speed/(ds_x*lattice_x); 

	//return(mepsilon[i][j][k] + wp*wp*(w*w-go*go)/((w*w+go*go)*(w*w+go*go)) );
		//corrections by S. L. Chuang (ver.8.785)
	//return(2*mepsilon[i][j][k] - wp*wp/(w*w+go*go) + wp*wp*(w*w-go*go)/((w*w+go*go)*(w*w+go*go)) );  		//additional a factor of 1/2 correction (ver.8.801)
	return(mepsilon[i][j][k] - 0.5*wp*wp/(w*w+go*go) + 0.5*wp*wp*(w*w-go*go)/((w*w+go*go)*(w*w+go*go)) );  
}

float eps_m2(int i,int j,int k) //for calculting kinetic energy of electrons (ver.8.78)
{
	float wp, w, go;

	if(pisize<=i && (xparity==1 || xparity==-1)) i=isize-i;  
	if(pjsize<=j && (yparity==1 || yparity==-1)) j=jsize-j; 
	if(pksize<=k && (zparity==1 || zparity==-1)) k=ksize-k;

	wp = momega[i][j][k];
	go = mgamma[i][j][k];
	w = global_W*2*3.1415926*light_speed/(ds_x*lattice_x); 

	return(wp*wp*(w*w-go*go)/((w*w+go*go)*(w*w+go*go)) );
}

float eps_m3(int i,int j,int k) //for calculting potential energy of electrons (ver.8.78)
{
	float wp, w, go;

	if(pisize<=i && (xparity==1 || xparity==-1)) i=isize-i;  
	if(pjsize<=j && (yparity==1 || yparity==-1)) j=jsize-j; 
	if(pksize<=k && (zparity==1 || zparity==-1)) k=ksize-k;

	return(mepsilon[i][j][k]);
}

void print_amp_and_phase(int mm)
{
	int i, j;
	FILE *amp, *phase;
	char name_freq[10];

	char name_Ex_amp[20], name_Ex_phase[20], name_Ey_amp[20], name_Ey_phase[20];
	char name_Hx_amp[20], name_Hx_phase[20], name_Hy_amp[20], name_Hy_phase[20];

	//////// making file names /////////
	sprintf(name_freq,".ap%02d",mm);

	sprintf(name_Ex_amp,"jEx_amp");
	sprintf(name_Ex_phase,"jEx_phase");
	strcat(name_Ex_amp,name_freq);
	strcat(name_Ex_phase,name_freq);

	sprintf(name_Ey_amp,"jEy_amp");
	sprintf(name_Ey_phase,"jEy_phase");
	strcat(name_Ey_amp,name_freq);
	strcat(name_Ey_phase,name_freq);

	sprintf(name_Hx_amp,"jHx_amp");
	sprintf(name_Hx_phase,"jHx_phase");
	strcat(name_Hx_amp,name_freq);
	strcat(name_Hx_phase,name_freq);

	sprintf(name_Hy_amp,"jHy_amp");
	sprintf(name_Hy_phase,"jHy_phase");
	strcat(name_Hy_amp,name_freq);
	strcat(name_Hy_phase,name_freq);

	/////////////////////////////////////
	/////////// Print Ex dat ////////////
	/////////////////////////////////////
	amp = fopen(name_Ex_amp,"wt");
	phase = fopen(name_Ex_phase,"wt");
	for(j=jsize-2; j>=1; j--)
	{
		for(i=1; i<=isize-2; i++)
		{
			if( (i >= pmlil && i <=isize-pmlir) && (j >= pmljl && j <= jsize-pmljr))
			{
				fprintf(amp,"%f\t", sqrt( pow(Ex_cos[i][j][mm],2) + pow(Ex_sin[i][j][mm],2) ));
				fprintf(phase,"%f\t", -atan2( Ex_sin[i][j][mm], Ex_cos[i][j][mm] ));
			}
			else
			{
				fprintf(amp,"%f\t", 0.0);
				fprintf(phase,"%f\t", 0.0);
			}
		}
		fprintf(amp, "\n");
		fprintf(phase, "\n");
	}
	fclose(amp); fclose(phase);

	/////////////////////////////////////
	/////////// Print Ey dat ////////////
	/////////////////////////////////////
	amp = fopen(name_Ey_amp,"wt");
	phase = fopen(name_Ey_phase,"wt");
	for(j=jsize-2; j>=1; j--)
	{
		for(i=1; i<=isize-2; i++)
		{
			if( (i >= pmlil && i <=isize-pmlir) && (j >= pmljl && j <= jsize-pmljr))
			{
				fprintf(amp,"%f\t", sqrt( pow(Ey_cos[i][j][mm],2) + pow(Ey_sin[i][j][mm],2) ));
				fprintf(phase,"%f\t", -atan2( Ey_sin[i][j][mm], Ey_cos[i][j][mm] ));
			}
			else
			{
				fprintf(amp, "%f\t", 0.0);
				fprintf(phase, "%f\t", 0.0);
			}
		}
		fprintf(amp, "\n");
		fprintf(phase, "\n");
	}
	fclose(amp); fclose(phase);

	/////////////////////////////////////
	/////////// Print Hx dat ////////////
	/////////////////////////////////////
	amp = fopen(name_Hx_amp,"wt");
	phase = fopen(name_Hx_phase,"wt");
	for(j=jsize-2; j>=1; j--)
	{
		for(i=1; i<=isize-2; i++)
		{
			if((i >= pmlil && i <=isize-pmlir) && (j >= pmljl && j <= jsize-pmljr))
			{
				fprintf(amp,"%f\t", sqrt( pow(Hx_cos[i][j][mm],2) + pow(Hx_sin[i][j][mm],2) ));
				fprintf(phase,"%f\t", -atan2( Hx_sin[i][j][mm], Hx_cos[i][j][mm] ));
			}
			else
			{
				fprintf(amp, "%f\t", 0.0);
				fprintf(phase, "%f\t", 0.0);
			}
		}
		fprintf(amp, "\n");
		fprintf(phase, "\n");
	}
	fclose(amp); fclose(phase);

	/////////////////////////////////////
	/////////// Print Hy dat ////////////
	/////////////////////////////////////
	amp = fopen(name_Hy_amp,"wt");
	phase = fopen(name_Hy_phase,"wt");
	for(j=jsize-2; j>=1; j--)
	{
		for(i=1; i<=isize-2; i++)
		{
			if( (i >= pmlil && i <=isize-pmlir) && (j >= pmljl && j <= jsize-pmljr))
			{
				fprintf(amp,"%f\t", sqrt( pow(Hy_cos[i][j][mm],2) + pow(Hy_sin[i][j][mm],2) ));
				fprintf(phase,"%f\t", -atan2( Hy_sin[i][j][mm], Hy_cos[i][j][mm] ));
			}
			else
			{
				fprintf(amp, "%f\t", 0.0);
				fprintf(phase, "%f\t", 0.0);
			}
		}
		fprintf(amp, "\n");
		fprintf(phase, "\n");
	}
	fclose(amp); fclose(phase);

	printf("print amp and phase [%d] ok....!\n",mm);

}

void print_real_and_imag_2n_size(int NROW, int mm)
{
	char string[80];  

	int row, col; // row num & column num
	int deltar; // (nrow-row)/2
	int deltac;

	int i,j;
	FILE *stream;

	char name_freq[10];

	char name_Ex_real[20], name_Ex_imag[20], name_Ey_real[20], name_Ey_imag[20];
	char name_Hx_real[20], name_Hx_imag[20], name_Hy_real[20], name_Hy_imag[20];

	//////// making file names /////////
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

	/////////////////////////////////////////////////
	// counting row num, col num ////////////////////
	/////////////////////////////////////////////////
	row = jsize-2; col = isize-2; 
	deltar = (NROW-row)/2 + (pmljl+10+pmljr+10)/2;  // triming out pml region that may cause numerical errors
	deltac = (NROW-col)/2 + (pmlil+10+pmlir+10)/2;

	/////////////////////////////////////////////////
	// File out enlarged filed data /////////////////
	/////////////////////////////////////////////////

	///////// Ex component ////////////	
	stream = fopen(name_Ex_real,"wt");
	for(j=NROW-1; j>=0; j--)
	{
		for(i=0; i<NROW; i++)
		{
			if( (i>=deltac && i<(NROW-deltac)) && (j>=deltar && j<(NROW-deltar)) )
				fprintf(stream, "%f\t", Ex_cos[i-deltac+(pmlil+10)][j-deltar+(pmlir+10)][mm] );
			else
				fprintf(stream, "%f\t", 0.0);				
		}
		fprintf(stream, "\n");
	}
	fclose(stream);

	stream = fopen(name_Ex_imag,"wt");
	for(j=NROW-1; j>=0; j--)
	{
		for(i=0; i<NROW; i++)
		{
			if( (i>=deltac && i<(NROW-deltac)) && (j>=deltar && j<(NROW-deltar)) )
				fprintf(stream, "%f\t", Ex_sin[i-deltac+(pmlil+10)][j-deltar+(pmlir+10)][mm] );
			else
				fprintf(stream, "%f\t", 0.0);				
		}
		fprintf(stream, "\n");
	}
	fclose(stream);

	///////// Ey component ////////////	
	stream = fopen(name_Ey_real,"wt");
	for(j=NROW-1; j>=0; j--)
	{
		for(i=0; i<NROW; i++)
		{
			if( (i>=deltac && i<(NROW-deltac)) && (j>=deltar && j<(NROW-deltar)) )
				fprintf(stream, "%f\t", Ey_cos[i-deltac+(pmlil+10)][j-deltar+(pmlir+10)][mm] );
			else
				fprintf(stream, "%f\t", 0.0);				
		}
		fprintf(stream, "\n");
	}
	fclose(stream);

	stream = fopen(name_Ey_imag,"wt");
	for(j=NROW-1; j>=0; j--)
	{
		for(i=0; i<NROW; i++)
		{
			if( (i>=deltac && i<(NROW-deltac)) && (j>=deltar && j<(NROW-deltar)) )
				fprintf(stream, "%f\t", Ey_sin[i-deltac+(pmlil+10)][j-deltar+(pmlir+10)][mm] );
			else
				fprintf(stream, "%f\t", 0.0);				
		}
		fprintf(stream, "\n");
	}
	fclose(stream);

	///////// Hx component ////////////	
	stream = fopen(name_Hx_real,"wt");
	for(j=NROW-1; j>=0; j--)
	{
		for(i=0; i<NROW; i++)
		{
			if( (i>=deltac && i<(NROW-deltac)) && (j>=deltar && j<(NROW-deltar)) )
				fprintf(stream, "%f\t", Hx_cos[i-deltac+(pmlil+10)][j-deltar+(pmlir+10)][mm] );
			else
				fprintf(stream, "%f\t", 0.0);				
		}
		fprintf(stream, "\n");
	}
	fclose(stream);

	stream = fopen(name_Hx_imag,"wt");
	for(j=NROW-1; j>=0; j--)
	{
		for(i=0; i<NROW; i++)
		{
			if( (i>=deltac && i<(NROW-deltac)) && (j>=deltar && j<(NROW-deltar)) )
				fprintf(stream, "%f\t", Hx_sin[i-deltac+(pmlil+10)][j-deltar+(pmlir+10)][mm] );
			else
				fprintf(stream, "%f\t", 0.0);				
		}
		fprintf(stream, "\n");
	}
	fclose(stream);

	///////// Hy component ////////////	
	stream = fopen(name_Hy_real,"wt");
	for(j=NROW-1; j>=0; j--)
	{
		for(i=0; i<NROW; i++)
		{
			if( (i>=deltac && i<(NROW-deltac)) && (j>=deltar && j<(NROW-deltar)) )
				fprintf(stream, "%f\t", Hy_cos[i-deltac+(pmlil+10)][j-deltar+(pmlir+10)][mm] );
			else
				fprintf(stream, "%f\t", 0.0);				
		}
		fprintf(stream, "\n");
	}
	fclose(stream);

	stream = fopen(name_Hy_imag,"wt");
	for(j=NROW-1; j>=0; j--)
	{
		for(i=0; i<NROW; i++)
		{
			if( (i>deltac && i<(NROW-deltac)) && (j>deltar && j<(NROW-deltar)) )
				fprintf(stream, "%f\t", Hy_sin[i-deltac+(pmlil+10)][j-deltar+(pmlir+10)][mm] );
			else
				fprintf(stream, "%f\t", 0.0);				
		}
		fprintf(stream, "\n");
	}
	fclose(stream);

	printf("make 2n_size data [%d] ok....!\n",mm);
}

void make_2n_size(int NROW, int mm)
{
	int deltar; // (NROW-row)/2
	int deltac;
	char ch;
	char string[80];  

	double **source; // source matrix

	int row, col; // row num & column num
	int i,j;

	FILE *stream;
	char name_freq[10];

	char name_tEx_real[20], name_tEx_imag[20], name_tEy_real[20], name_tEy_imag[20];
	char name_tHx_real[20], name_tHx_imag[20], name_tHy_real[20], name_tHy_imag[20];

	char name_Ex_real[20], name_Ex_imag[20], name_Ey_real[20], name_Ey_imag[20];
	char name_Hx_real[20], name_Hx_imag[20], name_Hy_real[20], name_Hy_imag[20];

	/////////////////////////////////////////////////
	//// Making file names   ////////////////////////
	/////////////////////////////////////////////////
	sprintf(name_freq,".ri%02d",mm);

	//// Source file name ////
	////----------------------
	sprintf(name_tEx_real,"tEx_real");
	sprintf(name_tEx_imag,"tEx_imag");
	strcat(name_tEx_real,name_freq);
	strcat(name_tEx_imag,name_freq);

	sprintf(name_tEy_real,"tEy_real");
	sprintf(name_tEy_imag,"tEy_imag");
	strcat(name_tEy_real,name_freq);
	strcat(name_tEy_imag,name_freq);

	sprintf(name_tHx_real,"tHx_real");
	sprintf(name_tHx_imag,"tHx_imag");
	strcat(name_tHx_real,name_freq);
	strcat(name_tHx_imag,name_freq);

	sprintf(name_tHy_real,"tHy_real");
	sprintf(name_tHy_imag,"tHy_imag");
	strcat(name_tHy_real,name_freq);
	strcat(name_tHy_imag,name_freq);

	//// Output file name ////
	////----------------------
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

	/////////////////////////////////////////////////
	// Count row and col ////////////////////////////
	/////////////////////////////////////////////////

	col = row = 0; 
	stream = fopen(name_tEx_real,"r");
	ch = getc(stream);
	while( ch != EOF )
	{
		if( ch == '\t' ) 
			col++;
		if( ch == '\n' )
		{
			row++;
		}
		ch = getc(stream);
	}
	fclose(stream);
	col = (int)(col / row); // in each row
	printf("t_data col = %d\n", col);
	printf("t_data row = %d\n", row);

	/////////////////////////////////////////////////
	// reading source ///////////////////////////////
	/////////////////////////////////////////////////

	deltar = (NROW-row)/2;
	deltac = (NROW-col)/2;

	source = (double **)malloc(sizeof(double *)*col);
	for(i=0; i<col; i++)
		source[i] = (double *)malloc(sizeof(double)*row);

	stream = fopen(name_tEx_real,"r");
	for(j=row-1; j>=0; j--)
	{
		for(i=0; i<col; i++)
		{
			fscanf(stream, "%s", &string);
			source[i][j] = atof(string);
		}
	}
	fclose(stream);

	stream = fopen(name_Ex_real,"wt");
	for(j=NROW-1; j>=0; j--)
	{
		for(i=0; i<NROW; i++)
		{
			if( (i>deltac && i<(NROW-deltac)) && (j>deltar && j<(NROW-deltar)) )
				fprintf(stream, "%lf\t", source[i-deltac][j-deltar]);
			else if( i == (NROW-1) )
				fprintf(stream, "%lf\t\n", 0.0);
			else
				fprintf(stream, "%lf\t", 0.0);				
		}
	}
	fclose(stream);

	stream = fopen(name_tEx_imag,"r");
	for(j=row-1; j>=0; j--)
	{
		for(i=0; i<col; i++)
		{
			fscanf(stream, "%s", &string);
			source[i][j] = atof(string);
		}
	}
	fclose(stream);

	stream = fopen(name_Ex_imag,"wt");
	for(j=NROW-1; j>=0; j--)
	{
		for(i=0; i<NROW; i++)
		{
			if( (i>deltac && i<(NROW-deltac)) && (j>deltar && j<(NROW-deltar)) )
				fprintf(stream, "%lf\t", source[i-deltac][j-deltar]);
			else if( i == (NROW-1) )
				fprintf(stream, "%lf\t\n", 0.0);
			else
				fprintf(stream, "%lf\t", 0.0);				
		}
	}
	fclose(stream);

	stream = fopen(name_tEy_real,"r");
	for(j=row-1; j>=0; j--)
	{
		for(i=0; i<col; i++)
		{
			fscanf(stream, "%s", &string);
			source[i][j] = atof(string);
		}
	}
	fclose(stream);

	stream = fopen(name_Ey_real,"wt");
	for(j=NROW-1; j>=0; j--)
	{
		for(i=0; i<NROW; i++)
		{
			if( (i>deltac && i<(NROW-deltac)) && (j>deltar && j<(NROW-deltar)) )
				fprintf(stream, "%lf\t", source[i-deltac][j-deltar]);
			else if( i == (NROW-1) )
				fprintf(stream, "%lf\t\n", 0.0);
			else
				fprintf(stream, "%lf\t", 0.0);				
		}
	}
	fclose(stream);

	stream = fopen(name_tEy_imag,"r");
	for(j=row-1; j>=0; j--)
	{
		for(i=0; i<col; i++)
		{
			fscanf(stream, "%s", &string);
			source[i][j] = atof(string);
		}
	}
	fclose(stream);

	stream = fopen(name_Ey_imag,"wt");
	for(j=NROW-1; j>=0; j--)
	{
		for(i=0; i<NROW; i++)
		{
			if( (i>deltac && i<(NROW-deltac)) && (j>deltar && j<(NROW-deltar)) )
				fprintf(stream, "%lf\t", source[i-deltac][j-deltar]);
			else if( i == (NROW-1) )
				fprintf(stream, "%lf\t\n", 0.0);
			else
				fprintf(stream, "%lf\t", 0.0);				
		}
	}
	fclose(stream);

	///////////////////////////////////////////

	stream = fopen(name_tHx_real,"r");
	for(j=row-1; j>=0; j--)
	{
		for(i=0; i<col; i++)
		{
			fscanf(stream, "%s", &string);
			source[i][j] = atof(string);
		}
	}
	fclose(stream);

	stream = fopen(name_Hx_real,"wt");
	for(j=NROW-1; j>=0; j--)
	{
		for(i=0; i<NROW; i++)
		{
			if( (i>deltac && i<(NROW-deltac)) && (j>deltar && j<(NROW-deltar)) )
				fprintf(stream, "%lf\t", source[i-deltac][j-deltar]);
			else if( i == (NROW-1) )
				fprintf(stream, "%lf\t\n", 0.0);
			else
				fprintf(stream, "%lf\t", 0.0);				
		}
	}
	fclose(stream);

	stream = fopen(name_tHx_imag,"r");
	for(j=row-1; j>=0; j--)
	{
		for(i=0; i<col; i++)
		{
			fscanf(stream, "%s", &string);
			source[i][j] = atof(string);
		}
	}
	fclose(stream);

	stream = fopen(name_Hx_imag,"wt");
	for(j=NROW-1; j>=0; j--)
	{
		for(i=0; i<NROW; i++)
		{
			if( (i>deltac && i<(NROW-deltac)) && (j>deltar && j<(NROW-deltar)) )
				fprintf(stream, "%lf\t", source[i-deltac][j-deltar]);
			else if( i == (NROW-1) )
				fprintf(stream, "%lf\t\n", 0.0);
			else
				fprintf(stream, "%lf\t", 0.0);				
		}
	}
	fclose(stream);

	stream = fopen(name_tHy_real,"r");
	for(j=row-1; j>=0; j--)
	{
		for(i=0; i<col; i++)
		{
			fscanf(stream, "%s", &string);
			source[i][j] = atof(string);
		}
	}
	fclose(stream);

	stream = fopen(name_Hy_real,"wt");
	for(j=NROW-1; j>=0; j--)
	{
		for(i=0; i<NROW; i++)
		{
			if( (i>deltac && i<(NROW-deltac)) && (j>deltar && j<(NROW-deltar)) )
				fprintf(stream, "%lf\t", source[i-deltac][j-deltar]);
			else if( i == (NROW-1) )
				fprintf(stream, "%lf\t\n", 0.0);
			else
				fprintf(stream, "%lf\t", 0.0);				
		}
	}
	fclose(stream);

	stream = fopen(name_tHy_imag,"r");
	for(j=row-1; j>=0; j--)
	{
		for(i=0; i<col; i++)
		{
			fscanf(stream, "%s", &string);
			source[i][j] = atof(string);
		}
	}
	fclose(stream);

	stream = fopen(name_Hy_imag,"wt");
	for(j=NROW-1; j>=0; j--)
	{
		for(i=0; i<NROW; i++)
		{
			if( (i>deltac && i<(NROW-deltac)) && (j>deltar && j<(NROW-deltar)) )
				fprintf(stream, "%lf\t", source[i-deltac][j-deltar]);
			else if( i == (NROW-1) )
				fprintf(stream, "%lf\t\n", 0.0);
			else
				fprintf(stream, "%lf\t", 0.0);				
		}
	}
	fclose(stream);

	free(source);
}

void print_real_and_imag(int mm)
{
	int i, j;
	FILE *real, *imag;
	char name_freq[10];

	char name_Ex_real[20], name_Ex_imag[20], name_Ey_real[20], name_Ey_imag[20];
	char name_Hx_real[20], name_Hx_imag[20], name_Hy_real[20], name_Hy_imag[20];

	//////// making file names /////////
	sprintf(name_freq,".ri%02d",mm);

	sprintf(name_Ex_real,"tEx_real");
	sprintf(name_Ex_imag,"tEx_imag");
	strcat(name_Ex_real,name_freq);
	strcat(name_Ex_imag,name_freq);

	sprintf(name_Ey_real,"tEy_real");
	sprintf(name_Ey_imag,"tEy_imag");
	strcat(name_Ey_real,name_freq);
	strcat(name_Ey_imag,name_freq);

	sprintf(name_Hx_real,"tHx_real");
	sprintf(name_Hx_imag,"tHx_imag");
	strcat(name_Hx_real,name_freq);
	strcat(name_Hx_imag,name_freq);

	sprintf(name_Hy_real,"tHy_real");
	sprintf(name_Hy_imag,"tHy_imag");
	strcat(name_Hy_real,name_freq);
	strcat(name_Hy_imag,name_freq);

	/////////////////////////////////////
	/////////// Print Ex dat ////////////
	/////////////////////////////////////
	real = fopen(name_Ex_real,"wt");
	imag = fopen(name_Ex_imag,"wt");
	for(j=jsize-2; j>=1; j--)
	{
		for(i=1; i<=isize-2; i++)
		{
			if( (i >= pmlil && i <=isize-pmlir) && (j >= pmljl && j <= jsize-pmljr))
			{
				fprintf(real,"%f\t", Ex_cos[i][j][mm]);
				fprintf(imag,"%f\t", Ex_sin[i][j][mm]);
			}
			else
			{
				fprintf(real,"%f\t", 0.0);
				fprintf(imag,"%f\t", 0.0);
			}
		}
		fprintf(real, "\n");
		fprintf(imag, "\n");
	}
	fclose(real); fclose(imag);

	/////////////////////////////////////
	/////////// Print Ey dat ////////////
	/////////////////////////////////////
	real = fopen(name_Ey_real,"wt");
	imag = fopen(name_Ey_imag,"wt");
	for(j=jsize-2; j>=1; j--)
	{
		for(i=1; i<=isize-2; i++)
		{
			if( (i >= pmlil && i <=isize-pmlir) && (j >= pmljl && j <= jsize-pmljr))
			{
				fprintf(real,"%f\t", Ey_cos[i][j][mm]);
				fprintf(imag,"%f\t", Ey_sin[i][j][mm]);
			}
			else
			{
				fprintf(real,"%f\t", 0.0);
				fprintf(imag,"%f\t", 0.0);
			}
		}
		fprintf(real, "\n");
		fprintf(imag, "\n");
	}
	fclose(real); fclose(imag);

	/////////////////////////////////////
	/////////// Print Hx dat ////////////
	/////////////////////////////////////
	real = fopen(name_Hx_real,"wt");
	imag = fopen(name_Hx_imag,"wt");
	for(j=jsize-2; j>=1; j--)
	{
		for(i=1; i<=isize-2; i++)
		{
			if( (i >= pmlil && i <=isize-pmlir) && (j >= pmljl && j <= jsize-pmljr))
			{
				fprintf(real,"%f\t", Hx_cos[i][j][mm]);
				fprintf(imag,"%f\t", Hx_sin[i][j][mm]);
			}
			else
			{
				fprintf(real,"%f\t", 0.0);
				fprintf(imag,"%f\t", 0.0);
			}
		}
		fprintf(real, "\n");
		fprintf(imag, "\n");
	}
	fclose(real); fclose(imag);

	/////////////////////////////////////
	/////////// Print Hy dat ////////////
	/////////////////////////////////////
	real = fopen(name_Hy_real,"wt");
	imag = fopen(name_Hy_imag,"wt");
	for(j=jsize-2; j>=1; j--)
	{
		for(i=1; i<=isize-2; i++)
		{
			if( (i >= pmlil && i <=isize-pmlir) && (j >= pmljl && j <= jsize-pmljr))
			{
				fprintf(real,"%f\t", Hy_cos[i][j][mm]);
				fprintf(imag,"%f\t", Hy_sin[i][j][mm]);
			}
			else
			{
				fprintf(real,"%f\t", 0.0);
				fprintf(imag,"%f\t", 0.0);
			}
		}
		fprintf(real, "\n");
		fprintf(imag, "\n");
	}
	fclose(real); fclose(imag);

	printf("print real and imag [%d] ok....!\n",mm);

}

int find_k_projection(int i, int j, int k_range)
{
	float initial;
	int m;

	initial = eps(i,j,k_range);

	for(m=k_range-1; m>0; m--)
	{
		if( meps(i,j,m)==0.0 && fabs(eps(i,j,m)-initial)>0.1 )
			return(m-1);
		if( meps(i,j,m)!=0.0 )
			return(m-1);
	}
	return(0);
}


