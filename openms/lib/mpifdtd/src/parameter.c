#include "./fdtd.hpp"

int m_pml,ma_pml;
float sigmaCPML,alphaCPML,kappaCPML;


int pmlil, pmlir, pmljl, pmljr, pmlkl, pmlkr;
int lattice_x, lattice_y, lattice_z;
int max_lattice_nz; // the finest non-uniform grid in the z axis
float xsize, ysize, zsize;
float xcenter, ycenter, zcenter;
float kx, ky, kz;
float orderxl, orderyl, orderzl;
float orderxr, orderyr, orderzr;
float sig_axl, sig_ayl, sig_azl;
float sig_axr, sig_ayr, sig_azr;
float ds_x, ds_y, ds_z, dt;
float dt_nm;
float *ds_nz;
float S_factor;
float pi, eo, uo, ups, light_speed;
int xparity=0, yparity=0, zparity=0;  //default

float wave_vector_x, wave_vector_y;
int use_periodic_x=0, use_periodic_y=0; //default

int ngrid_n=0; // for non_uniform_grid function
struct ngrid_info *ngrid;
float *base_z;
int *base_grid;

int lcm3(int a, int b, int c);
int lcm2(int m, int n);
int gcd2(int m, int n);
float *float_1d_memory_for_parameter(int imax);
void quick_sort_nz_start( struct ngrid_info *ngrid_s, int left, int right );
void generate_base_z();
void generate_base_grid();


void structure_size(float x,float y,float z)
{
	xsize=x;
	ysize=y;
	zsize=z;
}

void lattice_size(int lx, int ly, int lz)
{
	lattice_x=lx;
	lattice_y=ly;
	lattice_z=lz;

	///////////////////////////////////////////
	//// by default for non_uniform_grid() ////
	///////////////////////////////////////////
	max_lattice_nz=lattice_z;
}

void non_uniform_grid(char *component, float z_i, float z_f, int nlz)
{
	// only for z direction
	ngrid_n++;

	ngrid = (struct ngrid_info *)realloc(ngrid,sizeof(struct ngrid_info)*ngrid_n);

	ngrid[ngrid_n-1].lattice_nz = nlz;
	ngrid[ngrid_n-1].nz_start = z_i;
	ngrid[ngrid_n-1].nz_end = z_f;

	myprintf("ngrid_n = %d\n",ngrid_n);
}

void quick_sort_nz_start( struct ngrid_info *ngrid_s, int left, int right )
{
    int i, j, pivot;
    struct ngrid_info temp;
    pivot = ngrid_s[(int)(left+right)/2].nz_start;
    i = left; j = right;
    for( ; ; )
    {
        while( (ngrid_s[i].nz_start < pivot) && (i<right) ) i++;
        while( pivot < ngrid_s[j].nz_start ) j--;
        if(i>=j) break;
        temp = ngrid_s[i]; ngrid_s[i] = ngrid_s[j]; ngrid_s[j] = temp;
        i++; j--;
    }
    if( left < i-1 ) quick_sort_nz_start( ngrid_s, left, i-1 );
    if( j+1 < right ) quick_sort_nz_start( ngrid_s, j+1, right );
}

int non_uniform_z_to_i(float z)
{
	int n=0;

	/*//////////////////////////////////////////
	Example : when ngrid_n=3
	---------------------------

	lattice_z

	-------end[2]  ----------- base_z[3] & base_grid[3]   <-- index runs to (ngrid_n)
	non_uniform[2]
	-------start[2]

	lattice_z

	-------end[1] ----------- base_z[2] & base_grid[2]
	non_uniform[1]
	-------start[1]

	lattice_z

	-------end[0]  ----------- base_z[1] & base_grid[1]
	non_uniform[0]
	-------start[0]

	lattice_z

	-------------------------- base_z[0] & base_grid[0]

	//////////////////////////////////////////*/

	//////////////////////////////////////////////////////
	//// no use of non-uniform grid (special treatment)
	if(ngrid_n==0)
	{
		return( floor(0.5+(z+0.5*zsize)*lattice_z) );
	}
	//////////////////////////////////////////////////////

	//////////////////////////////////////////////////////
	//// use of non-uniform grid (when ngrid_n >=1 )
	while(n<ngrid_n)
	{
		if( z <= ngrid[n].nz_start )
		{
			return( floor(0.5+(z-base_z[n])*lattice_z) + base_grid[n]);
		}
		else if( (z > ngrid[n].nz_start) && (z <= ngrid[n].nz_end) )
		{
			return( floor(0.5+( (ngrid[n].nz_start-base_z[n])*lattice_z + (z-ngrid[n].nz_start)*ngrid[n].lattice_nz )) + base_grid[n]);
		}
		else
			n++;
	}

	// if z > base_z[ngrid_n]
	return( floor(0.5+( (z-base_z[n])*lattice_z))+ base_grid[n]);
	//////////////////////////////////////////////////////
}

float non_uniform_i_to_z(int i)
{
	int n=0;

	// Note that this function cannot return the original 'z' value, due to the rounding (loss of information).

	//////////////////////////////////////////////////////
	//// no use of non-uniform grid (special treatment)
	if(ngrid_n==0)
	{
		return( ((float)i)/((float)lattice_z) -0.5*zsize );
	}
	//////////////////////////////////////////////////////

	while(n<ngrid_n)
	{
		if( i <= non_uniform_z_to_i(ngrid[n].nz_start) )
		{
			return( ((float)(i-base_grid[n]))/((float)lattice_z) + base_z[n]);
		}
		else if( (i > non_uniform_z_to_i(ngrid[n].nz_start)) && (i <= non_uniform_z_to_i(ngrid[n].nz_end)) )
		{
			return( ((float)(non_uniform_z_to_i(ngrid[n].nz_start)-base_grid[n]))/((float)lattice_z) + ((float)(i-non_uniform_z_to_i(ngrid[n].nz_start)))/((float)ngrid[n].lattice_nz) + base_z[n] );
		}
		else
			n++;
	}

	// if i > base_grid[ngrid_n]
	return( ((float)(i-base_grid[n]))/((float)lattice_z) + base_z[n]);
}

int ngrid_lattice_nz_z(float z) // returns lattice_nz at the position z
{
	int n;
	int temp_return;

	temp_return = lattice_z;

	if(ngrid_n==0)
		return(temp_return);

	for(n=0; n<ngrid_n; n++)
	{
		if( (z > ngrid[n].nz_start) && (z <= ngrid[n].nz_end) )
			temp_return = ngrid[n].lattice_nz;
	}

	return(temp_return);
}

int ngrid_lattice_nz_i(int i) // returns lattice_nz at the position i
{
	int n;
	int temp_return;

	temp_return = lattice_z;

	if(ngrid_n==0)
		return(temp_return);

	for(n=0; n<ngrid_n; n++)
	{
		if( (i > non_uniform_z_to_i(ngrid[n].nz_start)) && (i <= non_uniform_z_to_i(ngrid[n].nz_end)) )
			temp_return = ngrid[n].lattice_nz;
	}

	return(temp_return);
}

int find_max_lattice_nz()
{
	int max_temp;
	int scan_i;
	int present_lattice;

	max_temp = ngrid_lattice_nz_i(0);

	if(ngrid_n==0)
		return(max_temp);

	for(scan_i=0; scan_i<=non_uniform_z_to_i(0.5*zsize); scan_i++)
	{
		present_lattice = ngrid_lattice_nz_i(scan_i);
		if(present_lattice > max_temp)
			max_temp = present_lattice;
	}

	return(max_temp);
}

void pml_size(int il,int ir,int jl,int jr,int kl,int kr)
{
	pmlil=il;
	pmlir=ir;
	pmljl=jl;
	pmljr=jr;
	pmlkl=kl;
	pmlkr=kr;
	//See the range of position[i][j][k] in memory.c.
	if(il==0) pmlil=0;
	if(ir==0) pmlir=0;
	if(jl==0) pmljl=0;
	if(jr==0) pmljr=0;
	if(kl==0) pmlkl=0;
	if(kr==0) pmlkr=0;
}

/*
init_grid(){
        int grid_size[3];

	grid_size[0] = floor(0.5+(xsize*lattice_x));
	grid_size[1] = floor(0.5+(ysize*lattice_y));
	////////////////////////////////////
	///// for non_uniform_grid() ///////
	////////////////////////////////////
	grid_size[2] = non_uniform_z_to_i(0.5*zsize);
	printf("non uniform grid_size[2] = %d\n", grid_size[2]); ////////////

#ifdef MPION
	int dn[3];

        int dn_remainder[3];
        for (int i = 0; i < 3; i++){
	   dn[i] = grid_size[i] / mpi_grid_proces[i];
	   dn_remainder[i] = grid_size[i] % mpi_grid_proces[i];

	   if (dn_remainder[i] != 0){
              if (mpi_grid_coords[0] < dn_remainder[i]){
	         mpi_grid_n1[i] = mpi_grid_coords[i] * (dn[i] + 1) + 1;
	         mpi_grid_n2[i] = (mpi_grid_coords[i] + 1) * (dn[i] + 1);
	      }
	      else
	      {
	         mpi_grid_n1[i] = mpi_grid_coords[i] * (dn[i] + 1) + 1 + dn_remainder[i];
	         mpi_grid_n2[i] = (mpi_grid_coords[i] + 1) * dn[i] + dn_remainder[i];
	      }
	   }
	   else
	   {
              mpi_grid_n1[i] = mpi_grid_coords[i] * dn[i] + 1;
	      mpi_grid_n2[i] = (mpi_grid_coords[i] + 1) * dn[i];
	   }
	}
#else
        for (int i = 0; i < 3; i++){
           mpi_grid_n1[i] = 1;
	   mpi_grid_n2[i] = grid_size[i];
        }
#endif

        // new we get the grid size on each node
        mygrid_nx = mpi_grid_n2[0] - mpi_grid_n1[0] + 1;
        mygrid_ny = mpi_grid_n2[1] - mpi_grid_n1[1] + 1;
        mygrid_nz = mpi_grid_n2[2] - mpi_grid_n1[2] + 1;

	xcenter = xsize / 2;
	ycenter = ysize / 2;
	zcenter = zsize / 2;  // always true even with non_uniform_grid()

	m_mygrid_nx = mygrid_nx;
	p_mygrid_nx = mygrid_nx - 1;
	c_mygrid_nx = mygrid_nx / 2;

	m_mygrid_ny = mygrid_ny;
	p_mygrid_ny = mygrid_ny - 1;
	c_mygrid_ny = mygrid_ny / 2;

	m_mygrid_nz = mygrid_nz;
	p_mygrid_nz = mygrid_nz - 1;
	c_mygrid_nz = mygrid_nz / 2;  // in case of using non_uniform_grid(), this will not be used.
}
*/

void set_default_parameter(float S)
{
	int lcm_temp;
	int i;
	FILE *stream;

	pi=3.141592;
	eo=8.854e-12;
	uo=pi*4.0e-7;
	ups=1.0;
	light_speed=1.0/sqrt(eo*uo);

	if(ngrid_n>=2)
		quick_sort_nz_start(ngrid,0,ngrid_n-1);

	////////////////////////////////////////////////
	// define base_z[n] & base_grid[n]
	///////////////////////////////////////////////
	generate_base_z();
	generate_base_grid();
	///////////////////////////////////////////////

	max_lattice_nz = find_max_lattice_nz();

	lcm_temp = lcm3(lattice_x, lattice_y, max_lattice_nz);

	ds_x = lcm_temp/lattice_x;
	ds_y = lcm_temp/lattice_y;
	///////////////////////////////////////////////////////////////////////////
	//in case of using non_uniform_grid(), ds_z takes the smallest feature size
	/////////////////////////////////////////////////////////////////////////// 
	ds_z = lcm_temp/max_lattice_nz;

	S_factor = S;

	dt=1/light_speed/S_factor; // S=Couriant parameter for numerical stability
	dt_nm=dt*1e-9;

	orderxl = 3.5; orderyl = 3.5; orderzl = 3.5;
	orderxr = 3.5; orderyr = 3.5; orderzr = 3.5;

	sig_axl = 1.0; sig_ayl = 1.0; sig_azl = 1.0;
	sig_axr = 1.0; sig_ayr = 1.0; sig_azr = 1.0;

	kx = 1.0; ky = 1.0; kz =1.0;

	// parameter for CPML
	m_pml=3; ma_pml=1;
	sigmaCPML=0.8*(m_pml+1)/(ds_x*1e-9*sqrt(uo/eo));
	alphaCPML=0.05;
	kappaCPML=5.0;
	myprintf("sigmaCPML= %f \n",sigmaCPML);

	//init_grid();

	///////////////////////////////////////////////////
	//// create non-uniform z-grid map (ds_nz[i]) /////
	///////////////////////////////////////////////////
	ds_nz = float_1d_memory_for_parameter(m_mygrid_nz);
	for(i=0; i<m_mygrid_nz; i++)
		ds_nz[i] = lcm_temp/ngrid_lattice_nz_i(i);//////////

	///////////////////////////////////////////////////
	myprintf("creating non uniform z map ok\n"); ////////////
	stream = fopen("dsz_map.dat","wt");
	for(i=m_mygrid_nz-1; i>=0; i--)
		fprintf(stream,"%g\n",ds_nz[i]);

	fclose(stream);

	////// Random number generation //////
	gsl_rng_env_setup();
	// In the below, 'rng_type' and 'rng_r' are defined in FDTDvar.h
	rng_type = gsl_rng_default;
	rng_r = gsl_rng_alloc (rng_type);
	gsl_rng_set(rng_r,1234);  // seed value = 1234
	//////////////////////////////////////

	myprintf("\nparameters...ok\n");
}

void set_sigma_order(float oxl, float oxr, float oyl, float oyr, float ozl, float ozr)
{
	orderxl = oxl; orderyl = oyl; orderzl = ozl;  //default value = 3.5
	orderxr = oxr; orderyr = oyr; orderzr = ozr;  //default value = 3.5
}

void set_sigma_max(float axl, float axr, float ayl, float ayr, float azl, float azr)
{
	sig_axl = axl; sig_ayl = ayl; sig_azl = azl;   //default value = 1.0
	sig_axr = axr; sig_ayr = ayr; sig_azr = azr;   //default value = 1.0
}

void set_kappa(float kappa_x, float kappa_y, float kappa_z)
{
	kx = kappa_x;  //defaut value = 1.0
	ky = kappa_y;  //defaut value = 1.0
	kz = kappa_z;  //defaut value = 1.0
}

void Hz_parity(int x,int y,int z)
{
	xparity=x;
	yparity=y;
	zparity=z;

	if(xparity==1 || xparity==-1)
	{
		m_mygrid_nx=c_mygrid_nx+2;
		p_mygrid_nx=c_mygrid_nx+1;
	}

	if(yparity==1 || yparity==-1)
	{
		m_mygrid_ny=c_mygrid_ny+2;
		p_mygrid_ny=c_mygrid_ny+1;
	}

	if(zparity==1 || zparity==-1)
	{
		m_mygrid_nz=c_mygrid_nz+2;
		p_mygrid_nz=c_mygrid_nz+1;
	}
}

void periodic_boundary(int x_on, int y_on, float k_x, float k_y)
{
	if( x_on == 1)
	{
		wave_vector_x = k_x;
		use_periodic_x = x_on; // 1:on, 0:off  default=0
		m_mygrid_nx = mygrid_nx + 2;
		p_mygrid_nx = mygrid_nx + 1;
		pmlil=-5;  //See the range of position[i][j][k] in memory.c
		pmlir=-5;
	}
	if( y_on == 1)
	{
		wave_vector_y = k_y;
		use_periodic_y = y_on;
		m_mygrid_ny = mygrid_ny + 2;
		p_mygrid_ny = mygrid_ny + 1;
		pmljl=-5;
		pmljr=-5;
	}
}

int lcm3(int a, int b, int c)
{
	return( lcm2(a, lcm2(b,c)) );
}

int lcm2(int m, int n)
{
	return( m*n/gcd2(m,n) );  //use the relation ; gcd(m,n)*lcm(m,n)=m*n
}

int gcd2(int m, int n)
{
	int temp;
	int r; //remainder

	if(n>m) //Make m>n
	{
		temp = m;
		m = n;
		n = temp;
	}
	while( (r= m%n) != 0)  //Euclid algorithm
	{
		m=n;
		n=r;
	}

	return(n);
}

float *float_1d_memory_for_parameter(int imax)
{
	int i;
	float *memory;

	memory=(float *)malloc(sizeof(float)*imax);
	for(i=0;i<imax;i++) memory[i]=0.0;

	return memory;
}

void generate_base_z()
{
	int i;

	base_z=(float *)malloc(sizeof(float)*(ngrid_n+1));

	base_z[0] = -0.5*zsize;
	myprintf("base_z[0] = %g\n",base_z[0]);

	for(i=1; i<=ngrid_n; i++)
	{
		base_z[i] = ngrid[i-1].nz_end;
		myprintf("base_z[%d] = %g\n",i,base_z[i]);
	}
}

void generate_base_grid()
{
	int i;

	base_grid=(int *)malloc(sizeof(int)*(ngrid_n+1));

	base_grid[0] = 0;
	myprintf("base_grid[0] = %d\n",base_grid[0]);

	for(i=1; i<=ngrid_n; i++)
	{
		if(i==1)
			base_grid[1] = (ngrid[0].nz_end-ngrid[0].nz_start)*ngrid[0].lattice_nz
				+ (ngrid[0].nz_start+0.5*zsize)*lattice_z;
		else
			base_grid[i] = base_grid[i-1] + (ngrid[i-1].nz_end-ngrid[i-1].nz_start)*ngrid[i-1].lattice_nz
				+ (ngrid[i-1].nz_start-ngrid[i-2].nz_end)*lattice_z;
		myprintf("base_grid[%d] = %d\n",i,base_grid[i]);
	}
}
