#include "./pFDTD.h"

int in_object(int n,float i,float j,float k);
void reading_matrix(char *matrix_file, int n);
float average_epsilon(float i,float j,float k);
void generate_random_2D(float x_min, float x_max, float y_min, float y_max, float radius, float *xr, float *yr, int i);
void generate_random_3D(float x_min, float x_max, float y_min, float y_max, float z_min, float z_max, float radius, float *xr, float *yr, float *zr, int i);
float rand_distance_2D(float x_scan, float y_scan, float x_temp, float y_temp);
float rand_distance_3D(float x_scan, float y_scan, float z_scan, float x_temp, float y_temp, float z_temp);

float back_epsilon;
int objectn=0;
struct obj *object;

///// For Euler rotation ///////
///// [n] : object index ///////
/// a = alpha, b=beta, c=gamma ///
float *cos_a, *cos_b, *cos_c;
float *sin_a, *sin_b, *sin_c;
//////////////////////////////////////

void background(float epsilon)
{
	int i,j,k;

	back_epsilon=epsilon;
	
	for(i=0;i<misize;i++)
	for(j=0;j<mjsize;j++)
	for(k=0;k<mksize;k++)
		epsilonx[i][j][k]=epsilony[i][j][k]=epsilonz[i][j][k]=back_epsilon;

	printf("background...ok\n");
}

void input_object(char *shape, char *matrix_file, float centerx, float centery, float centerz, float size1, float size2, float size3, float epsilon)
{	
	objectn++;

	object=(struct obj *)realloc(object,sizeof(struct obj)*objectn);
	
	cos_a=(float *)realloc(cos_a,sizeof(float)*objectn);
	cos_b=(float *)realloc(cos_b,sizeof(float)*objectn);
	cos_c=(float *)realloc(cos_c,sizeof(float)*objectn);
	sin_a=(float *)realloc(sin_a,sizeof(float)*objectn);
	sin_b=(float *)realloc(sin_b,sizeof(float)*objectn);
	sin_c=(float *)realloc(sin_c,sizeof(float)*objectn);
	
	strcpy(object[objectn-1].shape,shape);

	//////// common parameters ///////
	object[objectn-1].epsilon=epsilon;  
	//////////////////////////////////
	cos_a[objectn-1] = 1.0;
	cos_b[objectn-1] = 1.0;
	cos_c[objectn-1] = 1.0;
	sin_a[objectn-1] = 0.0;
	sin_b[objectn-1] = 0.0;
	sin_c[objectn-1] = 0.0;

	// in the below, we define various input parameters case by case. 

	if(strcmp(shape,"contour")==0)
	{
		object[objectn-1].centeri=centerx;           // matrix_center 
		object[objectn-1].centerj=centery;           // matrix_center 
		object[objectn-1].centerk=(centerz); // for non_uniform_grid(), the meaning changed
		object[objectn-1].size1=size1;     // discrimination level
		object[objectn-1].size2=(size2);   // height  // for non_uniform_grid() .....
		object[objectn-1].size3=size3;     // compression factor
		reading_matrix(matrix_file, objectn-1);	
	}
	else if(strcmp(shape,"rod")==0)
	{
		object[objectn-1].centeri=(centerx+xcenter)*lattice_x;  
		object[objectn-1].centerj=(centery+ycenter)*lattice_y;
		object[objectn-1].centerk=(centerz); // for non_uniform_grid(), the meaning changed
		object[objectn-1].size1=size1*lattice_x;  // radius
		object[objectn-1].size2=(size2);  // height  // for non_uniform_grid() ....
	}
	else if(strcmp(shape,"rodX")==0) //// not applicable for non_uniform_grid()
	{
		object[objectn-1].centeri=(centerx+xcenter)*lattice_x;  
		object[objectn-1].centerj=(centery+ycenter)*lattice_y;
		object[objectn-1].centerk=(centerz+zcenter)*lattice_z;
		object[objectn-1].size1=size1*lattice_y;  // radius
		object[objectn-1].size2=size2*lattice_x;  // x length
	}
	else if(strcmp(shape,"rodY")==0) //// not applicable for non_uniform_grid()
	{
		object[objectn-1].centeri=(centerx+xcenter)*lattice_x;  
		object[objectn-1].centerj=(centery+ycenter)*lattice_y;
		object[objectn-1].centerk=(centerz+zcenter)*lattice_z;
		object[objectn-1].size1=size1*lattice_x;  // radius
		object[objectn-1].size2=size2*lattice_y;  // y length
	}
	else if(strcmp(shape,"donut")==0)
	{
		object[objectn-1].centeri=(centerx+xcenter)*lattice_x;  
		object[objectn-1].centerj=(centery+ycenter)*lattice_y;
		object[objectn-1].centerk=(centerz); // for non_uniform_grid(), the meaning changed
		object[objectn-1].size1=size1*lattice_x;  // inner radius
		object[objectn-1].size2=size2*lattice_x;  // outer radius
		object[objectn-1].size3=(size3);  // height  // for non_uniform_grid() ....
	}
	else if(strcmp(shape,"sphere")==0) //// not applicable for non_uniform_grid()
	{
		object[objectn-1].centeri=(centerx+xcenter)*lattice_x;
		object[objectn-1].centerj=(centery+ycenter)*lattice_y;
		object[objectn-1].centerk=(centerz+zcenter)*lattice_z;
		object[objectn-1].size1=size1*lattice_x;  // radius
	}
	else if(strcmp(shape,"shell")==0) //// not applicable for non_uniform_grid()
	{
		object[objectn-1].centeri=(centerx+xcenter)*lattice_x;
		object[objectn-1].centerj=(centery+ycenter)*lattice_y;
		object[objectn-1].centerk=(centerz+zcenter)*lattice_z;
		object[objectn-1].size1=size1*lattice_x;  // inner radius
		object[objectn-1].size2=size2*lattice_x;  // outer radius
	}
	else if(strcmp(shape,"ellipse")==0)
	{
		object[objectn-1].centeri=(centerx+xcenter)*lattice_x;
		object[objectn-1].centerj=(centery+ycenter)*lattice_y;
		object[objectn-1].centerk=(centerz); // for non_uniform_grid(), the meaning changed
		object[objectn-1].size1=size1*lattice_x;  // radius
		object[objectn-1].size2=(size2);  // height (centerz); // for non_uniform_grid() ....
		object[objectn-1].size3=size3;  // aspect ratio=ry/rx
	}
	else if(strcmp(shape,"ellipsoidal")==0) //// not applicable for non_uniform_grid()
	{
		object[objectn-1].centeri=(centerx+xcenter)*lattice_x;
		object[objectn-1].centerj=(centery+ycenter)*lattice_y;
		object[objectn-1].centerk=(centerz+zcenter)*lattice_z;
		object[objectn-1].size1=size1*lattice_x;  // rx
		object[objectn-1].size2=size2*lattice_y;  // ry
		object[objectn-1].size3=size3*lattice_z;  // rz
	}
	else if(strcmp(shape,"cone")==0)
	{
		object[objectn-1].centeri=(centerx+xcenter)*lattice_x;
		object[objectn-1].centerj=(centery+ycenter)*lattice_y;
		object[objectn-1].centerk=(centerz); // for non_uniform_grid()
		object[objectn-1].size1=size1*lattice_x;  // r1
		object[objectn-1].size2=size2*lattice_x;  // r2
		object[objectn-1].size3=(size3);  // for non_uniform_grid(), height
	}
	else // "block"
	{
		object[objectn-1].centeri=(centerx+xcenter)*lattice_x;
		object[objectn-1].centerj=(centery+ycenter)*lattice_y;
		object[objectn-1].centerk=(centerz); // for non_uniform_grid(), the meaning changed
		object[objectn-1].size1=size1*lattice_x;
		object[objectn-1].size2=size2*lattice_y;
		object[objectn-1].size3=(size3);  // for non_uniform_grid(), the meaning changed
	}
}

void input_object_Euler_rotation(char *shape, char *matrix_file, float centerx, float centery, float centerz, float size1, float size2, float size3, float alpha, float beta, float gamma, float epsilon)
// alpha, beta, gamma : 3- Euler angles 
{	
	objectn++;

	object=(struct obj *)realloc(object,sizeof(struct obj)*objectn);

	cos_a=(float *)realloc(cos_a,sizeof(float)*objectn);
	cos_b=(float *)realloc(cos_b,sizeof(float)*objectn);
	cos_c=(float *)realloc(cos_c,sizeof(float)*objectn);
	sin_a=(float *)realloc(sin_a,sizeof(float)*objectn);
	sin_b=(float *)realloc(sin_b,sizeof(float)*objectn);
	sin_c=(float *)realloc(sin_c,sizeof(float)*objectn);
	
	strcpy(object[objectn-1].shape,shape);

	//////// common parameters ///////
	object[objectn-1].epsilon=epsilon;  
	//////////////////////////////////
	cos_a[objectn-1] = cos(alpha*pi/180);
	cos_b[objectn-1] = cos(beta*pi/180);
	cos_c[objectn-1] = cos(gamma*pi/180);
	sin_a[objectn-1] = sin(alpha*pi/180);
	sin_b[objectn-1] = sin(beta*pi/180);
	sin_c[objectn-1] = sin(gamma*pi/180);

	//// in the below, we define various input parameters case by case. ////
	//// Contrary to "input_object()", 'Euler rotaion' cannot be applied to "sphere" and "contour"////

	if(strcmp(shape,"rod")==0)
	{
		object[objectn-1].centeri=(centerx+xcenter)*lattice_x;  
		object[objectn-1].centerj=(centery+ycenter)*lattice_y;
		object[objectn-1].centerk=(centerz+zcenter)*lattice_z;
		object[objectn-1].size1=size1*lattice_x;  // radius
		object[objectn-1].size2=size2*lattice_z;  // height
		strcpy(object[objectn-1].shape,"Rrod"); // Euler Rotation indicator 
	}
	else if(strcmp(shape,"ellipse")==0)
	{
		object[objectn-1].centeri=(centerx+xcenter)*lattice_x;
		object[objectn-1].centerj=(centery+ycenter)*lattice_y;
		object[objectn-1].centerk=(centerz+zcenter)*lattice_z;
		object[objectn-1].size1=size1*lattice_x;  // radius
		object[objectn-1].size2=size2*lattice_z;  // height
		object[objectn-1].size3=size3;  // aspect ratio=ry/rx
		strcpy(object[objectn-1].shape,"Rellipse"); // Euler Rotation indicator 
	}
	else if(strcmp(shape,"ellipsoidal")==0)
	{
		object[objectn-1].centeri=(centerx+xcenter)*lattice_x;
		object[objectn-1].centerj=(centery+ycenter)*lattice_y;
		object[objectn-1].centerk=(centerz+zcenter)*lattice_z;
		object[objectn-1].size1=size1*lattice_x;  // rx
		object[objectn-1].size2=size2*lattice_y;  // ry
		object[objectn-1].size3=size3*lattice_z;  // rz
		strcpy(object[objectn-1].shape,"Rellipsoidal"); // Euler Rotation indicator 
	}
	else if(strcmp(shape,"block")==0)
	{
		object[objectn-1].centeri=(centerx+xcenter)*lattice_x;
		object[objectn-1].centerj=(centery+ycenter)*lattice_y;
		object[objectn-1].centerk=(centerz+zcenter)*lattice_z;
		object[objectn-1].size1=size1*lattice_x;
		object[objectn-1].size2=size2*lattice_y;
		object[objectn-1].size3=size3*lattice_z;
		strcpy(object[objectn-1].shape,"Rblock"); // Euler Rotation indicator 
	}
	else
	{}
}

void random_object(char *shape, float radius, float height, float epsilon, float x_min, float x_max, float y_min, float y_max, float z_min, float z_max, int gen_number, int seed)
{
	int i;	
	float *xr, *yr, *zr;  //random number coordinates

	srand(seed); //initialization of random number generator

	if(strcmp(shape,"rod")==0)
	{
		xr = (float *)malloc(sizeof(float)*gen_number);
		yr = (float *)malloc(sizeof(float)*gen_number);

		for(i=0; i<gen_number; i++)
		{
			generate_random_2D(x_min, x_max, y_min, y_max, radius, xr, yr, i);
			input_object("rod", EMP, xr[i], yr[i], 0+shift, radius, height, 0, epsilon);
		}
	}

	if(strcmp(shape,"sphere")==0)
	{
		xr = (float *)malloc(sizeof(float)*gen_number);
		yr = (float *)malloc(sizeof(float)*gen_number);
		zr = (float *)malloc(sizeof(float)*gen_number);

		for(i=0; i<gen_number; i++)
		{
			generate_random_3D(x_min, x_max, y_min, y_max, z_min, z_max, radius, xr, yr, zr, i);
			input_object("sphere", EMP, xr[i], yr[i], zr[i], radius, 0, 0, epsilon);
		}
	}	
}

void generate_random_2D(float x_min, float x_max, float y_min, float y_max, float radius, float *xr, float *yr, int i)
{
	int scan; 
	float xr_temp, yr_temp; 	

	xr_temp = ( (float)rand() / ((float)(RAND_MAX)+(float)(1)) )*(x_max-x_min) + x_min;	
	yr_temp = ( (float)rand() / ((float)(RAND_MAX)+(float)(1)) )*(y_max-y_min) + y_min;	

	if(i>0)
	{
		while(1)
		{
			for(scan=0; scan<i; scan++)
				if( rand_distance_2D(xr[scan], yr[scan], xr_temp, yr_temp) < 2*radius )
				{	
					scan = -10; 
					break;
				}

			if(scan == i)
			{
				xr[i] = xr_temp;
				yr[i] = yr_temp;
				break; //out of the while loop
			}
			else
			{
				xr_temp = ( (float)rand() / ((float)(RAND_MAX)+(float)(1)) )*(x_max-x_min) + x_min;	
				yr_temp = ( (float)rand() / ((float)(RAND_MAX)+(float)(1)) )*(y_max-y_min) + y_min;	
			}
		}
	}

	if(i==0)
	{
		xr[i] = xr_temp;
		yr[i] = yr_temp;
	}
}

float rand_distance_2D(float x_scan, float y_scan, float x_temp, float y_temp)
{
	return( sqrt((x_scan-x_temp)*(x_scan-x_temp) + (y_scan-y_temp)*(y_scan-y_temp)) );
}

void generate_random_3D(float x_min, float x_max, float y_min, float y_max, float z_min, float z_max, float radius, float *xr, float *yr, float *zr, int i)
{
	int scan; 
	float xr_temp, yr_temp, zr_temp; 	

	xr_temp = ( (float)rand() / ((float)(RAND_MAX)+(float)(1)) )*(x_max-x_min) + x_min;	
	yr_temp = ( (float)rand() / ((float)(RAND_MAX)+(float)(1)) )*(y_max-y_min) + y_min;
	zr_temp = ( (float)rand() / ((float)(RAND_MAX)+(float)(1)) )*(z_max-z_min) + z_min;		

	if(i>0)
	{
		while(1)
		{
			for(scan=0; scan<i; scan++)
				if( rand_distance_3D(xr[scan], yr[scan], zr[scan], xr_temp, yr_temp, zr_temp) < 2*radius )
				{	
					scan = -10; 
					break;
				}

			if(scan == i)
			{
				xr[i] = xr_temp;
				yr[i] = yr_temp;
				zr[i] = zr_temp;
				break; //out of the while loop
			}
			else
			{
				xr_temp = ( (float)rand() / ((float)(RAND_MAX)+(float)(1)) )*(x_max-x_min) + x_min;	
				yr_temp = ( (float)rand() / ((float)(RAND_MAX)+(float)(1)) )*(y_max-y_min) + y_min;
				zr_temp = ( (float)rand() / ((float)(RAND_MAX)+(float)(1)) )*(z_max-z_min) + z_min;	
			}
		}
	}

	if(i==0)
	{
		xr[i] = xr_temp;
		yr[i] = yr_temp;
		zr[i] = zr_temp;
	}
}

float rand_distance_3D(float x_scan, float y_scan, float z_scan, float x_temp, float y_temp, float z_temp)
{
	return( sqrt((x_scan-x_temp)*(x_scan-x_temp) + (y_scan-y_temp)*(y_scan-y_temp) + (z_scan-z_temp)*(z_scan-z_temp)) );
}

void make_epsilon()
{
	int i,j,k;

	printf("-----------------------\n");	
	printf("making the epsilon...\n");

	for(i=0;i<misize;i++)
	{
		printf("%d% \n",100*i/misize); 

		for(j=0;j<mjsize;j++)
			for(k=0;k<mksize;k++)
			{
				epsilonx[i][j][k]=average_epsilon(i+0.5,j,k);
				epsilony[i][j][k]=average_epsilon(i,j+0.5,k);
				epsilonz[i][j][k]=average_epsilon(i,j,k+0.5);
			}
	}

	free(object);

	printf("make_epsilon...ok\n");
}

float average_epsilon(float i,float j,float k)
{
	int n,m,partial=0;
	float ii,jj,kk,epsilon,partial_epsilon[1000];

	epsilon=back_epsilon;
	for(m=0;m<1000;m++) partial_epsilon[m]=back_epsilon;

	for(n=0;n<objectn;n++)
	{
	
		if( in_object(n,i-0.5,j-0.5,k-0.5)==0 && in_object(n,i-0.5,j-0.5,k+0.5)==0 && 
			in_object(n,i-0.5,j+0.5,k-0.5)==0 && in_object(n,i-0.5,j+0.5,k+0.5)==0 &&
			in_object(n,i+0.5,j-0.5,k-0.5)==0 && in_object(n,i+0.5,j-0.5,k+0.5)==0 &&	
			in_object(n,i+0.5,j+0.5,k-0.5)==0 && in_object(n,i+0.5,j+0.5,k+0.5)==0);

		else if(in_object(n,i-0.5,j-0.5,k-0.5)==1 && in_object(n,i-0.5,j-0.5,k+0.5)==1 && 
				in_object(n,i-0.5,j+0.5,k-0.5)==1 && in_object(n,i-0.5,j+0.5,k+0.5)==1 &&
				in_object(n,i+0.5,j-0.5,k-0.5)==1 && in_object(n,i+0.5,j-0.5,k+0.5)==1 &&
				in_object(n,i+0.5,j+0.5,k-0.5)==1 && in_object(n,i+0.5,j+0.5,k+0.5)==1)
		{
			epsilon=object[n].epsilon;
			for(m=0;m<1000;m++) partial_epsilon[m]=object[n].epsilon;
			partial=0;
		}

		else
		{
			for(m=0,ii=i-0.45;ii<i+0.5;ii=ii+0.1)
			for(jj=j-0.45;jj<j+0.5;jj=jj+0.1)
			for(kk=k-0.45;kk<k+0.5;kk=kk+0.1,m++)
				if(in_object(n,ii,jj,kk)==1) partial_epsilon[m]=object[n].epsilon;
			partial=1;
		}
	}

	if(partial==1)
		for(epsilon=0,m=0;m<1000;m++)
			epsilon=epsilon+partial_epsilon[m]/1000;
	
	return epsilon;
}

int in_object(int n,float i,float j,float k)
{
	int I, J; 
	float X, Y, Z;
	int temp;

	if(strcmp(object[n].shape,"Rrod")!=0 && strcmp(object[n].shape,"Rellipse")!=0 && strcmp(object[n].shape,"Rellipsoidal")!=0 && strcmp(object[n].shape,"Rblock")!=0)
	{
		if(strcmp(object[n].shape,"rod")==0)
		{
			if( (i-object[n].centeri)*(i-object[n].centeri)+(j-object[n].centerj)*(j-object[n].centerj)<=object[n].size1*object[n].size1 && non_uniform_z_to_i(object[n].centerk-0.5*object[n].size2)<=k && k<=non_uniform_z_to_i(object[n].centerk+0.5*object[n].size2) ) return 1;
			else return 0;
		}

		else if(strcmp(object[n].shape,"rodX")==0)
		{
			if( (j-object[n].centerj)*(j-object[n].centerj)+(k-object[n].centerk)*(k-object[n].centerk)<=object[n].size1*object[n].size1 && object[n].centeri-object[n].size2/2<=i && i<=object[n].centeri+object[n].size2/2) return 1;
			else return 0;
		}

		else if(strcmp(object[n].shape,"rodY")==0)
		{
			if( (k-object[n].centerk)*(k-object[n].centerk)+(i-object[n].centeri)*(i-object[n].centeri)<=object[n].size1*object[n].size1 && object[n].centerj-object[n].size2/2<=j && j<=object[n].centerj+object[n].size2/2) return 1;
			else return 0;
		}

		else if(strcmp(object[n].shape,"donut")==0)
		{
			if( (i-object[n].centeri)*(i-object[n].centeri)+(j-object[n].centerj)*(j-object[n].centerj)>=object[n].size1*object[n].size1 && (i-object[n].centeri)*(i-object[n].centeri)+(j-object[n].centerj)*(j-object[n].centerj)<=object[n].size2*object[n].size2 && non_uniform_z_to_i(object[n].centerk-0.5*object[n].size3)<=k && k<=non_uniform_z_to_i(object[n].centerk+0.5*object[n].size3)) return 1;
			else return 0;
		}

		else if(strcmp(object[n].shape,"sphere")==0)
		{
			if( (i-object[n].centeri)*(i-object[n].centeri)+(j-object[n].centerj)*(j-object[n].centerj)+(k-object[n].centerk)*(k-object[n].centerk)<=object[n].size1*object[n].size1 ) return 1;
			else return 0;
		}

		else if(strcmp(object[n].shape,"shell")==0)
		{
			if( (i-object[n].centeri)*(i-object[n].centeri)+(j-object[n].centerj)*(j-object[n].centerj)+(k-object[n].centerk)*(k-object[n].centerk)>=object[n].size1*object[n].size1 && (i-object[n].centeri)*(i-object[n].centeri)+(j-object[n].centerj)*(j-object[n].centerj)+(k-object[n].centerk)*(k-object[n].centerk)<=object[n].size2*object[n].size2 ) return 1;
			else return 0;
		}

		else if(strcmp(object[n].shape,"ellipse")==0)
		{
			if( (i-object[n].centeri)*(i-object[n].centeri)+(j-object[n].centerj)*(j-object[n].centerj)/(object[n].size3*object[n].size3)<=object[n].size1*object[n].size1 && non_uniform_z_to_i(object[n].centerk-0.5*object[n].size2)<=k && k<=non_uniform_z_to_i(object[n].centerk+0.5*object[n].size2)) return 1;
			else return 0;
		}

		else if(strcmp(object[n].shape,"ellipsoidal")==0)
		{
			if( (i-object[n].centeri)*(i-object[n].centeri)/((object[n].size1)*(object[n].size1))+(j-object[n].centerj)*(j-object[n].centerj)/((object[n].size2)*(object[n].size2))+(k-object[n].centerk)*(k-object[n].centerk)/((object[n].size3)*(object[n].size3))<=1 ) return 1;
			else return 0;
		}

		else if(strcmp(object[n].shape,"cone")==0)
		{
			///////////////////////////////////////////////
			//  object[n].centerk & object[n].size3 --> in units of z
			//  size3 : height 
			///////////////////////////////////////////////
			temp = (object[n].size1-object[n].size2)*(non_uniform_i_to_z(k)-object[n].centerk)/object[n].size3+0.5*(object[n].size1+object[n].size2); // radius varying as a function of 'k'

			if( (i-object[n].centeri)*(i-object[n].centeri)+(j-object[n].centerj)*(j-object[n].centerj) <= temp*temp && (k <= non_uniform_z_to_i(object[n].centerk+0.5*object[n].size3)) && (k >= non_uniform_z_to_i(object[n].centerk-0.5*object[n].size3))) return 1;
			else return 0;
		}

		else if(strcmp(object[n].shape,"block")==0)
		{
			if( object[n].centeri-object[n].size1/2<=i && i<=object[n].centeri+object[n].size1/2 &&
				object[n].centerj-object[n].size2/2<=j && j<=object[n].centerj+object[n].size2/2 &&
				non_uniform_z_to_i(object[n].centerk-0.5*object[n].size3)<=k && k<=non_uniform_z_to_i(object[n].centerk+0.5*object[n].size3) ) return 1;
			else return 0;
		}

		else if(strcmp(object[n].shape,"contour")==0)
		{
			if( (-object[n].centeri*CUTFACTOR <= (i-(xcenter*lattice_x))/object[n].size3) && ((i-(xcenter*lattice_x))/object[n].size3 <= (object[n].col-object[n].centeri)*CUTFACTOR) && (-object[n].centerj*CUTFACTOR <= (j-(ycenter*lattice_y))/object[n].size3) && ((j-(ycenter*lattice_y))/object[n].size3 <= (object[n].row-object[n].centerj)*CUTFACTOR) && (non_uniform_z_to_i(object[n].centerk-0.5*object[n].size2) <= k) && (k <= non_uniform_z_to_i(object[n].centerk+0.5*object[n].size2)) )
			{
				I = floor( object[n].centeri + (i - (xcenter*lattice_x))/object[n].size3 ); 
				J = floor( object[n].centerj + (j - (ycenter*lattice_y))/object[n].size3 );
		       		if( object[n].matrix[I][J] <= object[n].size1 )
				return 1;
			}
			else return 0;
		}
		else 
		{
		}
	}
	else 	
////if(strcmp(object[n].shape,"Rrod")==0 || strcmp(object[n].shape,"Rellipse")==0 || 
////strcmp(object[n].shape,"Rellipsoidal")==0 || strcmp(object[n].shape,"Rblock")==0)
	{
		//// Transform (i,j,k) --> (X,Y,Z), the coordinate with respect to the object
		X = (cos_a[n]*cos_c[n]-sin_a[n]*cos_b[n]*sin_c[n])*(i-object[n].centeri) 
			+ (sin_a[n]*cos_c[n]+cos_a[n]*cos_b[n]*sin_c[n])*(j-object[n].centerj)
			+ (sin_b[n]*sin_c[n])*(k-object[n].centerk) + object[n].centeri;
		Y =  -(cos_a[n]*sin_c[n]+sin_a[n]*cos_b[n]*cos_c[n])*(i-object[n].centeri)
			-(sin_a[n]*sin_c[n]-cos_a[n]*cos_b[n]*cos_c[n])*(j-object[n].centerj)
			+(sin_b[n]*cos_c[n])*(k-object[n].centerk) + object[n].centerj;
		Z = (sin_a[n]*sin_b[n])*(i-object[n].centeri) - (cos_a[n]*sin_b[n])*(j-object[n].centerj)
		 	+ cos_b[n]*(k-object[n].centerk) + object[n].centerk;

		if(strcmp(object[n].shape,"Rrod")==0)
		{
			if( (X-object[n].centeri)*(X-object[n].centeri)+(Y-object[n].centerj)*(Y-object[n].centerj)<=object[n].size1*object[n].size1 && object[n].centerk-object[n].size2/2<=Z && Z<=object[n].centerk+object[n].size2/2) return 1;
			else return 0;
		}

		if(strcmp(object[n].shape,"Rellipse")==0)
		{
			if( (X-object[n].centeri)*(X-object[n].centeri)+(Y-object[n].centerj)*(Y-object[n].centerj)/(object[n].size3*object[n].size3)<=object[n].size1*object[n].size1 && object[n].centerk-object[n].size2/2<=Z && Z<=object[n].centerk+object[n].size2/2) return 1;
			else return 0;
		}

		if(strcmp(object[n].shape,"Rellipsoidal")==0)
		{
			if( (X-object[n].centeri)*(X-object[n].centeri)/((object[n].size1)*(object[n].size1))+(Y-object[n].centerj)*(Y-object[n].centerj)/((object[n].size2)*(object[n].size2))+(Z-object[n].centerk)*(Z-object[n].centerk)/((object[n].size3)*(object[n].size3))<=1 ) return 1;
			else return 0;
		}

		else if(strcmp(object[n].shape,"Rblock")==0)
		{
			if( object[n].centeri-object[n].size1/2<=X && X<=object[n].centeri+object[n].size1/2 &&
				object[n].centerj-object[n].size2/2<=Y && Y<=object[n].centerj+object[n].size2/2 &&
				object[n].centerk-object[n].size3/2<=Z && Z<=object[n].centerk+object[n].size3/2) return 1;
			else return 0;
		}
	}
}		   

void reading_matrix(char *matrix_file, int n)
{
	FILE *stream;
	int i,j;
	int col, row;
	char ch;
	char string[20];
	
	stream = fopen(matrix_file,"rt");

	col=0; row=0;	
	/////// Counting row and col ///////
	ch = getc(stream);
	while( ch != EOF )
	{
		if( ch == '\t' ) 
			col++;
		if( ch == '\n' )
		{	
			col++;    // Origin standard (with TAB separation)
			row++;
		}
		ch = getc(stream);
	}
	col = (int)(col / row); // in each row
	printf("%s matrix file\n",matrix_file);
	printf("  matrix col = %d\n", col);
	printf("  matrix row = %d\n", row);
	printf("\n");
	fclose(stream);

	object[n].col=col; object[n].row=row;

	//////// matrix memory allocation /////
	object[n].matrix = (float **)malloc(sizeof(float *)*col);
	for(i=0; i<col; i++)
		object[n].matrix[i] = (float *)malloc(sizeof(float)*row);
	
	stream = fopen(matrix_file,"rt");
	//////// Reading matrix file data //////
	for(j=row-1; j>=0; j--)
	{
		for(i=0; i<col; i++)
		{
			fscanf(stream,"%s",string);
			object[n].matrix[i][j] = atof(string);
		}
	}

	fclose(stream);	

}
