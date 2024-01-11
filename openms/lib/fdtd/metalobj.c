#include "./pFDTD.h"

int in_mobject(int n,float i,float j,float k);
void reading_matrix_m(char *matrix_file, int n);
int mobjn=0; // number of metal objects
struct mobj *mobject;  // declaration of 1D-array, 'mobject'

// molecular objects
void input_molecular(char *shape, float centerx, float centery, float centerz, float size1, float size2, float size3)
{

    mobjn++;

    mobject=(struct mobj *)realloc(mobject,sizeof(struct mobj)*mobjn);
    strcpy(mobject[mobjn-1].shape,shape);

    mobject[mobjn-1].epsilon_b=100;
    mobject[mobjn-1].omega_p=0;
    mobject[mobjn-1].gamma_0=0;

    printf("\n position of %d th molecular: \n", mobjn);

    if(strcmp(shape,"point")==0)
    {
        mobject[mobjn-1].centeri=(centerx+xcenter)*lattice_x;
        mobject[mobjn-1].centerj=(centery+ycenter)*lattice_y;
        mobject[mobjn-1].centerk=(centerz+zcenter)*lattice_z;

        printf("%f %f \n", centerx, xcenter);
        printf("%f \n", mobject[mobjn-1].centeri);
        printf("%f \n", mobject[mobjn-1].centerj);
        printf("%f \n", mobject[mobjn-1].centerk);
    }
    else
    {
    }
}

void input_Drude_medium(char *shape, float centerx, float centery, float centerz, float size1, float size2, float size3, float epsilon_b, float omega_p, float gamma_0, float lattice_n)
{
    float omega_pn, gamma_0n; //reduced frequency, NOT normalized frequency

    ////////////////////////////////////////////
    /// How to convert into the reduced unit .
    ////////////////////////////////////////////
    /*
            {phase change}=w_p * dt_R = w_F * dt_F 
             Here, w_p = angular frequency in unit of Hz
                   dt_R = time step in unit of sec.
                   w_F = reduced frequency in FDTD
                   dt_F = FDTD time step 1 
             Remember that dt_F is NOT assumed to be 1 
                    BUT represented in unit of (sec q m^-1) (See Eq.49 of manual)

             Now that, 
                   dt_F = 1/(c * S(q^-1))
                   dt_R = 1/(c * S(m^-1))
                        = 1/(c * S(q^-1) * ds_x * lattice_x / lattice_n ) 
             Therefore we get,
                   dt_R/dt_F = lattice_n/(ds_x*lattice_x)              
                   w_F = w_p * lattice_n/(ds_x*lattice_x) 
    ///////////////////////////////////////////////////// */

    omega_pn = omega_p*lattice_n/(ds_x*lattice_x);
    gamma_0n = gamma_0*lattice_n/(ds_x*lattice_x);    

    if(mobjn==0) //printf once!
    {
        printf("----------------\n",omega_pn);    
        printf("omega_pn = %g\n",omega_pn);
        printf("gamma_0n = %g\n",gamma_0n);
    }

    mobjn++;

    mobject=(struct mobj *)realloc(mobject,sizeof(struct mobj)*mobjn);

    strcpy(mobject[mobjn-1].shape,shape);

    mobject[mobjn-1].epsilon_b=epsilon_b;  
    mobject[mobjn-1].omega_p=omega_pn;  //conversion to 'FDTD' frequency
    mobject[mobjn-1].gamma_0=gamma_0n;     //         "      "

    if(strcmp(shape,"rod")==0)
    {
        mobject[mobjn-1].centeri=(centerx+xcenter)*lattice_x;
        mobject[mobjn-1].centerj=(centery+ycenter)*lattice_y;
        mobject[mobjn-1].centerk=(centerz);  // for non_uniform_grid(), the meaning changed
        mobject[mobjn-1].size1=size1*lattice_x;  // radius
        mobject[mobjn-1].size2=(size2);  // height  // for non_uniform_grid(), the meaning changed
    }
    else if(strcmp(shape,"rodX")==0) //// not applicable for non_uniform_grid()
    {
        mobject[mobjn-1].centeri=(centerx+xcenter)*lattice_x;  
        mobject[mobjn-1].centerj=(centery+ycenter)*lattice_y;
        mobject[mobjn-1].centerk=(centerz+zcenter)*lattice_z;
        mobject[mobjn-1].size1=size1*lattice_x;  // radius
        mobject[mobjn-1].size2=size2*lattice_z;  // x length
    }
    else if(strcmp(shape,"rodY")==0) //// not applicable for non_uniform_grid()
    {
        mobject[mobjn-1].centeri=(centerx+xcenter)*lattice_x;  
        mobject[mobjn-1].centerj=(centery+ycenter)*lattice_y;
        mobject[mobjn-1].centerk=(centerz+zcenter)*lattice_z;
        mobject[mobjn-1].size1=size1*lattice_x;  // radius
        mobject[mobjn-1].size2=size2*lattice_z;  // y length
    }
    else if(strcmp(shape,"donut")==0)
    {
        mobject[mobjn-1].centeri=(centerx+xcenter)*lattice_x;  
        mobject[mobjn-1].centerj=(centery+ycenter)*lattice_y;
        mobject[mobjn-1].centerk=(centerz);  // for non_uniform_grid(), the meaning changed
        mobject[mobjn-1].size1=size1*lattice_x;  // inner radius
        mobject[mobjn-1].size2=size2*lattice_x;  // outer radius
        mobject[mobjn-1].size3=(size3);  // height  // for non_uniform_grid(), the meaning changed
    }
    else if(strcmp(shape,"sphere")==0) //// not applicable for non_uniform_grid()
    {
        mobject[mobjn-1].centeri=(centerx+xcenter)*lattice_x;
        mobject[mobjn-1].centerj=(centery+ycenter)*lattice_y;
        mobject[mobjn-1].centerk=(centerz+zcenter)*lattice_z;
        mobject[mobjn-1].size1=size1*lattice_x;  // radius
    }
    else if(strcmp(shape,"shell")==0) //// not applicable for non_uniform_grid()
    {
        mobject[mobjn-1].centeri=(centerx+xcenter)*lattice_x;
        mobject[mobjn-1].centerj=(centery+ycenter)*lattice_y;
        mobject[mobjn-1].centerk=(centerz+zcenter)*lattice_z;
        mobject[mobjn-1].size1=size1*lattice_x;  // inner radius
        mobject[mobjn-1].size2=size2*lattice_x;  // outer radius
    }
    else if(strcmp(shape,"ellipse")==0)
    {
        mobject[mobjn-1].centeri=(centerx+xcenter)*lattice_x;
        mobject[mobjn-1].centerj=(centery+ycenter)*lattice_y;
        mobject[mobjn-1].centerk=(centerz); // for non_uniform_grid(), the meaning changed
        mobject[mobjn-1].size1=size1*lattice_x;  // radius
        mobject[mobjn-1].size2=(size2);  // height (centerz); // for non_uniform_grid() ....
        mobject[mobjn-1].size3=size3;  // aspect ratio=ry/rx
    }
    else if(strcmp(shape,"ellipsoidal")==0) //// not applicable for non_uniform_grid()
    {
        mobject[mobjn-1].centeri=(centerx+xcenter)*lattice_x;
        mobject[mobjn-1].centerj=(centery+ycenter)*lattice_y;
        mobject[mobjn-1].centerk=(centerz+zcenter)*lattice_z;
        mobject[mobjn-1].size1=size1*lattice_x;  // rx
        mobject[mobjn-1].size2=size2*lattice_y;  // ry
        mobject[mobjn-1].size3=size3*lattice_z;  // rz
    }
    else if(strcmp(shape,"cone")==0)
    {
        mobject[mobjn-1].centeri=(centerx+xcenter)*lattice_x;
        mobject[mobjn-1].centerj=(centery+ycenter)*lattice_y;
        mobject[mobjn-1].centerk=(centerz); // for non_uniform_grid()
        mobject[mobjn-1].size1=size1*lattice_x;  // r1
        mobject[mobjn-1].size2=size2*lattice_x;  // r2
        mobject[mobjn-1].size3=(size3);  // for non_uniform_grid() // height
    }
    else if(strcmp(shape,"block")==0)
    {
        mobject[mobjn-1].centeri=(centerx+xcenter)*lattice_x;
        mobject[mobjn-1].centerj=(centery+ycenter)*lattice_y;
        mobject[mobjn-1].centerk=(centerz); // for non_uniform_grid(), the meaning changed 
        mobject[mobjn-1].size1=size1*lattice_x;
        mobject[mobjn-1].size2=size2*lattice_y;
        mobject[mobjn-1].size3=(size3); // for non_uniform_grid(), the meaning changed
    }
    else
    {
    }
}

void input_Drude_medium2(char *shape, char *matrix_file, float centerx, float centery, float centerz, float size1, float size2, float size3, float epsilon_b, float omega_p, float gamma_0, float lattice_n)
{    
    float omega_pn, gamma_0n; //reduced frequency, NOT normalized frequency

    ////////////////////////////////////////////
    /// How to convert into the reduced unit .
    ////////////////////////////////////////////
    /*
            {phase change}=w_p * dt_R = w_F * dt_F 
             Here, w_p = angular frequency in unit of Hz
                   dt_R = time step in unit of sec.
                   w_F = reduced frequency in FDTD
                   dt_F = FDTD time step 1 
             Remember that dt_F is NOT assumed to be 1 
                    BUT represented in unit of (sec q m^-1) (See Eq.49 of manual)
             
             Now that, 
                   dt_F = 1/(c * S(q^-1))
                   dt_R = 1/(c * S(m^-1))
                        = 1/(c * S(q^-1) * ds_x * lattice_x / lattice_n ) 
             Therefore we get,
                   dt_R/dt_F = lattice_n/(ds_x*lattice_x)              
                   w_F = w_p * lattice_n/(ds_x*lattice_x) 
    ///////////////////////////////////////////////////// */

    omega_pn = omega_p*lattice_n/(ds_x*lattice_x);
    gamma_0n = gamma_0*lattice_n/(ds_x*lattice_x);    

    if(mobjn==0) //printf once!
    {
        printf("----------------\n",omega_pn);    
        printf("omega_pn = %g\n",omega_pn);
        printf("gamma_0n = %g\n",gamma_0n);
    }

    mobjn++;

    mobject=(struct mobj *)realloc(mobject,sizeof(struct mobj)*mobjn);

    strcpy(mobject[mobjn-1].shape,shape);

    mobject[mobjn-1].epsilon_b=epsilon_b;  
    mobject[mobjn-1].omega_p=omega_pn;  //conversion to 'FDTD' frequency
    mobject[mobjn-1].gamma_0=gamma_0n;     //         "      "

    if(strcmp(shape,"contour")==0) //// from ver. 8.80
    {
        mobject[mobjn-1].centeri=centerx;           // matrix_center 
        mobject[mobjn-1].centerj=centery;           // matrix_center 
        mobject[mobjn-1].centerk=(centerz); // for non_uniform_grid(), the meaning changed
        mobject[mobjn-1].size1=size1;     // discrimination level
        mobject[mobjn-1].size2=(size2);   // height  // for non_uniform_grid() .....
        mobject[mobjn-1].size3=size3;     // compression factor
        reading_matrix_m(matrix_file, mobjn-1);    
    }
    if(strcmp(shape,"rod")==0)
    {
        mobject[mobjn-1].centeri=(centerx+xcenter)*lattice_x;
        mobject[mobjn-1].centerj=(centery+ycenter)*lattice_y;
        mobject[mobjn-1].centerk=(centerz);  // for non_uniform_grid(), the meaning changed
        mobject[mobjn-1].size1=size1*lattice_x;  // radius
        mobject[mobjn-1].size2=(size2);  // height  // for non_uniform_grid(), the meaning changed
    }
    else if(strcmp(shape,"rodX")==0) //// not applicable for non_uniform_grid()
    {
        mobject[mobjn-1].centeri=(centerx+xcenter)*lattice_x;  
        mobject[mobjn-1].centerj=(centery+ycenter)*lattice_y;
        mobject[mobjn-1].centerk=(centerz+zcenter)*lattice_z;
        mobject[mobjn-1].size1=size1*lattice_x;  // radius
        mobject[mobjn-1].size2=size2*lattice_z;  // x length
    }
    else if(strcmp(shape,"rodY")==0) //// not applicable for non_uniform_grid()
    {
        mobject[mobjn-1].centeri=(centerx+xcenter)*lattice_x;  
        mobject[mobjn-1].centerj=(centery+ycenter)*lattice_y;
        mobject[mobjn-1].centerk=(centerz+zcenter)*lattice_z;
        mobject[mobjn-1].size1=size1*lattice_x;  // radius
        mobject[mobjn-1].size2=size2*lattice_z;  // y length
    }
    else if(strcmp(shape,"donut")==0)
    {
        mobject[mobjn-1].centeri=(centerx+xcenter)*lattice_x;  
        mobject[mobjn-1].centerj=(centery+ycenter)*lattice_y;
        mobject[mobjn-1].centerk=(centerz);  // for non_uniform_grid(), the meaning changed
        mobject[mobjn-1].size1=size1*lattice_x;  // inner radius
        mobject[mobjn-1].size2=size2*lattice_x;  // outer radius
        mobject[mobjn-1].size3=(size3);  // height  // for non_uniform_grid(), the meaning changed
    }
    else if(strcmp(shape,"sphere")==0) //// not applicable for non_uniform_grid()
    {
        mobject[mobjn-1].centeri=(centerx+xcenter)*lattice_x;
        mobject[mobjn-1].centerj=(centery+ycenter)*lattice_y;
        mobject[mobjn-1].centerk=(centerz+zcenter)*lattice_z;
        mobject[mobjn-1].size1=size1*lattice_x;  // radius
    }
    else if(strcmp(shape,"shell")==0) //// not applicable for non_uniform_grid()
    {
        mobject[mobjn-1].centeri=(centerx+xcenter)*lattice_x;
        mobject[mobjn-1].centerj=(centery+ycenter)*lattice_y;
        mobject[mobjn-1].centerk=(centerz+zcenter)*lattice_z;
        mobject[mobjn-1].size1=size1*lattice_x;  // inner radius
        mobject[mobjn-1].size2=size2*lattice_x;  // outer radius
    }
    else if(strcmp(shape,"ellipse")==0)
    {
        mobject[mobjn-1].centeri=(centerx+xcenter)*lattice_x;
        mobject[mobjn-1].centerj=(centery+ycenter)*lattice_y;
        mobject[mobjn-1].centerk=(centerz); // for non_uniform_grid(), the meaning changed
        mobject[mobjn-1].size1=size1*lattice_x;  // radius
        mobject[mobjn-1].size2=(size2);  // height (centerz); // for non_uniform_grid() ....
        mobject[mobjn-1].size3=size3;  // aspect ratio=ry/rx
    }
    else if(strcmp(shape,"ellipsoidal")==0) //// not applicable for non_uniform_grid()
    {
        mobject[mobjn-1].centeri=(centerx+xcenter)*lattice_x;
        mobject[mobjn-1].centerj=(centery+ycenter)*lattice_y;
        mobject[mobjn-1].centerk=(centerz+zcenter)*lattice_z;
        mobject[mobjn-1].size1=size1*lattice_x;  // rx
        mobject[mobjn-1].size2=size2*lattice_y;  // ry
        mobject[mobjn-1].size3=size3*lattice_z;  // rz
    }
    else if(strcmp(shape,"cone")==0)
    {
        mobject[mobjn-1].centeri=(centerx+xcenter)*lattice_x;
        mobject[mobjn-1].centerj=(centery+ycenter)*lattice_y;
        mobject[mobjn-1].centerk=(centerz); // for non_uniform_grid()
        mobject[mobjn-1].size1=size1*lattice_x;  // r1
        mobject[mobjn-1].size2=size2*lattice_x;  // r2
        mobject[mobjn-1].size3=(size3);  // for non_uniform_grid() // height
    }
    else if(strcmp(shape,"block")==0)
    {
        mobject[mobjn-1].centeri=(centerx+xcenter)*lattice_x;
        mobject[mobjn-1].centerj=(centery+ycenter)*lattice_y;
        mobject[mobjn-1].centerk=(centerz); // for non_uniform_grid(), the meaning changed 
        mobject[mobjn-1].size1=size1*lattice_x;
        mobject[mobjn-1].size2=size2*lattice_y;
        mobject[mobjn-1].size3=(size3); // for non_uniform_grid(), the meaning changed
    }
    else
    {
    }
}

void make_metal_structure()
{
    int i,j,k;
    int n;

    float me, mw, mr;

    printf("-------------------------------\n");
    printf("making the metal structure...\n");

    ///////////////////////////////////////////////////
    ///        Medium Rules for (i,j,k)             ///
    ///---------------------------------            ///
    ///  mepsilon=momega= 0.0 --> dielectric        ///
    ///  mepsilon!=0.0 --> metal                    ///
    ///  mepsilon=1000 --> PCM                      ///
    ///  mepsilon=0.0 & momega=0.0 --> metal eraser ///
    ///////////////////////////////////////////////////

    for(i=0;i<misize;i++)
    {
        printf("%d% \n",100*i/misize);

        for(j=0;j<mjsize;j++)
            for(k=0;k<mksize;k++)
            {
                me = mw = mr = 0.0; // initialization : non-metal
                for(n=0;n<mobjn;n++)
                    if(in_mobject(n,i,j,k)==1)
                    {
                        me=mobject[n].epsilon_b;
                        mw=mobject[n].omega_p;
                        mr=mobject[n].gamma_0;
                    }
                mepsilon[i][j][k]=me;
                momega[i][j][k]=mw;
                mgamma[i][j][k]=mr;
            }
    }

    free(mobject);

    printf("make_metal_structure...ok\n");
}

int in_mobject(int n,float i,float j,float k)
{
    int I, J;
    float X, Y, Z;
    int temp;

    if(strcmp(mobject[n].shape,"rod")==0)
    {
        if( (i-mobject[n].centeri)*(i-mobject[n].centeri)+(j-mobject[n].centerj)*(j-mobject[n].centerj)<=mobject[n].size1*mobject[n].size1 &&
            non_uniform_z_to_i(mobject[n].centerk-0.5*mobject[n].size2)<=k && k<=non_uniform_z_to_i(mobject[n].centerk+0.5*mobject[n].size2) ) return 1;
        else return 0;
    }

    else if(strcmp(mobject[n].shape,"point")==0)
    {
        if( (i-mobject[n].centeri)*(i-mobject[n].centeri)<1.E-6)
        {
              if( (j-mobject[n].centerj)*(j-mobject[n].centerj)<1.E-6 && 
                (k-mobject[n].centerk)*(k-mobject[n].centerk)<1.E-6 )  printf("test1 %f %f %f \n", i, j, k);
        }

        if( (i-mobject[n].centeri)*(i-mobject[n].centeri)+(j-mobject[n].centerj)*(j-mobject[n].centerj)+(k-mobject[n].centerk)*(k-mobject[n].centerk)<1.E-6 ) return 1;
        else return 0;
    }

    else if(strcmp(mobject[n].shape,"rodX")==0)
    {
        if( (j-mobject[n].centerj)*(j-mobject[n].centerj)+(k-mobject[n].centerk)*(k-mobject[n].centerk)<=mobject[n].size1*mobject[n].size1 && mobject[n].centeri-mobject[n].size2/2<=i && i<=mobject[n].centeri+mobject[n].size2/2) return 1;
        else return 0;
    }

    else if(strcmp(mobject[n].shape,"rodY")==0)
    {
        if( (k-mobject[n].centerk)*(k-mobject[n].centerk)+(i-mobject[n].centeri)*(i-mobject[n].centeri)<=mobject[n].size1*mobject[n].size1 && mobject[n].centerj-mobject[n].size2/2<=j && j<=mobject[n].centerj+mobject[n].size2/2) return 1;
        else return 0;
    }

    else if(strcmp(mobject[n].shape,"donut")==0)
    {
            if( (i-mobject[n].centeri)*(i-mobject[n].centeri)+(j-mobject[n].centerj)*(j-mobject[n].centerj)>=mobject[n].size1*mobject[n].size1 && (i-mobject[n].centeri)*(i-mobject[n].centeri)+(j-mobject[n].centerj)*(j-mobject[n].centerj)<=mobject[n].size2*mobject[n].size2 && non_uniform_z_to_i(mobject[n].centerk-0.5*mobject[n].size3)<=k && k<=non_uniform_z_to_i(mobject[n].centerk+0.5*mobject[n].size3)) return 1;
            else return 0;
    }

    else if(strcmp(mobject[n].shape,"sphere")==0)
    {
        if( (i-mobject[n].centeri)*(i-mobject[n].centeri)+(j-mobject[n].centerj)*(j-mobject[n].centerj)+(k-mobject[n].centerk)*(k-mobject[n].centerk)<=mobject[n].size1*mobject[n].size1 ) return 1;
        else return 0;
    }

    else if(strcmp(mobject[n].shape,"shell")==0)
    {
        if( (i-mobject[n].centeri)*(i-mobject[n].centeri)+(j-mobject[n].centerj)*(j-mobject[n].centerj)+(k-mobject[n].centerk)*(k-mobject[n].centerk)>=mobject[n].size1*mobject[n].size1 && (i-mobject[n].centeri)*(i-mobject[n].centeri)+(j-mobject[n].centerj)*(j-mobject[n].centerj)+(k-mobject[n].centerk)*(k-mobject[n].centerk)<=mobject[n].size2*mobject[n].size2) return 1;
        else return 0;
    }

    else if(strcmp(mobject[n].shape,"ellipse")==0)
    {
        if( (i-mobject[n].centeri)*(i-mobject[n].centeri)+(j-mobject[n].centerj)*(j-mobject[n].centerj)/(mobject[n].size3*mobject[n].size3)<=mobject[n].size1*mobject[n].size1 && non_uniform_z_to_i(mobject[n].centerk-0.5*mobject[n].size2)<=k && k<=non_uniform_z_to_i(mobject[n].centerk+0.5*mobject[n].size2)) return 1;
        else return 0;
    }

    else if(strcmp(mobject[n].shape,"ellipsoidal")==0)
    {
        if( (i-mobject[n].centeri)*(i-mobject[n].centeri)/((mobject[n].size1)*(mobject[n].size1))+(j-mobject[n].centerj)*(j-mobject[n].centerj)/((mobject[n].size2)*(mobject[n].size2))+(k-mobject[n].centerk)*(k-mobject[n].centerk)/((mobject[n].size3)*(mobject[n].size3))<=1 ) return 1;
        else return 0;
    }

    else if(strcmp(mobject[n].shape,"cone")==0)
    {
        ///////////////////////////////////////////////
        //  mobject[n].centerk & mobject[n].size3 --> in units of z
        //  size3 : height
        ///////////////////////////////////////////////
        temp = (mobject[n].size1-mobject[n].size2)*(non_uniform_i_to_z(k)-mobject[n].centerk)/mobject[n].size3+0.5*(mobject[n].size1+mobject[n].size2); // radius varying as a function of 'k'

        if( (i-mobject[n].centeri)*(i-mobject[n].centeri)+(j-mobject[n].centerj)*(j-mobject[n].centerj) <= temp*temp && (k <= non_uniform_z_to_i(mobject[n].centerk+0.5*mobject[n].size3)) && (k >= non_uniform_z_to_i(mobject[n].centerk-0.5*mobject[n].size3)) ) return 1;
        else return 0;
    }

    else if(strcmp(mobject[n].shape,"block")==0)
    {
        if( mobject[n].centeri-mobject[n].size1/2<=i && i<=mobject[n].centeri+mobject[n].size1/2 &&
            mobject[n].centerj-mobject[n].size2/2<=j && j<=mobject[n].centerj+mobject[n].size2/2 &&
            non_uniform_z_to_i(mobject[n].centerk-0.5*mobject[n].size3)<=k && k<=non_uniform_z_to_i(mobject[n].centerk+0.5*mobject[n].size3) ) return 1;
        else return 0;
    }

    else if(strcmp(mobject[n].shape,"contour")==0)
    {
        if( (-mobject[n].centeri*CUTFACTOR <= (i-(xcenter*lattice_x))/mobject[n].size3) && ((i-(xcenter*lattice_x))/mobject[n].size3 <= (mobject[n].col-mobject[n].centeri)*CUTFACTOR) && (-mobject[n].centerj*CUTFACTOR <= (j-(ycenter*lattice_y))/mobject[n].size3) && ((j-(ycenter*lattice_y))/mobject[n].size3 <= (mobject[n].row-mobject[n].centerj)*CUTFACTOR) && (non_uniform_z_to_i(mobject[n].centerk-0.5*mobject[n].size2) <= k) && (k <= non_uniform_z_to_i(mobject[n].centerk+0.5*mobject[n].size2)) )
        {
            I = floor( mobject[n].centeri + (i - (xcenter*lattice_x))/mobject[n].size3 ); 
            J = floor( mobject[n].centerj + (j - (ycenter*lattice_y))/mobject[n].size3 );
                   if( mobject[n].matrix[I][J] <= mobject[n].size1 )
            return 1;
        }
        else return 0;
    }

    else
    {
    }
}

void reading_matrix_m(char *matrix_file, int n)
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

    mobject[n].col=col; mobject[n].row=row;

    //////// matrix memory allocation /////
    mobject[n].matrix = (float **)malloc(sizeof(float *)*col);
    for(i=0; i<col; i++)
        mobject[n].matrix[i] = (float *)malloc(sizeof(float)*row);

    stream = fopen(matrix_file,"rt");
    //////// Reading matrix file data //////
    for(j=row-1; j>=0; j--)
    {
        for(i=0; i<col; i++)
        {
            fscanf(stream,"%s",string);
            mobject[n].matrix[i][j] = atof(string);
        }
    }

    fclose(stream);

}

