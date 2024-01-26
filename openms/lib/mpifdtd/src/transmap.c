#include "./fdtd.hpp"

int counting_row_col(char *name, int mm);
void tr_reading_data(int rcnum, float **umdata, char *name, int mm);
void ad_reading_data(int tnum, float **sumdata, char *name, int mm);
void into_polar(float **umdata, float **findata, int rcnum, int tnum, int mm);
void tr_writing_data(float **findata, int tnum, char *name, int mm);
void ad_writing_data(float **sumdata, int tnum, char *name);
int circle_out(float i, float j, int tnum);
float calc_theta(float i, float j, int tnum);
float calc_phi(float i, float j, int tnum);
float interpol(float theta, float phi, int rcnum, float **umdata);

//----------------------------------------------------------------

void transform_farfield(int NROW, int tnum, char *name, int mm)
{
	int i;
	int rcnum;
	float **umdata, **findata;  // umdata = source, findata = result

	rcnum = counting_row_col(name, mm);
	printf("rcnum=%d\n",rcnum);
	umdata = (float **)malloc(sizeof(float *)*rcnum);
	for(i=0; i<rcnum; i++)
		umdata[i] = (float *)malloc(sizeof(float)*rcnum);	
	findata = (float **)malloc(sizeof(float *)*tnum);
	for(i=0; i<tnum; i++)
		findata[i] = (float *)malloc(sizeof(float)*tnum);
		
	tr_reading_data(rcnum, umdata, name, mm);
	printf("reading ok...\n");
	into_polar(umdata, findata, rcnum, tnum, mm);
	printf("into polar ok ...\n");
	tr_writing_data(findata, tnum, name, mm);
	printf("writing ok ...\n");

	free(umdata); free(findata);

	printf("transform farfield data [%d] ok....!\n",mm);
}

void add_farfield(int tnum, char *name)
{
	int i;
	int mm;
	float **sumdata;

	sumdata = (float **)calloc(tnum,sizeof(float *));   //calloc --> initialization 
	for(i=0; i<tnum; i++)
		sumdata[i] = (float *)calloc(tnum,sizeof(float));

	for(mm=0; mm<SpecN; mm++)
		ad_reading_data(tnum, sumdata, name, mm);

	ad_writing_data(sumdata, tnum, name);

	free(sumdata);
}

int counting_row_col(char *name, int mm)
{
	char ch;
	char name_freq[10];
	char name_head[20];
	FILE *stream;
	int col=0, row=0;

	// making file name
	sprintf(name_freq,".ri%02d",mm);
	sprintf(name_head,"%s",name);
	strcat(name_head,name_freq);

	// reading ..
	stream = fopen(name_head,"rt");

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
	fclose(stream);
	
	return(row);
}

void tr_reading_data(int rcnum, float **umdata, char *name, int mm)
{
	int i, j;
	char name_freq[10];
	char name_head[20];
	FILE *stream;
	char string[40];

	// making file name
	sprintf(name_freq,".ri%02d",mm);
	sprintf(name_head,"%s",name);
	strcat(name_head,name_freq);

	// reading ..
	stream = fopen(name_head,"rt");
	for(j=rcnum-1; j>=0; j--)
	{
		for(i=0; i<rcnum; i++)
		{
			fscanf(stream, "%s", string);
			umdata[i][j] = atof(string);
		}
	}
	fclose(stream);
}

void ad_reading_data(int tnum, float **sumdata, char *name, int mm)
{
	int i, j;
	char name_freq[10];
	char name_head[20];
	FILE *stream;
	char string[40];

	// making file name
	sprintf(name_freq,".sas%02d",mm);
	sprintf(name_head,"%s",name);
	strcat(name_head,name_freq);

	// reading ..
	stream = fopen(name_head,"rt");
	for(j=tnum-1; j>=0; j--)
	{
		for(i=0; i<tnum; i++)
		{
			fscanf(stream, "%s", string);
			sumdata[i][j] += atof(string);
		}
	}
	fclose(stream);
}

void into_polar(float **umdata, float **findata, int rcnum, int tnum, int mm)
{
	int i, j;
	float theta, phi;
	int r; // r=rcnum/2

	r = (int)(rcnum/2);

	for(i=0; i<tnum; i++)
	{
		for(j=0; j<tnum; j++)
		{
			if( circle_out(i,j,tnum) == 0 )
				findata[i][j] = 0.0;
			else
			{
				theta = calc_theta(i,j,tnum);
				if(fabs(theta)<0.02)
					findata[i][j] = umdata[r][r];
				else
				{
					phi = calc_phi(i,j,tnum);
					findata[i][j] = interpol(theta,phi,rcnum,umdata);
				}
			}
		}
	}
}

void tr_writing_data(float **findata, int tnum, char *name, int mm)
{
	int i,j;
	FILE *stream;
	char name_freq[10];
	char name_head[20];

	// making file name
	sprintf(name_freq,".sas%02d",mm);
	sprintf(name_head,"%s",name);
	strcat(name_head,name_freq);

	stream = fopen(name_head,"wt");
	for(j=tnum-1; j>=0; j--)
	{
		for(i=0; i<tnum; i++)
			fprintf(stream, "%g\t", findata[i][j]);
		fprintf(stream, "\n");
	}
	fclose(stream);
}

void ad_writing_data(float **sumdata, int tnum, char *name)
{
	int i,j;
	FILE *stream;
	char name_head[20];

	//making file name
	sprintf(name_head,"%s",name);
	strcat(name_head,"sum.sas");

	stream = fopen(name_head,"wt");
	for(j=tnum-1; j>=0; j--)
	{
		for(i=0; i<tnum; i++)
			fprintf(stream, "%g\t", sumdata[i][j]);
		fprintf(stream, "\n");
	}
	fclose(stream);
}

int circle_out(float i, float j, int tnum)
{
	int tr; // tr=tnum/2;
	tr = (int)(tnum/2);

	if( ((i-tr)*(i-tr) + (j-tr)*(j-tr) - (tr-1)*(tr-1))>0.1 )
		return(0);
	else 
		return(1);
}

float calc_theta(float i, float j, int tnum)
{
	int tr; // tr=tnum/2;
	float temp;

	tr = (int)(tnum/2);

	temp = sqrt( (i-tr)*(i-tr) + (j-tr)*(j-tr) );

	return( temp*pi/2/tr );
}

float calc_phi(float i, float j, int tnum)
{
	float testphi;
	float gauge;
	float temp;
	int tr; // tr=tnum/2;
	tr = (int)(tnum/2);

	temp = sqrt( (i-tr)*(i-tr) + (j-tr)*(j-tr) );

    	testphi = acos( (i-tr)/temp );
    	gauge = (j-tr) - temp*sin(testphi);
	if( fabs(gauge) < 0.01 )
		return( testphi );
	else
		return( 2*pi - testphi );
}

float interpol(float theta, float phi, int rcnum, float **umdata)
{
	double A, B, C;
	double p, q;
	int fp, fq, cp, cq;
	int r; // r=tnum/2;
	r = (int)(rcnum/2);

	p = r*sin(theta)*cos(phi);  
	q = r*sin(theta)*sin(phi);

	fp = (int)floor(p)+r;
	fq = (int)floor(q)+r;
	cp = (int)ceil(p)+r;
	cq = (int)ceil(q)+r;

	A = umdata[fp][fq] + (p+r-fp)*(umdata[cp][fq]-umdata[fp][fq]);
	B = umdata[fp][cq] + (p+r-fp)*(umdata[cp][cq]-umdata[fp][cq]);
	C = A + (q+r-fq)*(B-A);

    	return(C);
}
