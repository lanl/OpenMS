/* # of multi-frequencies */
int SpecN;

/* vertical origin shift in the presence of the vertical asymmetry */
float shift;

/* FDTD update time variable */
long t;

/* variables used for separating the vertical loss and the horizontal loss */
float Sumx, Sumy, Sumz, SideSumx, SideSumy, SumUpper, SumLower;

/*variables for GSL random number generator*/
const gsl_rng_type *rng_type;
gsl_rng *rng_r;
