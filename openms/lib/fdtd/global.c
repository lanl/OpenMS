#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>

int SpecN;
float shift;
long t;
float Sumx, Sumy, Sumz, SideSumx, SideSumy, SumUpper, SumLower;
const gsl_rng_type *rng_type;
gsl_rng *rng_r;
