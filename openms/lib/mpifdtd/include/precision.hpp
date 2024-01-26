
#ifndef PRECISION_H
#define PRECISION_H

// Define PRECISION_DOUBLE in your build settings or before including this file
// to use double precision. Otherwise, float precision will be used.
#ifdef PRECISION_DOUBLE
    using precision = double;
#else
    using precision = float;
#endif

#endif // PRECISION_H

