
#ifndef INIT_H
#define INIT_H

#include "fdtd.hpp"

// Initialize MPI
void init_mpi(int argc, char* argv[]);

// Finalize MPI
void finalize_mpi();

void initialize();

void setup_MPI();

// void init_pre();
// void init();
// #ifndef SERIAL
// void init_MPI();
// #endif
// void init_grid();
// void init_time();
// void init_domain();
// void init_fields();
// void init_coeff();
// void init_sim();
// void init_pml();
//
// void init_FT_fields();
//
// void init_workdir();
//
// // note: the geometry file is depreciated
// void geometry(double x, double y, double z, int &n);

#endif
