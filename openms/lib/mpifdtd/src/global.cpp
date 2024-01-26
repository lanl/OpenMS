
#include "fdtd.hpp"

#ifdef MPION
int mpi_rank, mpi_size;

MPI_Comm   mpi_grid_comm;

int   mpi_grid_proces[3];
int   mpi_nxprocs, mpi_nyprocs, mpi_nzprocs;
int   mpi_grid_coords[3];
int   mpi_grid_n1[3];
int   mpi_grid_n2[3];
int   mpi_efield_isend, mpi_efield_irecv, mpi_efield_jsend, mpi_efield_jrecv;
int   mpi_efield_ksend, mpi_efield_krecv, mpi_hfield_isend, mpi_hfield_irecv;
int   mpi_hfield_jsend, mpi_hfield_jrecv, mpi_hfield_ksend, mpi_hfield_krecv;

#endif
