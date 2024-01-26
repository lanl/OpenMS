
#include "fdtd.hpp"

// Custom print function
void myprintf(const char *format, ...) {
    va_list args;
    va_start(args, format); // Initialize the argument list

#ifdef MPION
    // When MPI is used:
    int mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank); // Get the current process's rank

    if (mpi_rank == 0) {
        // Only the process with rank 0 will print the message
        vprintf(format, args);
    }
#else
    // When MPI is not used:
    vprintf(format, args);
#endif

    va_end(args); // Clean up the argument list
}

// TODO: use verbosity to control the output level
//

