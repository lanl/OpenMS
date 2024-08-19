#include <mpi.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <omp.h>

namespace py = pybind11;

void initialize_mpi() {
    int is_initialized;
    MPI_Initialized(&is_initialized);
    if (!is_initialized) {
        MPI_Init(NULL, NULL);
    }
}

void finalize_mpi() {
    int is_finalized;
    MPI_Finalized(&is_finalized);
    if (!is_finalized) {
        MPI_Finalize();
    }
}

int get_mpi_rank() {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
}


int mpi_sum() {
    // MPI initialization and finalization are handled outside this function
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int total;
    MPI_Reduce(&world_rank, &total, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    return world_rank == 0 ? total : 0;
}

int parallel_sum(const std::vector<int>& numbers) {

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Calculate the portion of the array each process will handle
    size_t chunk_size = numbers.size() / world_size;
    size_t start = world_rank * chunk_size;
    size_t end = (world_rank == world_size - 1) ? numbers.size() : (start + chunk_size);

    // Sum the local portion of the array
    int local_sum = 0;
    #pragma omp parallel for reduction(+:local_sum)
    for (size_t i = start; i < end; ++i) {
        local_sum += numbers[i];
    }

    // Aggregate the sums from all processes
    int total_sum;
    MPI_Reduce(&local_sum, &total_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    return total_sum; // Now, every process will return the total sum
}


PYBIND11_MODULE(mpi_example, m) {
    m.def("parallel_sum", &parallel_sum, "A function which sums an array of integers using MPI and OpenMP");
    m.def("mpi_sum", &mpi_sum, "A function which sums MPI ranks");
    m.def("initialize_mpi", &initialize_mpi, "Initialize MPI environment");
    m.def("finalize_mpi", &finalize_mpi, "Finalize MPI environment");
    m.def("get_mpi_rank", &get_mpi_rank, "Get the MPI rank of the current process");
}

