#include <pybind11/pybind11.h>
#include "fdtd.hpp"


// biding examples
namespace py = pybind11;

void initialize_and_run_fdtd() {
    int argc = 1;  // Initialize argc and argv as needed for MPI
    char* argv[1] = {nullptr};

    init_mpi(argc, argv);  // Initialize MPI

    // Call your C/C++ functions here
    init_pre();
    input(argc, argv);
    init();
    FDTD();
    cleanup();

    finalize_mpi();  // Finalize MPI
}

PYBIND11_MODULE(your_module_name, m) {
    m.def("initialize_and_run_fdt", &initialize_and_run_fdt, "Initialize MPI and run FDTD");
}


