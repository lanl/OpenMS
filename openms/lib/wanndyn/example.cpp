// examples/example.cpp
#include "we2pd_banner.hpp"
#include "wanndyn.hpp"
#include "constants.hpp"

#include <iostream>

int main() {
    // RAII guard: prints header now and footer at scope exit (even on early-return)
    we2pd::banner::Scope banner;

    constexpr std::size_t n = 4;
    WannDyn::DenseDensityMatrix rho(n);
    rho.zero();

    // Make rho = identity (note: 0-based indices in C++)
    for (std::size_t i = 0; i < n; ++i) {
        rho(i, i) = parameters::real(1.0);
    }

    // Compute and print the trace
    parameters::real trace = 0.0;
    for (std::size_t i = 0; i < n; ++i) trace += rho(i, i);

    std::cout << "Density matrix size: " << rho.n_mo() << " x " << rho.n_mo() << "\n";
    std::cout << "Trace(rho) = " << trace << "\n";

    // Show a sample element access (column-major storage under the hood)
    std::size_t i = 1, j = 2;
    std::cout << "rho(" << i << "," << j << ") = " << rho(i, j) << "\n";

    // Optionally invoke the library routine (currently placeholder):
    // WannDyn::we2pd();

    return 0;
}

