#if defined(USE_ACCELERATE)
    #include <Accelerate/Accelerate.h>
#elif defined(USE_OPENBLAS)
    #include <cblas.h>
#else
    #error "No BLAS backend defined. Define either USE_ACCELERATE or USE_OPENBLAS."
#endif
#include <omp.h>
#include <vector>
#include <complex>
#include <stdexcept>
#include <cstring>
#include <algorithm>
#include "linalg.hpp"

namespace py = pybind11;


py::array_t<std::complex<double>> tensordot_complex(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> A,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> B
) {
    if (A.ndim() != 2 || B.ndim() != 3)
        throw std::runtime_error("Expected A (2D) and B (3D)");

    ssize_t z = A.shape(0);   // A: (z, n)
    ssize_t n = A.shape(1);   // A: (z, n)
    ssize_t p = B.shape(1);   // B: (n, p, q)
    ssize_t q = B.shape(2);

    if (B.shape(0) != n)
        throw std::runtime_error("Shape mismatch: A.shape(1) != B.shape(0)");

    py::array_t<std::complex<double>> result({z, p, q});
    auto A_buf = A.unchecked<2>();
    auto B_buf = B.unchecked<3>();

    auto C_ptr = static_cast<std::complex<double>*>(result.request().ptr);

    #pragma omp parallel for
    for (ssize_t zi = 0; zi < z; ++zi) {
        std::fill_n(C_ptr + zi * p * q, p * q, std::complex<double>(0.0, 0.0));

        for (ssize_t ni = 0; ni < n; ++ni) {
            const std::complex<double> alpha = A_buf(zi, ni);

            for (ssize_t pi = 0; pi < p; ++pi) {
                for (ssize_t qi = 0; qi < q; ++qi) {
                    C_ptr[zi * p * q + pi * q + qi] += alpha * B_buf(ni, pi, qi);
                }
            }
        }
    }


    /*
    for (ssize_t zi = 0; zi < z; ++zi) {
        std::fill_n(&C_buf(zi, 0, 0), p * q, std::complex<double>(0.0, 0.0));

        for (ssize_t ni = 0; ni < n; ++ni) {
            const std::complex<double> alpha = A_buf(zi, ni);
            const std::complex<double>* B_mat = &B_buf(ni, 0, 0);
            std::complex<double>* C_mat = &C_buf(zi, 0, 0);
            cblas_zaxpy(p * q, &alpha, B_mat, 1, C_mat, 1);
        }
    }
    */
    return result;
}
