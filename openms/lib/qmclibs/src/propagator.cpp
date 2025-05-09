#include "propagator.hpp"
#if defined(USE_ACCELERATE)
    #include <Accelerate/Accelerate.h>
#elif defined(USE_OPENBLAS)
    #include <cblas.h>
#else
    #error "No BLAS backend defined. Define either USE_ACCELERATE or USE_OPENBLAS."
#endif
#include <omp.h>
#include <cstring>  // for std::memcpy
#include <vector>
#include <complex>
#include <stdexcept>
#include <pybind11/numpy.h>

namespace py = pybind11;

// may use namespace in the future

void propagate_onebody_blas_complex(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> op,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> phi
) {
    if (op.ndim() != 2 || phi.ndim() != 3)
        throw std::runtime_error("Expected op to be 2D and phi to be 3D");

    if (!phi.flags() & py::array::c_style)
        throw std::runtime_error("phi must be contiguous in memory");
    if (!op.flags() & py::array::c_style)
        throw std::runtime_error("op must be contiguous in memory");


    ssize_t n = op.shape(0);
    ssize_t n2 = op.shape(1);
    ssize_t nw = phi.shape(0);
    ssize_t phi_n = phi.shape(1);
    ssize_t m = phi.shape(2);

    if (n != n2 || phi_n != n)
        throw std::runtime_error("Shape mismatch between op and phi");

    auto* A = reinterpret_cast<const void*>(op.data());

    #pragma omp parallel for
    for (ssize_t iw = 0; iw < nw; ++iw) {
        auto* B = reinterpret_cast<const void*>(phi.data(iw, 0, 0));

        std::vector<std::complex<double>> C(n * m);

        std::complex<double> alpha(1.0, 0.0);
        std::complex<double> beta(0.0, 0.0);

        // C = A * B
        cblas_zgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            n, m, n,
            &alpha,
            A, n,
            B, m,
            &beta,
            reinterpret_cast<void*>(C.data()), m
        );

        // Copy result back into phi[iw]
        std::memcpy(phi.mutable_data(iw, 0, 0), C.data(), sizeof(std::complex<double>) * n * m);
    }
}

py::array_t<std::complex<double>> propagate_onebody_blas_complex_return(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> op,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> phi
) {
    if (op.ndim() != 2 || phi.ndim() != 3)
        throw std::runtime_error("Expected op to be 2D and phi to be 3D");

    ssize_t n = op.shape(0);
    ssize_t n2 = op.shape(1);
    ssize_t nw = phi.shape(0);
    ssize_t phi_n = phi.shape(1);
    ssize_t m = phi.shape(2);

    if (n != n2 || phi_n != n)
        throw std::runtime_error("Shape mismatch between op and phi");

    auto* A = reinterpret_cast<const void*>(op.data());

    // Allocate output array
    py::array_t<std::complex<double>> result({nw, n, m});
    auto buf = result.request();

    #pragma omp parallel for
    for (ssize_t iw = 0; iw < nw; ++iw) {
        auto* B = reinterpret_cast<const void*>(phi.data(iw, 0, 0));
        auto* C = reinterpret_cast<void*>(
            static_cast<std::complex<double>*>(buf.ptr) + iw * n * m
        );

        std::complex<double> alpha(1.0, 0.0);
        std::complex<double> beta(0.0, 0.0);

        cblas_zgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            n, m, n,
            &alpha,
            A, n,
            B, m,
            &beta,
            C, m
        );
    }

    return result;
}


void propagate_onebody_blas(py::array_t<double, py::array::c_style | py::array::forcecast> op,
                            py::array_t<double, py::array::c_style | py::array::forcecast> phi) {
    if (op.ndim() != 2 || phi.ndim() != 3)
        throw std::runtime_error("Expected op to be 2D and phi to be 3D");

    ssize_t n = op.shape(0);
    ssize_t n2 = op.shape(1);
    ssize_t nw = phi.shape(0);
    ssize_t phi_n = phi.shape(1);
    ssize_t m = phi.shape(2);

    if (n != n2 || phi_n != n)
        throw std::runtime_error("Shape mismatch between op and phi");

    double* A = static_cast<double*>(op.mutable_data());

    #pragma omp parallel for
    for (ssize_t iw = 0; iw < nw; ++iw) {
        double* B = static_cast<double*>(phi.mutable_data(iw, 0, 0));
        std::vector<double> C(n * m);

        // C = A * B
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    n, m, n,
                    1.0, A, n,
                    B, m,
                    0.0, C.data(), m);

        // Copy result back into phi[iw]
        std::memcpy(B, C.data(), sizeof(double) * n * m);
    }
}


py::array_t<double> propagate_onebody_blas_return(
    py::array_t<double, py::array::c_style | py::array::forcecast> op,
    py::array_t<double, py::array::c_style | py::array::forcecast> phi
) {
    // Check input shapes
    if (op.ndim() != 2 || phi.ndim() != 3)
        throw std::runtime_error("op must be 2D and phi must be 3D");

    ssize_t n = op.shape(0);
    ssize_t n2 = op.shape(1);
    ssize_t nw = phi.shape(0);
    ssize_t phi_n = phi.shape(1);
    ssize_t m = phi.shape(2);

    if (n != n2 || phi_n != n)
        throw std::runtime_error("Dimension mismatch between op and phi");

    // Allocate output: shape (nw, n, m)
    py::array_t<double> result({nw, n, m});
    auto result_ptr = result.mutable_unchecked<3>();

    double* A = static_cast<double*>(op.mutable_data());

    #pragma omp parallel for
    for (ssize_t iw = 0; iw < nw; ++iw) {
        double* B = static_cast<double*>(phi.mutable_data(iw, 0, 0));
        double* C = static_cast<double*>(result.mutable_data(iw, 0, 0));

        // Perform C = A @ B
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    n, m, n,
                    1.0, A, n,
                    B, m,
                    0.0, C, m);
    }

    return result;
}



// for two-body

/*
py::array_t<double> propagate_exp_op_taylor_return(
    py::array_t<double, py::array::c_style | py::array::forcecast> phiw,
    py::array_t<double, py::array::c_style | py::array::forcecast> op,
    int order
) {
    if (phiw.ndim() != 3 || op.ndim() != 3)
        throw std::runtime_error("phiw and op must be 3D");

    ssize_t nwalkers = phiw.shape(0);
    ssize_t ndim = phiw.shape(1);

    if (op.shape(0) != nwalkers || op.shape(1) != ndim || op.shape(2) != ndim)
        throw std::runtime_error("Dimension mismatch between phiw and op");

    // Output array (copy of input)
    py::array_t<double> result = py::array_t<double>(phiw.shape());
    std::memcpy(result.mutable_data(), phiw.data(), sizeof(double) * nwalkers * ndim);

    // Work buffer: temp vector for Taylor terms
    #pragma omp parallel for
    for (ssize_t iw = 0; iw < nwalkers; ++iw) {
        const double* A = op.data(iw, 0, 0);
        std::vector<double> temp(ndim);
        std::memcpy(temp.data(), phiw.data(iw, 0), sizeof(double) * ndim);

        for (int k = 1; k <= order; ++k) {
            std::vector<double> new_temp(ndim);
            cblas_dgemv(CblasRowMajor, CblasNoTrans,
                        ndim, ndim,
                        1.0, A, ndim,
                        temp.data(), 1,
                        0.0, new_temp.data(), 1);

            for (ssize_t i = 0; i < ndim; ++i)
                result.mutable_at(iw, i) += new_temp[i] / static_cast<double>(k);

            temp = std::move(new_temp);
        }
    }
    return result;
}

*/


void propagate_exp_op_taylor(
    py::array_t<double, py::array::c_style | py::array::forcecast> phiw,
    py::array_t<double, py::array::c_style | py::array::forcecast> op,
    int order
) {
    if (phiw.ndim() != 3 || op.ndim() != 3)
        throw std::runtime_error("Expected phiw to be 3D (nw, n, m), and op to be 3D (nw, n, n)");

    ssize_t nw = phiw.shape(0);
    ssize_t n = phiw.shape(1);
    ssize_t m = phiw.shape(2);

    if (op.shape(0) != nw || op.shape(1) != n || op.shape(2) != n)
        throw std::runtime_error("Dimension mismatch between phiw and op");

    #pragma omp parallel for
    for (ssize_t iw = 0; iw < nw; ++iw) {
        const double* A = op.data(iw, 0, 0);
        double* phi = phiw.mutable_data(iw, 0, 0);

        std::vector<double> temp(n * m);
        std::memcpy(temp.data(), phi, sizeof(double) * n * m);

        for (int k = 1; k <= order; ++k) {
            std::vector<double> new_temp(n * m);
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        n, m, n,
                        1.0, A, n,
                        temp.data(), m,
                        0.0, new_temp.data(), m);

            for (ssize_t i = 0; i < n * m; ++i)
                phi[i] += new_temp[i] / static_cast<double>(k);

            temp = std::move(new_temp);
        }
    }
}


void propagate_exp_op_taylor_complex(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> phiw,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> op,
    int order
) {
    if (phiw.ndim() != 3 || op.ndim() != 3)
        throw std::runtime_error("phiw must be 3D (nw, n, m), and op must be 3D (nw, n, n)");

    ssize_t nw = phiw.shape(0);
    ssize_t n = phiw.shape(1);
    ssize_t m = phiw.shape(2);

    if (op.shape(0) != nw || op.shape(1) != n || op.shape(2) != n)
        throw std::runtime_error("Dimension mismatch between phiw and op");

    const std::complex<double> one(1.0, 0.0);
    const std::complex<double> zero(0.0, 0.0);

    #pragma omp parallel for
    for (ssize_t iw = 0; iw < nw; ++iw) {
        const std::complex<double>* A = op.data(iw, 0, 0);
        std::complex<double>* phi = phiw.mutable_data(iw, 0, 0);

        std::vector<std::complex<double>> temp(phi, phi + n * m);

        for (int k = 1; k <= order; ++k) {
            std::vector<std::complex<double>> new_temp(n * m);

            cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        n, m, n,
                        reinterpret_cast<const void*>(&one),
                        reinterpret_cast<const void*>(A), n,
                        reinterpret_cast<const void*>(temp.data()), m,
                        reinterpret_cast<const void*>(&zero),
                        reinterpret_cast<void*>(new_temp.data()), m);

            /*
            double factor = 1.0 / static_cast<double>(k);
            for (ssize_t i = 0; i < n * m; ++i){
                new_temp[i] *= factor;
                phi[i] += new_temp[i];
            }
            */
            std::complex<double> scale(1.0 / static_cast<double>(k), 0.0);
            cblas_zscal(n * m, &scale, new_temp.data(), 1);
            cblas_zaxpy(n * m, &one, new_temp.data(), 1, phi, 1);

            temp = std::move(new_temp);
        }
    }
}


// batched version
// Apple accelerate does not support batch
/*
void propagate_exp_op_taylor_complex_batched(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> phiw,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> op,
    int order
) {
    if (phiw.ndim() != 3 || op.ndim() != 3)
        throw std::runtime_error("phiw must be 3D (nw, n, m), and op must be 3D (nw, n, n)");

    int nw = static_cast<int>(phiw.shape(0));
    int n  = static_cast<int>(phiw.shape(1));
    int m  = static_cast<int>(phiw.shape(2));

    if (op.shape(0) != nw || op.shape(1) != n || op.shape(2) != n)
        throw std::runtime_error("Dimension mismatch between phiw and op");

    const std::complex<double> one(1.0, 0.0);
    const std::complex<double> zero(0.0, 0.0);

    // Allocate buffers
    std::vector<std::vector<std::complex<double>>> temp(nw, std::vector<std::complex<double>>(n * m));
    std::vector<std::vector<std::complex<double>>> result(nw, std::vector<std::complex<double>>(n * m));

    for (int iw = 0; iw < nw; ++iw) {
        std::memcpy(temp[iw].data(), phiw.mutable_data(iw, 0, 0), sizeof(std::complex<double>) * n * m);
    }

    // Batched zgemm inputs
    std::vector<const void*> A_array(nw), B_array(nw);
    std::vector<void*> C_array(nw);

    std::vector<CBLAS_TRANSPOSE> transA(nw, CblasNoTrans);
    std::vector<CBLAS_TRANSPOSE> transB(nw, CblasNoTrans);

    std::vector<int> M(nw, n), N(nw, m), K(nw, n);
    std::vector<int> lda(nw, n), ldb(nw, m), ldc(nw, m);
    std::vector<const void*> alpha(nw, reinterpret_cast<const void*>(&one));
    std::vector<const void*> beta(nw, reinterpret_cast<const void*>(&zero));

    const int group_count = 1;
    std::vector<int> group_size = { nw };

    for (int k = 1; k <= order; ++k) {
        for (int iw = 0; iw < nw; ++iw) {
            A_array[iw] = reinterpret_cast<const void*>(op.data(iw, 0, 0));
            B_array[iw] = reinterpret_cast<const void*>(temp[iw].data());
            C_array[iw] = reinterpret_cast<void*>(result[iw].data());
        }

        cblas_zgemm_batch(CblasRowMajor,
                          transA.data(), transB.data(),
                          M.data(), N.data(), K.data(),
                          alpha.data(), A_array.data(), lda.data(),
                          B_array.data(), ldb.data(),
                          beta.data(), C_array.data(), ldc.data(),
                          group_count, group_size.data());

        // Scale and accumulate result into phiw
        std::complex<double> factor(1.0 / static_cast<double>(k), 0.0);
        for (int iw = 0; iw < nw; ++iw) {
            std::complex<double>* phi = phiw.mutable_data(iw, 0, 0);
            cblas_zscal(n * m,
                        reinterpret_cast<const void*>(&factor),
                        result[iw].data(), 1);

            cblas_zaxpy(n * m,
                        reinterpret_cast<const void*>(&one),
                        result[iw].data(), 1,
                        phi, 1);

            temp[iw].swap(result[iw]);
        }
    }
}

*/

// force bias
