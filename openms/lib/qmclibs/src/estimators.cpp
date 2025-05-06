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
#include "estimators.hpp"

namespace py = pybind11;

py::array_t<std::complex<double>> exx_rltensor_Ghalf_complex(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> rltensor,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> Ghalf
) {
    if (rltensor.ndim() != 3 || Ghalf.ndim() != 3)
        throw std::runtime_error("Expected 3D arrays");

    ssize_t nchol = rltensor.shape(0);
    ssize_t nao = rltensor.shape(1);
    ssize_t no = rltensor.shape(2);
    ssize_t nwalkers = Ghalf.shape(0);

    if (Ghalf.shape(1) != nao || Ghalf.shape(2) != no)
        throw std::runtime_error("Shape mismatch between rltensor and Ghalf");

    py::array_t<std::complex<double>> exx({nwalkers});
    auto* exx_ptr = exx.mutable_data();


    #pragma omp parallel for
    for (ssize_t i = 0; i < nwalkers; ++i) {
        std::complex<double> acc = 0.0;

        std::vector<std::complex<double>> LG(no * no);
        for (ssize_t l = 0; l < nchol; ++l) {
            const std::complex<double>* L = rltensor.data(l, 0, 0);
            const std::complex<double>* G = Ghalf.data(i, 0, 0);

            const std::complex<double> one(1.0, 0.0), zero(0.0, 0.0);

            // LG = L.T @ G (L is real, G is complex)
            cblas_zgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        no, no, nao,
                        &one,
                        L, no,
                        G, no,
                        &zero,
                        LG.data(), no);

            // Compute LG_flat @ LG^T_flat
            for (ssize_t x = 0; x < no; ++x) {
                for (ssize_t y = 0; y < no; ++y) {
                    acc += LG[x * no + y] * LG[y * no + x];  // LG[i,j] * LG[j,i]
                }
            }
            /*
            //  Create pointer to LG^T without transposing: just reinterpret indexing
            std::complex<double> tmp = 0.0;
            // cblas_zdotc_sub(
            cblas_zdotu_sub(
                no * no,
                reinterpret_cast<const void*>(LG.data()),
                1,
                reinterpret_cast<const void*>(LG.data()),
                no + 1,  // stride between transposed elements
                &tmp
            );
            acc += tmp;
            */
        }
        exx_ptr[i] = 0.5 * acc;
    }

    return exx;
}


py::array_t<double> exx_rltensor_Ghalf_real(
    py::array_t<double, py::array::c_style | py::array::forcecast> rltensor,
    py::array_t<double, py::array::c_style | py::array::forcecast> Ghalf
) {
    if (rltensor.ndim() != 3 || Ghalf.ndim() != 3)
        throw std::runtime_error("Expected 3D arrays");

    ssize_t nchol = rltensor.shape(0);
    ssize_t nao = rltensor.shape(1);
    ssize_t no = rltensor.shape(2);
    ssize_t nwalkers = Ghalf.shape(0);

    if (Ghalf.shape(1) != nao || Ghalf.shape(2) != no)
        throw std::runtime_error("Shape mismatch between rltensor and Ghalf");

    py::array_t<double> exx({nwalkers});
    auto* exx_ptr = exx.mutable_data();

    #pragma omp parallel for
    for (ssize_t i = 0; i < nwalkers; ++i) {
        double acc = 0.0;

        for (ssize_t l = 0; l < nchol; ++l) {
            const double* L = rltensor.data(l, 0, 0);
            const double* G = Ghalf.data(i, 0, 0);

            std::vector<double> LG(no * no);

            const double one = 1.0, zero = 0.0;

            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        no, no, nao,
                        one,
                        L, no,
                        G, no,
                        zero,
                        LG.data(), no);

            for (ssize_t x = 0; x < no; ++x) {
                for (ssize_t y = 0; y < no; ++y) {
                    acc += LG[x * no + y] * LG[y * no + x];
                }
            }
            /*
            double tmp = 0.0;
            cblas_ddot(no * no,
                       LG.data(),
                       1,
                       LG.data(),
                       no + 1);
            acc += tmp;
            */

        }

        exx_ptr[i] = 0.5 * acc;
    }

    return exx;
}

