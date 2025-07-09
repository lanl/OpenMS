#include "qedhf.hpp"

#if defined(USE_ACCELERATE)
    #include <Accelerate/Accelerate.h>
#elif defined(USE_OPENBLAS)
    #include <cblas.h>
#else
    #error "No BLAS backend defined. Define either USE_ACCELERATE or USE_OPENBLAS."
#endif
#include <omp.h>
#include <stdexcept>
#include <vector>
#include <cstring>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <cmath>
#include <numeric>
#include <gsl/gsl_sf_laguerre.h>
#include <gsl/gsl_sf_gamma.h>  // For gsl_sf_fact

inline int packed_index(int i, int j) {
    if (i > j) std::swap(i, j);
    return j * (j + 1) / 2 + i;
}


std::vector<double> pack_symmetric(const py::array_t<double>& pdm) {
    auto r = pdm.unchecked<2>();
    const ssize_t dim = r.shape(0);
    std::vector<double> packed;
    packed.reserve(dim * (dim + 1) / 2);

    for (ssize_t i = 0; i < dim; ++i) {
        for (ssize_t j = 0; j <= i; ++j) {
            packed.push_back(r(i, j));
        }
    }
    return packed;
}



std::pair<py::array_t<double>, py::array_t<double>> displacement_matrix_cpp(
    const std::vector<int>& nboson_states,
    int imode,
    py::array_t<double, py::array::c_style | py::array::forcecast> freq,
    py::array_t<double, py::array::c_style | py::array::forcecast> eta,
    py::array_t<double, py::array::c_style | py::array::forcecast> pdm,
    py::array_t<double, py::array::c_style | py::array::forcecast> vsq,
    double shift
) {
    const double* freq_ptr = freq.data();
    const double* vsq_ptr = vsq.data();
    const double* eta_ptr = eta.data();
    const double* pdm_ptr = pdm.data();

    ssize_t nao = eta.shape(1);
    ssize_t mdim = nboson_states.at(imode);
    ssize_t packed_dim = mdim * (mdim + 1) / 2;

    py::array_t<double> disp_mat({packed_dim, nao, nao});
    py::array_t<double> exp_val({nao, nao});

    auto* disp_data = disp_mat.mutable_data();
    auto* exp_data = exp_val.mutable_data();

    const double tau = std::exp(vsq_ptr[imode]);
    const double tmp = tau / freq_ptr[imode];

    // Precompute factor matrix (nao x nao)
    std::vector<double> factor(nao * nao);
    for (ssize_t p = 0; p < nao; ++p) {
        for (ssize_t q = 0; q < nao; ++q) {
            double diff_eta = eta_ptr[imode * nao + p] - eta_ptr[imode * nao + q] + shift;
            factor[p * nao + q] = tmp * diff_eta;
        }
    }

    // Fill displacement matrix
    for (int m = 0; m < mdim; ++m) {
        for (int n = 0; n <= m; ++n) {
            int idx = packed_index(m, n);
            double* slice = disp_data + idx * nao * nao;

            for (ssize_t i = 0; i < nao * nao; ++i) {
                double f = factor[i];
                double val = 0.0;

                if (m == n) {
                    val = gsl_sf_laguerre_n(m, 0.0, f * f);
                } else {
                    double ratio = gsl_sf_fact(n) / gsl_sf_fact(m);
                    val = 2.0 * std::sqrt(ratio)
                        * std::pow(-f, m - n)
                        * gsl_sf_laguerre_n(n, m - n, f * f);
                }

                slice[i] = val;
            }
        }
    }

    // Multiply by exp(-0.5 * factor^2)
    for (ssize_t i = 0; i < nao * nao; ++i) {
        double g = std::exp(-0.5 * factor[i] * factor[i]);
        for (int idx = 0; idx < packed_dim; ++idx) {
            disp_data[idx * nao * nao + i] *= g;
        }
    }

    // Contract with symmetric PDM
    for (ssize_t pq = 0; pq < nao * nao; ++pq) {
        double acc = 0.0;
        for (int idx = 0; idx < packed_dim; ++idx) {
            acc += pdm_ptr[idx] * disp_data[idx * nao * nao + pq];
        }
        exp_data[pq] = acc;
    }

    return {disp_mat, exp_val};
}


//std::pair<py::array_t<double>, py::array_t<double>> displacement_matrix(
//    const std::vector<int>& nboson_states,
//    int mode,
//    py::array_t<double> factor,
//    py::array_t<double> pdm
//) {
//    const int mdim = nboson_states[mode];
//    const size_t packed_mdim = mdim * (mdim + 1) / 2;
//
//    auto factor_buf = factor.unchecked<1>();  // assuming 1D for simplicity
//    ssize_t fsize = factor_buf.shape(0);
//
//    py::array_t<double> disp_mat({packed_mdim, fsize});
//    auto disp = disp_mat.mutable_unchecked<2>();
//
//    for (int i_m = 0; i_m < mdim; ++i_m) {
//        for (int i_n = 0; i_n <= i_m; ++i_n) {
//            size_t idx = packed_index(i_m, i_n);
//
//            for (ssize_t k = 0; k < fsize; ++k) {
//                double A = factor_buf(k);
//                double val = 0.0;
//
//                if (i_m == i_n) {
//                    val = gsl_sf_laguerre_n(i_m, 0.0, A * A);
//                } else {
//                    double ratio = gsl_sf_fact(i_n) / gsl_sf_fact(i_m);
//                    val = 2.0 * std::sqrt(ratio)
//                        * std::pow(-A, i_m - i_n)
//                        * gsl_sf_laguerre_n(i_n, i_m - i_n, A * A);
//                }
//
//                disp(idx, k) = val;
//            }
//        }
//    }
//
//    // Multiply by exp(-0.5 * A^2)
//    for (size_t idx = 0; idx < packed_mdim; ++idx) {
//        for (ssize_t k = 0; k < fsize; ++k) {
//            double A = factor_buf(k);
//            disp(idx, k) *= std::exp(-0.5 * A * A);
//        }
//    }
//
//    // Contract with photon density matrix
//    auto packed_pdm = pack_symmetric(pdm);
//    py::array_t<double> exp_val({fsize});
//    auto exp = exp_val.mutable_unchecked<1>();
//
//    for (ssize_t k = 0; k < fsize; ++k) {
//        double sum = 0.0;
//        for (size_t idx = 0; idx < packed_mdim; ++idx) {
//            sum += packed_pdm[idx] * disp(idx, k);
//        }
//        exp(k) = sum;
//    }
//
//    return std::make_pair(disp_mat, exp_val);
//}

/*
py::array_t<double> displacement_val(
    int imode,
    py::array_t<int, py::array::c_style | py::array::forcecast> nboson_states,
    py::array_t<double, py::array::c_style | py::array::forcecast> eta,
    py::array_t<double, py::array::c_style | py::array::forcecast> brho,
    float shift,
) {
    ssize_t nmode = eta.shape(0);
    ssize_t nao = eta.shape(1);
    ssize_t mdim = nboson_states(imode);
    ssize_t packed_mdim = mdim * (mdim + 1) / 2; //

    // compute factor
    py::array_t<double> factor({nao, nao});
    py::array_t<double> disp_mat({packed_mdim, nao, nao});

    for (ssize_t p = 0; p < nao, ++im){
        for (ssize_t q = 0; q < nao; ++in){
            factor(p, q) = eta(imode, p) - eta(imode, q);
         }
    }
    factor += shift;

    for (ssize_t im = 0; im < nao, ++im){
        for (ssize_t in = im; in < nao; ++in){
            // TODO: implement factorial
            if (im == in){
                val = genlaguerre(n=i_m, alpha=0)(factor**2);
            } esle {
                ratio = 1.0; // factorial(i_n, exact=True) / factorial(i_m, exact=True);
                // Matrix elements
                val = 2.0 * numpy.sqrt(ratio) * (-factor)**(i_m - i_n) \
                      * genlaguerre(n=i_n, alpha=(i_m - i_n))(factor**2);
            }
            disp_mat[idx] = val;
        }
    }

}

*/
