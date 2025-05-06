#include "trial.hpp"

#if defined(USE_ACCELERATE)
    #include <Accelerate/Accelerate.h>
#elif defined(USE_OPENBLAS)
    #include <cblas.h>
#else
    #error "No BLAS backend defined. Define either USE_ACCELERATE or USE_OPENBLAS."
#endif
#include <omp.h>
#include <complex>
#include <stdexcept>
#include <vector>
#include <cstring>
#include <pybind11/eigen.h>
#include <Eigen/Dense>


namespace py = pybind11;

py::array_t<std::complex<double>> trial_walker_ovlp_base(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> phiw,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> psi
) {
    if (phiw.ndim() != 3 || psi.ndim() != 2)
        throw std::runtime_error("phiw must be 3D and psi must be 2D");

    ssize_t nw = phiw.shape(0);
    ssize_t n = phiw.shape(1);
    ssize_t no = phiw.shape(2);

    if (psi.shape(0) != n || psi.shape(1) != no)
        throw std::runtime_error("psi must be of shape (n, no)");

    py::array_t<std::complex<double>> ovlp({nw, no, no});

    // Manually conjugate psi once
    auto* psi_data = psi.data();
    std::vector<std::complex<double>> psi_conj(n * no);
    for (ssize_t i = 0; i < n * no; ++i)
        psi_conj[i] = std::conj(psi.data()[i]);

    const std::complex<double> one(1.0, 0.0);
    const std::complex<double> zero(0.0, 0.0);

    #pragma omp parallel for
    for (ssize_t z = 0; z < nw; ++z) {
        const std::complex<double>* phi_z = phiw.data(z, 0, 0);
        std::complex<double>* ovlp_z = ovlp.mutable_data(z, 0, 0);

        cblas_zgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    no, no, n,
                    reinterpret_cast<const void*>(&one),
                    reinterpret_cast<const void*>(phi_z), no,
                    reinterpret_cast<const void*>(psi_conj.data()), no,
                    reinterpret_cast<const void*>(&zero),
                    reinterpret_cast<void*>(ovlp_z), no);
    }

    return ovlp;
}


std::tuple<py::array_t<std::complex<double>>, py::array_t<std::complex<double>>>
trial_walker_ovlp_gf_base(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> phiw,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> psi
) {
    if (phiw.ndim() != 3 || psi.ndim() != 2)
        throw std::runtime_error("phiw must be 3D and psi must be 2D");

    ssize_t nw = phiw.shape(0);
    ssize_t p  = phiw.shape(1);
    ssize_t i  = phiw.shape(2);
    ssize_t j  = psi.shape(1);

    if (psi.shape(0) != p)
        throw std::runtime_error("psi shape mismatch with phiw");

    // Allocate output
    py::array_t<std::complex<double>> ovlp({nw, i, j});
    py::array_t<std::complex<double>> Ghalf({nw, p, j});

    // Conjugate psi once
    std::vector<std::complex<double>> psi_conj(p * j);
    for (ssize_t idx = 0; idx < p * j; ++idx)
        psi_conj[idx] = std::conj(psi.data()[idx]);

    #pragma omp parallel for
    for (ssize_t z = 0; z < nw; ++z) {
        const std::complex<double>* phi_z = phiw.data(z, 0, 0);
        std::complex<double>* ovlp_z = ovlp.mutable_data(z, 0, 0);
        std::complex<double>* ghalf_z = Ghalf.mutable_data(z, 0, 0);

        const std::complex<double> one(1.0, 0.0);
        const std::complex<double> zero(0.0, 0.0);

        // Step 1: ovlp[z] = phiw[z].T @ psi_conj
        cblas_zgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    i, j, p,
                    &one,
                    phi_z, i,
                    psi_conj.data(), j,
                    &zero,
                    ovlp_z, j);

        // Step 2: inv(ovlp[z])
        Eigen::Map<const Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> ovlp_matrix(ovlp_z, i, j);
        Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> inv_ovlp = ovlp_matrix.inverse();

        // Step 3: Ghalf[z] = phiw[z] @ inv(ovlp[z]).T
        Eigen::Map<const Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> phi_mat(phi_z, p, i);
        Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> ghalf_mat(ghalf_z, p, j);
        ghalf_mat.noalias() = phi_mat * inv_ovlp.transpose();
    }

    return std::make_tuple(std::move(ovlp), std::move(Ghalf));
}
