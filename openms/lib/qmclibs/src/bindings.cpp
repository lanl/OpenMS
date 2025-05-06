
#include <pybind11/pybind11.h>

// qmc propagator
#include "propagator.hpp"
#include "trial.hpp"
#include "estimators.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_qmclib, m) {
    m.def("propagate_onebody", &propagate_onebody_blas, "BLAS-based propagation using op @ phi[iw]");
    // m.def("propagate_onebody_complex", &propagate_onebody_blas_complex, "BLAS-based propagation using op @ phi[iw]");
    m.def("propagate_onebody_complex", &propagate_onebody_blas_complex_return);
    m.def("propagate_onebody_return", &propagate_onebody_blas_return, "Return-based OpenMP + BLAS propagation");
    //m.def("propagate_exp_op_return", &propagate_exp_op_taylor_return, "Apply exp(A) via Taylor expansion");

    // real
    m.def("propagate_exp_op", &propagate_exp_op_taylor, "Apply exp(op) to phiw in-place using Taylor expansion");

    // complex
    m.def("propagate_exp_op_complex", &propagate_exp_op_taylor_complex, "In-place complex Taylor exponential propagation");

    // functions for trial/walker
    m.def("trial_walker_ovlp_base", &trial_walker_ovlp_base, "Compute walker-trial overlap matrices");
    m.def("trial_walker_ovlp_gf_base", &trial_walker_ovlp_gf_base,
          "Compute trial-walker overlaps and Ghalf");

    // estimators
    // m.def("qmc_exx_chols", &exx_chols, "Calculate exx_chols", pybind11::arg("ltensor"), pybind11::arg("Gf"));
    m.def("exx_rltensor_Ghalf_complex", &exx_rltensor_Ghalf_complex, "Compute EXX energy from rltensor and Ghalf (complex)");
    m.def("exx_rltensor_Ghalf", &exx_rltensor_Ghalf_real, "Compute EXX energy from rltensor and Ghalf (real)");
}


