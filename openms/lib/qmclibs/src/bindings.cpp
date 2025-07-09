
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>

// qmc propagator
#include "propagator.hpp"
#include "trial.hpp"
#include "estimators.hpp"
#include "linalg.hpp"
#include "qedhf.hpp"

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

    //m.def("propagate_exp_op_complex_batched",
    //      &propagate_exp_op_taylor_complex_batched,
    //      "Batched matrix exponential propagation using Taylor expansion");

    // functions for trial/walker
    m.def("trial_walker_ovlp_base", &trial_walker_ovlp_base, "Compute walker-trial overlap matrices");
    m.def("trial_walker_ovlp_gf_base", &trial_walker_ovlp_gf_base,
          "Compute trial-walker overlaps and Ghalf");

    // estimators
    // m.def("qmc_exx_chols", &exx_chols, "Calculate exx_chols", pybind11::arg("ltensor"), pybind11::arg("Gf"));
    m.def("exx_rltensor_Ghalf_complex", &exx_rltensor_Ghalf_complex, "Compute EXX energy from rltensor and Ghalf (complex)");
    m.def("exx_rltensor_Ghalf", &exx_rltensor_Ghalf_real, "Compute EXX energy from rltensor and Ghalf (real)");

    m.def("ecoul_rltensor_uhf_complex", &ecoul_rltensor_uhf_complex);

    m.def("ecoul_rltensor_Ghalf_complex", &ecoul_rltensor_Ghalf_complex,
          "Compute Coulomb energy from Ghalfa/b and rltensor(a/b)"); //,
          //py::arg("rltensora"),
          //py::arg("Ghalfa"),
          //py::arg("rltensorb") = py::none(),
          //py::arg("Ghalfb") = py::none());

    // linalg or tensor lib
    m.def("tensordot_complex", &tensordot_complex);

    // qedhf lib

    m.def("displacement_matrix", &displacement_matrix_cpp,
          py::arg("nboson_states"),
          py::arg("imode"),
          py::arg("freq"),
          py::arg("eta"),
          py::arg("pdm"),
          py::arg("vsq"),
          py::arg("shift") = 0.0,
          "Compute displacement matrix");


}
