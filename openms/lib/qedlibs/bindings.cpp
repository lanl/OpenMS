
#include "qed.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>   // convert eigen3 to numpy array
#include "vt_qedhf.hpp"
#include "eigen_example.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_qedhf, m) {
    m.def("cpp_method_impl", &cpp_method_impl);
    m.def("update_fock", &update_fock_energy_gradient_vt_qedhf);
    m.def("gaussian_factor_vt_qedhf", &gaussian_factor_vt_qedhf);
    m.def("test_gaussian", &test_gaussian);
    m.def("eigen3_exampleFunction", &eigen3_exampleFunction);
}

