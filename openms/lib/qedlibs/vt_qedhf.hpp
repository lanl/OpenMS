/*

 @ 2023. Triad National Security, LLC. All rights reserved.

 This program was produced under U.S. Government contract 89233218CNA000001
 for Los Alamos National Laboratory (LANL), which is operated by Triad
 National Security, LLC for the U.S. Department of Energy/National Nuclear
 Security Administration. All rights in the program are reserved by Triad
 National Security, LLC, and the U.S. Department of Energy/National Nuclear
 Security Administration. The Government is granted for itself and others acting
 on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this
 material to reproduce, prepare derivative works, distribute copies to the
 public, perform publicly and display publicly, and to permit others to do so.

 Author: Yu Zhang <zhy@lanl.gov>
 */

// #include "cint.h"

#define DM_PLAIN        0
#define DM_HERMITIAN    1
#define DM_ANTI         2

// #include "optimizer.h"

// cpp kernel for Variational QEDHF one- and two-body integrals (TODO)


#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <iostream>

namespace py = pybind11;

//void cpp_method_impl(py::object self);
std::vector<std::vector<double>> double_gmat(py::object py_obj);
std::vector<std::vector<double>> cpp_method_impl(py::object py_obj, const std::vector<std::vector<double>>& g);


