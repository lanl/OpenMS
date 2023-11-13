#ifndef EXAMPLES_MODULE_H
#define EXAMPLES_MODULE_H

#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

std::vector<std::vector<double>> cpp_method_impl(py::object py_obj);

#endif // EXAMPLES_MODULE_H

