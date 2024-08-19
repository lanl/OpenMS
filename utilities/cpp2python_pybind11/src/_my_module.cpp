// _my_module.cpp

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <_my_module.h>

namespace py = pybind11;

/*
std::vector<std::vector<double>> cpp_method_impl(const std::vector<std::vector<double>>& g) {
    std::vector<std::vector<double>> result = g;
    for (size_t p = 0; p < result.size(); ++p) {
        for (size_t q = 0; q < result[p].size(); ++q) {
            result[p][q] = 2 * g[p][q];
        }
    }
    return result;
}
*/

std::vector<std::vector<double>> cpp_method_impl(py::object py_obj) {
    std::vector<std::vector<double>> g = py_obj.attr("g").cast<std::vector<std::vector<double>>>();
    std::vector<std::vector<double>> result = g;

    for (size_t p = 0; p < result.size(); ++p) {
        for (size_t q = 0; q < result[p].size(); ++q) {
            result[p][q] = 2 * g[p][q];
        }
    }
    return result;
}

//PYBIND11_MODULE(_my_module, m) {
//    m.def("cpp_method_impl", &cpp_method_impl);
//}


