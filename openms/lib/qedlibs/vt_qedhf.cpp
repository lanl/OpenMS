#include "vt_qedhf.hpp"

namespace py = pybind11;

/*
void cpp_method_impl(py::object self) {
    // Your C++ implementation here
    // You can access other methods of the Python object using self.attr(...)
    std::cout << "This test_cpp_impl method is implemented in C++" << std::endl;

}

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

*/

std::vector<std::vector<double>> cpp_method_impl(py::object py_obj, const std::vector<std::vector<double>>& g) {
    std::vector<std::vector<double>> result = g;

    for (size_t p = 0; p < result.size(); ++p) {
        for (size_t q = 0; q < result[p].size(); ++q) {
            result[p][q] = 2 * g[p][q];
        }
    }
    return result;
}


