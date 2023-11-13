// bindings.cpp
#include <pybind11/pybind11.h>
#include "myclass.h"
#include "_my_module.h"

namespace py = pybind11;

PYBIND11_MODULE(_my_module, m) {
    py::class_<MyClass>(m, "MyClass")
        .def(py::init<const std::string &, int>())
        .def("setName", &MyClass::setName)
        .def("getName", &MyClass::getName)
        .def("setValue", &MyClass::setValue)
        .def("getValue", &MyClass::getValue);
    m.def("cpp_method_impl", &cpp_method_impl);
}

