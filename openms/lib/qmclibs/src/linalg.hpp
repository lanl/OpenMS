#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <complex>

pybind11::array_t<std::complex<double>> tensordot_complex(
    pybind11::array_t<std::complex<double>, pybind11::array::c_style | pybind11::array::forcecast> A,
    pybind11::array_t<std::complex<double>, pybind11::array::c_style | pybind11::array::forcecast> B
);


