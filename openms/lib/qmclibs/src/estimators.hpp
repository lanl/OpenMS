#pragma once

#include <pybind11/numpy.h>
#include <complex>

// L and G are both complex
pybind11::array_t<std::complex<double>> exx_rltensor_Ghalf_complex(
    pybind11::array_t<std::complex<double>, pybind11::array::c_style | pybind11::array::forcecast> rltensor,
    pybind11::array_t<std::complex<double>, pybind11::array::c_style | pybind11::array::forcecast> Ghalf);


// L and G are both real
pybind11::array_t<double> exx_rltensor_Ghalf_real(
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> rltensor,
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> Ghalf);

