#pragma once

#include <pybind11/numpy.h>


// overlap

#pragma once
#include <pybind11/numpy.h>
#include <complex>

pybind11::array_t<std::complex<double>> trial_walker_ovlp_base(
    pybind11::array_t<std::complex<double>, pybind11::array::c_style | pybind11::array::forcecast> phiw,
    pybind11::array_t<std::complex<double>, pybind11::array::c_style | pybind11::array::forcecast> psi);

#pragma once
#include <pybind11/numpy.h>
#include <complex>
#include <tuple>

std::tuple<
    pybind11::array_t<std::complex<double>>,
    pybind11::array_t<std::complex<double>>
>
trial_walker_ovlp_gf_base(
    pybind11::array_t<std::complex<double>, pybind11::array::c_style | pybind11::array::forcecast> phiw,
    pybind11::array_t<std::complex<double>, pybind11::array::c_style | pybind11::array::forcecast> psi);

