#pragma once

#include <pybind11/numpy.h>
#include <complex>

// Function to propagate phi using op via BLAS
// phi is modified in-place
void propagate_onebody_blas(pybind11::array_t<double,
    pybind11::array::c_style | pybind11::array::forcecast> op,
    pybind11::array_t<double,
    pybind11::array::c_style | pybind11::array::forcecast> phi);


void propagate_onebody_blas_complex(
    pybind11::array_t<std::complex<double>, pybind11::array::c_style | pybind11::array::forcecast> op,
    pybind11::array_t<std::complex<double>, pybind11::array::c_style | pybind11::array::forcecast> phi
);


pybind11::array_t<std::complex<double>> propagate_onebody_blas_complex_return(
    pybind11::array_t<std::complex<double>, pybind11::array::c_style | pybind11::array::forcecast> op,
    pybind11::array_t<std::complex<double>, pybind11::array::c_style | pybind11::array::forcecast> phi
);


// return a new phi, phi_out = f(op, phi_in)
pybind11::array_t<double> propagate_onebody_blas_return(
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> op,
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> phi
);


// for two-body term
void propagate_exp_op_taylor(
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> phiw,
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> op,
    int order);


/*
pybind11::array_t<double> propagate_exp_op_taylor_return(
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> phiw,
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> op,
    int order);
*/


void propagate_exp_op_taylor_complex(
    pybind11::array_t<std::complex<double>, pybind11::array::c_style | pybind11::array::forcecast> phiw,
    pybind11::array_t<std::complex<double>, pybind11::array::c_style | pybind11::array::forcecast> op,
    int order);


void propagate_exp_op_taylor_complex_batched(
    pybind11::array_t<std::complex<double>, pybind11::array::c_style | pybind11::array::forcecast> phiw,
    pybind11::array_t<std::complex<double>, pybind11::array::c_style | pybind11::array::forcecast> op,
    int order);
