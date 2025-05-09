#pragma once

#include <pybind11/numpy.h>
#include <complex>

// L and G are both complex
pybind11::array_t<std::complex<double>> exx_rltensor_Ghalf_complex(
    pybind11::array_t<std::complex<double>, pybind11::array::c_style | pybind11::array::forcecast> rltensor,
    pybind11::array_t<std::complex<double>, pybind11::array::c_style | pybind11::array::forcecast> Ghalf);

pybind11::array_t<std::complex<double>> ecoul_rltensor_uhf_complex(
    pybind11::array_t<std::complex<double>, pybind11::array::c_style | pybind11::array::forcecast> rltensora,
    pybind11::array_t<std::complex<double>, pybind11::array::c_style | pybind11::array::forcecast> Ghalfa,
    pybind11::array_t<std::complex<double>, pybind11::array::c_style | pybind11::array::forcecast> rltensorb,
    pybind11::array_t<std::complex<double>, pybind11::array::c_style | pybind11::array::forcecast> Ghalfb
);

pybind11::array_t<std::complex<double>> ecoul_rltensor_Ghalf_complex(
    pybind11::array_t<std::complex<double>, pybind11::array::c_style | pybind11::array::forcecast> rltensora,
    pybind11::array_t<std::complex<double>, pybind11::array::c_style | pybind11::array::forcecast> Ghalfa //,
    //std::optional<pybind11::array_t<std::complex<double>, pybind11::array::c_style | pybind11::array::forcecast>> rltensorb = std::nullopt,
    //std::optional<pybind11::array_t<std::complex<double>, pybind11::array::c_style | pybind11::array::forcecast>> Ghalfb = std::nullopt
);

// L and G are both real
pybind11::array_t<double> exx_rltensor_Ghalf_real(
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> rltensor,
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> Ghalf);

