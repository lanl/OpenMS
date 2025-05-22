#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include "Tile.hpp"
#include "TiledTensor.hpp"

namespace py = pybind11;
using namespace tensorlib;

template <typename T, int Rank>
void bind_tile(py::module_ &m, const std::string &class_name) {
    using TileT = Tile<T, Rank>;
    using TensorType = typename TileT::TensorType;

    py::class_<TileT>(m, class_name.c_str())
        .def(py::init<>())
        .def(py::init<const std::array<int, Rank>&>())
        .def("shape", &TileT::shape, py::return_value_policy::reference_internal)
        .def("data", [](TileT &self) -> py::array {
            std::vector<ssize_t> shape(self.shape().begin(), self.shape().end());
            std::vector<ssize_t> strides(Rank);

            strides[Rank - 1] = sizeof(T);  // Last dimension
            for (int i = Rank - 2; i >= 0; --i) {
                strides[i] = strides[i + 1] * shape[i + 1];
            }

            return py::array(
                py::buffer_info(
                    self.data().data(),       // Pointer to buffer
                    sizeof(T),                // Size of one scalar
                    py::format_descriptor<T>::format(), // Format (e.g. 'f' for float)
                    Rank,                     // Number of dimensions
                    shape,                    // Shape of tensor
                    strides                   // Strides (in bytes)
                )
            );
        })

        /*
        .def("data", [](TileT &self) -> py::array {
            Eigen::DSizes<ptrdiff_t, Rank> dims;
            for (int i = 0; i < Rank; ++i)
                dims[i] = self.shape()[i];
            return py::array_t<T>(
                py::buffer_info(
                    self.data().data(),
                    sizeof(T),
                    py::format_descriptor<T>::format(),
                    Rank,
                    std::vector<ssize_t>(self.shape().begin(), self.shape().end()),
                    std::vector<ssize_t>(Rank, sizeof(T))  // Simplified strides
                )
            );
        })
        */
        .def("transpose", &TileT::transpose)
        .def("fill", &TileT::fill)
        .def("__add__", &TileT::operator+)
        .def("__mul__", &TileT::operator*);
}

// for tiledtensor
template <typename T, int Rank>
void bind_tiled_tensor(py::module_ &m, const std::string &class_name) {
    using TiledTensorT = TiledTensor<T, Rank>;
    using Index = typename TiledTensorT::Index;
    using TileT = Tile<T, Rank>;

    py::class_<TiledTensorT>(m, class_name.c_str())
        .def(py::init<const std::array<int, Rank>&,
                      const std::array<int, Rank>&,
                      std::function<void(const Index&, TileT&)>,
                      std::function<bool(const Index&)>>(),
             py::arg("shape"),
             py::arg("tile_shape"),
             py::arg("initializer") = nullptr,
             py::arg("tile_filter") = nullptr)

        .def("set_tile", &TiledTensorT::set_tile)
        .def("get_tile", [](const TiledTensorT &self, const Index &idx) -> py::object {
            auto *tile = self.get_tile(idx);
            if (tile)
                return py::cast(*tile);
            return py::none();
        })

        .def("tile_indices", &TiledTensorT::tile_indices)
        .def("shape", &TiledTensorT::shape, py::return_value_policy::reference_internal)
        .def("tile_shape", &TiledTensorT::tile_shape, py::return_value_policy::reference_internal);
}


template <int Rank>
void bind_utilities(py::module_ &m) {
    m.def("generate_all_tile_indices", &generate_all_tile_indices<Rank>,
          py::arg("shape"), py::arg("tile_shape"),
          ("Generate all tile indices for rank-" + std::to_string(Rank)).c_str());
}

PYBIND11_MODULE(pytensorlib, m) {
    // float
    bind_tile<float, 2>(m, "Tile2f");
    bind_tile<float, 3>(m, "Tile3f");
    bind_tile<float, 4>(m, "Tile4f");

    // double
    bind_tile<double, 2>(m, "Tile2d");
    bind_tile<double, 3>(m, "Tile3d");
    bind_tile<double, 4>(m, "Tile4d");

    // tensor float
    bind_tiled_tensor<float, 2>(m, "TiledTensor2f");
    bind_tiled_tensor<float, 3>(m, "TiledTensor3f");

    // tensor (double)
    bind_tiled_tensor<double, 2>(m, "TiledTensor2d");
    bind_tiled_tensor<double, 3>(m, "TiledTensor3d");

    bind_utilities<2>(m);
}
