#pragma once

#include <array>
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

namespace tensorlib {

template <typename T, int Rank>
class Tile {
public:
    using TensorType = Eigen::Tensor<T, Rank, Eigen::RowMajor>;

    Tile() = default;

    Tile(const std::array<int, Rank>& shape)
        : shape_(shape), tensor_(make_tensor(shape)) {}

    Tile(const TensorType& tensor, const std::array<int, Rank>& shape)
        : shape_(shape), tensor_(tensor) {}

    const std::array<int, Rank>& shape() const { return shape_; }
    const TensorType& data() const { return tensor_; }
    TensorType& data() { return tensor_; }

    Tile<T, Rank> operator+(const Tile<T, Rank>& other) const {
        assert(shape_ == other.shape_ && "Shape mismatch in Tile addition");
        return Tile<T, Rank>(tensor_ + other.tensor_, shape_);
    }

    Tile<T, Rank> operator*(const T& scalar) const {
        return Tile<T, Rank>(tensor_ * scalar, shape_);
    }

    Tile<T, Rank> transpose(const std::array<int, Rank>& perm) const {
        auto shuffled = tensor_.shuffle(perm);
        std::array<int, Rank> new_shape;
        for (int i = 0; i < Rank; ++i)
            new_shape[i] = shape_[perm[i]];

        return Tile<T, Rank>(shuffled, new_shape);
    }

    void fill(const T& value) {
        tensor_.setConstant(value);
    }

private:
    std::array<int, Rank> shape_;
    TensorType tensor_;

    static TensorType make_tensor(const std::array<int, Rank>& shape) {
        Eigen::DSizes<Eigen::Index, Rank> dims;
        for (int i = 0; i < Rank; ++i)
            dims[i] = shape[i];
        TensorType tensor(dims);
        tensor.setZero();
        return tensor;
    }
};

} // namespace tensorlib
