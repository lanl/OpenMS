#pragma once

#include "Tile.hpp"
#include <array>
#include <map>
#include <vector>
#include <algorithm>
#include <cassert>
#include <iostream>

namespace tensorlib {

template <int Rank>
std::vector<std::array<int, Rank>> generate_all_tile_indices(
    const std::array<int, Rank>& shape,
    const std::array<int, Rank>& tile_shape) {

    std::array<int, Rank> tile_counts;
    for (int i = 0; i < Rank; ++i) {
        tile_counts[i] = (shape[i] + tile_shape[i] - 1) / tile_shape[i];
    }

    std::vector<std::array<int, Rank>> indices;
    std::array<int, Rank> index = {};

    while (true) {
        indices.push_back(index);

        // Increment multi-dimensional index
        int d = Rank - 1;
        while (d >= 0) {
            index[d]++;
            if (index[d] < tile_counts[d]) break;
            index[d] = 0;
            --d;
        }

        if (d < 0) break;
    }

    return indices;
}


template <typename T, int Rank>
class TiledTensor {
public:
    using TileType = Tile<T, Rank>;
    using Index = std::array<int, Rank>;

    // Default constructor with automatic tile generation (empty data)
    TiledTensor(const std::array<int, Rank>& shape,
               const std::array<int, Rank>& tile_shape)
        : shape_(shape), tile_shape_(tile_shape) {
        initialize_tile_indices(nullptr);
    }

    /*
    // Constructor with custom tile initializer lambda
    TiledTensor(const std::array<int, Rank>& shape,
               const std::array<int, Rank>& tile_shape,
               std::function<void(const Index&, TileType&)> initializer)
        : shape_(shape), tile_shape_(tile_shape) {
        initialize_tile_indices(&initializer);
    }
    */

    TiledTensor(const std::array<int, Rank>& shape,
               const std::array<int, Rank>& tile_shape,
               std::function<void(const Index&, TileType&)> initializer = nullptr,
               std::function<bool(const Index&)> tile_filter = nullptr)
        : shape_(shape), tile_shape_(tile_shape) {
        initialize_tile_indices(&initializer, tile_filter);
    }


    void set_tile(const Index& idx, const TileType& tile) {
        tiles_[idx] = tile;
    }

    const TileType* get_tile(const Index& idx) const {
        auto it = tiles_.find(idx);
        return it != tiles_.end() ? &it->second : nullptr;
    }

    std::vector<Index> tile_indices() const {
        std::vector<Index> result;
        for (const auto& kv : tiles_) result.push_back(kv.first);
        return result;
    }

    const std::array<int, Rank>& shape() const { return shape_; }
    const std::array<int, Rank>& tile_shape() const { return tile_shape_; }

private:
    std::array<int, Rank> shape_;
    std::array<int, Rank> tile_shape_;
    std::map<Index, TileType> tiles_;

    void initialize_tile_indices(std::function<void(const Index&, TileType&)>* initializer,
        std::function<bool(const Index&)> tile_filter) {
        // auto indices = generate_all_tile_indices(shape_, tile_shape_);
        auto indices = generate_all_tile_indices<Rank>(shape_, tile_shape_);
        for (const auto& index : indices) {
            if (tile_filter && tile_filter(index)) {
                // Skip this tile
                continue;
            }
            // Compute shape of tile (adjust for edge tiles)
            std::array<int, Rank> actual_tile_shape = tile_shape_;
            for (int d = 0; d < Rank; ++d) {
                int offset = index[d] * tile_shape_[d];
                actual_tile_shape[d] = std::min(tile_shape_[d], shape_[d] - offset);
            }

            TileType tile(actual_tile_shape);
            if (initializer) {
                (*initializer)(index, tile);
            }

            tiles_[index] = tile;
        }
    }

};

template <typename T, size_t N>
std::ostream& operator<<(std::ostream& os, const std::array<T, N>& arr) {
    os << "{ ";
    for (size_t i = 0; i < N; ++i) {
        os << arr[i];
        if (i + 1 < N) os << ", ";
    }
    os << " }";
    return os;
}

} // namespace tensorlib
