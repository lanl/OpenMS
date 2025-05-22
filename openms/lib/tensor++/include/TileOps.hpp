#pragma once

#include "Tile.hpp"
#include <memory>
#include <vector>

namespace tensorlib {

template <typename T, int Rank>
class TileOps {
public:
    virtual ~TileOps() = default;

    virtual Tile<T, Rank> contract(const Tile<T, Rank>& A,
                                   const Tile<T, Rank>& B,
                                   const std::vector<int>& A_contract_dims,
                                   const std::vector<int>& B_contract_dims) = 0;

    virtual Tile<T, Rank> transpose(const Tile<T, Rank>& A,
                                    const std::array<int, Rank>& perm) = 0;

    virtual Tile<T, Rank> add(const Tile<T, Rank>& A,
                              const Tile<T, Rank>& B) = 0;

    virtual Tile<T, Rank> scale(const Tile<T, Rank>& A,
                                T scalar) = 0;
};

template <typename T, int Rank>
std::shared_ptr<TileOps<T, Rank>> make_default_tile_ops();

} // namespace tensorlib
