#include "TileOps.hpp"
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

namespace tensorlib {

// Tensor contraction for each tile

template <typename T, int Rank>
class EigenTileOps : public TileOps<T, Rank> {
public:

    // contract function
    Tile<T, Rank> contract(const Tile<T, Rank>& A,
                           const Tile<T, Rank>& B,
                           const std::vector<int>& A_contract_dims,
                           const std::vector<int>& B_contract_dims) override {

        /*
        using IndexPair = Eigen::IndexPair<int>;
        const size_t num_pairs = A_contract_dims.size();

        // Dynamically build contraction pairs
        std::vector<IndexPair> pairs;
        for (size_t i = 0; i < num_pairs; ++i)
            pairs.emplace_back(A_contract_dims[i], B_contract_dims[i]);

        auto expr = A.data().contract(B.data(), pairs);
        auto result = expr.eval();  // evaluates the contraction expression
        std::array<int, Rank> out_shape;
        for (int i = 0; i < Rank; ++i)
            out_shape[i] = result.dimension(i);
        return Tile<T, Rank>(result, out_shape);
        */
    }

    // transpose
    Tile<T, Rank> transpose(const Tile<T, Rank>& A,
                            const std::array<int, Rank>& perm) override {
        return A.transpose(perm);
    }

    Tile<T, Rank> add(const Tile<T, Rank>& A,
                      const Tile<T, Rank>& B) override {
        return A + B;
    }

    Tile<T, Rank> scale(const Tile<T, Rank>& A,
                        T scalar) override {
        return A * scalar;
    }
};

template <typename T, int Rank>
std::shared_ptr<TileOps<T, Rank>> make_default_tile_ops() {
    return std::make_shared<EigenTileOps<T, Rank>>();
}

// Explicit instantiation for 2D and 3D tiles
template std::shared_ptr<TileOps<float, 2>> make_default_tile_ops<float, 2>();
template std::shared_ptr<TileOps<float, 3>> make_default_tile_ops<float, 3>();

} // namespace tensorlib
