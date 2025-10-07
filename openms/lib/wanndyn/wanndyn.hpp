// wanndyn.hpp
#ifndef WE2PD_WANNDYN_HPP
#define WE2PD_WANNDYN_HPP

#include <cstddef>
#include <stdexcept>
#include <vector>
#include "constants.hpp"  // uses parameters::real and parameters::dp

namespace WannDyn {

// Fortran: type, abstract :: densitymatrix
class DensityMatrix {
public:
    using value_type = parameters::real;

    // Keep abstract (pure virtual dtor)
    virtual ~DensityMatrix() = 0;

    std::size_t n_mo() const noexcept { return n_mo_; }

    // Column-major access (Fortran-like); 0-based indices in C++
    value_type& operator()(std::size_t i, std::size_t j) {
        return rho_[index(i, j)];
    }
    const value_type& operator()(std::size_t i, std::size_t j) const {
        return rho_[index(i, j)];
    }

    void zero() noexcept { std::fill(rho_.begin(), rho_.end(), value_type(0)); }

    value_type* data() noexcept { return rho_.data(); }
    const value_type* data() const noexcept { return rho_.data(); }
    std::size_t size() const noexcept { return rho_.size(); }

protected:
    explicit DensityMatrix(std::size_t n_mo) : n_mo_(n_mo), rho_(n_mo * n_mo) {}

    std::size_t index(std::size_t i, std::size_t j) const {
        if (i >= n_mo_ || j >= n_mo_) throw std::out_of_range("DensityMatrix index out of range");
        // Column-major: i + n_mo * j
        return i + n_mo_ * j;
    }

    std::size_t n_mo_{0};
    std::vector<value_type> rho_;
};

// Simple concrete dense implementation you can instantiate
class DenseDensityMatrix final : public DensityMatrix {
public:
    explicit DenseDensityMatrix(std::size_t n_mo) : DensityMatrix(n_mo) {}
    ~DenseDensityMatrix() override = default;
};

// Fortran: subroutine we2pd()
void we2pd();

} // namespace WannDyn

#endif // WE2PD_WANNDYN_HPP
