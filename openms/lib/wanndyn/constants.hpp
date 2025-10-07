#ifndef WE2PD_CONSTATS_HPP
#define WE2PD_CONSTATS_HPP

#include <cstddef>
#include <limits>

namespace parameters {

// Meaning: 8-byte real (IEEE double)
constexpr int dp = 8;

// Use this as the canonical floating type (double precision)
using real = double;

// Sanity checks (similar to selected_real_kind(15, 307))
static_assert(sizeof(real) == dp, "Expected 8-byte double on this platform.");
static_assert(std::numeric_limits<real>::digits10 >= 15,
              "double lacks ~15 decimal digits of precision.");
static_assert(std::numeric_limits<real>::max_exponent10 >= 307,
              "double lacks required exponent range.");

} // namespace parameters

#endif // WE2PD_CONSTATS_HPP
