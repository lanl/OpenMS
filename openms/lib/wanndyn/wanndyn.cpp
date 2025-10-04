// WannDyn_class.cpp
#include "wanndyn.hpp"
#include "we2pd_banner.hpp"

namespace WannDyn {

DensityMatrix::~DensityMatrix() = default;

void we2pd() {
    we2pd::banner::Scope banner_guard; // prints header now, ending at scope exit
    // ... computation ...
}

} // namespace WannDyn
