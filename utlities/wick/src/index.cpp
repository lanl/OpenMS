
#include "../src/index.hpp"

namespace qedsqa {


Idx::Idx(int idx, const std::string& sp, bool ferm)
    : index(idx), space(sp), fermion(ferm) {}

std::ostream& operator<<(std::ostream& os, const Idx& idx) {
    return os << idx.index << "(" << idx.space << ")";
}

bool Idx::operator==(const Idx& other) const {
    return this->index == other.index && this->space == other.space;
}

bool Idx::operator!=(const Idx& other) const {
    return !(*this == other);
}

bool Idx::operator<(const Idx& other) const {
    if (this->space < other.space) {
        return true;
    } else if (this->space == other.space) {
        return this->index < other.index;
    }
    return false;
}

bool Idx::operator<=(const Idx& other) const {
    return *this < other || *this == other;
}

bool Idx::operator>(const Idx& other) const {
    return !(*this <= other);
}

bool Idx::operator>=(const Idx& other) const {
    return !(*this < other);
}

Idx idx_copy(const Idx& idx) {
    return Idx(idx.index, idx.space, idx.fermion);
}

bool is_occupied(const Idx& idx, const std::string& occ) {
    if (occ.empty()) {
        return idx.space.find('o') != std::string::npos;
    } else {
        return idx.space == occ;
    }
}




}  // namespace qedsqa
