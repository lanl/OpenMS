
#ifndef QEDSQA_INDEX_H
#define QEDSQA_INDEX_H

#include <functional>
#include <string>
#include <iostream>

namespace qedsqa {

class Index;
using WstrList = std::initializer_list<std::wstring_view>;
using IndexList = std::initializer_list<Index>;


class Idx {
public:
    // Member attributes
    int index;
    std::string space;
    bool fermion;

    // Constructor
    Idx(int idx, const std::string& sp, bool ferm = true);

    // Overloading the << operator for easy printing
    friend std::ostream& operator<<(std::ostream& os, const Idx& idx);

    // Overloading comparison operators
    bool operator==(const Idx& other) const;
    bool operator!=(const Idx& other) const;
    bool operator<(const Idx& other) const;
    bool operator<=(const Idx& other) const;
    bool operator>(const Idx& other) const;
    bool operator>=(const Idx& other) const;
};

// Function declarations
Idx idx_copy(const Idx& idx);
bool is_occupied(const Idx& idx, const std::string& occ = "");


}  // namespace qedsqa

namespace std {
    template<>
    struct hash<qedsqa::Idx> {
        size_t operator()(const qedsqa::Idx& idx) const {
            // Hash combine method - combine hashes of individual attributes.
            size_t h1 = hash<int>()(idx.index);
            size_t h2 = hash<std::string>()(idx.space);
            return h1 ^ (h2 << 1);  // Or another combining method
        }
    };
} // std namespace




#endif  // QEDSQA_INDEX_H
