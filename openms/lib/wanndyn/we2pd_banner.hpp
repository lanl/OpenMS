// we2pd_banner.hpp  (header-only)
#ifndef WE2PD_BANNER_HPP
#define WE2PD_BANNER_HPP

#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <string>

namespace we2pd { namespace banner {

inline std::tm localtime_portable(std::time_t t) {
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    return tm;
}

inline void printHeader(std::ostream& out = std::cout) {
    // First line is 80 spaces, matching the Fortran banner
    out << std::string(80, ' ') << "\n"
        << "********************************************************************************\n"
        << "**                                                                            **\n"
        << "**                                   WE^2PD                                   **\n"
        << "**                                                                            **\n"
        << "**    Maximally localized Wannier function based Density Matrix Dynamics in   **\n"
        << "**    in the presence of electron-electron and electron-phonon scatterings.   **\n"
        << "**                                                                            **\n"
        << "**                         Copyright (c)  Yu Zhang, PhD,                      **\n"
        << "**                             Email: zhy@lanl.gov                            **\n"
        << "**                                                                            **\n"
        << "**   Theoretical Division, Los Alamos National Laboratory, Los Alamos         **\n"
        << "**                                                                            **\n"
        << "**   This propgram is used to calculate the hot-carrier generation from       **\n"
        << "**   generation plasmon decay and its injection to other materials via        **\n"
        << "**   the interface. The electron transport is calculated within the NEGF      **\n"
        << "**   formalism. The electronic structure employs tight-binding, DFTB or       **\n"
        << "**   ab-initio tight-binding (localized wannier function as basis).           **\n"
        << "**                                                                            **\n"
        << "**                                                                            **\n"
        << "********************************************************************************\n";

    auto now  = std::chrono::system_clock::now();
    std::time_t tt = std::chrono::system_clock::to_time_t(now);
    std::tm tm = localtime_portable(tt);

    out << " Calculation Started on  " << std::put_time(&tm, "%d/%m/%Y  %H:%M:%S") << "\n\n";
}

inline void printEnding(std::ostream& out = std::cout) {
    auto now  = std::chrono::system_clock::now();
    std::time_t tt = std::chrono::system_clock::to_time_t(now);
    std::tm tm = localtime_portable(tt);

    // Leading newline to match the Fortran format record with a slash (/)
    out << "\n Calculation Finished on  " << std::put_time(&tm, "%d/%m/%Y  %H:%M:%S") << "\n"
        << "********************************************************************************\n"
        << "**                                                                            **\n"
        << "**                             Program Ended                                  **\n"
        << "**                                                                            **\n"
        << "********************************************************************************\n";
}

// RAII guard to always print header/ending in a scope
class Scope {
public:
    explicit Scope(std::ostream& os = std::cout) : out_(&os) { printHeader(*out_); }
    ~Scope() noexcept {
        try { printEnding(*out_); } catch (...) { /* swallow to keep noexcept */ }
    }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
private:
    std::ostream* out_;
};

}} // namespace we2pd::banner

#endif // WE2PD_BANNER_HPP
