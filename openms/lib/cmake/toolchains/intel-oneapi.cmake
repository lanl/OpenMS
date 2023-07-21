#
# Generic Toolchain for Intel OneAPI (LLVM-based toolchain)
#
# REQUIREMENTS:
# - in PATH:
#   * icx
#   * icpx
#   * ifx (optional)
#   * mpiicc (optional; specify MPI_C_COMPILER explicitly to use non-Intel MPI)
#   * mpiicpc (optional; specify MPI_CXX_COMPILER explicitly to use non-Intel MPI)
# - environment variables:
#   * EIGEN3_DIR or (deprecated) EIGEN_DIR: the Eigen3 directory
#   * BOOST_DIR: the Boost root directory
#

# Intel OneAPI setvars.sh does not define INTEL_DIR, assign manually
set(INTEL_DIR "/opt/intel/oneapi" CACHE PATH "Intel tools root directory")

####### Compilers
include(${CMAKE_CURRENT_LIST_DIR}/_icx.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/_impi.cmake)

####### Compile flags
include(${CMAKE_CURRENT_LIST_DIR}/_std_c_flags.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/_std_cxx_flags.cmake)

####### Boost
include(${CMAKE_CURRENT_LIST_DIR}/_boost.cmake)

####### BLAS/LAPACK Libraries
set(INTEGER4 TRUE CACHE BOOL "Set Fortran integer size to 4 bytes")
set(BLA_STATIC OFF CACHE BOOL "Whether to use static linkage for BLAS, LAPACK, and related libraries")
include(${CMAKE_CURRENT_LIST_DIR}/_tbb.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/_mkl.cmake)

####### Eigen
include(${CMAKE_CURRENT_LIST_DIR}/_eigen.cmake)
