#
# Generic Toolchain using MKL + TBB on cray machines 
#
# REQUIREMENTS:
# - load PrgEnv-gnu or PrgEnv-intel to place Cray compiler wrappers (cc, CC) in PATH
# - load modules:
#   - module add boost
#   - module add cmake
#   - export CRAYPE_LINK_TYPE=dynamic

# Set compilers (assumes the compilers are in the PATH)
set(CMAKE_C_COMPILER cc)
set(CMAKE_CXX_COMPILER CC)
# setting these breaks FindMPI in cmake 3.9 and later
#set(MPI_C_COMPILER cc)
#set(MPI_CXX_COMPILER CC)
set(MPI_CXX_SKIP_MPICXX ON)

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
