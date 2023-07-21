# Set compile flags
set(CMAKE_C_FLAGS_INIT             "-std=c99  -m64 -I/usr/include" CACHE STRING "Inital C compile flags")
set(CMAKE_C_FLAGS_DEBUG            "-g -Wall -Wno-sign-compare" CACHE STRING "Inital C debug compile flags")
set(CMAKE_C_FLAGS_MINSIZEREL       "-Os -march=native -DNDEBUG" CACHE STRING "Inital C minimum size release compile flags")
set(CMAKE_C_FLAGS_RELEASE          "-O3 -march=native -DNDEBUG" CACHE STRING "Inital C release compile flags")
set(CMAKE_C_FLAGS_RELWITHDEBINFO   "-O2 -g -Wall -Wno-sign-compare" CACHE STRING "Inital C release with debug info compile flags")
set(CMAKE_CXX_FLAGS_INIT           "" CACHE STRING "Inital C++ compile flags")
set(CMAKE_CXX_FLAGS_DEBUG          "-g -Wall -Wno-sign-compare" CACHE STRING "Inital C++ debug compile flags")
set(CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -march=native -DNDEBUG" CACHE STRING "Inital C++ minimum size release compile flags")
# clang issue with mismatched alloc/free in Eigen goes away if NDEBUG is not defined ... just a workaround
set(CMAKE_CXX_FLAGS_RELEASE        "-O3 -march=native" CACHE STRING "Inital C++ release compile flags")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -Wall -Wno-sign-compare" CACHE STRING "Inital C++ release with debug info compile flags")

# Libraries

set(BLAS_LINKER_FLAGS "-L/usr/lib/libblas" "-lblas" "-L/usr/lib/lapack" "-llapack" "-L/usr/lib" "-llapacke" CACHE STRING "BLAS linker flags")
set(BLAS_LIBRARIES ${BLAS_LINKER_FLAGS} CACHE STRING "BLAS libraries")
set(LAPACK_LIBRARIES ${BLAS_LINKER_FLAGS} CACHE STRING "LAPACK libraries")
set(LAPACK_INCLUDE_DIRS "/usr/include" CACHE STRING "LAPACK include directories")
set(LAPACK_COMPILE_DEFINITIONS MADNESS_LINALG_USE_LAPACKE TILEDARRAY_EIGEN_USE_LAPACKE CACHE STRING "LAPACK preprocessor definitions")
set(INTEGER4 TRUE CACHE BOOL "Set Fortran integer size to 4 bytes")
set(BLA_STATIC OFF CACHE BOOL "Whether to use static linkage for BLAS, LAPACK, and related libraries")

# for wavefunction91's FindLAPACK
set( lapack_LIBRARIES ${BLAS_LINKER_FLAGS} )

# BLACS
set( blacs_LIBRARIES      "-L$ENV{INSTALL_PREFIX}/scalapack/lib;-lscalapack;${lapack_LIBRARIES};-L/usr/lib/gcc/x86_64-linux-gnu/8;-lgfortran;-lm"  CACHE STRING "BLACS libraries")

# ScaLAPACK
set( scalapack_LIBRARIES  "${blacs_LIBRARIES}"  CACHE STRING "ScaLAPACK libraries")
# used by https://github.com/wavefunction91/linalg-cmake-modules
set( ScaLAPACK_LIBRARIES  "${scalapack_LIBRARIES};MPI::MPI_C"  CACHE STRING "ScaLAPACK libraries")
