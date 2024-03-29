# these cache/environment variables control the behavior of this module:
# INTEL_DIR (cache, env)
# MKL_ROOT_DIR (cache, or env MKLROOT)
# MKL_THREADING (cache)
# MKL_BLACS_MPI_ABI (cache)
# INTEGER4 (cache)
# BLA_STATIC (cache)

############## Only usable on UNIX platforms
if (NOT UNIX)
  message(FATAL_ERROR "_mkl.cmake is only usable on UNIX platforms")
endif(NOT UNIX)
if (DEFINED CMAKE_SYSTEM_NAME)
  set(_cmake_system_name ${CMAKE_SYSTEM_NAME})
else(DEFINED CMAKE_SYSTEM_NAME)
  set(_cmake_system_name ${CMAKE_HOST_SYSTEM_NAME})
endif(DEFINED CMAKE_SYSTEM_NAME)

############## Top tools dir
# prefer OneAPI
if (NOT INTEL_DIR)
  if (DEFINED ENV{INTEL_DIR})
    set(_intel_dir $ENV{INTEL_DIR})
  else(DEFINED ENV{INTEL_DIR})
    if (EXISTS /opt/intel/oneapi)
      set(_intel_dir /opt/intel/oneapi)
    elseif(EXISTS /opt/intel)
      set(_intel_dir /opt/intel)
    else ()
      set(_intel_dir )
    endif()
  endif(DEFINED ENV{INTEL_DIR})
  if (NOT EXISTS "${_intel_dir}")
    set(_intel_dir )
  endif()
  set(INTEL_DIR "${_intel_dir}" CACHE PATH "Intel tools root directory")
endif(NOT INTEL_DIR)

############## MKL
if (DEFINED ENV{MKLROOT})
  set(_mkl_root_dir "$ENV{MKLROOT}")
else(DEFINED ENV{MKLROOT})
  if (EXISTS "${INTEL_DIR}/mkl/latest") # OneAPI
    set(_mkl_root_dir "${INTEL_DIR}/mkl/latest")
  elseif (EXISTS "${INTEL_DIR}/mkl")
    set(_mkl_root_dir "${INTEL_DIR}/mkl")
  else ()
    set(_mkl_root_dir )
  endif()
endif(DEFINED ENV{MKLROOT})
if (NOT EXISTS "${_mkl_root_dir}")
  set(_mkl_root_dir )
endif()
set(MKL_ROOT_DIR "${_mkl_root_dir}" CACHE PATH "Intel MKL root directory")

if (EXISTS "${MKL_ROOT_DIR}/lib/intel64") # Linux
  set(_mkl_lib_dir "${MKL_ROOT_DIR}/lib/intel64")
elseif (EXISTS "${MKL_ROOT_DIR}/lib") # MacOS
  set(_mkl_lib_dir "${MKL_ROOT_DIR}/lib")
endif()

# is this OneAPI?
if (EXISTS "${_mkl_lib_dir}/libmkl_sycl.a")
  set(_mkl_oneapi 1)
else()
  set(_mkl_oneapi 0)
endif()

set(MKL_THREADING "SEQ" CACHE STRING "MKL backend: SEQ(default), TBB or OMP")
if (MKL_THREADING STREQUAL "SEQ")
  set(_mkl_backend_id "sequential")
elseif (MKL_THREADING STREQUAL "TBB")
  include(${CMAKE_CURRENT_LIST_DIR}/_tbb.cmake)
  set(_mkl_backend_id "tbb_thread")
  foreach(_libdir ${TBB_LIBRARY_DIRS})
    list(APPEND _tbb_libdir_dashLs "-L${_libdir}")
    list(APPEND _tbb_libdir_rpaths "-Wl,-rpath,${_libdir}")
  endforeach(_libdir)
else()
  set(_mkl_backend_id "intel_thread")
endif()

if (INTEGER4)
  set(_mkl_fortran_int lp64)
else(INTEGER4)
  set(_mkl_fortran_int ilp64)
endif(INTEGER4)

# initialize BLA_STATIC, if needed
if (BUILD_SHARED_LIBS)
  set(_bla_static FALSE)
else (BUILD_SHARED_LIBS)
  set(_bla_static TRUE)
endif (BUILD_SHARED_LIBS)
set(BLA_STATIC ${_bla_static} CACHE BOOL "Whether to use static linkage for BLAS, LAPACK, and related libraries")

if (NOT BLA_STATIC)
  set(_mkl_libraries "-lmkl_intel_${_mkl_fortran_int}" "-lmkl_${_mkl_backend_id}" "-lmkl_core" "-L${_mkl_lib_dir}")
  if (_cmake_system_name STREQUAL Linux)
    list(PREPEND _mkl_libraries "-Wl,--no-as-needed")
  elseif(_cmake_system_name STREQUAL Darwin)
    list(APPEND _mkl_libraries "-Wl,-rpath,${_mkl_lib_dir}")
  endif()
else(NOT BLA_STATIC)
  set(_mkl_static_library_suffix ".a")
  set(_mkl_libraries "${_mkl_lib_dir}/libmkl_intel_${_mkl_fortran_int}${_mkl_static_library_suffix}" "${_mkl_lib_dir}/libmkl_${_mkl_backend_id}${_mkl_static_library_suffix}" "${_mkl_lib_dir}/libmkl_core${_mkl_static_library_suffix}")
  if (_cmake_system_name STREQUAL Linux)
    list(PREPEND _mkl_libraries "-Wl,--start-group")
    list(APPEND _mkl_libraries "-Wl,--end-group")
  endif(_cmake_system_name STREQUAL Linux)
endif(NOT BLA_STATIC)
if (MKL_THREADING STREQUAL "TBB")
  list(APPEND _mkl_libraries "-ltbb" "${_tbb_libdir_dashLs}")
  if (BUILD_SHARED_LIBS AND APPLE)
    list(APPEND _mkl_libraries "${_tbb_libdir_rpaths}")
  endif(BUILD_SHARED_LIBS AND APPLE)
elseif (MKL_THREADING STREQUAL "OMP")
  # only using Intel OMP lib
  list(APPEND _mkl_libraries "-liomp5")
endif()
list(APPEND _mkl_libraries "-lpthread;-lm;-ldl")

# LAPACK
set(BLAS_LIBRARIES "${_mkl_libraries}" CACHE STRING "BLAS libraries")
set(LAPACK_LIBRARIES "${_mkl_libraries}" CACHE STRING "LAPACK libraries")
set(LAPACK_INCLUDE_DIRS ${MKL_ROOT_DIR}/include CACHE STRING "LAPACK include directories")
set(_mkl_compile_definitions "MADNESS_LINALG_USE_LAPACKE;MKL_INT=$<IF:$<BOOL:${INTEGER4}>,int,long>")
set(LAPACK_COMPILE_DEFINITIONS "${_mkl_compile_definitions}" CACHE STRING "LAPACK preprocessor definitions")

# for wavefunction91's FindLAPACK
set(lapack_LIBRARIES "${LAPACK_LIBRARIES}" CACHE STRING "LAPACK libraries")

# BLACS
set(MKL_BLACS_MPI_ABI "intelmpi" CACHE STRING "MKL BLACS assumes this MPI ABI: intelmpi(default), sgimpt or openmpi")
set(_mkl_blacs_mpi_abi "${MKL_BLACS_MPI_ABI}")
if (NOT BLA_STATIC)
  set(_blacs_lib "-lmkl_blacs_${_mkl_blacs_mpi_abi}_${_mkl_fortran_int};${LAPACK_LIBRARIES}")
else(NOT BLA_STATIC)
  set(_blacs_lib "${_mkl_lib_dir}/libmkl_blacs_${_mkl_blacs_mpi_abi}_${_mkl_fortran_int}${_mkl_static_library_suffix};${LAPACK_LIBRARIES}")
  if (_cmake_system_name STREQUAL Linux)
    list(PREPEND _blacs_lib "-Wl,--start-group")
    list(APPEND _blacs_lib "-Wl,--end-group")
  endif(_cmake_system_name STREQUAL Linux)
endif(NOT BLA_STATIC)
set(blacs_LIBRARIES "${_blacs_lib}" CACHE STRING "BLACS libraries")

# ScaLAPACK
if (NOT BLA_STATIC)
  set(_scalapack_lib "-lmkl_scalapack_${_mkl_fortran_int}")
else(NOT BLA_STATIC)
  set(_scalapack_lib "${_mkl_lib_dir}/libmkl_scalapack_${_mkl_fortran_int}${_mkl_static_library_suffix}")
endif(NOT BLA_STATIC)
set(scalapack_LIBRARIES "${_scalapack_lib};${blacs_LIBRARIES}" CACHE STRING "ScaLAPACK libraries")
# used by https://github.com/wavefunction91/linalg-cmake-modules
set(ScaLAPACK_LIBRARIES "${scalapack_LIBRARIES};MPI::MPI_C" CACHE STRING "ScaLAPACK libraries")
