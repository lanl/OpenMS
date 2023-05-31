#; -*-CMake-*-
#
# This file is part of OpenmS package for compiling the low-level libraries
# 
# Copyright 2023. Triad National Security, LLC. All rights reserved. 
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# cmake file for buildig c, c++, fortran libs (TODO)
#


cmake_minimum_required(VERSION 3.15.0)

# Set OpenMS version ===========================================================
set(OPENMS_MAJOR_VERSION 1)
set(OPENMS_MINOR_VERSION 0)
set(OPENMS_MICRO_VERSION 0)
set(OPENMS_PRERELEASE_ID )

set(OPENMS_VERSION "${OPENMS_MAJOR_VERSION}.${OPENMS_MINOR_VERSION}.${OPENMS_MICRO_VERSION}")
if (OPENMS_PRERELEASE_ID)
  set(OPENMS_EXT_VERSION "${OPENMS_VERSION}-${OPENMS_PRERELEASE_ID}")
else(OPENMS_PRERELEASE_ID)
  set(OPENMS_EXT_VERSION "${OPENMS_VERSION}")
endif(OPENMS_PRERELEASE_ID)

project(openms
        VERSION ${OPENMS_VERSION}
	DESCRIPTION "OpenMS: python package for solving multiscale/multi-physics light-matter interactions problems in both closed and open quantum systems"
        HOMEPAGE_URL "https://openms-lmi.readthedocs.io/en/latest/")
enable_language(C)  # C needed even for basic platform introspection

# MAD default number of threads ================================================
set(MAD_NUM_THREADS 2)

# Preload versions/tags of all dependencies ====================================
include(versions.cmake)

# Safety net for dev workflow: accidental install will not affect FindOrFetch*
if (NOT DEFINED CACHE{CMAKE_FIND_NO_INSTALL_PREFIX})
  set(CMAKE_FIND_NO_INSTALL_PREFIX ON CACHE BOOL "Whether find_* commands will search CMAKE_INSTALL_PREFIX and CMAKE_STAGING_PREFIX; see https://cmake.org/cmake/help/latest/variable/CMAKE_FIND_NO_INSTALL_PREFIX.html#variable:CMAKE_FIND_NO_INSTALL_PREFIX")
endif()

# Bring camek toolkit ==========================================================
# FetchContent is used to download and include external projects or libraries  
include(FetchContent)
if (DEFINED PROJECT_BINARY_DIR)
  set(CMAKE_KIT_PREFIX_DIR PROJECT_BINARY_DIR)
else ()
  set(CMAKE_KIT_PREFIX_DIR CMAKE_CURRENT_BINARY_DIR)
endif()
FetchContent_Declare(
    cmake_kit
    #QUIET
    GIT_REPOSITORY      https://github.com/ValeevGroup/kit-cmake.git
    GIT_TAG             ${TA_TRACKED_VGCMAKEKIT_TAG}
    SOURCE_DIR ${${CMAKE_KIT_PREFIX_DIR}}/cmake/vg
    BINARY_DIR ${${CMAKE_KIT_PREFIX_DIR}}/cmake/vg-build
    SUBBUILD_DIR ${${CMAKE_KIT_PREFIX_DIR}}/cmake/vg-subbuild
)
FetchContent_MakeAvailable(cmake_kit)
list(APPEND CMAKE_MODULE_PATH "${cmake_kit_SOURCE_DIR}/modules")

# ==============================================================================
# old one, may be removed (todo)
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RELWITHDEBINFO)
endif()

set(CMAKE_BUILD_TYPE RELEASE)
set(CMAKE_VERBOSE_MAKEFILE OFF)

if (CMAKE_COMPILER_IS_GNUCC) # Does it skip the link flag on old OsX?
  # CMAKE_SHARED_LINKER_FLAGS_INIT to combine LDFLAGS options.
  # https://cmake.org/cmake/help/v3.7/variable/CMAKE_SHARED_LINKER_FLAGS_INIT.html
  if(UNIX AND NOT APPLE AND NOT DEFINED ENV{LDFLAGS})
    set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-as-needed")
  endif()
endif()
set(CMAKE_C_FLAGS "-Wall ${CMAKE_C_FLAGS}")

option(BUILD_MARCH_NATIVE "gcc flag -march=native" off)
if (BUILD_MARCH_NATIVE)
  include(CheckCCompilerFlag)
  CHECK_C_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
  if(COMPILER_SUPPORTS_MARCH_NATIVE)
    if ("${CMAKE_C_COMPILER_ID}" STREQUAL "Intel")
      message("Add CFLAGS -march=native -unroll-aggressive")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -unroll-aggressive -ipo")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-math-errno")
    else()
      message("Add CFLAGS -march=native -ftree-vectorize")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native -funroll-loops -ftree-vectorize")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-math-errno")
    endif()
  endif()
else()
  if ("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
    include(CheckCCompilerFlag)
    CHECK_C_COMPILER_FLAG("-msse3" COMPILER_SUPPORTS_SSE3)
    if(COMPILER_SUPPORTS_SSE3)
      # Avoids error "‘SIMDD’ undeclared here (not in a function)" in the qcint two-electron integral interface
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -msse3")
    endif()
  endif()
endif()

# Set install paths ============================================================
# not done yet (TODO)
set(PROJECT_INSTALL_DIR "${PROJECT_SOURCE_DIR}/deps")
message("Library install path is ${PROJECT_INSTALL_DIR}")
set(OPENMS_INSTALL_BINDIR "bin"
	CACHE PATH "OpenMS binary install directory")
set(OPENMS_INSTALL_INCLUDEDIR "include"
	CACHE PATH "OpenMS INCLUDE install directory")
set(OPENMS_INSTALL_LIBDIR "lib"
	CACHE PATH "OpenMS LIB install directory")
set(OPENMS_INSTALL_SHAREDIR "share/openms/${OPENMS_EXT_VERSION}"
	CACHE PATH "OpenMS DATA install directory")
set(OPENMS_INSTALL_DATADIR "${OPENMS_INSTALL_SHAREDIR}/data"
	CACHE PATH "OpenMS DATA install directory")
set(OPENMS_INSTALL_DOCDIR "${OPENMS_INSTALL_SHAREDIR}/doc"
	CACHE PATH "OpenMS DOC install directory")
set(OPENMS_INSTALL_CMAKEDIR "lib/cmake/openms"
	CACHE PATH "OpenMS CMAKE install directory")


# Add module directory and modules =============================================
# TODO

# Load extra CMake features ====================================================
include(CMakeDependentOption)
include(CMakePackageConfigHelpers)
include(FeatureSummary)

set(MPI_CXX_SKIP_MPICXX TRUE CACHE BOOL "MPI_CXX_SKIP_MPICXX")

# Configure options ============================================================
# all features are summaried at the end!



option(ENABLE_MPI "Enable MPI" ON)
option(ENABLE_QMD "Enable QMD/MQC/CTMQC" ON)
option(ENABLE_MEEP "Using MEEP for solvng maxwell equations" ON)
option(ENABLE_TA "Using TiledArray lib for tensor contraction" ON)
option(ENABLE_TASCALAPACK "Enable ScaLAPACK Bindings in TiledArray" OFF)
option(ENABLE_TACUDA "Enable use of CUDA with TiledArray" OFF)
option(ENABLE_TAPYTHON "Enable Python binding of TA lib" ON)
option(ENABLE_TBB "Enable use of TBB with MADNESS" OFF)
option(TA_TENSOR_MEM_TRACE "Turn on instrumented tracing of TA::Tensor memory use" OFF)
option(TA_TENSOR_MEM_PROFILE "Turn on instrumented profiling of TA::Tensor memory use" ${TA_TENSOR_MEM_TRACE})
option(BUILD_INFDTD "Using internal FDTDc for solvng maxwell equations" ON)

if (ENABLE_TA)
	set(EIGEN3_PATH $ENV{EIGEN3_PATH} CACHE INTERNAL "EIGEN3_PATH for TA lib" FORCE)
	message("EIGEN3_PATH is ${EIGEN3_PATH}")
	if(NOT EIGEN3_PATH)
		message(FATAL_ERROR "EIGEN3_PATH is not defined!!! Please download eigen3 (https://gitlab.com/libeigen/eigen/-/archive/3.3.9/eigen-3.3.9.tar.gz) and privde the path as EIGEN3_PATH (no need to compile it)")
	endif()
endif()

add_feature_info(TA ENABLE_TA "TiledArray: block-sparse tensor framework for modern (distributed-memory and heterogeneous) computing")
add_feature_info(QMD ENABLE_QMD "QMD: Module of Mixed quantum-classical dynamics and quatum nuclear dynamics")
add_feature_info(TAPYTHON ENABLE_TAPYTHON "Python binding of TiledArray lib")
add_feature_info(MPI ENABLE_MPI "Message-Passing Interface supports distributed-memory parallel programs")
add_feature_info(MEEP ENABLE_MEEP "MEEP: a free and open-source software package for electromagnetics simulation")
add_feature_info(TAScaLAPACK ENABLE_TASCALAPACK "ScaLAPACK provides distributed linear algebra")
add_feature_info(TACUDA ENABLE_TACUDA "NVIDIA CUDA support for GPU in TiledArray")
add_feature_info(TBB ENABLE_TBB "Intel Thread-Building Blocks (TBB) supports programming shared-memory systems")
add_feature_info(TA_MEM_TRACE TA_TENSOR_MEM_TRACE "instrumented tracing of TA::Tensor memory use")
add_feature_info(TA_MEM_PROFILE TA_TENSOR_MEM_PROFILE "instrumented profiling of TA::Tensor memory use")
add_feature_info(INTFDTD BUILD_INFDTD "Built-in (internal) FDTD solvers with limited functionalities")

# miscellaneous cmake platform-neutral and platform-specific configuration =============================
set(CMAKE_FIND_NO_INSTALL_PREFIX TRUE)  # do not search in CMAKE_INSTALL_PREFIX
set(CMAKE_SKIP_RPATH FALSE)
set(CMAKE_SKIP_BUILD_RPATH  FALSE)
set(CMAKE_SKIP_INSTALL_RPATH FALSE)
set(CMAKE_NO_SYSTEM_FROM_IMPORTED TRUE)  # do not use -isystem by default to avoid include dir ordering issues as well as other issues like https://gcc.gnu.org/onlinedocs/cpp/System-Headers.html

if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
  # look for frameworks and appbundles last
  set(CMAKE_FIND_FRAMEWORK LAST)
  set(CMAKE_FIND_APPBUNDLE LAST)
endif()

# Build variables  =============================================================

# find MKLROOT (more advanced TBA)
if (DEFINED ENV{MKLROOT})
	set(MKLROOT $ENV{MKLROOT})
	message("MKLROOT is " ${MKLROOT})
endif()

set(CMAKE_INCLUDE_CURRENT_DIR ON)

# find the python path
find_package(Python3 COMPONENTS Interpreter Development) # REQUIRED VERSION 3.8)

message(STATUS "Python executable: ${Python3_EXECUTABLE}")
message(STATUS "Python include dirs: ${Python3_INCLUDE_DIRS}")
message(STATUS "Python libraries: ${Python3_LIBRARIES}")


# Architecture specified cmake flags.  See also the templates in
# openms/lib/cmake_arch_config
if(EXISTS "${PROJECT_SOURCE_DIR}/cmake.arch.inc")
  include("${PROJECT_SOURCE_DIR}/cmake.arch.inc")
endif()

if (NOT BLAS_LIBRARIES)
#enable_language(Fortran)
find_package(BLAS)
check_function_exists(ffsll HAVE_FFS)
endif()

if (NOT BLAS_LIBRARIES)
  message(FATAL_ERROR "A required library with BLAS API not found.")
else()
  message(STATUS "BLAS libraries: ${BLAS_LIBRARIES}")
endif()

# if unable to find mkl library, just create BLAS_LIBRARIES here, e.g.
# set(BLAS_LIBRARIES "-L/path/to/mkl/lib -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lmkl_avx -lm")
# or
# set(BLAS_LIBRARIES "                  /path/to/mkl/lib/intel64/libmkl_intel_lp64.so")
# set(BLAS_LIBRARIES "${BLAS_LIBRARIES};/path/to/mkl/lib/intel64/libmkl_sequential.so")
# set(BLAS_LIBRARIES "${BLAS_LIBRARIES};/path/to/mkl/lib/intel64/libmkl_core.so")
# set(BLAS_LIBRARIES "${BLAS_LIBRARIES};/path/to/mkl/lib/intel64/libmkl_avx.so")
# set(BLAS_LIBRARIES "${BLAS_LIBRARIES};/path/to/mkl/lib/intel64/libmkl_def.so")
# set(BLAS_LIBRARIES "-Wl,-rpath=${MKLROOT}/lib/intel64/ ${BLAS_LIBRARIES}")
#
# These settings can be written in the cmake.arch.inc file. This config file
# will automatically load all settings specified in cmake.arch.inc
#

option(ENABLE_OPENMP "Compiling C extensions with openmp" ON)
set(OPENMP_C_PROPERTIES "")
if(ENABLE_OPENMP)
  find_package(OpenMP)
  if(OPENMP_FOUND)
    set(HAVE_OPENMP 1)
    set(OPENMP_C_PROPERTIES OpenMP::OpenMP_C)
  endif()
endif()

#find_package(PythonInterp REQUIRED)
#find_package(PythonLibs REQUIRED)
#execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import numpy; print(numpy.get_include())"
#  OUTPUT_VARIABLE NUMPY_INCLUDE)
#include_directories(${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE})

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/deps/include)
include_directories(${CMAKE_INSTALL_PREFIX}/include)
link_directories(${PROJECT_SOURCE_DIR}/deps/lib ${PROJECT_SOURCE_DIR}/deps/lib64)
link_directories(${CMAKE_INSTALL_PREFIX}/lib ${CMAKE_INSTALL_PREFIX}/lib64)

configure_file(
  "${PROJECT_SOURCE_DIR}/config.h.in"
  "${PROJECT_BINARY_DIR}/config.h")
# to find config.h
include_directories("${PROJECT_BINARY_DIR}")

# See also https://gitlab.kitware.com/cmake/community/wikis/doc/cmake/RPATH-handling
if (WIN32)
  #?
elseif (APPLE)
  set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
  set(CMAKE_INSTALL_RPATH "@loader_path;@loader_path/deps/lib;@loader_path/deps/lib64")
  set(CMAKE_BUILD_RPATH "@loader_path;@loader_path/deps/lib;@loader_path/deps/lib64")
else ()
  set(CMAKE_SKIP_BUILD_RPATH  True)
  set(CMAKE_BUILD_WITH_INSTALL_RPATH True)
  set(CMAKE_INSTALL_RPATH "\$ORIGIN:\$ORIGIN/deps/lib:\$ORIGIN/deps/lib64")
endif ()

# Overwrite CMAKE_C_CREATE_SHARED_LIBRARY in Modules/CMakeCInformation.cmake
# to remove the SONAME flag in the so file. The soname information causes
# dynamic link error when importing libcint library.
set(C_LINK_TEMPLATE "<CMAKE_C_COMPILER> <CMAKE_SHARED_LIBRARY_C_FLAGS> <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> <CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS> -o <TARGET> <OBJECTS> <LINK_LIBRARIES>")
set(CXX_LINK_TEMPLATE "<CMAKE_CXX_COMPILER> <CMAKE_SHARED_LIBRARY_CXX_FLAGS> <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> <CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS> -o <TARGET> <OBJECTS> <LINK_LIBRARIES>")

include(ExternalProject)
# Avoid warning about DOWNLOAD_EXTRACT_TIMESTAMP in CMake 3.24:
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
	cmake_policy(SET CMP0135 NEW)
endif()

option(BUILD_TAMM "Using TAMM lib for tensor contraction" OFF)
option(MEEP_GIT "Using github version of meep" OFF)
set(BUILD_TARGETS)

if(BUILD_INFDTD)
  ## cmakefile for building build-in FDTD code
  add_subdirectory(./fdtd/ libfdtd)
  #target_link_libraries(intfdtd PUBLIC fdtd)
  list(APPEND BUILD_TARGETS libfdtd)
endif()

if(ENABLE_QMD)
  message("Lib for QMD and MQC dynamics (TBA)")
endif()

# find FFTW_PATH (will be replaced with find_package (todo,to  add a toolchain))
if (DEFINED FFTW_LIBRARIES)
  message("FFTW libraries are defined (in arch.inc):")
else()
  message("Looking for FFTW libraries...")
  find_library(FFTW_LIBRARIES NAMES fftw3 libfftw3)
endif()
get_filename_component(FFTW_LIBRARY_DIR ${FFTW_LIBRARIES} DIRECTORY)
get_filename_component(FFTW_PARENT_DIR ${FFTW_LIBRARY_DIR} DIRECTORY)
set(FFTW_INCLUDE_DIR ${FFTW_PARENT_DIR}/include)
message("FFTW libraries: ${FFTW_LIBRARIES}")
message("FFTW library path: ${FFTW_LIBRARY_DIR}")
message("FFTW include path: ${FFTW_INCLUDE_DIR}")

if(ENABLE_MEEP)
    # if building meep, we need to build libctl, harminv, mpb as well,
    # a) build libctl
  ExternalProject_Add(libctl
    URL https://github.com/NanoComp/libctl/releases/download/v4.2.0/libctl-4.2.0.tar.gz
    PREFIX ${PROJECT_BINARY_DIR}/deps
    INSTALL_DIR ${PROJECT_SOURCE_DIR}/deps
    #./configure  --enable-shared=yes --prefix= --without-guile
    CONFIGURE_COMMAND ../libctl/configure --enable-shared=yes --prefix=<INSTALL_DIR>  --without-guile
    BUILD_COMMAND make
    INSTALL_COMMAND make install
    TEST_COMMAND make check
    #CMAKE_ARGS -DCMAKE_BUILD_TYPE=RELEASE -DBUILD_SHARED_LIBS=1
    #        -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
    #        -DCMAKE_INSTALL_LIBDIR:PATH=lib
    #	    --without-guile
    #        -DCMAKE_C_CREATE_SHARED_LIBRARY=${C_LINK_TEMPLATE}
    #        -DENABLE_XHOST:STRING=${BUILD_MARCH_NATIVE}
  )
  # b) build harminv
  ExternalProject_Add(harminv
    URL https://github.com/NanoComp/harminv/releases/download/v1.4.1/harminv-1.4.1.tar.gz
    PREFIX ${PROJECT_BINARY_DIR}/deps
    INSTALL_DIR ${PROJECT_SOURCE_DIR}/deps
    CONFIGURE_COMMAND ../harminv/configure --enable-shared=yes --prefix=<INSTALL_DIR>  --without-guile
    BUILD_COMMAND make
    INSTALL_COMMAND make install
    TEST_COMMAND make check
  )

  # c) build  mpb
  set(LDFLAGS  "-L${FFTW_LIBRARY_DIR} -L${PROJECT_SOURCE_DIR}/deps/lib")
  set(CPPFLAGS "-I${FFTW_INCLUDE_DIR} -I${PROJECT_SOURCE_DIR}/deps/include")

  ExternalProject_Add(mpb
    URL https://github.com/NanoComp/mpb/releases/download/v1.11.1/mpb-1.11.1.tar.gz
    PREFIX ${PROJECT_BINARY_DIR}/deps
    INSTALL_DIR ${PROJECT_SOURCE_DIR}/deps
    CONFIGURE_COMMAND 
    ${CMAKE_COMMAND} -E env LDFLAGS=${LDFLAGS} 
    ${CMAKE_COMMAND} -E env CPPFLAGS=${CPPFLAGS}
      ../mpb/configure --enable-shared=yes --without-scheme --with-fftw=${FFTW_LIBRARY_DIR} --enable-maintainer-mode --without-guile --without-libctl --prefix=<INSTALL_DIR>  
    BUILD_COMMAND make
    INSTALL_COMMAND make install
    TEST_COMMAND make check
  )

  # d) build meep
  set(ADDF LDFLAGS=)
  #set(LDFLAGS "-L$ENV{HOME}/local/fftw3.3.10/lib/ -L${PROJECT_SOURCE_DIR}/deps/lib")
  set(LDFLAGS  "-L${FFTW_LIBRARY_DIR} -L${PROJECT_SOURCE_DIR}/deps/lib")
  set(CPPFLAGS "-I${FFTW_INCLUDE_DIR} -I${PROJECT_SOURCE_DIR}/deps/include  -I${Python3_INCLUDE_DIRS}")

  ExternalProject_Add(meep
    PREFIX ${PROJECT_BINARY_DIR}/deps
    INSTALL_DIR ${PROJECT_SOURCE_DIR}/deps
    BUILD_IN_SOURCE  True
    # github
    ##GIT_REPOSITORY https://github.com/NanoComp/meep.git
    #GIT_REPOSITORY https://github.com/ares201005/meep.git
    #GIT_TAG master
    # use released version
    URL https://github.com/NanoComp/meep/releases/download/v1.25.0/meep-1.25.0.tar.gz

    CONFIGURE_COMMAND 
    ${CMAKE_COMMAND} -E env LDFLAGS=${LDFLAGS} 
    ${CMAKE_COMMAND} -E env CPPFLAGS=${CPPFLAGS}
    	../meep/configure  --enable-shared=yes --without-scheme --with-libctl=<INSTALL_DIR> --with-mpb=yes --with-mpi=no  --with-python=yes --prefix=<INSTALL_DIR> 
       #../meep/autogen.sh  --enable-shared=yes --without-scheme --with-libctl=<INSTALL_DIR> --with-mpb=yes --with-mpi=no  --with-python=yes --prefix=<INSTALL_DIR> 
    BUILD_COMMAND make
    INSTALL_COMMAND make install
    #TEST_COMMAND make check
  )
  add_dependencies(meep mpb)
  add_dependencies(meep libctl)

  list(APPEND BUILD_TARGETS libctl)
  list(APPEND BUILD_TARGETS harminv)
  list(APPEND BUILD_TARGETS mpb)
  list(APPEND BUILD_TARGETS meep)
endif()

# cmake for building bml (TODO)
if(BUILD_BML)
  ExternalProject_Add(libbml
    PREFIX ${PROJECT_BINARY_DIR}/deps
    INSTALL_DIR ${PROJECT_SOURCE_DIR}/deps
    GIT_REPOSITORY https://github.com/lanl/bml.git
    GIT_TAG master
  )
endif()

# build Tiledarray 
# TODO
# find path to python
# specify toolchainpath

if(ENABLE_TA)
  message("TA lib will be build")
  message("EIGEN3 PATH is:      " ${EIGEN3_PATH})
  message("toolchain used is:   " ${toolchainpath})
  message("Use scalapack in TA: " ${ENABLE_TASCALAPACK})
  message("Enable CUDA in TA:   " ${ENABLE_TACUDA})
  message("Enable MPI in TA:    " ${ENABLE_MPI})
  message("MKLROOT is:          " ${MKLROOT})

  ExternalProject_Add(libta
    PREFIX ${PROJECT_BINARY_DIR}/deps
    INSTALL_DIR ${PROJECT_SOURCE_DIR}/deps
    #GIT_REPOSITORY https://github.com/ValeevGroup/tiledarray.git
    GIT_REPOSITORY https://github.com/ares201005/tiledarray.git
    GIT_TAG master
    # set cmake arguments for TA
    CMAKE_ARGS 
        -B build 
	-DCMAKE_INSTALL_PREFIX=${PROJECT_INSTALL_DIR}
	-DEIGEN3_INCLUDE_DIR=${EIGEN3_PATH}
        -DCMAKE_TOOLCHAIN_FILE=${toolchainpath}/gcc-mpi-mkl-tbb.cmake
        -DBUILD_DEPS_FROM_SOURC=ON 
	-DMKLROOT=${MKLROOT}
	-DENABLE_CUDA=${ENABLE_TACUDA}
	-DENABLE_MPI=${ENABLE_MPI}
	-DTA_PYTHON=${ENABLE_TAPYTHON}
	-DENABLE_SCALAPACK=${ENABLE_TASCALAPACK}
	-DPYTHON_EXECUTABLE=${Python3_EXECUTABLE}

    BUILD_COMMAND  cmake --build build --target python-tiledarray
    INSTALL_COMMAND cmake --build build --target install
    TEST_COMMAND "" # empty command, will not run tests during build
    #TEST_COMMAND cmake --build build --target check
    TEST_AFTER_INSTALL TRUE # Run tests after installation with "make check" 
    #CMAKE_CACHE_ARGS
  )

  list(APPEND BUILD_TARGETS libta)
endif()


# build TAMM 
if(BUILD_TAMM) # git@github.com:NWChemEx-Project/TAMM.git
  ExternalProject_Add(libtamm
    PREFIX ${PROJECT_BINARY_DIR}/deps
    INSTALL_DIR ${PROJECT_SOURCE_DIR}/deps
    GIT_REPOSITORY https://github.com/NWChemEx-Project/TAMM.git
    GIT_TAG main
  )
  
endif() # ENABLE_LIBXC

## build target
#add_custom_target(build
#  DEPENDS ${MY_PROJECT_TARGETS}
#)

add_custom_target(test
  COMMAND ${CMAKE_COMMAND} --build . --target test --config $<CONFIGURATION>
#  DEPENDS build_external_projects
)


message("")
message("=========================================================")
feature_summary(WHAT ALL
	DESCRIPTION "=== OpenMS Package/Feature Info ===")

if(EXISTS "${PROJECT_SOURCE_DIR}/cmake.user.inc")
  include("${PROJECT_SOURCE_DIR}/cmake.user.inc")
endif()
