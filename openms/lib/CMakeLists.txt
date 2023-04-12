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
cmake_minimum_required (VERSION 3.5)
project (openms)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RELWITHDEBINFO)
endif()
#set(CMAKE_BUILD_TYPE DEBUG)
set(CMAKE_VERBOSE_MAKEFILE OFF)
if (CMAKE_COMPILER_IS_GNUCC) # Does it skip the link flag on old OsX?
  # TODO: If updating to minimum requirement cmake>=3.7, use
  # CMAKE_SHARED_LINKER_FLAGS_INIT to combine LDFLAGS options.
  # https://cmake.org/cmake/help/v3.7/variable/CMAKE_SHARED_LINKER_FLAGS_INIT.html
  # see also issue #661
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


set(CMAKE_INCLUDE_CURRENT_DIR ON)

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
option(BUILD_MEEP "Using MEEP for solvng maxwell equations" ON)
option(BUILD_TAMM "Using TAMM lib for tensor contraction" ON)
option(BUILD_INFDTD "Using internal FDTDc for solvng maxwell equations" ON)
option(MEEP_GIT "Using github version of meep" OFF)

if(BUILD_INFDTD)
	## cmakefile for building build-in FDTD code
  add_subdirectory(./fdtd/ libfdtd)
  #target_link_libraries(intfdtd PUBLIC fdtd)

endif()

if(BUILD_MEEP)
    # if building meep, we need to build libctl, harminv, mpb as well,
    # a) build libctl (TODO)
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
  set(LDFLAGS  "-L$ENV{HOME}/local/fftw3.3.10/lib/   -L${PROJECT_SOURCE_DIR}/deps/lib")
  set(CPPFLAGS "-I$ENV{HOME}/ocal/fftw3.3.10/include -I${PROJECT_SOURCE_DIR}/deps/include")
  ExternalProject_Add(mpb
    URL https://github.com/NanoComp/mpb/releases/download/v1.11.1/mpb-1.11.1.tar.gz
    PREFIX ${PROJECT_BINARY_DIR}/deps
    INSTALL_DIR ${PROJECT_SOURCE_DIR}/deps
    CONFIGURE_COMMAND 
    ${CMAKE_COMMAND} -E env LDFLAGS=${LDFLAGS} 
    ${CMAKE_COMMAND} -E env CPPFLAGS=${CPPFLAGS}
        ../mpb/configure --enable-shared=yes --without-scheme --with-fftw=/home/zhy/local/fftw3.3.10/ --enable-maintainer-mode --without-guile --without-libctl --prefix=<INSTALL_DIR>  
    BUILD_COMMAND make
    INSTALL_COMMAND make install
    TEST_COMMAND make check
  )

  # d) build meep
  set(ADDF LDFLAGS=)
  #set(LDFLAGS "-L$ENV{HOME}/local/fftw3.3.10/lib/ -L${PROJECT_SOURCE_DIR}/deps/lib")
  set(LDFLAGS  "-L$ENV{HOME}/local/fftw3.3.10/lib/ -L${PROJECT_SOURCE_DIR}/deps/lib")
  set(CPPFLAGS "-I$ENV{HOME}/ocal/fftw3.3.10/include -I${PROJECT_SOURCE_DIR}/deps/include  -I$HOME/anaconda3/include/python3.8/")

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
endif()

# cmake for building bml (TODO)


if(BUILD_TAMM) # git@github.com:NWChemEx-Project/TAMM.git
  ExternalProject_Add(libtamm
    PREFIX ${PROJECT_BINARY_DIR}/deps
    INSTALL_DIR ${PROJECT_SOURCE_DIR}/deps
    GIT_REPOSITORY https://github.com/NWChemEx-Project/TAMM.git
    GIT_TAG main
  )
  
endif() # ENABLE_LIBXC


if(EXISTS "${PROJECT_SOURCE_DIR}/cmake.user.inc")
  include("${PROJECT_SOURCE_DIR}/cmake.user.inc")
endif()
