# Set default install location to parent of source directory (qmclib/../)

get_filename_component(PARENT_DIR "${CMAKE_CURRENT_SOURCE_DIR}" DIRECTORY)
set(CMAKE_INSTALL_PREFIX "${PARENT_DIR}" CACHE PATH "Install path prefix" FORCE)


cmake_minimum_required(VERSION 3.14)
project(qedlib LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Use Homebrew LLVM
# You must call cmake with:
#   export HOMEBREW_PREFIX=$(brew --prefix)
#   export CC=$(brew --prefix llvm)/bin/clang
#   export CXX=$(brew --prefix llvm)/bin/clang++

# Find pybind11 using FetchContent (no install required)
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.12.0
)
FetchContent_MakeAvailable(pybind11)


#=============================
# Try to find system GSL
#=============================
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/Modules")
message(STATUS "CMAKE_CURRENT_SOURCE_DIR is ${CMAKE_CURRENT_SOURCE_DIR}")
find_package(GSL REQUIRED)

message(STATUS "GSL FOUND or not ${GSL_FOUND}")

if (NOT GSL_FOUND)
    message(STATUS "System GSL not found. Will download and build GSL locally.")

    # Define path to install GSL locally
    set(GSL_INSTALL_DIR "${PROJECT_SOURCE_DIR}/deps")
    set(GSL_PREFIX_PATH "${GSL_INSTALL_DIR}")
    set(GSL_INCLUDE_DIR "${GSL_INSTALL_DIR}/include")
    set(GSL_LIBRARY_DIR "${GSL_INSTALL_DIR}/lib")

else()
    message(STATUS "Found system GSL.")
    message(STATUS "GSL_INCLUDE_DIR path is   : ${GSL_INCLUDE_DIR}")
endif()


# ==================== BLAS Backend Detection ====================

# macOS + Apple Silicon: try Accelerate first
if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    message(STATUS "Detected macOS on Apple Silicon (arm64), attempting to use Accelerate framework")

    find_library(ACCELERATE_FRAMEWORK Accelerate)

    if(ACCELERATE_FRAMEWORK)
        message(STATUS "Using Accelerate framework: ${ACCELERATE_FRAMEWORK}")
        set(USE_ACCELERATE TRUE)
    else()
        message(WARNING "Accelerate not found; falling back to OpenBLAS")
    endif()
endif()

# find_package(OpenBLAS REQUIRED)
# Fallback or default: OpenBLAS
if(NOT USE_ACCELERATE)
    # TODO: use a find_openblas module to achieve this
    find_library(OPENBLAS_LIBRARY NAMES openblas PATHS /usr/lib /usr/local/lib /opt/homebrew/lib)
    find_path(OPENBLAS_INCLUDE_DIR NAMES cblas.h PATHS /usr/include /usr/local/include /opt/homebrew/include)

    if(NOT OPENBLAS_LIBRARY OR NOT OPENBLAS_INCLUDE_DIR)
        message(FATAL_ERROR "Could not find OpenBLAS. Please install it and set OPENBLAS_LIBRARY and OPENBLAS_INCLUDE_DIR manually.")
    endif()

    message(STATUS "Using OpenBLAS: ${OPENBLAS_LIBRARY}")
    message(STATUS "OpenBLAS include dir: ${OPENBLAS_INCLUDE_DIR}")
endif()
message(STATUS "BLAS backend: ${USE_ACCELERATE} -> Accelerate, fallback: OpenBLAS")

# ==================== OpenMP ====================

# Manually set OpenMP (macOS + LLVM)
# set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I${HOMEBREW_PREFIX}/include")
# set(OpenMP_C_LIB_NAMES "omp")
# set(OpenMP_omp_LIBRARY "${HOMEBREW_PREFIX}/lib/libomp.dylib")
find_package(OpenMP REQUIRED)

#================ Eigen ===============
# Locate Eigen
# Require user to provide Eigen3 path explicitly
if(NOT DEFINED EIGEN3_INCLUDE_DIR)
    message(FATAL_ERROR "You must provide the Eigen3 include path using -DEIGEN3_INCLUDE_DIR=/path/to/eigen3")
endif()

message(STATUS "Using Eigen3 include path: ${EIGEN3_INCLUDE_DIR}")


# Flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -ffast-math -fopenmp")


# =================== module ===============
# Create the pybind11 module
pybind11_add_module(qmclib MODULE
    src/bindings.cpp
    src/trial.cpp
    src/propagator.cpp
    src/estimators.cpp
    src/linalg.cpp
    src/qedhf.cpp
)

set_target_properties(
  qmclib
  PROPERTIES
  #PREFIX ""
  OUTPUT_NAME _qmclib
)



# ==================== Link Libraries ====================
target_include_directories(qmclib PRIVATE ${GSL_INCLUDE_DIR})
target_include_directories(qmclib PRIVATE ${EIGEN3_INCLUDE_DIR})

target_link_libraries(qmclib PRIVATE ${GSL_LIBRARIES})
target_link_libraries(qmclib PRIVATE OpenMP::OpenMP_CXX)
if(USE_ACCELERATE)
    target_compile_definitions(qmclib PRIVATE
        USE_ACCELERATE
    )
    #target_compile_definitions(qmclib PRIVATE
    #    USE_ACCELERATE
    #    ACCELERATE_NEW_LAPACK
    #    ACCELERATE_LAPACK_ILP64
    #)
    target_link_libraries(qmclib PRIVATE ${ACCELERATE_FRAMEWORK})
else()
    target_include_directories(qmclib PRIVATE ${OPENBLAS_INCLUDE_DIR})
    target_link_libraries(qmclib PRIVATE ${OPENBLAS_LIBRARY})
    target_compile_definitions(qmclib PRIVATE USE_OPENBLAS)
endif()


#
install(TARGETS qmclib
        LIBRARY DESTINATION .   # Install directly into CMAKE_INSTALL_PREFIX
)
