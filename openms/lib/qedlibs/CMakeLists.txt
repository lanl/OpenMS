project(qedhf)
cmake_minimum_required(VERSION 3.12)

set(CMAKE_CXX_STANDARD 14)

include(FetchContent)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.8.1  # specify the version/tag
)
FetchContent_MakeAvailable(pybind11)


# Find required packages
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
#find_package(Python COMPONENTS Interpreter Development REQUIRED)
#find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development)
find_package(Python3 COMPONENTS Interpreter Development) # REQUIRED VERSION 3.8)
message(STATUS "Python executable: ${Python3_EXECUTABLE}")
message(STATUS "Python include dirs: ${Python3_INCLUDE_DIRS}")
message(STATUS "Python libraries: ${Python3_LIBRARIES}")

#find_package(OpenBLAS REQUIRED)

if (DEFINED ENV{MKLROOT})
        set(MKLROOT $ENV{MKLROOT})
        message("MKLROOT is " ${MKLROOT})
endif()

#find_package(MKL REQUIRED)
find_package(MKL HINTS $ENV{MKLROOT} PATH_SUFFIXES lib)


#eigen3
set(EIGEN3_PATH $ENV{EIGEN3_PATH} CACHE INTERNAL "EIGEN3_PATH for TA lib" FORCE)
message("EIGEN3_PATH is ${EIGEN3_PATH}")
if(NOT EIGEN3_PATH)
        message(FATAL_ERROR "EIGEN3_PATH is not defined!!! Please download eigen3 (https://gitlab.com/libeigen/eigen/-/archive/3.3.9/eigen-3.3.9.tar.gz) and privde the path as EIGEN3_PATH (no need to compile it)")
endif()

pybind11_add_module(qedhf MODULE
vt_qedhf.cpp
qed.cpp
# qmc.cpp
bindings.cpp)

set_target_properties(
  qedhf
  PROPERTIES
  #PREFIX ""
  OUTPUT_NAME _qedhf
)


target_include_directories(qedhf PRIVATE ${EIGEN3_PATH})


message("OpenBLAS_LIB" ${OpenBLAS_LIB})
#target_link_libraries(qedhf PRIVATE ${MPI_CXX_LIBRARIES} Python::Python pybind11::pybind11 OpenMP::OpenMP_CXX)
#target_link_libraries(qedhf PRIVATE ${MPI_CXX_LIBRARIES} Python::Python pybind11::pybind11 OpenMP::OpenMP_CXX ${OpenBLAS_LIB})
target_link_libraries(qedhf PRIVATE ${MPI_CXX_LIBRARIES} pybind11::pybind11 OpenMP::OpenMP_CXX)


