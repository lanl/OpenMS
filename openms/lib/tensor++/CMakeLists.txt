# Set default install location to parent of source directory (../)

get_filename_component(PARENT_DIR "${CMAKE_CURRENT_SOURCE_DIR}" DIRECTORY)
set(CMAKE_INSTALL_PREFIX "${PARENT_DIR}" CACHE PATH "Install path prefix" FORCE)


cmake_minimum_required(VERSION 3.14)
project(tensorlib LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# =============================================
# build option
# =============================================

# option(BUILD_EXAMPLES "Build example executables" OFF)
option(BUILD_EXAMPLES "Build example executables" ON)
option(ENABLE_OPENMP "Enable OpenMP" OFF)

# Find pybind11 using FetchContent (no install required)
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.12.0
)
FetchContent_MakeAvailable(pybind11)


# =============================================
# BLAS backend detection
# =============================================

# =============================================
# OpenMP
# =============================================
if (ENABLE_OPENMP)
    # Manually set OpenMP (macOS + LLVM)
    # set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I${HOMEBREW_PREFIX}/include")
    # set(OpenMP_C_LIB_NAMES "omp")
    # set(OpenMP_omp_LIBRARY "${HOMEBREW_PREFIX}/lib/libomp.dylib")
    find_package(OpenMP REQUIRED)
endif()


# =============================================
# Eigen
# =============================================

# Locate Eigen
# Require user to provide Eigen3 path explicitly
if(NOT DEFINED EIGEN3_INCLUDE_DIR)
    message(FATAL_ERROR "You must provide the Eigen3 include path using -DEIGEN3_INCLUDE_DIR=/path/to/eigen3")
endif()

message(STATUS "Using Eigen3 include path: ${EIGEN3_INCLUDE_DIR}")


# =============================================
# Include header files
# =============================================
include_directories(include)

# =============================================
# Gather source files for the core library
# =============================================
file(GLOB_RECURSE LIB_SOURCES
    src/Tile.cpp
    src/TiledTensor.cpp
    src/TileOps_Eigen.cpp
    src/EinsumPlanner.cpp
    src/Einsum.cpp
)

# =============================================
# Build and link the core library
# =============================================
add_library(tensorlib STATIC ${LIB_SOURCES})

if (ENABLE_OPENMP)
    target_link_libraries(tensorlib PUBLIC OpenMP::OpenMP_CXX)
endif()

# target_include_directories(tensorlib PUBLIC
#    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
#    $<INSTALL_INTERFACE:include>
#)
target_include_directories(tensorlib PUBLIC ${EIGEN3_INCLUDE_DIR})

# =============================================
# Install the core library
# =============================================
install(TARGETS tensorlib
        EXPORT tensorlibTargets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)

install(DIRECTORY include/ DESTINATION include)

# =============================================
# Build Python bindings (module)
# =============================================
add_subdirectory(python)


# =============================================
# Optionally build examples
# =============================================
message("=====================================")
message("BUILD_EXAMPLES = ${BUILD_EXAMPLES}")
message("=====================================")

if(BUILD_EXAMPLES)
    message("=====================================")
    message(STATUS "Building selected examples...")
    message("=====================================")

    # file(GLOB EXAMPLE_SOURCES "example/*.cpp")
    # Manually list example source files
    set(EXAMPLE_SOURCES
        examples/example_2dtile.cpp
        examples/example_3dtile.cpp
        examples/example_tiledtensor.cpp
        # examples/einsum_batch.cpp
        # examples/test_cpu_vs_gpu.cpp
    )

    foreach(EXAMPLE_SRC IN LISTS EXAMPLE_SOURCES)
        get_filename_component(EXAMPLE_NAME ${EXAMPLE_SRC} NAME_WE)

        add_executable(${EXAMPLE_NAME} ${EXAMPLE_SRC})
        target_link_libraries(${EXAMPLE_NAME} PRIVATE tensorlib)

        if(${EXAMPLE_NAME} STREQUAL "test_cpu_vs_gpu")
            target_link_libraries(${EXAMPLE_NAME} PRIVATE tensorlib_cuda)
        endif()

        # Ensure example builds with the correct include path
        target_include_directories(${EXAMPLE_NAME} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/include
        )

        set_target_properties(${EXAMPLE_NAME} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/examples"
        )
    endforeach()
endif()
