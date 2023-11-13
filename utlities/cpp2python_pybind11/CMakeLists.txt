project(my_module)
cmake_minimum_required(VERSION 3.12)

set(CMAKE_CXX_STANDARD 14)

include(FetchContent)

FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.8.1  # specify the version/tag
)
FetchContent_MakeAvailable(pybind11)


# Find required packages
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Python COMPONENTS Interpreter Development REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)


#pybind11_add_module(my_module _my_module.cpp)
pybind11_add_module(my_module MODULE
src/_my_module.cpp
src/myclass.cpp
src/bindings.cpp)

set_target_properties(
  my_module
  PROPERTIES
  #PREFIX ""
  OUTPUT_NAME _my_module
)



#####

# Define the target
add_library(mpi_example SHARED
src/mpi_example.cpp)

# Set properties for the target
set_target_properties(mpi_example PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

# Include directories
target_include_directories(mpi_example PRIVATE ${MPI_INCLUDE_PATH} ${Python_INCLUDE_DIRS} ${pybind11_INCLUDE_DIRS})

# Link libraries
target_link_libraries(mpi_example PRIVATE ${MPI_CXX_LIBRARIES} Python::Python pybind11::pybind11)
target_link_libraries(mpi_example PRIVATE ${MPI_CXX_LIBRARIES} Python::Python pybind11::pybind11 OpenMP::OpenMP_CXX)

# Add compile options
target_compile_options(mpi_example PRIVATE -Wall -O3)

# Define the output file name
# Retrieve the extension suffix for the Python module (e.g., .so, .pyd)
execute_process(COMMAND "${Python_EXECUTABLE}" -c
                "from distutils.sysconfig import get_config_var; print(get_config_var('EXT_SUFFIX'))"
                OUTPUT_VARIABLE Python_EXTENSION_SUFFIX
                OUTPUT_STRIP_TRAILING_WHITESPACE)
#set_target_properties(mpi_example PROPERTIES PREFIX "" ) #SUFFIX "${Python_EXTENSION_SUFFIX}")
set_target_properties(mpi_example PROPERTIES PREFIX "" SUFFIX "${Python_EXTENSION_SUFFIX}")

