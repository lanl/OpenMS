cmake_minimum_required(VERSION 3.10)  # Ensure CMake version compatibility
project(WE2PD VERSION 1.0 LANGUAGES Fortran)

# Enable Fortran
enable_language(Fortran)

# Add directories containing .mod files
include_directories(${CMAKE_Fortran_MODULE_DIRECTORY})

# Set the sources
set(SOURCES
    parameters.f90
    printHeader.f90
    wanndyn.f90
)


# Create the library
add_library(${PROJECT_NAME} ${SOURCES})

# Specify the executable target and its sources
add_executable(we2pd.exe example.f90 ${SOURCES})

# Enable testing if desired
enable_testing()

# Define a basic test (assuming MyApp outputs something verifiable)
# add_test(NAME MyAppTest COMMAND MyApp)


# Add the module directory to the target's include directories
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_Fortran_MODULE_DIRECTORY})
