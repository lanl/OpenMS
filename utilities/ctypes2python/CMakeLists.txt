cmake_minimum_required(VERSION 3.0)
project(ExampleLibrary)

# Set the C standard
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED True)

# Create a shared library from example.c
add_library(example SHARED example.c)

# Set the output name of the library to libexample
set_target_properties(example PROPERTIES OUTPUT_NAME "example")

