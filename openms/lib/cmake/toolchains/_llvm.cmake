# these cache/environment variables control the behavior of this module:
# CLANG_VERSION (env) : if defined, will look for clang-$ENV{CLANG_VERSION} instead of clang; ditto for clang++

############## Only usable on UNIX platforms
if (NOT UNIX)
  message(FATAL_ERROR "_llvm.cmake is only usable on UNIX platforms")
endif(NOT UNIX)

# on MacOS assume Brew
if (APPLE)
  if (DEFINED ENV{CLANG_VERSION})
    if ("$ENV{CLANG_VERSION}" AND EXISTS "/usr/local/bin/clang-$ENV{CLANG_VERSION}")
      set(_cmake_c_compiler /usr/local/bin/clang-$ENV{CLANG_VERSION})
      set(_cmake_cxx_compiler /usr/local/bin/clang++-$ENV{CLANG_VERSION})
    endif()
  elseif (EXISTS /usr/local/bin/clang)
    set(_cmake_c_compiler /usr/local/bin/clang)
    set(_cmake_cxx_compiler /usr/local/bin/clang++)
  endif()
endif(APPLE)
# if no special definition found, assume it's in PATH
if (NOT DEFINED _cmake_c_compiler)
  if (DEFINED ENV{CLANG_VERSION})
    set(_cmake_c_compiler clang-$ENV{CLANG_VERSION})
  else()
    set(_cmake_c_compiler clang)
  endif()
endif(NOT DEFINED _cmake_c_compiler)
if (NOT DEFINED _cmake_cxx_compiler)
  if (DEFINED ENV{CLANG_VERSION})
    set(_cmake_cxx_compiler clang++-$ENV{CLANG_VERSION})
  else()
    set(_cmake_cxx_compiler clang++)
  endif()
endif(NOT DEFINED _cmake_cxx_compiler)

set(CMAKE_C_COMPILER "${_cmake_c_compiler}" CACHE STRING "C compiler")
set(CMAKE_CXX_COMPILER "${_cmake_cxx_compiler}" CACHE STRING "C++ compiler")
