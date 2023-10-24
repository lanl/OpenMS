cmake_minimum_required(VERSION 3.15.0) # need list(PREPEND for toolchains

###############################################################################
# Bring Group cmake toolkit (TODO)
###############################################################################

include(external/versions.cmake)
include(FetchContent)

if (DEFINED PROJECT_BINARY_DIR)
    set(CMAKE_KIT_PREFIX_DIR PROJECT_BINARY_DIR)
else ()
    set(CMAKE_KIT_PREFIX_DIR CMAKE_CURRENT_BINARY_DIR)
endif ()
#FetchContent_Declare(
#        cmake_kit
#        QUIET
#        GIT_REPOSITORY https://github.com/ValeevGroup/kit-cmake.git
#        GIT_TAG ${QEDSQA_TRACKED_CMAKEKIT_TAG}
#        SOURCE_DIR ${${CMAKE_KIT_PREFIX_DIR}}/cmake/
#        BINARY_DIR ${${CMAKE_KIT_PREFIX_DIR}}/cmake/build
#        SUBBUILD_DIR ${${CMAKE_KIT_PREFIX_DIR}}/cmake/subbuild
#)
#FetchContent_MakeAvailable(cmake_kit)
#list(APPEND CMAKE_MODULE_PATH "${cmake_kit_SOURCE_DIR}/modules")

# Set project version
set(QedSQA_MAJOR_VERSION 0)
set(QedSQA_MINOR_VERSION 0)
set(QedSQA_MICRO_VERSION 1)
set(QedSQA_PRERELEASE_ID alpha.1)
set(QedSQA_VERSION "${QedSQA_MAJOR_VERSION}.${QedSQA_MINOR_VERSION}.${QedSQA_MICRO_VERSION}")
if (QedSQA_PRERELEASE_ID)
    set(QedSQA_EXT_VERSION "${QedSQA_VERSION}-${QedSQA_PRERELEASE_ID}")
else (QedSQA_PRERELEASE_ID)
    set(QedSQA_EXT_VERSION "${QedSQA_VERSION}")
endif (QedSQA_PRERELEASE_ID)

## extract git metadata
#include(GetGitMetadata)
#cmake_git_metadata()


# make QedSQA project
project(QedSQA LANGUAGES CXX VERSION "${QedSQA_VERSION}" DESCRIPTION "SEcond QUANTization toolkit")

# need C++17, insist on strict standard
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ ISO Standard version")
if (NOT (CMAKE_CXX_STANDARD EQUAL 17 OR CMAKE_CXX_STANDARD EQUAL 20))
    message(FATAL_ERROR "C++ 2017 ISO Standard or higher is required to compile QedSQA")
endif ()
# C++20 is only configurable via compile features with cmake 3.12 and older
if (CMAKE_CXX_STANDARD EQUAL 20 AND CMAKE_VERSION VERSION_LESS 3.12.0)
    cmake_minimum_required(VERSION 3.12.0)
endif ()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF CACHE BOOL "Whether to use extensions of C++  ISO Standard version")

# Enable ccache if not already enabled by symlink masquerading
if (NOT CMAKE_CXX_COMPILER MATCHES ".*/ccache$")
    find_program(CCACHE_EXECUTABLE ccache)
    if (CCACHE_EXECUTABLE)
        mark_as_advanced(CCACHE_EXECUTABLE)
        message(STATUS "Found ccache: ${CCACHE_EXECUTABLE}")
        set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_EXECUTABLE}" CACHE STRING "Compiler launcher to use for compiling C++")
        set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_EXECUTABLE}" CACHE STRING "Compiler launcher to use for compiling C")
    else ()
        set(CCACHE_EXECUTABLE "")
    endif ()
endif ()

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules/")

set(FETCHCONTENT_UPDATES_DISCONNECTED OFF CACHE BOOL "Enables UPDATE_DISCONNECTED behavior for all content population")
include(FetchContent)
#include(AddCustomTargetSubproject)
include(FeatureSummary)
include(CMakePackageConfigHelpers)

##########################
# Standard build variables
##########################
set(QedSQA_INSTALL_BINDIR "bin"
        CACHE PATH "QedSQA BIN install directory")
set(QedSQA_INSTALL_INCLUDEDIR "include"
        CACHE PATH "QedSQA INCLUDE install directory")
set(QedSQA_INSTALL_LIBDIR "lib"
        CACHE PATH "QedSQA LIB install directory")
set(QedSQA_INSTALL_SHAREDIR "share/qedsqa/${QedSQA_MAJOR_VERSION}.${QedSQA_MINOR_VERSION}.${QedSQA_MICRO_VERSION}"
        CACHE PATH "QedSQA SHARE install directory")
set(QedSQA_INSTALL_DATADIR "${QedSQA_INSTALL_SHAREDIR}/data"
        CACHE PATH "QedSQA DATA install directory")
set(QedSQA_INSTALL_DOCDIR "${QedSQA_INSTALL_SHAREDIR}/doc"
        CACHE PATH "QedSQA DOC install directory")
set(QedSQA_INSTALL_CMAKEDIR "lib/cmake/qedsqa"
        CACHE PATH "QedSQA CMAKE install directory")


##########################
# QedSQA package options
##########################





##########################
# Prerequisites
##########################

# std::thread requires (on some platforms?) thread support which is not
# provided automatically
find_package(Threads REQUIRED)

# PSTL (used by g++-9 and clang++-8 in c++17 mode) needs TBB
if ((${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU" AND
        ${CMAKE_CXX_COMPILER_VERSION} VERSION_GREATER 9) OR
(${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang" AND
        ${CMAKE_CXX_COMPILER_VERSION} VERSION_GREATER 8))
    if (ENABLE_TBB)  # but user must enable the search for TBB since this is an additional source of build entropy
        find_package(TBB REQUIRED)
        # TBB::tbb by default is not GLOBAL, so to allow users of LINALG_LIBRARIES to safely use it we need to make it global
        # more discussion here: https://gitlab.kitware.com/cmake/cmake/-/issues/17256
        set_target_properties(TBB::tbb PROPERTIES IMPORTED_GLOBAL TRUE)
    endif (ENABLE_TBB)
endif ()

# check of <execution> header is usable

#include(CheckCXXFeatures)
#check_cxx_execution_header(QedSQA)

# Ranges-V3

#include(FindOrFetchRangeV3)

#  # need header-only Boost + (compiled) Boost.Regex and Boost.Locale
#  # NB Boost.Container is broken in v1.70
#  if (NOT TARGET Boost::boost OR NOT TARGET Boost::regex OR NOT TARGET Boost::locale)
#      find_package(Boost ${QEDSQA_TRACKED_BOOST_VERSION} CONFIG COMPONENTS regex locale)
#      if (Boost_VERSION_STRING VERSION_LESS ${QEDSQA_TRACKED_BOOST_VERSION} OR NOT TARGET Boost::boost OR NOT TARGET Boost::regex OR NOT TARGET Boost::locale)
#          find_package(Boost ${QEDSQA_TRACKED_BOOST_VERSION} REQUIRED COMPONENTS regex locale)
#          message(STATUS "Found Boost (version ${Boost_VERSION}) via FindBoost module")
#          set(Boost_USE_CONFIG FALSE)
#      else ()
#          message(STATUS "Found Boost (version ${Boost_VERSION}) via CONFIG ${Boost_CONFIG}")
#          set(Boost_USE_CONFIG TRUE)
#      endif ()
#      if (Boost_VERSION_MACRO EQUAL 107000)
#          message(FATAL_ERROR "Found Boost 1.70, but its Boost.Container is broken: "
#                  "use any version ${QEDSQA_TRACKED_BOOST_VERSION} "
#                  "or higher, other than 1.70")
#      endif ()
#      # Boost.Move is broken in 1.77 and 1.78 unless using c++20
#      # fixed in 1.79 via https://github.com/boostorg/move/commit/78f26da1f3a5a3831e9e70efe83f9c56eef94e8c
#      if (CMAKE_CXX_STANDARD LESS 20)
#          if (Boost_VERSION_MACRO GREATER_EQUAL 107700 AND Boost_VERSION_MACRO LESS 107900)
#              message(FATAL_ERROR "Found Boost 1.77 <= version < 1.79, but its Boost.Move is broken with pre-C++20: use a version older than 1.77 or newer than 1.78")
#          endif ()
#      endif ()
#  endif ()


message("project binary dir=" ${PROJECT_BINARY_DIR})

set(QedSQA_src
        #src/version.hpp
        ${PROJECT_BINARY_DIR}/src/version.hpp
        src/version.cpp
        src/index.cpp
        src/index.hpp
        src/expression.cpp
        src/expression.hpp
        src/term.cpp
        src/term.hpp
        src/operator.cpp
        src/operator.hpp
	src/convenience.hpp
	src/convenience.cpp
	)


add_library(QedSQA
        ${QedSQA_src}
	)

#target_link_libraries(QedSQA PUBLIC range-v3::range-v3 Boost::regex Boost::locale Boost::boost QedSQA-bliss Threads::Threads)


set_source_files_properties(
        src/version.cpp
        PROPERTIES COMPILE_DEFINITIONS
        "QEDSQA_GIT_REVISION=\"${QEDSQA_GIT_REVISION}\";QEDSQA_GIT_DESCRIPTION=\"${QEDSQA_GIT_DESCRIPTION}\""
	)




##########################
# export QedSQA
##########################

configure_file(
        ${PROJECT_SOURCE_DIR}/src/version.hpp.in
        ${PROJECT_BINARY_DIR}/src/version.hpp
)
install(FILES ${PROJECT_BINARY_DIR}/src/version.hpp
        DESTINATION "${QEDSQA_INSTALL_INCLUDEDIR}/QedSQA")


