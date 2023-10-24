# BLAS++ / LAPACK++

if (NOT TARGET blaspp)
    find_package(blaspp QUIET CONFIG)

    if (TARGET blaspp)
        message(STATUS "Found blaspp CONFIG at ${blaspp_CONFIG}")
    else (TARGET blaspp)
        cmake_minimum_required(VERSION 3.14.0)  # for FetchContent_MakeAvailable
        include(FetchContent)
        include(${CMAKE_CURRENT_LIST_DIR}/versions.cmake)
        FetchContent_Declare(blaspp
                GIT_REPOSITORY https://bitbucket.org/icl/blaspp.git
                GIT_TAG ${VGCMAKEKIT_TRACKED_BLASPP_TAG}
                )

        FetchContent_MakeAvailable(blaspp)

        # set blaspp_CONFIG to the install location so that we know where to find it
        set(blaspp_CONFIG ${CMAKE_INSTALL_PREFIX}/lib/blaspp/blasppConfig.cmake)
    endif (TARGET blaspp)
endif (NOT TARGET blaspp)

if (NOT TARGET lapackpp)
    find_package(OpenMP QUIET) #XXX Open LAPACKPP issue for this...
    find_package(lapackpp QUIET CONFIG)
    if (TARGET lapackpp)
        message(STATUS "Found lapackpp CONFIG at ${lapackpp_CONFIG}")
    else (TARGET lapackpp)
        cmake_minimum_required(VERSION 3.14.0)  # for FetchContent_MakeAvailable
        include(FetchContent)
        include(${CMAKE_CURRENT_LIST_DIR}/versions.cmake)
        FetchContent_Declare(lapackpp
                GIT_REPOSITORY https://bitbucket.org/icl/lapackpp.git
                GIT_TAG ${VGCMAKEKIT_TRACKED_LAPACKPP_TAG}
                )

        FetchContent_MakeAvailable(lapackpp)

        # set lapackpp_CONFIG to the install location so that we know where to find it
        set(lapackpp_CONFIG ${CMAKE_INSTALL_PREFIX}/lib/lapackpp/lapackppConfig.cmake)
    endif (TARGET lapackpp)
endif (NOT TARGET lapackpp)

##################### Introspect BLAS/LAPACK libs

# Check if BLAS/LAPACK is MKL
include(CheckFunctionExists)
include(CMakePushCheckState)
cmake_push_check_state(RESET)
set(CMAKE_REQUIRED_LIBRARIES "${blaspp_libraries}" m)
check_function_exists(mkl_dimatcopy BLAS_IS_MKL)
cmake_pop_check_state()

# blaspp_header library is a target that permits #include'ing library-specific headers, e.g. mkl.h
if (NOT TARGET blaspp_headers)

    add_library(blaspp_headers INTERFACE)

    if (BLAS_IS_MKL)
        foreach (_lib ${blaspp_libraries})
            if (EXISTS ${_lib} AND _lib MATCHES libmkl_)
                string(REGEX REPLACE "/lib/(intel64_lin/|intel64/|)libmkl_.*" "" _mklroot "${_lib}")
            elseif (_lib MATCHES "^-L")
                string(REGEX REPLACE "^-L" "" _mklroot "${_lib}")
                string(REGEX REPLACE "/lib(/intel64_lin|/intel64|)(/|)" "" _mklroot "${_mklroot}")
            endif ()
            if (_mklroot)
                break()
            endif (_mklroot)
        endforeach ()

        set(_mkl_include)
        if (EXISTS "${_mklroot}/include")
            set(_mkl_include "${_mklroot}/include")
        elseif (EXISTS "/usr/include/mkl") # ubuntu package
            set(_mkl_include "/usr/include/mkl")
        endif ()
        if (_mkl_include AND EXISTS "${_mkl_include}")
            target_include_directories(blaspp_headers INTERFACE "${_mkl_include}")
        endif (_mkl_include AND EXISTS "${_mkl_include}")

    endif (BLAS_IS_MKL)

    install(TARGETS blaspp_headers EXPORT blaspp_headers)
    export(EXPORT blaspp_headers FILE "${PROJECT_BINARY_DIR}/blaspp_headers-targets.cmake")
    install(EXPORT blaspp_headers
            FILE "blaspp_headers-targets.cmake"
            DESTINATION "lib/blaspp" # current install location of blaspp
            )

endif (NOT TARGET blaspp_headers)
