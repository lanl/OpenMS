# finds BLAS, LAPACK and, if ENABLE_SCALAPACK=TRUE, ScaLAPACK
# include(FetchWfn91LinAlgModules) if want to use NWChemEx/WFN91's linear algebra discovery modules

if (ENABLE_SCALAPACK)
    find_package(ScaLAPACK REQUIRED)
    # Propagate ScaLAPACK -> BLAS/LAPACK if not set
    # (ScaLAPACK necessarily contains a BLAS/LAPACK linker by standard)
    # TODO: Tell David to write a macro that hides this verbosity from user space
    if (NOT BLAS_LIBRARIES)
        set(BLAS_LIBRARIES "${ScaLAPACK_LIBRARIES}" CACHE STRING "BLAS LIBRARIES")
    endif ()
    if (NOT LAPACK_LIBRARIES)
        set(LAPACK_LIBRARIES "${ScaLAPACK_LIBRARIES}" CACHE STRING "LAPACK LIBRARIES")
    endif ()
else (ENABLE_SCALAPACK)
    find_package(LAPACK REQUIRED)
    # Propagate LAPACK -> BLAS if not set
    # (LAPACK necessarily contains a BLAS linker by standard)
    # TODO: Tell David to write a macro that hides this verbosity from user space
    if (NOT BLAS_LIBRARIES)
        set(BLAS_LIBRARIES "${LAPACK_LIBRARIES}" CACHE STRING "BLAS LIBRARIES")
    endif ()
endif (ENABLE_SCALAPACK)
