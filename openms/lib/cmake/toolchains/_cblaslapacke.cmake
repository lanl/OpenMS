set(BLAS_LIBRARIES "-lcblas" "-lblas" CACHE STRING "BLAS libraries")
set(LAPACK_LIBRARIES "-llapacke" "-lcblas" "-llapack" "-lblas" CACHE STRING "LAPACK libraries")
set(LAPACK_COMPILE_DEFINITIONS "MADNESS_LINALG_USE_LAPACKE" CACHE STRING "LAPACK preprocessor definitions")
