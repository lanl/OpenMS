
#========================
# find gsl lib
#========================
include(ExternalProject)

message("Debug: CAMEK_PREFIX_PATH is ${CMAKE_PREFIX_PATH}")
set(CMAKE_MODULE_PATH ${CMAKE_PREFIX_PATH})

# maybe a better idea is to let user to install the gsl and provide it via LD_LIBRARY_PATH
find_package(GSL COMPONENTS libgsl libgslcblas PATHS ${CMAKE_PREFIX_PATH} QUIET)
if(GSL_FOUND)
    message("GSL found: path is " ${GSL_LIBRARIES})
else()
    message(WARNING "GSL library not found. It will be downloaded and built")
    ExternalProject_Add(gsl
        URL ftp://ftp.gnu.org/gnu/gsl/gsl-2.4.tar.gz
	PREFIX ${PROJECT_BINARY_DIR}/gsl
        INSTALL_DIR ${PROJECT_SOURCE_DIR}/deps/
        CONFIGURE_COMMAND ../gsl/configure --prefix=<INSTALL_DIR>
        BUILD_COMMAND make
        INSTALL_COMMAND make install
        TEST_COMMAND make check
    )
endif()

set(gsl_lib "${PROJECT_SOURCE_DIR}/deps/lib/libgsl.a")
set(gsl_cblas "${PROJECT_SOURCE_DIR}/deps/lib/libgslcblas.a")

set(HEADERS-C
pFDTDvar.h
pFDTD.h
)

set(SOURCES-C
input.c
incoherent.c
farfield.c
energy.c
output.c
metalobj.c
memory.c
libpFDTD.a
transmap.c
timeupdate.c
source.c
sigma.c
parameter.c)

add_library(fdtd SHARED
  ${SOURCES-C})


set_target_properties(fdtd PROPERTIES
	LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/deps/lib)

target_link_libraries(fdtd ${gsl_lib} ${gsl_cblas} ) 

install(TARGETS fdtd
	ARCHIVE DESTINATION ${PROJECT_SOURCE_DIR}/deps/lib
	LIBRARY DESTINATION ${PROJECT_SOURCE_DIR}/deps/lib)

install(FILES ${HEADERS-C} DESTINATION ${PROJECT_SOURCE_DIR}/deps/include)

## swig python interface(todo)
#FIND_PACKAGE(SWIG REQUIRED)
#INCLUDE(${SWIG_USE_FILE})

#FIND_PACKAGE(PythonLibs)
#INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_PATH})

#SET(CMAKE_SWIG_FLAGS "")

#SET_SOURCE_FILES_PROPERTIES(pFDTD.i PROPERTIES CPLUSPLUS ON)
#SET_SOURCE_FILES_PROPERTIES(pFDTD.i PROPERTIES SWIG_FLAGS "-includeall")
#SWIG_ADD_MODULE(pFDTD python pFDTD.i ${SOURCES-C})
#SWIG_LINK_LIBRARIES(pFDTD ${PYTHON_LIBRARIES})

