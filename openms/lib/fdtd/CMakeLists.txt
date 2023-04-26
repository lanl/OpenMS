
#========================
# find gsl lib
#========================
include(ExternalProject)

message("Debug: CAMEK_PREFIX_PATH is ${CMAKE_PREFIX_PATH}")
set(CMAKE_MODULE_PATH ${CMAKE_PREFIX_PATH})

include(FindGSL)

## maybe a better idea is to let user to install the gsl and provide it via LD_LIBRARY_PATH
##find_package(GSL COMPONENTS libgsl libgslcblas PATHS ${CMAKE_PREFIX_PATH} QUIET)
if(GSL_FOUND)
    message("GSL found!")
    message("GSL_LIBRARIES=" ${GSL_LIBRARIES})
    message("GSL_LIBRARY="  ${GSL_LIBRARY})
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
    include(FindGSL)
endif()

#set(gsl_lib "${PROJECT_SOURCE_DIR}/deps/lib/libgsl.a")
#set(gsl_cblas "${PROJECT_SOURCE_DIR}/deps/lib/libgslcblas.a")

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

add_library(fdtd SHARED ${SOURCES-C})

set_target_properties(fdtd PROPERTIES
	LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/deps/lib)

target_link_libraries(fdtd  ${GSL_LIBRARIES} gsl gslcblas) 
#target_link_libraries(fdtd ${gsl_lib} ${gsl_cblas} ) 

install(TARGETS fdtd
	ARCHIVE DESTINATION ${PROJECT_SOURCE_DIR}/deps/lib
	LIBRARY DESTINATION ${PROJECT_SOURCE_DIR}/deps/lib)

install(FILES ${HEADERS-C} DESTINATION ${PROJECT_SOURCE_DIR}/deps/include)


#----------------------------------
# build python binding lib
#----------------------------------

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  set(CMAKE_C_FLAGS "-c -O4")
  #set(CMAKE_C_FLAGS "-c -fPIC -O4")
endif()

# Note: pFDTD_wrap.c is generated from pFDTD.i, don't change it!!!!!!
set(WRAPPER-C
    pFDTD_wrap.c
)

include_directories(${Python3_INCLUDE_DIRS})

# Create _fdtdc Python extension module
add_library(_fdtdc SHARED ${SOURCES-C} ${WRAPPER-C})

#add_custom_command(TARGET _fdtdc COMMAND swig -python ${CMAKE_CURRENT_SOURCE_DIR}/pFDTD.i)
#add_custom_command(TARGET _fdtdc POST_BUILD COMMAND swig -python ${CMAKE_CURRENT_SOURCE_DIR}/pFDTD.i)

# Link against GSL library
target_link_libraries(_fdtdc ${GSL_LIBRARIES} gsl gslcblas)


# Install _fdtdc Python extension module
set_target_properties(_fdtdc PROPERTIES PREFIX "")
set_target_properties(_fdtdc PROPERTIES OUTPUT_NAME "_fdtdc")

install(TARGETS _fdtdc
	ARCHIVE DESTINATION ${PROJECT_SOURCE_DIR}/fdtd/
	LIBRARY DESTINATION ${PROJECT_SOURCE_DIR}/fdtd)

# -----------not used ------------
#FIND_PACKAGE(SWIG REQUIRED)
#INCLUDE(${SWIG_USE_FILE})

#FIND_PACKAGE(PythonLibs)
#INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_PATH})

#SET(CMAKE_SWIG_FLAGS "")

#SET_SOURCE_FILES_PROPERTIES(pFDTD.i PROPERTIES CPLUSPLUS ON)
#SET_SOURCE_FILES_PROPERTIES(pFDTD.i PROPERTIES SWIG_FLAGS "-includeall")
#SWIG_ADD_MODULE(pFDTD python pFDTD.i ${SOURCES-C})
#SWIG_LINK_LIBRARIES(pFDTD ${PYTHON_LIBRARIES})

