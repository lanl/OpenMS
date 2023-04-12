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


# find gsl libs (todo)
set(gsl_lib "/home/zhy/local/gsl-2.4/lib/libgsl.a")
set(gsl_cblas "/home/zhy/local/gsl-2.4/lib/libgslcblas.a")

set_target_properties(fdtd PROPERTIES
	LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/deps/lib)

target_link_libraries(fdtd ${gsl_lib} ${gsl_cblas} ) 

install(TARGETS fdtd
	ARCHIVE DESTINATION ${PROJECT_SOURCE_DIR}/deps/lib
	LIBRARY DESTINATION ${PROJECT_SOURCE_DIR}/deps/lib)

install(FILES ${HEADERS-C} DESTINATION ${PROJECT_SOURCE_DIR}/deps/include)


# swig python interface(todo)
#FIND_PACKAGE(SWIG REQUIRED)
#INCLUDE(${SWIG_USE_FILE})
#
#FIND_PACKAGE(PythonLibs)
#INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_PATH})
#
#SET(CMAKE_SWIG_FLAGS "")
#
#SET_SOURCE_FILES_PROPERTIES(pFDTD.i PROPERTIES CPLUSPLUS ON)
#SET_SOURCE_FILES_PROPERTIES(pFDTD.i PROPERTIES SWIG_FLAGS "-includeall")
#SWIG_ADD_MODULE(pFDTD python pFDTD.i ${SOURCES-C})
#SWIG_LINK_LIBRARIES(pFDTD ${PYTHON_LIBRARIES})

