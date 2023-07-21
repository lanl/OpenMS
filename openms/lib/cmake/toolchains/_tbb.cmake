############## Only usable on UNIX platforms
if (NOT UNIX)
  message(FATAL_ERROR "_tbb.cmake is only usable on UNIX platforms")
endif(NOT UNIX)
if (DEFINED CMAKE_SYSTEM_NAME)
  set(_cmake_system_name ${CMAKE_SYSTEM_NAME})
else(DEFINED CMAKE_SYSTEM_NAME)
  set(_cmake_system_name ${CMAKE_HOST_SYSTEM_NAME})
endif(DEFINED CMAKE_SYSTEM_NAME)

############## Top tools dir
# prefer OneAPI
if (NOT INTEL_DIR)
  if (DEFINED ENV{INTEL_DIR})
    set(_intel_dir $ENV{INTEL_DIR})
  else(DEFINED ENV{INTEL_DIR})
    if (EXISTS /opt/intel/oneapi)
      set(_intel_dir /opt/intel/oneapi)
    elseif(EXISTS /opt/intel)
      set(_intel_dir /opt/intel)
    else ()
      set(_intel_dir )
    endif()
  endif(DEFINED ENV{INTEL_DIR})
  if (NOT EXISTS "${_intel_dir}")
    set(_intel_dir )
  endif()
  set(INTEL_DIR "${_intel_dir}" CACHE PATH "Intel tools root directory")
endif(NOT INTEL_DIR)

############## TBB
if (DEFINED ENV{TBBROOT})
  set(_tbb_root_dir "$ENV{TBBROOT}")
else(DEFINED ENV{TBBROOT})
  if (EXISTS "${INTEL_DIR}/tbb/latest") # OneAPI
    set(_tbb_root_dir "${INTEL_DIR}/tbb/latest")
  elseif (EXISTS "${INTEL_DIR}/tbb")
    set(_tbb_root_dir "${INTEL_DIR}/tbb")
  else ()
    set(_tbb_root_dir )
  endif()
endif(DEFINED ENV{TBBROOT})
set(TBB_ROOT_DIR "${_tbb_root_dir}" CACHE PATH "Intel TBB root directory")

if(${_cmake_system_name} MATCHES "Darwin")
  # libc++ subdir for legacy tbb
  set(TBB_LIBRARY_DIRS "${TBB_ROOT_DIR}/lib/libc++;${TBB_ROOT_DIR}/lib"
      CACHE PATH "Intel TBB library directory list")
elseif(${_cmake_system_name} MATCHES "Linux")
  if(EXISTS ${TBB_ROOT_DIR}/lib/intel64/gcc4.8) # Intel packaged TBB
    set(TBB_LIBRARY_DIRS "${TBB_ROOT_DIR}/lib/intel64/gcc4.8"
        CACHE PATH "Intel TBB library directory list")
  elseif (EXISTS ${TBB_ROOT_DIR}/lib/intel64/gcc4.7)  # Intel packaged TBB
    set(TBB_LIBRARY_DIRS "${TBB_ROOT_DIR}/lib/intel64/gcc4.7"
        CACHE PATH "Intel TBB library directory list")
  elseif(EXISTS ${TBB_ROOT_DIR}/lib/intel64/gcc4.4) # Intel packaged TBB
    set(TBB_LIBRARY_DIRS "${TBB_ROOT_DIR}/lib/intel64/gcc4.4"
        CACHE PATH "Intel TBB library directory list")
  elseif(EXISTS ${TBB_ROOT_DIR}/lib/intel64/gcc4.1) # Intel packaged TBB
    set(TBB_LIBRARY_DIRS "${TBB_ROOT_DIR}/lib/intel64/gcc4.1"
        CACHE PATH "Intel TBB library directory list")
  else() # Intel OSS TBB
    set(TBB_LIBRARY_DIRS "${TBB_ROOT_DIR}/lib"
        CACHE PATH "Intel TBB library directory list")
  endif()
else()
  set(TBB_LIBRARY_DIRS "${TBB_ROOT_DIR}/lib"
      CACHE PATH "Intel TBB library directory list")
endif()

set(ENABLE_TBB ON CACHE BOOL "Whether to enable the use of Intel TBB")
