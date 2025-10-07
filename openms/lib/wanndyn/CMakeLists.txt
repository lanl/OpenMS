cmake_minimum_required(VERSION 3.10)
project(WE2PD VERSION 1.0 LANGUAGES CXX)

# --- C++ standard ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Let users choose static/shared via -DBUILD_SHARED_LIBS=ON/OFF (default OFF)
# add_library will honor BUILD_SHARED_LIBS automatically.

# --- Sources/headers ---
set(WE2PD_HEADERS
    constants.hpp
    wanndyn.hpp
    we2pd_banner.hpp
)
set(WE2PD_SOURCES
    wanndyn.cpp
)

# --- Library target ---
add_library(we2pd ${WE2PD_SOURCES} ${WE2PD_HEADERS})
add_library(we2pd::we2pd ALIAS we2pd)

target_include_directories(we2pd
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_compile_features(we2pd PUBLIC cxx_std_17)

# --- Example executable (links to the library) ---
add_executable(we2pd_example example.cpp)
target_link_libraries(we2pd_example PRIVATE we2pd)

# Optional: make the executable name consistent across platforms
set_target_properties(we2pd_example PROPERTIES OUTPUT_NAME we2pd)

# --- (Optional) tests ---
option(WE2PD_BUILD_TESTS "Build test targets" OFF)
if(WE2PD_BUILD_TESTS)
    enable_testing()
    add_executable(we2pd_smoketest tests/smoketest.cpp)
    target_link_libraries(we2pd_smoketest PRIVATE we2pd)
    add_test(NAME Smoke COMMAND we2pd_smoketest)
endif()

# --- (Optional) install rules ---
# install(TARGETS we2pd
#     EXPORT we2pdTargets
#     ARCHIVE DESTINATION lib
#     LIBRARY DESTINATION lib
#     RUNTIME DESTINATION bin
# )
# install(FILES ${WE2PD_HEADERS} DESTINATION include)
# install(EXPORT we2pdTargets NAMESPACE we2pd:: DESTINATION lib/cmake/we2pd)
