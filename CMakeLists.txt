cmake_minimum_required(VERSION 3.10)
project(rtac_tools VERSION 0.1)

find_package(Eigen3 REQUIRED)
find_package(happly REQUIRED)

list(APPEND rtac_tools_headers
    include/rtac_tools/types/common.h
    include/rtac_tools/types/Pose.h
    include/rtac_tools/types/Mesh.h
    include/rtac_tools/files.h
)

add_library(rtac_tools SHARED
    src/types/common.cpp
    src/files.cpp
)
set_target_properties(rtac_tools PROPERTIES
    PUBLIC_HEADER "${rtac_tools_headers}"
)
target_link_libraries(rtac_tools
    PUBLIC
        Eigen3::Eigen
    PRIVATE
        stdc++fs
)
if(TARGET happly)
    target_link_libraries(rtac_tools PUBLIC happly)
    target_compile_definitions(rtac_tools PUBLIC RTAC_TOOLS_PLY_FILES)
else()
    message(WARNING "The happly target was not found. You won't be able to read/write .ply file.")
endif()
target_include_directories(rtac_tools
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_compile_features(rtac_tools PRIVATE cxx_std_17)

add_subdirectory(tests)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/installation.cmake)
install_target(rtac_tools
    ADDITIONNAL_CONFIG_FILES cmake/installation.cmake
)




