cmake_minimum_required(VERSION 3.10)
project(rtac_base VERSION 0.1)

find_package(Eigen3 REQUIRED)
find_package(happly REQUIRED)

list(APPEND rtac_base_headers
    include/rtac_base/types/common.h
    include/rtac_base/types/Pose.h
    include/rtac_base/types/PointCloud.h
    include/rtac_base/types/Mesh.h
    include/rtac_base/types/Shape.h
    include/rtac_base/types/Rectangle.h
    include/rtac_base/algorithm.h
    include/rtac_base/geometry.h
    include/rtac_base/files.h
    include/rtac_base/misc.h
)

add_library(rtac_base SHARED
    src/types/common.cpp
    src/files.cpp
    src/misc.cpp
)
set_target_properties(rtac_base PROPERTIES
    PUBLIC_HEADER "${rtac_base_headers}"
)
target_link_libraries(rtac_base
    PUBLIC
        Eigen3::Eigen
    PRIVATE
        stdc++fs
)
if(TARGET happly)
    target_link_libraries(rtac_base PUBLIC happly)
    target_compile_definitions(rtac_base PUBLIC RTAC_BASE_PLY_FILES)
else()
    message(WARNING "The happly target was not found. You won't be able to read/write .ply file.")
endif()
target_include_directories(rtac_base
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_compile_features(rtac_base PRIVATE cxx_std_17)

add_subdirectory(tests)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/installation.cmake)
install_target(rtac_base
    ADDITIONNAL_CONFIG_FILES cmake/installation.cmake
)




