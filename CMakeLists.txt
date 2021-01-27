cmake_minimum_required(VERSION 3.10)
project(rtac_base VERSION 0.1)

option(WITH_CUDA "Compile and install cuda-dependent code." ON)

find_package(Eigen3 REQUIRED)
find_package(happly REQUIRED)
# Loading installation script (just loading function, not calling)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/rtac_installation.cmake)

list(APPEND rtac_base_headers
    include/rtac_base/types/Handle.h
    include/rtac_base/types/common.h
    include/rtac_base/types/Pose.h
    include/rtac_base/types/PointCloudBase.h
    include/rtac_base/types/PointCloud.h
    include/rtac_base/types/Mesh.h
    include/rtac_base/types/Shape.h
    include/rtac_base/types/Rectangle.h
    include/rtac_base/types/SharedVector.h
    include/rtac_base/algorithm.h
    include/rtac_base/geometry.h
    include/rtac_base/files.h
    include/rtac_base/time.h
    include/rtac_base/ply_files.h
)

add_library(rtac_base SHARED
    src/files.cpp
    src/time.cpp
    src/ply_files.cpp
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
    list(APPEND ADDITIONAL_CONFIG_COMMANDS
        "find_package(happly REQUIRED)"
    )
else()
    message(WARNING "The happly target was not found. You won't be able to read/write .ply file.")
endif()
target_include_directories(rtac_base
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_compile_features(rtac_base PRIVATE cxx_std_17)

if(${WITH_CUDA})
    add_subdirectory(cuda)
endif()

add_subdirectory(tests)

rtac_install_target(rtac_base
    ADDITIONAL_CONFIG_COMMANDS ${ADDITIONAL_CONFIG_COMMANDS}
    ADDITIONAL_CONFIG_FILES cmake/rtac_installation.cmake
)




