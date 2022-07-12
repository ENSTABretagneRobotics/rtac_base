cmake_minimum_required(VERSION 3.10)
project(rtac_base VERSION 0.1)

option(WITH_CUDA "Compile and install cuda-dependent code." ON)

# This script fixes some issues related to LD_LIBRARY_PATH # DOES NOT WORK
# include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/rpath_config.cmake)
# Loading installation script (just loading function, not calling)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/rtac_installation.cmake)
# This script handles Doxygen documentation (add a BUILD_DOCUMENTATION option)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/rtac_build_docs.cmake)

find_package(Eigen3 REQUIRED)
find_package(PNG)

list(APPEND rtac_base_headers
    include/rtac_base/types/Handle.h
    include/rtac_base/types/Shape.h
    include/rtac_base/types/Rectangle.h
    include/rtac_base/types/Bounds.h
    include/rtac_base/types/common.h
    include/rtac_base/types/Pose.h
    include/rtac_base/types/Point.h
    include/rtac_base/types/Image.h
    include/rtac_base/types/PointCloudBase.h
    include/rtac_base/types/PointCloud.h
    include/rtac_base/types/Mesh.h
    include/rtac_base/types/MappedPointer.h
    include/rtac_base/types/Buildable.h
    include/rtac_base/types/BuildTarget.h
    include/rtac_base/types/CallbackQueue.h
    include/rtac_base/types/VectorView.h
    include/rtac_base/types/TuplePointer.h
    include/rtac_base/types/Array2D.h
    include/rtac_base/types/GridMap.h
    include/rtac_base/types/MappedGrid.h
    include/rtac_base/files.h
    include/rtac_base/time.h
    include/rtac_base/ply_files.h
    include/rtac_base/happly.h
    include/rtac_base/type_utils.h
    include/rtac_base/geometry.h
    include/rtac_base/interpolation.h
    include/rtac_base/cuda_defines.h
    include/rtac_base/nmea_utils.h
    include/rtac_base/navigation.h
    include/rtac_base/external/obj_codec.h
)

add_library(rtac_base SHARED
    src/types/BuildTarget.cpp
    src/files.cpp
    src/time.cpp
    src/ply_files.cpp
    src/external/obj_codec.cpp
)
set_target_properties(rtac_base PROPERTIES
    RTAC_PUBLIC_HEADERS "${rtac_base_headers}"
)
target_link_libraries(rtac_base
    PUBLIC
        Eigen3::Eigen
    PRIVATE
        stdc++fs
)
target_include_directories(rtac_base
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_compile_features(rtac_base PUBLIC cxx_std_17)

if(${WITH_CUDA})
    add_subdirectory(cuda)
endif()

if(PNG_FOUND) 
    message(STATUS "Found libpng, version ${PNG_VERSION_STRING}")
    target_include_directories(rtac_base PUBLIC ${PNG_INCLUDE_DIRS})
    target_link_libraries(rtac_base PUBLIC ${PNG_LIBRARIES})
    target_compile_definitions(rtac_base PUBLIC ${PNG_DEFINITIONS} RTAC_PNG)
    list(APPEND rtac_base_headers
        include/rtac_base/external/png_codec.h
    )
    set_target_properties(rtac_base PROPERTIES
        RTAC_PUBLIC_HEADERS "${rtac_base_headers}"
    )
endif()


add_subdirectory(tests)

rtac_install_target(rtac_base
    ADDITIONAL_CONFIG_FILES cmake/rtac_installation.cmake
)
# target_add_docs(rtac_base)




