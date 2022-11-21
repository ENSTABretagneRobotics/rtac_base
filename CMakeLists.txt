cmake_minimum_required(VERSION 3.16)
project(rtac_base VERSION 0.1)

option(WITH_CUDA   "Compile and install cuda-dependent code." ON)
option(BUILD_TESTS "Build unit tests" ON)

# Loading installation script (just loading function, not calling)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/rtac_installation.cmake)
# This script handles Doxygen documentation (add a BUILD_DOCUMENTATION option)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/rtac_build_docs.cmake)

find_package(Eigen3 3.4 REQUIRED)
find_package(PNG)
find_package(JPEG)


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
    include/rtac_base/files.h
    include/rtac_base/time.h
    include/rtac_base/ply_files.h
    include/rtac_base/happly.h
    include/rtac_base/type_utils.h
    include/rtac_base/geometry.h
    include/rtac_base/interpolation.h
    include/rtac_base/interpolation_impl.h
    include/rtac_base/cuda_defines.h
    include/rtac_base/nmea_utils.h
    include/rtac_base/navigation.h

    include/rtac_base/signal_helpers.h

    include/rtac_base/external/obj_codec.h
    include/rtac_base/external/ImageCodec.h

    include/rtac_base/types/Complex.h
)

add_library(rtac_base SHARED
    src/types/BuildTarget.cpp
    src/files.cpp
    src/time.cpp
    src/ply_files.cpp

    src/external/obj_codec.cpp
    src/external/ImageCodec.cpp
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

if(WITH_CUDA)
    add_subdirectory(cuda)
endif()

if(PNG_FOUND) 
    message(STATUS "Found libpng, version ${PNG_VERSION_STRING}")
    target_include_directories(rtac_base PUBLIC ${PNG_INCLUDE_DIRS})
    target_link_libraries(rtac_base PUBLIC ${PNG_LIBRARIES})
    target_compile_definitions(rtac_base PUBLIC ${PNG_DEFINITIONS} RTAC_PNG)
    target_sources(rtac_base PRIVATE
        src/external/png_codec.cpp
    )
    list(APPEND rtac_base_headers
        include/rtac_base/external/png_codec.h
    )
    set_target_properties(rtac_base PROPERTIES
        RTAC_PUBLIC_HEADERS "${rtac_base_headers}"
    )
endif()

if(JPEG_FOUND) 
    message(STATUS "Found libjpeg, version ${JPEG_VERSION}")
    target_link_libraries(rtac_base PUBLIC JPEG::JPEG)
    target_compile_definitions(rtac_base PUBLIC RTAC_JPEG)
    target_sources(rtac_base PRIVATE
        src/external/jpg_codec.cpp
    )
    list(APPEND rtac_base_headers
        include/rtac_base/external/jpg_codec.h
    )
    set_target_properties(rtac_base PROPERTIES
        RTAC_PUBLIC_HEADERS "${rtac_base_headers}"
    )
endif()

if(BUILD_TESTS)
    add_subdirectory(tests)
endif()

rtac_install_target(rtac_base
    HEADER_FILES      ${rtac_base_headers}
    ADDITIONAL_CONFIG_FILES cmake/rtac_installation.cmake
    ADDITIONAL_CONFIG_FILES cmake/rtac_generate_asm.cmake
)
# target_add_docs(rtac_base)




