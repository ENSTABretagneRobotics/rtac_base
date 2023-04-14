cmake_minimum_required(VERSION 3.16)
project(rtac_base VERSION 0.2)

option(WITH_CUDA   "Compile and install cuda-dependent code." ON)
option(WITH_ASYNC  "Compile and install async components (will use GCC '-pthread' compile and link option)" ON)
option(BUILD_TESTS "Build unit tests" ON)
option(BUILD_DOCS  "Build Doxygen documentation" OFF)

# Loading installation script (just loading function, not calling)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/rtac_installation.cmake)

find_package(Eigen3 3.4 REQUIRED)
find_package(yaml-cpp REQUIRED)

find_package(PNG)
find_package(JPEG)

list(APPEND rtac_base_headers
    include/rtac_base/cuda_defines.h
    include/rtac_base/Exception.h
    include/rtac_base/utilities.h

    include/rtac_base/types/Shape.h
    include/rtac_base/types/Rectangle.h
    include/rtac_base/types/Bounds.h
    include/rtac_base/types/Linspace.h
    include/rtac_base/types/Point.h
    include/rtac_base/types/PointFormat.h
    include/rtac_base/types/Complex.h
    include/rtac_base/types/PODWrapper.h

    include/rtac_base/containers/VectorView.h
    include/rtac_base/containers/HostVector.h
    include/rtac_base/containers/Image.h
    include/rtac_base/containers/DimExpression.h
    include/rtac_base/containers/ScaledImage.h
    include/rtac_base/containers/utilities.h

    include/rtac_base/type_utils.h
    include/rtac_base/types/MappedPointer.h
    include/rtac_base/types/BuildTarget.h
    include/rtac_base/types/CallbackQueue.h

    include/rtac_base/types/PointCloudBase.h
    include/rtac_base/types/PointCloud.h
    include/rtac_base/types/Mesh.h
    include/rtac_base/types/SonarPing2D.h
    include/rtac_base/types/SonarPing.h
    include/rtac_base/types/Edge.h
    include/rtac_base/types/DualMesh.h

    include/rtac_base/types/common.h
    include/rtac_base/types/Pose.h
    include/rtac_base/geometry.h
    include/rtac_base/interpolation.h
    include/rtac_base/interpolation_impl.h
    include/rtac_base/signal_helpers.h

    include/rtac_base/files.h
    include/rtac_base/time.h
    include/rtac_base/bin_codecs.h

    include/rtac_base/external/ply_files.h
    include/rtac_base/external/happly.h
    include/rtac_base/external/obj_codec.h
    include/rtac_base/external/ImageCodec.h
    include/rtac_base/external/tinyxml2.h
    include/rtac_base/external/metashape.h
    include/rtac_base/external/kalibr.h
    include/rtac_base/external/nmea_utils.h

    include/rtac_base/navigation.h
)

add_library(rtac_base SHARED
    src/types/BuildTarget.cpp
    src/files.cpp
    src/time.cpp
    src/bin_codecs.cpp
    src/Mesh.cpp

    src/external/nmea_utils.cpp
    src/external/ply_files.cpp
    src/external/obj_codec.cpp
    src/external/ImageCodec.cpp
    src/external/tinyxml2.cpp
    src/external/metashape.cpp
    src/external/kalibr.cpp
)
target_link_libraries(rtac_base
    PUBLIC
        Eigen3::Eigen
        yaml-cpp
    PRIVATE
        stdc++fs
)
target_include_directories(rtac_base
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_compile_features(rtac_base PUBLIC cxx_std_14)
# target_compile_features(rtac_base PUBLIC cxx_std_17)
target_compile_definitions(rtac_base PUBLIC 
    # $<$<COMPILE_LANGUAGE:CXX>:RTAC_CXX_STD=17>
    $<$<COMPILE_LANGUAGE:CXX>:RTAC_CXX_STD=14>
)

if(WITH_ASYNC)
    list(APPEND rtac_base_headers
        include/rtac_base/async/AsyncFunction.h
        include/rtac_base/async/AsyncWorker.h
    )
    target_sources(rtac_base PRIVATE src/async/AsyncWorker.cpp)
    target_compile_options(rtac_base PUBLIC -pthread)
    target_link_options   (rtac_base PUBLIC -pthread)
endif()

if(WITH_CUDA)
    target_compile_definitions(rtac_base PUBLIC "-DRTAC_CUDA_ENABLED")
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

find_package(PROJ QUIET CONFIG)
if(TARGET PROJ::proj)
    set(WITH_PROJ ON)
    target_link_libraries(rtac_base PUBLIC PROJ::proj)
    list(APPEND rtac_base_headers
        include/rtac_base/external/proj.h
    )
    target_sources(rtac_base PRIVATE
        src/external/proj.cpp
    )
endif()



if(BUILD_TESTS)
    add_subdirectory(tests)
endif()

if(BUILD_DOCS)
    # This script handles Doxygen documentation (add a BUILD_DOCUMENTATION option)
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/rtac_build_docs.cmake)
    rtac_build_docs(rtac_base)
endif()

rtac_install_target(rtac_base
    HEADER_FILES      ${rtac_base_headers}
    ADDITIONAL_CONFIG_FILES cmake/rtac_installation.cmake
    ADDITIONAL_CONFIG_FILES cmake/rtac_generate_asm.cmake
    ADDITIONAL_CONFIG_FILES cmake/rtac_build_docs.cmake
    ADDITIONAL_CONFIG_FILES cmake/Doxyfile.in
)




