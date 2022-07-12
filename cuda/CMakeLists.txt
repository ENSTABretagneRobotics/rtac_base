cmake_minimum_required(VERSION 3.10)
project(rtac_cuda VERSION 0.1 LANGUAGES CUDA CXX)

if(NOT TARGET rtac_base)
    find_package(rtac_base REQUIRED)
endif()

list(APPEND rtac_cuda_headers
    include/rtac_base/cuda/utils.h
    include/rtac_base/cuda/DeviceVector.h
    include/rtac_base/cuda/HostVector.h
    include/rtac_base/cuda/DeviceMesh.h
    include/rtac_base/cuda/DeviceObject.h
    include/rtac_base/cuda/Texture2D.h
    include/rtac_base/cuda/functors.h
    include/rtac_base/cuda/FunctorCompound.h
    include/rtac_base/cuda/Mapping.h
    include/rtac_base/cuda/mappings.h
    include/rtac_base/cuda/reductions.hcu
    include/rtac_base/cuda/operators.h
    include/rtac_base/cuda/algorithm.hcu
)

find_package(CUDA REQUIRED)

add_library(rtac_cuda SHARED
    src/utils.cu
)
set_target_properties(rtac_cuda PROPERTIES
    RTAC_PUBLIC_HEADERS "${rtac_cuda_headers}"
)
target_include_directories(rtac_cuda PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${CUDA_TOOLKIT_INCLUDE}
)
target_compile_options(rtac_cuda PUBLIC
    # $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_61,code=sm_61>
    $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_61,code=sm_61 --expt-relaxed-constexpr>
)
target_link_libraries(rtac_cuda PUBLIC
    ${CUDA_LIBRARIES}
    rtac_base
)

add_subdirectory(tests)

rtac_install_target(rtac_cuda)

