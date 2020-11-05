cmake_minimum_required(VERSION 3.10)
project(rtac_cuda VERSION 0.1 LANGUAGES CUDA CXX)

if(NOT TARGET rtac_base)
    find_package(rtac_base REQUIRED)
endif()

list(APPEND rtac_cuda_headers
    include/rtac_base/cuda/utils.h
    include/rtac_base/cuda/DeviceVector.h
    include/rtac_base/cuda/HostVector.h
)

add_library(rtac_cuda SHARED
    src/utils.cu
)
set_target_properties(rtac_cuda PROPERTIES
    PUBLIC_HEADER "${rtac_cuda_headers}"
)
target_include_directories(rtac_cuda PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_options(rtac_cuda PUBLIC
    $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_61,code=sm_61>
)
target_link_libraries(rtac_cuda PUBLIC rtac_base)

add_subdirectory(tests)

rtac_install_target(rtac_cuda)

