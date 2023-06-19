cmake_minimum_required(VERSION 3.16)
project(rtac_cuda VERSION 1.0 LANGUAGES CUDA CXX)

if(NOT TARGET rtac_base)
    find_package(rtac_base REQUIRED)
endif()

list(APPEND rtac_cuda_headers
    include/rtac_base/cuda/utils.h
    include/rtac_base/cuda/geometry.h
    include/rtac_base/cuda/CudaVector.h
    include/rtac_base/cuda/TextureVector.h
    include/rtac_base/cuda/TextureDim.h
    include/rtac_base/cuda/PinnedVector.h
    include/rtac_base/cuda/DeviceMesh.h
    include/rtac_base/cuda/CudaPing.h
    include/rtac_base/cuda/DeviceReference.h
    include/rtac_base/cuda/DeviceObject.h
    include/rtac_base/cuda/Texture2D.h
    include/rtac_base/cuda/TextureView2D.h
    include/rtac_base/cuda/reductions.hcu
    include/rtac_base/cuda/operators.h
    include/rtac_base/cuda/algorithm.hcu
    include/rtac_base/cuda/vec_math.h

    include/rtac_base/cuda/DeviceDualMesh.h

    include/rtac_base/cuda/texture_utils.h
    include/rtac_base/cuda/vector_utils.h
)

find_package(CUDAToolkit REQUIRED)
if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.24.0")
    set(CMAKE_CUDA_ARCHITECTURES native)
elseif(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.23.0")
    set(CMAKE_CUDA_ARCHITECTURES all)
else()
    set(CMAKE_CUDA_ARCHITECTURES "52")
endif()

add_library(rtac_cuda SHARED
    src/utils.cu
    src/texture_utils.cu
    src/vector_utils.cu
    src/DeviceDualMesh.cu
)
set_target_properties(rtac_cuda PROPERTIES
    RTAC_PUBLIC_HEADERS "${rtac_cuda_headers}"
)
target_include_directories(rtac_cuda PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_options(rtac_cuda PUBLIC
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)
if("${CUDAToolkit_VERSION}" VERSION_LESS "11.0.0")
    target_compile_definitions(rtac_cuda PUBLIC 
        $<$<COMPILE_LANGUAGE:CUDA>:RTAC_CXX_STD=14>
    )
else()
    target_compile_definitions(rtac_cuda PUBLIC
        $<$<COMPILE_LANGUAGE:CUDA>:RTAC_CXX_STD=17>
    )
endif()

target_link_libraries(rtac_cuda PUBLIC
    CUDA::cudart
    rtac_base
)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/rtac_generate_ptx.cmake)

if(BUILD_TESTS)
    add_subdirectory(tests)
endif()

rtac_install_target(rtac_cuda
    HEADER_FILES ${rtac_cuda_headers}
    ADDITIONAL_CONFIG_FILES cmake/rtac_generate_ptx.cmake
)

