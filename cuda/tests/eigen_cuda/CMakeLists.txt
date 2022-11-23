

set(target_name eigen_cuda_${PROJECT_NAME})
add_executable(${target_name}
    src/eigen_cuda.cu
    src/eigen_cuda.cpp
)
target_link_libraries(${target_name} PUBLIC rtac_cuda)
rtac_generate_ptx(eigen_cuda_ptx 
    SOURCES src/eigen_cuda.cu
    LINK_LIBRARIES rtac_cuda
)
