

set(target_name vector_float_nd_${PROJECT_NAME})
add_executable(${target_name}
    src/vector_float_nd.cu
    src/vector_float_nd.cpp
)
target_link_libraries(${target_name} rtac_cuda)
set_target_properties(${target_name} PROPERTIES CUDA_ARCHITECTURES native)
