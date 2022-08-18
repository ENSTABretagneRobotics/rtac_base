

set(target_name mapping_test_${PROJECT_NAME})
add_executable(${target_name}
    src/mapping_test.cpp
    src/mapping_test.cu
)
target_link_libraries(${target_name} PRIVATE
    rtac_cuda
)
set_target_properties(${target_name} PROPERTIES CUDA_ARCHITECTURES native)


