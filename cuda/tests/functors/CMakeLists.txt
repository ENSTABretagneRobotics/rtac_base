

set(target_name ${PROJECT_NAME}_functors_test)
add_executable(${target_name}
    src/functors_test.cpp
    src/functors_test.cu
)
target_link_libraries(${target_name} PRIVATE
    rtac_cuda
)
set_target_properties(${target_name} PROPERTIES CUDA_ARCHITECTURES native)

