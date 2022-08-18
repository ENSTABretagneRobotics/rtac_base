

set(target_name tuplepointer_${PROJECT_NAME})
add_executable(${target_name}
    src/tuplepointer.cpp
    src/tuplepointer.cu
)
target_link_libraries(${target_name} PRIVATE rtac_cuda)
set_target_properties(${target_name} PROPERTIES CUDA_ARCHITECTURES native)

