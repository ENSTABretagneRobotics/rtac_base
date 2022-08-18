
set(target_name vector_view_${PROJECT_NAME})
add_executable(${target_name}
    src/vector_view.cpp
    src/vector_view.cu
)
target_link_libraries(${target_name} PRIVATE
    rtac_cuda
)
set_target_properties(${target_name} PROPERTIES CUDA_ARCHITECTURES native)
