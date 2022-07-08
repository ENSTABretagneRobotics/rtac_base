

set(target_name array_scale_${PROJECT_NAME})
add_executable(${target_name}
    src/array_scale.cu
)
target_link_libraries(${target_name} PRIVATE rtac_cuda)

