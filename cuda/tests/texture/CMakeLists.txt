

set(target_name texture_test_${PROJECT_NAME})
add_executable(${target_name}
    src/texture_test.cpp
    src/texture_test.cu
)
target_link_libraries(${target_name} PRIVATE
    rtac_cuda
)

