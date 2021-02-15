

set(target_name ${PROJECT_NAME}_texture_test)
add_executable(${target_name}
    src/texture_test.cpp
    src/texture_test.cu
)
target_link_libraries(${target_name} PRIVATE
    rtac_cuda
)

