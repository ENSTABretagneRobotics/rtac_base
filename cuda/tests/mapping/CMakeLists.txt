

set(target_name ${PROJECT_NAME}_mapping_test)
add_executable(${target_name}
    src/mapping_test.cpp
    src/mapping_test.cu
)
target_link_libraries(${target_name} PRIVATE
    rtac_cuda
)

