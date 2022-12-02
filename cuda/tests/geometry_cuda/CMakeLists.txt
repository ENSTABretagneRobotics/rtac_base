

set(target_name geometry_${PROJECT_NAME})
add_executable(${target_name}
    src/geometry_cuda.cu
    src/geometry_cuda.cpp
)
target_link_libraries(${target_name} PRIVATE rtac_cuda)
rtac_generate_ptx(${target_name}_ptx
    SOURCES src/geometry_cuda.cu
    LINK_LIBRARIES rtac_cuda
)
