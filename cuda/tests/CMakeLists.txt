
list(APPEND test_names
    src/devicevector_test.cpp
    src/devicemesh_test.cpp
    src/linear_algebra.cpp
    src/image_test.cpp
    src/cuda_devicevector_test.cu
    src/reductions_test.cu
    src/pinned_vector.cu
)

foreach(filename ${test_names})
    get_filename_component(test_name ${filename} NAME_WE)
    set(target_name ${test_name}_${PROJECT_NAME})
    add_executable(${target_name} ${filename})
    target_link_libraries(${target_name} rtac_cuda)
    # set_target_properties(${target_name} PROPERTIES CUDA_ARCHITECTURES native)
endforeach()

add_subdirectory(texture)
add_subdirectory(functors)
add_subdirectory(mapping)
add_subdirectory(vector_view)
add_subdirectory(tuple_pointer)
add_subdirectory(grid_map)
add_subdirectory(vector_float_nd)


