
list(APPEND test_names
    src/devicevector_test.cpp
    src/devicemesh_test.cpp
    src/linear_algebra.cpp
    src/image_test.cpp
    src/cuda_devicevector_test.cu
    src/reductions_test.cu
    src/pinned_vector.cu
    src/texture1d.cu
    src/constant_object.cu
    src/scaledimage.cu
    src/texture_function.cu
    src/texture_vector.cu
    src/texture_float2.cu
    src/point_format.cu
    src/ping_type.cu
)

foreach(filename ${test_names})
    get_filename_component(test_name ${filename} NAME_WE)
    set(target_name ${test_name}_${PROJECT_NAME})
    add_executable(${target_name} ${filename})
    target_link_libraries(${target_name} rtac_cuda)
    # set_target_properties(${target_name} PROPERTIES CUDA_ARCHITECTURES native)
endforeach()

add_subdirectory(texture)
add_subdirectory(vector_view)
add_subdirectory(eigen_cuda)
add_subdirectory(geometry_cuda)


