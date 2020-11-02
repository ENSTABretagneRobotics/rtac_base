cmake_minimum_required(VERSION 3.10)
project(rtac_base_tests VERSION 0.1)

list(APPEND test_names
    types_test
    files_test
    algorithm_test
    misc_test
    pointcloud_test
    ply_files_test
    time_test
    sharedvector_test
)

if(TARGET happly)
    list(APPEND test_names
        ply_test
    )
endif()

list(APPEND test_deps
    rtac_base
)

foreach(name ${test_names})
    add_executable(${name} src/${name}.cpp)
    target_link_libraries(${name} ${test_deps})
endforeach(name)

enable_language(CUDA)

list(APPEND cuda_test_names
    cuda_vectors_test
)

foreach(name ${cuda_test_names})
    add_executable(${name} src/${name}.cu)
    target_link_libraries(${name} ${test_deps})
    target_compile_options(${name} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_61,code=sm_61>
    )
endforeach(name)



