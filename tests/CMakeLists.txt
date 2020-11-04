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
    vectormap_test
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



