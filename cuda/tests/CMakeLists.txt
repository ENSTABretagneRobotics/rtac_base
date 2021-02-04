
list(APPEND test_names
    devicevector_test
    devicemesh_test
)

list(APPEND test_deps
    rtac_cuda
)

foreach(name ${test_names})
    add_executable(${name} src/${name}.cpp)
    target_link_libraries(${name} ${test_deps})
endforeach(name)

list(APPEND cuda_test_names
    cuda_devicevector_test
    cuda_sharedvector_test
)

foreach(name ${cuda_test_names})

    # Generating a unique target name to avoid name colision with other rtac
    # packages.
    get_filename_component(executable_name ${name} NAME_WE)
    set(test_target_name ${PROJECT_NAME}_test_${executable_name})

    add_executable(${test_target_name} src/${name})
    target_link_libraries(${test_target_name} ${test_deps})
    set_target_properties(${test_target_name} PROPERTIES OUTPUT_NAME ${executable_name})

endforeach(name)


