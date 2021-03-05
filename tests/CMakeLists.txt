
list(APPEND test_names
    misc_test.cpp
    files_test.cpp
    mathtypes_test.cpp
    geometry_test.cpp
    type_utils_test.cpp
    ply_files_test.cpp
    mesh.cpp
    # ply_test.cpp
    pointcloud_test.cpp
    sharedvector_test.cpp
    mappedpointer_test.cpp
    buildables_test.cpp
)

list(APPEND test_deps
    rtac_base
)

foreach(name ${test_names})

    # Generating a unique target name to avoid name colision with other rtac
    # packages.
    get_filename_component(executable_name ${name} NAME_WE)
    set(test_target_name ${PROJECT_NAME}_test_${executable_name})

    add_executable(${test_target_name} src/${name})
    target_link_libraries(${test_target_name} ${test_deps})
    set_target_properties(${test_target_name} PROPERTIES OUTPUT_NAME ${executable_name})

endforeach(name)



