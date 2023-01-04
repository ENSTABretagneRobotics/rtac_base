list(APPEND container_test_files
    src/vectorview.cpp
    src/hostvector.cpp
)

foreach(filename ${container_test_files})
    get_filename_component(name ${filename} NAME_WE)
    set(target_name ${name}_container_${PROJECT_NAME})
    add_executable(${target_name} ${filename})
    target_link_libraries(${target_name} PRIVATE rtac_base)
endforeach()
