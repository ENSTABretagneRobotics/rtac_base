list(APPEND test_files
    src/misc_test.cpp
    src/files_test.cpp
    src/mathtypes_test.cpp
    src/geometry_test.cpp
    src/ply_files_test.cpp
    src/image_test.cpp
    src/mesh.cpp
    src/pointcloud_test.cpp
    src/mappedpointer_test.cpp
    src/buildtarget_test.cpp
    src/vector_view.cpp
    src/complex_test.cpp
    src/bounds_test.cpp
    src/ppmformat_test.cpp
    src/nmea_utils.cpp
    src/navigation_test.cpp
)

list(APPEND test_deps
    rtac_base
)

find_package(matplotlib_cpp QUIET)
if(TARGET matplotlib_cpp::matplotlib_cpp)
    list(APPEND test_files src/interpolation_test.cpp )
    list(APPEND test_deps matplotlib_cpp::matplotlib_cpp)
else()
    message(WARNING "matplotlib_cpp not found : cannot build interpolation_test")
endif()

foreach(filename ${test_files})
    get_filename_component(name ${filename} NAME_WE)
    set(target_name ${name}_${PROJECT_NAME})

    add_executable(${target_name} ${filename})
    target_link_libraries(${target_name} ${test_deps})
endforeach()

add_subdirectory(external)
