cmake_minimum_required(VERSION 3.10)
project(rtac_tools_tests VERSION 0.1)

list(APPEND test_names
    types-test
    files-test
    # xtensor-blas-test
)

list(APPEND test_deps
    rtac_tools
)

foreach(name ${test_names})
    add_executable(${name} src/${name}.cpp)
    target_link_libraries(${name} ${test_deps})
endforeach(name)






