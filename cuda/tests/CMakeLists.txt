cmake_minimum_required(VERSION 3.10)
project(perex_cuda_tests VERSION 0.1 LANGUAGES CUDA CXX)

list(APPEND test_names
    devicevector_test
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
    add_executable(${name} src/${name}.cu)
    target_link_libraries(${name} ${test_deps})
endforeach(name)


