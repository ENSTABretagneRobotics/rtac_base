

set(target_name array_test_${PROJECT_NAME})
add_executable(${target_name}
    src/functions0.cpp
    src/functions1.cpp
    src/array_tests.cpp
)
target_link_libraries(${target_name}
    rtac_base
)
