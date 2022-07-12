

set(target_name ndpoint_test_${PROJECT_NAME})
add_executable(${target_name} 
    src/ndpoint.cpp
    src/ndpoint_test.cpp
)
target_link_libraries(${target_name} PRIVATE rtac_base)
