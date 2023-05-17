

if(${PNG_FOUND})
    set(target_name png_codec_${PROJECT_NAME})
    add_executable(${target_name}
        src/png_codec.cpp
    )
    target_link_libraries(${target_name} PRIVATE rtac_base)
endif()

if(${JPEG_FOUND})
    set(target_name jpg_codec_${PROJECT_NAME})
    add_executable(${target_name}
        src/jpg_codec.cpp
    )
    target_link_libraries(${target_name} PRIVATE rtac_base)
endif()

if(${JPEG_FOUND} AND ${PNG_FOUND})
    set(target_name image_codec_${PROJECT_NAME})
    add_executable(${target_name}
        src/image_codec.cpp
    )
    target_link_libraries(${target_name} PRIVATE rtac_base)
endif()

set(target_name kalibr_${PROJECT_NAME})
add_executable(${target_name}
    src/kalibr_test.cpp
)
target_link_libraries(${target_name} PRIVATE rtac_base)

if(TARGET PROJ::proj)
    set(target_name proj_test_${PROJECT_NAME})
    add_executable(${target_name}
        src/proj_test.cpp
    )
    target_link_libraries(${target_name} PRIVATE rtac_base)
endif()

set(target_name npy_${PROJECT_NAME})
add_executable(${target_name}
    src/npy_test.cpp
)
target_link_libraries(${target_name} PRIVATE rtac_base)
