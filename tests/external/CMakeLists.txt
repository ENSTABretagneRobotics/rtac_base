

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
