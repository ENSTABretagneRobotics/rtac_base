function(rtac_generate_ptx TARGET_NAME)

    set(multiValueArgs SOURCES LINK_LIBRARIES)
    cmake_parse_arguments(ARGUMENT 
                          "${options}"
                          "${oneValueArgs}"
                          "${multiValueArgs}"
                           ${ARGN})

    add_library(${TARGET_NAME} OBJECT ${ARGUMENT_SOURCES})
    target_link_libraries(${TARGET_NAME} PRIVATE ${ARGUMENT_LINK_LIBRARIES})
    set_target_properties(${TARGET_NAME} PROPERTIES CUDA_PTX_COMPILATION ON)
    target_compile_options(${TARGET_NAME} PUBLIC
        $<$<COMPILE_LANGUAGE:CUDA>: ${ARGUMENTS_CUDA_OPTIONS}>
    )

endfunction()
