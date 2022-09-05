
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

    # get_target_property(target_bin_dir ${TARGET_NAME} BINARY_DIR)
    # set(object_files_dir ${target_bin_dir})
    # cmake_path(GET CMAKE_FILES_DIRECTORY RELATIVE_PART cmake_files_dir)
    # cmake_path(APPEND object_files_dir ${cmake_files_dir} ${TARGET_NAME}.dir)
    # foreach(source ${ARGUMENT_SOURCES})
    #     set(object_path ${object_files_dir})
    #     cmake_path(APPEND object_path ${source}.o)
    #     set(ptx_output_path ${target_bin_dir})
    #     cmake_path(APPEND ptx_output_path ${TARGET_NAME}_ptx ${source}.ptx)
    #     get_filename_component(ptx_output_dir ${ptx_output_path} DIRECTORY)

    #     ########### the add_custom_command with ALWAYS run if the OUTPUT is not
    #     ########### generated ! The DEPENDS is there to rebuild the OUTPUT if
    #     ########### it changed.
    #     add_custom_command(OUTPUT ${ptx_output_path}
    #                        DEPENDS ${object_path}
    #                        COMMAND ${CMAKE_COMMAND} -E make_directory ${ptx_output_dir}
    #                        COMMAND objdump -d ${object_path} > ${ptx_output_path}
    #                        COMMENT "Disassembling ${source}")
    #     list(APPEND generated_ptx_files ${ptx_output_path})
    # endforeach()
    # set(ptx_target ${TARGET_NAME}_PTX)
    # add_custom_target(${ptx_target} ALL DEPENDS ${generated_ptx_files})
    # add_dependencies(${ptx_target} ${TARGET_NAME})

endfunction()
