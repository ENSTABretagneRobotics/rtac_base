
function(rtac_generate_asm TARGET_NAME)


    set(multiValueArgs SOURCES LINK_LIBRARIES)
    cmake_parse_arguments(ARGUMENT 
                          "${options}"
                          "${oneValueArgs}"
                          "${multiValueArgs}"
                           ${ARGN})

    add_library(${TARGET_NAME} OBJECT ${ARGUMENT_SOURCES})
    target_link_libraries(${TARGET_NAME} PRIVATE ${ARGUMENT_LINK_LIBRARIES})

    get_target_property(target_bin_dir ${TARGET_NAME} BINARY_DIR)
    set(object_files_dir ${target_bin_dir})
    cmake_path(GET CMAKE_FILES_DIRECTORY RELATIVE_PART cmake_files_dir)
    cmake_path(APPEND object_files_dir ${cmake_files_dir} ${TARGET_NAME}.dir)
    foreach(source ${ARGUMENT_SOURCES})
        set(object_path ${object_files_dir})
        cmake_path(APPEND object_path ${source}.o)
        set(asm_output_path ${target_bin_dir})
        cmake_path(APPEND asm_output_path ${TARGET_NAME}_asm ${source}.asm)
        get_filename_component(asm_output_dir ${asm_output_path} DIRECTORY)

        ########### the add_custom_command with ALWAYS run if the OUTPUT is not
        ########### generated ! The DEPENDS is there to rebuild the OUTPUT if
        ########### it changed.
        add_custom_command(OUTPUT ${asm_output_path}
                           DEPENDS ${object_path}
                           COMMAND ${CMAKE_COMMAND} -E make_directory ${asm_output_dir}
                           COMMAND objdump -d ${object_path} > ${asm_output_path}
                           COMMENT "Disassembling ${source}")
        list(APPEND generated_asm_files ${asm_output_path})
    endforeach()
    set(asm_target ${TARGET_NAME}_ASM)
    add_custom_target(${asm_target} ALL DEPENDS ${generated_asm_files})
    add_dependencies(${asm_target} ${TARGET_NAME})

endfunction()
