

function(target_add_docs TARGET_NAME)
    find_package(Doxygen REQUIRED)

    if(NOT DOXYGEN_FOUND)
        message(FATAL_ERROR "BUILD_DOCUMENTATION option is enabled but "
                            "Doxygen was not found. Please install Doxygen or "
                            "set -DBUILD_DOCUMENTATION to off")
    endif()

    get_target_property(DOXYGEN_EXECUTABLE Doxygen::doxygen IMPORTED_LOCATION)
    message(STATUS "DOXYGEN_EXECUTABLE : ${DOXYGEN_EXECUTABLE}")

    # Genewrating documentation in PRE_BUILD mode so the documentation is ready
    # if we have an issue with the build of TARGET_NAME itself.
    add_custom_command(TARGET ${TARGET_NAME} PRE_BUILD  
                       COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile
                       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                       COMMENT "Building documentation for ${TARGET_NAME}"
                       VERBATIM USES_TERMINAL)
endfunction()


