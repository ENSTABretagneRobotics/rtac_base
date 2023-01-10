

set(RTAC_BUILD_DOCS_DIR ${CMAKE_CURRENT_LIST_DIR})

function(rtac_build_docs TARGET_NAME)
    find_package(Doxygen REQUIRED)

    if(NOT DOXYGEN_FOUND)
        message(FATAL_ERROR "BUILD_DOCUMENTATION option is enabled but "
                            "Doxygen was not found. Please install Doxygen or "
                            "set -DBUILD_DOCUMENTATION to off")
    endif()

    get_target_property(DOXYGEN_EXECUTABLE Doxygen::doxygen IMPORTED_LOCATION)
    message(STATUS "DOXYGEN_EXECUTABLE : ${DOXYGEN_EXECUTABLE}")
    
    set(DOXYFILE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile)
    configure_file(${RTAC_BUILD_DOCS_DIR}/Doxyfile.in ${DOXYFILE_PATH} @ONLY)

    # Generating documentation in PRE_BUILD mode so the documentation is ready
    # if we have an issue with the build of TARGET_NAME itself.
    add_custom_command(TARGET ${TARGET_NAME} PRE_BUILD  
                       COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYFILE_PATH}
                       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                       COMMENT "Building documentation for ${TARGET_NAME}"
                       VERBATIM USES_TERMINAL)
endfunction()


