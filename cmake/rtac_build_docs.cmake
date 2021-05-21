

function(target_add_docs TARGET_NAME)
    find_package(Doxygen REQUIRED)

    if(NOT DOXYGEN_FOUND)
        message(FATAL_ERROR "BUILD_DOCUMENTATION option is enabled but "
                            "Doxygen was not found. Please install Doxygen or "
                            "set -DBUILD_DOCUMENTATION to off")
    endif()

    get_target_property(DOXYGEN_EXECUTABLE Doxygen::doxygen IMPORTED_LOCATION)
    # message(STATUS "DOXYGEN_EXECUTABLE : ${DOXYGEN_EXECUTABLE}")

    add_custom_target(${TARGET_NAME}_docs ALL
                      COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile
                      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                      COMMENT "Building documentation for ${TARGET_NAME}"
                      VERBATIM)
    add_dependencies(${TARGET_NAME}_docs ${TARGET_NAME})
endfunction()


