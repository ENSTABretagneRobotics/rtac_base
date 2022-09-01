function(rtac_install_target TARGET_NAME)

    set(oneValueArgs HEADERS_DIRECTORY NAMESPACE)
	set(multiValueArgs HEADER_FILES ADDITIONAL_CONFIG_FILES)
	cmake_parse_arguments(ARGUMENT              # parsed argument variable prefix
                          "${options}"          # empty
                          "${oneValueArgs}"     # HEADERS_DIRECTORY
	                      "${multiValueArgs}"   # HEADER_FILES ADDITIONAL_CONFIG_FILES
                          ${ARGN})

    # message(STATUS "=========== HEADERS_DIRECTORY       : ${ARGUMENT_HEADERS_DIRECTORY}")
    # message(STATUS "=========== NAMESPACE               : ${ARGUMENT_NAMESPACE}")
    # message(STATUS "=========== HEADER_FILES            : ${ARGUMENT_HEADER_FILES}")
    # message(STATUS "=========== ADDITIONAL_CONFIG_FILES : ${ARGUMENT_ADDITIONAL_CONFIG_FILES}")

	include(GNUInstallDirs)

    # RPATH related configuration (see https://gitlab.kitware.com/cmake/community/-/wikis/doc/cmake/RPATH-handling for details)
    set_target_properties(${TARGET_NAME} PROPERTIES
        SKIP_BUILD_RPATH FALSE
        BUILD_WITH_INSTALL_RPATH FALSE
        INSTALL_RPATH "${CMAKE_INSTALL_LIBDIR}"
        INSTALL_RPATH_USE_LINK_PATH TRUE
    )
    list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_LIBDIR}" isSystemDir)
    if("${isSystemDir}" STREQUAL "-1")
        set_target_properties(${TARGET_NAME} PROPERTIES
            INSTALL_RPATH "${CMAKE_INSTALL_LIBDIR}" # redundant with above ??
        )
    endif()

    set(export_name ${TARGET_NAME}Targets)
    install(TARGETS     ${TARGET_NAME}
            DESTINATION ${CMAKE_INSTALL_LIBDIR}
            EXPORT      ${export_name}
    )
    install(EXPORT ${export_name}
            FILE ${export_name}.cmake
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${TARGET_NAME}
    )

    foreach(config_file ${ARGUMENT_ADDITIONAL_CONFIG_FILES})
        get_filename_component(filename ${config_file} NAME)
        list(APPEND ADDITIONAL_CONFIG_FILES ${filename})
    endforeach()

    # Getting a version either from target or project
    get_target_property(target_version ${TARGET_NAME} VERSION)
    if(NOT target_version)
        set(target_version ${CMAKE_PROJECT_VERSION})
    endif()

    include(CMakePackageConfigHelpers)
    configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Config.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}Config.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${TARGET_NAME}
        PATH_VARS CMAKE_INSTALL_INCLUDEDIR CMAKE_INSTALL_LIBDIR
    )
    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}ConfigVersion.cmake
        VERSION ${target_version}
        COMPATIBILITY AnyNewerVersion
    )
    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}Config.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}ConfigVersion.cmake
        ${ARGUMENT_ADDITIONAL_CONFIG_FILES}
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${TARGET_NAME}
    )
    export(EXPORT ${export_name}
           FILE ${CMAKE_CURRENT_BINARY_DIR}/${export_name}.cmake
    )

    if(ARGUMENT_HEADERS_DIRECTORY)
        install(DIRECTORY ${ARGUMENT_HEADERS_DIRECTORY}
                DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        )
    endif()

    if(ARGUMENT_HEADER_FILES)
        foreach(header ${ARGUMENT_HEADER_FILES})
            get_filename_component(header_dir ${header} DIRECTORY)
            install(FILES ${header}
                    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/../${header_dir}")
        endforeach()
    endif()

endfunction()

