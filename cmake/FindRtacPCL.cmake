
find_package(PCL REQUIRED)

list(APPEND PCL_MODULE_LIST
	COMMON
	OCTREE
	IO
	KDTREE
	SEARCH
	SAMPLE_CONSENSUS
	FILTERS
	2D
	GEOMETRY
	FEATURES
	ML
	SEGMENTATION
	VISUALIZATION
	SURFACE
	REGISTRATION
	KEYPOINTS
	TRACKING
	RECOGNITION
	STEREO
	APPS
	IN_HAND_SCANNER
	MODELER
	POINT_CLOUD_EDITOR
	OUTOFCORE
	PEOPLE
)
# message(STATUS "${PCL_MODULE_LIST}")

foreach(module ${PCL_MODULE_LIST})
	if(${PCL_FOUND} AND NOT TARGET PCL::${module})
	    add_library(PCL::${module} INTERFACE IMPORTED)
	    # issue with zlib ? have to filter https://github.com/PointCloudLibrary/pcl/issues/2989
	    set(LIBRARIES "")
	    foreach(library ${PCL_${module}_LIBRARIES})
	        if(${library} STREQUAL "optimized" OR ${library} STREQUAL "debug")
	            continue()
	        endif()
	        list(APPEND LIBRARIES ${library})
	    endforeach(library)
	    set_target_properties(PCL::${module} PROPERTIES
	        INTERFACE_LINK_LIBRARIES "${LIBRARIES}"
	    )
	    # set_property(TARGET PCL::${module} PROPERTY
	    #     INTERFACE_LINK_LIBRARIES "${PCL_${module}_LIBRARIES}"
	    #     # INTERFACE_LINK_LIBRARIES "${LIBRARIES}"
	    # )
	    set_target_properties(PCL::${module} PROPERTIES
	        INTERFACE_INCLUDE_DIRECTORIES "${PCL_INCLUDE_DIRS}"
	    )
	endif()
endforeach(module)

# if(${PCL_FOUND} AND NOT TARGET PCL::IO)
#     add_library(PCL::IO INTERFACE IMPORTED)
#     # issue with zlib ? have to filter https://github.com/PointCloudLibrary/pcl/issues/2989
#     set(LIBRARIES "")
#     foreach(library ${PCL_IO_LIBRARIES})
#         if(${library} STREQUAL "optimized" OR ${library} STREQUAL "debug")
#             continue()
#         endif()
#         list(APPEND LIBRARIES ${library})
#     endforeach(library)
#     set_target_properties(PCL::IO PROPERTIES
#         INTERFACE_LINK_LIBRARIES "${LIBRARIES}"
#     )
#     # set_property(TARGET PCL::IO PROPERTY
#     #     INTERFACE_LINK_LIBRARIES "${PCL_IO_LIBRARIES}"
#     #     # INTERFACE_LINK_LIBRARIES "${LIBRARIES}"
#     # )
#     set_target_properties(PCL::IO PROPERTIES
#         INTERFACE_INCLUDE_DIRECTORIES "${PCL_INCLUDE_DIRS}"
#     )
# endif()

# if(${PCL_FOUND} AND NOT TARGET PCL_IO)
#     add_library(PCL_IO INTERFACE)
#     target_link_libraries(PCL_IO
#         INTERFACE ${PCL_IO_LIBRARIES}
#     )
#     set_target_properties(PCL_IO PROPERTIES
#         INTERFACE_INCLUDE_DIRECTORIES "${PCL_INCLUDE_DIRS}"
#     )
#     # add_library(PCL::IO ALIAS PCL_IO)
# endif()

