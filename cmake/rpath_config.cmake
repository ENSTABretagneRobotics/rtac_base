# THIS DOES NOT WORK !

# This CMake script is loaded before the creation of any rtac project
# depending on rtac_base.

### IN SHORT ##############
# It allows execution of binaries in the build tree to execute using the shared
# libraries in the same build tree, regardless if there is a version of these
# libraries already installed under LD_LIBRARY_PATH.

### LESS SHORT ############# (skip if you don't have issues)
# When a binary is executed, it must first load the shared libraries it depends
# on (.so, .dll ...). To find them on the system, the executable looks up in
# several location lists, RPATH (stored in the executable), LD_LIBRARY_PATH
# (environment variable) and RUNPATH (stored in the executable). RPATH has
# precedence over LD_LIBRARY_PATH which itself has precedence over RUNPATH. All
# of this is still true with shared libraries depending on other shared
# libraries. On installed releases of modern softwares, only RUNPATH should be
# used.

# A typical situation when developing a shared library is to create executable
# tests in the same build tree as the shared library. The default behavior of
# CMake is to set the RUNPATH of these executables to the shared library build
# location at build time, and then change it at install time to the
# installation directory. This allows to execute the tests in the build
# directory using the shared library located in the build tree, but to use
# the installed version of the library if the installed binary is executed.
# Specifically, if an older version of your shared library is already
# installed, it will be ignored when executing the tests from the build
# directory, preferring the locally built instead.

# But everything changes when LD_LIBRARY_PATH is set.

# When LD_LIRBARY_PATH is set to a location where an old version of the
# dependant shared library is installed, there will be issues with executing
# the test binaries in the build directory. Indeed, LD_LIBRARY_PATH having
# precedence over RUNPATH, the test binary will execute using the INSTALLED
# version of your shared library located under LD_LIBRARY_PATH but not the one
# you just built into your build directory. The effect is that any changes you
# make to your library will not be testable until you install it, which is an
# undesired behavior.

# This cmake script manipulates RPATH to ensure the freshly built library from
# the binary directory is always used instead of an installed version,
# regardless of LD_LIBRARY_PATH. (More details here :
# https://gitlab.kitware.com/cmake/community/-/wikis/doc/cmake/RPATH-handling).

# On LD_LIRBARY_PATH : LD_LIBRARY_PATH is used to take precedence over RUNPATH
# when a binary is looking for its dependencies at runtime. However this should
# be  considered as a developing tool and installed binaries should not rely on
# LD_LIBRARY_PATH in most cases. In your case, you may have LD_LIBRARY_PATH set
# on your system for a good reason (but often bad ones which are out of your
# control). 

# LESS SHORT END ##############################

# the following was taken from
# https://gitlab.kitware.com/cmake/community/-/wikis/doc/cmake/RPATH-handling

# use, i.e. don't skip the full RPATH for the build tree
set(CMAKE_SKIP_BUILD_RPATH FALSE)

# when building, don't use the install RPATH already
# (but later on when installing)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# the RPATH to be used when installing, but only if it's not a system directory
list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
if("${isSystemDir}" STREQUAL "-1")
    set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
endif("${isSystemDir}" STREQUAL "-1")

