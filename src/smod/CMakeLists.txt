cmake_minimum_required(VERSION 3.1)

# opencv
find_package(OpenCV 3 REQUIRED)
include_directories (${OpenCV_INCLUDE_DIRS})

# openmp
find_package(OpenMP REQUIRED)

# include MIPP headers
set(MIPP_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/sup/MIPP" CACHE PATH "Path of MIPP headers")
include_directories (${MIPP_INCLUDE_DIR})

add_library(
    smod SHARED
    line2dup.cpp
    shapeinfo.cpp
    nmsboxes.cpp
    smod.cpp
)

set_target_properties(smod PROPERTIES
    PREFIX "lib"
    IMPORT_PREFIX "lib"
)

target_include_directories(smod PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/smod>
)
target_include_directories(smod PRIVATE ${CMAKE_INSTALL_INCLUDEDIR}/mipp)
target_link_libraries(smod PRIVATE ${OpenCV_LIBS} OpenMP::OpenMP_CXX icp)

install(
    TARGETS smod
    EXPORT libsmod
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/smod
)

install(
    FILES line2dup.h shapeinfo.h smod.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/smod
)

install(
    EXPORT libsmod
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/smod
)

set(INCLUDE_DIRS ${CMAKE_INSTALL_INCLUDEDIR}/smod)
set(LIBRARIES ${CMAKE_INSTALL_LIBDIR}/libsmod.dll.a)

include(CMakePackageConfigHelpers)
# generate the config file that is includes the exports
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/config.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/libsmodConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PATH_VARS INCLUDE_DIRS LIBRARIES
    NO_SET_AND_CHECK_MACRO
    NO_CHECK_REQUIRED_COMPONENTS_MACRO
)

# generate the version file for the config file
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/libsmodConfigVersion.cmake"
    VERSION "1.0"
    COMPATIBILITY AnyNewerVersion
)

# install the configuration file
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/libsmodConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/libsmodConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
