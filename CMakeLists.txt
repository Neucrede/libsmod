cmake_minimum_required(VERSION 3.1)
set (CMAKE_CXX_STANDARD 14)

project(smod)

include(GNUInstallDirs)

# arm or x86
IF(${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm")
    SET(PLATFORM_COMPILE_FLAGS "-mfpu=neon")
ELSE()
    # SET(PLATFORM_COMPILE_FLAGS "-march=native")

    # some places of the algorithm are designed for 128 SIMD
    # so 128 SSE may slightly faster than 256 AVX, you may want this
    SET(PLATFORM_COMPILE_FLAGS "-msse -msse2 -msse3 -mssse3 -msse4")
ENDIF()

SET(COMMON_COMPILE_FLAGS "-Wno-sign-compare")
SET(CMAKE_CXX_FLAGS "${PLATFORM_COMPILE_FLAGS} ${COMMON_COMPILE_FLAGS} $ENV{CXXFLAGS} -fPIC")
SET(CMAKE_CXX_FLAGS_DEBUG "-O0 -g2 -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "-O3")

add_subdirectory(${PROJECT_SOURCE_DIR}/src/icp)
add_subdirectory(${PROJECT_SOURCE_DIR}/src/smod)
add_subdirectory(${PROJECT_SOURCE_DIR}/src/test)
