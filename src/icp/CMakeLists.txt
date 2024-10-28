include_directories (${PROJECT_SOURCE_DIR}/src)

# opencv
find_package(OpenCV 3 REQUIRED)
list(APPEND icp_inc ${OpenCV_INCLUDE_DIRS})
list(APPEND icp_lib ${OpenCV_LIBS})

# openmp
find_package(OpenMP REQUIRED)

# eigen
# find_package(Eigen3 REQUIRED)
set(EIGEN3_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/sup/eigen-3.3.9" CACHE PATH "EIGEN3_INCLUDE_DIR")
include_directories(${EIGEN3_INCLUDE_DIR})

# src
SET(icp_srcs icp.cpp edge_scene.cpp)

# lib & test exe
add_library(icp STATIC ${icp_srcs})

target_include_directories(icp PUBLIC ${icp_inc} ./)
target_link_libraries(icp PUBLIC ${icp_lib})
target_link_libraries(icp PRIVATE ${icp_lib} OpenMP::OpenMP_CXX)

install(
    TARGETS icp
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)