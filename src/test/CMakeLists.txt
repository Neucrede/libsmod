include_directories (${PROJECT_SOURCE_DIR}/src)

add_executable(
    testprg
    test.cpp
)

# opencv
find_package(OpenCV 3 REQUIRED)
include_directories (${OpenCV_INCLUDE_DIRS})

target_link_libraries(testprg smod ${OpenCV_LIBS})

install(
    TARGETS testprg
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)