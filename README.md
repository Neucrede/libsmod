# SMOD

## About
SMOD (Shape Model Object Detector) enables users to search an image for a pre-registered template, and obtain information on position, orientation and scaling of the objects found.

## License
The source code is licensed under **BSD 2-Clause License**.

## Acknowledgement
Line2Dup and ICP were ported then modified from [sim3](https://github.com/meiqua/shape_based_matching/tree/sim3) branch of [shape_based_matching](https://github.com/meiqua/shape_based_matching) project hosted by [meiqua](https://github.com/meiqua).

## Prerequisities
- Intel / AMD CPU that supports SSE4.2 instruction set.
- C++ 14 compatible compilers with OpenMP support.
- OpenCV 3 or above.
- CMake 3.1 or above.

## Installation
1. Create a directory named `build` inside the directory that contains this ReadMe file.
2. Open a terminal window, `cd` to `build` directory created just before and type in `cmake ..` then press `ENTER` key.
3. Wait until CMake completed its job, deal with errors when needed.
4. Modify `CMakeCache.txt` if needed. You may want to change the value of  `CMAKE_BUILD_TYPE` to `Release` or `RelWithDebInfo` since default `Debug` build is extremely slow due to various C/C++ assertion calls.
5. Type in `cmake --build . --target install` then press `ENTER` key and wait.
6. If Doxygen was installed on you computer, run `doxygen` in `doc` directory to generate HTML documentation.
7. Done.

## Using SMOD with CMake
Add the following lines in `CMakeLists.txt`.
```
find_package(libsmod REQUIRED)
target_include_directories(your_project_name PRIVATE ${libsmod_INCLUDE_DIRS})
target_link_libraries(your_project_name PRIVATE ${libsmod_LIBS})
```

## Usage
For details on data structures and functions, refer to generated HTML documentation.

See the sample code in `/src/test/test.cpp`.

