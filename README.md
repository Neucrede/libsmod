# Shape Model Object Detector

## About
SMOD is an open source shape based 2D object detector that enables users to search an image for a pre-registered template, and obtain information on position, orientation and scaling of the objects found.

## License
The source code is licensed under **BSD 2-Clause License**.

## Acknowledgement
Line2Dup and ICP (renamed from CUDA-ICP after CUDA related stuffs being removed) were modified from [sim3](https://github.com/meiqua/shape_based_matching/tree/sim3) branch of [shape_based_matching](https://github.com/meiqua/shape_based_matching) project hosted by [meiqua](https://github.com/meiqua).

## Prerequisities
- Intel / AMD CPU that supports SSE4.2 instruction set.
- C++ 14 compatible compilers with OpenMP support.
- OpenCV 3 or above.
- CMake 3.1 or above.
- MIPP (supplied in /sup directory).
- Eigen (supplied in /sup directory).

## Install
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

## Benchmark
Using `blades_and_gaskets` and `poker_cards` testing sets supplied in `/test` directory, it takes about 0.1s on my laptop computer to find at most 5 objects for each source image being examined without refinement. 

```
OS Name:                   Microsoft Windows 10 Enterprise LTSC
OS Version:                10.0.17763 N/A Build 17763
OS Manufacturer:           Microsoft Corporation
OS Configuration:          Standalone Workstation
OS Build Type:             Multiprocessor Free
System Manufacturer:       Dell Inc.
System Model:              Latitude 7280
System Type:               x64-based PC
Processor(s):              1 Processor(s) Installed.
                           [01]: Intel64 Family 6 Model 78 Stepping 3 GenuineIntel ~2607 Mhz
Total Physical Memory:     8,070 MB
```

