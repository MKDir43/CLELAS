cmake_minimum_required(VERSION 3.10)
project(CLELAS)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build optimized binaries unless a build type is given (e.g. -DCMAKE_BUILD_TYPE=Debug)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(BUILD_SAMPLE "Build sample programs" ON)
option(BUILD_TESTS "Build tests" OFF)

add_subdirectory(cpu)
add_subdirectory(opencl)

if(BUILD_SAMPLE)
    add_subdirectory(sample)
endif()

if(BUILD_TESTS)
    add_subdirectory(test)
endif()
