cmake_minimum_required(VERSION 3.10)
project(elas-opencl)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Option to enable kernel execution logging
option(DEBUG_KERNEL "Enable kernel execution logging" OFF)
if(DEBUG_KERNEL)
    add_definitions(-DDEBUG_KERNEL)
endif()

find_package(OpenCV REQUIRED)
find_package(OpenCL REQUIRED)

file(GLOB ELAS_SRC "src/*.cpp")
file(GLOB KERNEL_SRC "kernels/*.cpp")
file(GLOB OPENCL_KERNELS "kernels/*.cl")

# Copy kernel files into build/kernels/
set(CLELAS_KERNEL_DIR ${CMAKE_BINARY_DIR}/kernels)
file(MAKE_DIRECTORY ${CLELAS_KERNEL_DIR})
foreach(KERNEL ${OPENCL_KERNELS})
    get_filename_component(KERNEL_NAME ${KERNEL} NAME)
    configure_file(${KERNEL} ${CLELAS_KERNEL_DIR}/${KERNEL_NAME} COPYONLY)
endforeach()

add_library(elas_opencl_lib ${ELAS_SRC} ${KERNEL_SRC})
# Public headers (src/) do not include OpenCL; the kernels and the OpenCL
# C++ bindings are internal to the library.
target_include_directories(elas_opencl_lib
    PUBLIC src ${OpenCV_INCLUDE_DIRS}
    PRIVATE kernels ${OpenCL_INCLUDE_DIRS})
target_compile_definitions(elas_opencl_lib PRIVATE
    CL_HPP_ENABLE_EXCEPTIONS
    CL_HPP_TARGET_OPENCL_VERSION=200
    CL_HPP_MINIMUM_OPENCL_VERSION=200
    CL_TARGET_OPENCL_VERSION=200
    CLELAS_KERNEL_DIR="${CLELAS_KERNEL_DIR}")
target_link_libraries(elas_opencl_lib PUBLIC ${OpenCV_LIBS} PRIVATE ${OpenCL_LIBRARIES})
