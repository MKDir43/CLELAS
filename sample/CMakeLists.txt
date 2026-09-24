cmake_minimum_required(VERSION 3.10)
project(elas-samples)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(Boost COMPONENTS program_options system filesystem REQUIRED)

# CPU sample
add_executable(image_cpu image_cpu.cpp)
target_include_directories(image_cpu PRIVATE ${CMAKE_SOURCE_DIR}/cpu/src)
target_link_libraries(image_cpu elas_cpu_lib ${OpenCV_LIBS})

# OpenCL sample
add_executable(image_opencl image_opencl.cpp)
target_include_directories(image_opencl PRIVATE ${CMAKE_SOURCE_DIR}/opencl/src ${Boost_INCLUDE_DIRS})
target_link_libraries(image_opencl elas_opencl_lib ${OpenCV_LIBS} ${Boost_LIBRARIES})

# OAK-D samples (require depthai)
find_package(depthai QUIET)
if(depthai_FOUND)
    # OAK-D + OpenCL
    add_executable(oakd_opencl oakd_opencl.cpp)
    target_include_directories(oakd_opencl PRIVATE ${CMAKE_SOURCE_DIR}/opencl/src)
    target_link_libraries(oakd_opencl elas_opencl_lib depthai::opencv ${OpenCV_LIBS})

    # OAK-D + CPU
    add_executable(oakd_cpu oakd_cpu.cpp)
    target_include_directories(oakd_cpu PRIVATE ${CMAKE_SOURCE_DIR}/cpu/src)
    target_link_libraries(oakd_cpu elas_cpu_lib depthai::opencv ${OpenCV_LIBS})
else()
    message(WARNING "depthai not found, oakd_opencl/oakd_cpu will not be built")
endif()
