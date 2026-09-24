cmake_minimum_required(VERSION 3.10)
project(elas-tests)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(GTest REQUIRED)
find_package(Boost COMPONENTS filesystem system REQUIRED)

enable_testing()

# CPU test
add_executable(test_stereo_matching_cpu
    src/main.cpp
    src/test_stereo_matching_cpu.cpp
)
target_include_directories(test_stereo_matching_cpu PRIVATE
    src
    ${CMAKE_SOURCE_DIR}/cpu/src
    ${GTEST_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
)
target_link_libraries(test_stereo_matching_cpu
    elas_cpu_lib
    ${OpenCV_LIBS}
    GTest::GTest
    GTest::Main
    ${Boost_LIBRARIES}
)
add_test(NAME test_stereo_matching_cpu COMMAND test_stereo_matching_cpu)

# OpenCL test
add_executable(test_stereo_matching_opencl
    src/main.cpp
    src/test_stereo_matching_opencl.cpp
)
target_include_directories(test_stereo_matching_opencl PRIVATE
    src
    ${CMAKE_SOURCE_DIR}/opencl/src
    ${GTEST_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
)
target_link_libraries(test_stereo_matching_opencl
    elas_opencl_lib
    ${OpenCV_LIBS}
    GTest::GTest
    GTest::Main
    ${Boost_LIBRARIES}
)
add_test(NAME test_stereo_matching_opencl COMMAND test_stereo_matching_opencl)
