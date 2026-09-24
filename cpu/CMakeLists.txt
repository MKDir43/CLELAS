cmake_minimum_required(VERSION 3.10)
project(elas-cpu)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

file(GLOB ELAS_CPU_SRC "src/*.cpp")
add_library(elas_cpu_lib ${ELAS_CPU_SRC})
target_include_directories(elas_cpu_lib PUBLIC src ${OpenCV_INCLUDE_DIRS})
target_link_libraries(elas_cpu_lib ${OpenCV_LIBS})
