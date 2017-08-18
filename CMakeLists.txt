cmake_minimum_required(VERSION 3.0)
project(NerdNet)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

enable_testing()

include(cmake/find_headers.cmake)
include(cmake/find_libraries.cmake)

add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(example)

file(COPY ${CMAKE_SOURCE_DIR}/data DESTINATION ${CMAKE_BINARY_DIR})
file(COPY ${CMAKE_SOURCE_DIR}/run_experiment.py DESTINATION ${CMAKE_BINARY_DIR})
