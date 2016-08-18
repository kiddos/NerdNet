cmake_minimum_required(VERSION 3.0)

enable_testing()

include(cmake/find_libraries.cmake)
include(cmake/find_headers.cmake)
include(cmake/find_clang.cmake)

add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(example)

file(COPY ${CMAKE_SOURCE_DIR}/data DESTINATION ${CMAKE_BINARY_DIR})
file(COPY ${CMAKE_SOURCE_DIR}/run_experiment.py DESTINATION ${CMAKE_BINARY_DIR})
