cmake_minimum_required(VERSION 3.14)

project(nnet VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
  add_compile_options(/W4)
else()
  add_compile_options(-Wall -Wextra)
endif()

# Find Armadillo (required dependency)
find_package(Armadillo REQUIRED)

# Find OpenMP parallelization (optional)
find_package(OpenMP)

enable_testing()

add_subdirectory(src)
add_subdirectory(example)
add_subdirectory(test)
