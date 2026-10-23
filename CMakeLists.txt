cmake_minimum_required(VERSION 3.14)

project(nnet VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
  add_compile_options(/W4)
else()
  add_compile_options(-Wall -Wextra)
endif()

# Fetch Armadillo via FetchContent
include(FetchContent)
FetchContent_Declare(
  armadillo
  GIT_REPOSITORY https://gitlab.com/conradsnicta/armadillo-code.git
  GIT_TAG        15.6.2
  GIT_SHALLOW    TRUE
)

# Build static armadillo library
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries" FORCE)
FetchContent_MakeAvailable(armadillo)

# Find OpenMP parallelization (optional)
find_package(OpenMP)

enable_testing()

add_subdirectory(src)
add_subdirectory(example)
add_subdirectory(test)
