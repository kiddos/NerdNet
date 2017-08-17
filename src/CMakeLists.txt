set(CMAKE_INCLUDE_CURRENT_DIR ON)

# armadillo linear algebra
find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})
message(STATUS "Armadillo Library Version: ${ARMADILLO_VERSION_STRING}")
link_libraries(${ARMADILLO_LIBRARIES})

# OpenBLAS for performance
find_library(OpenBLAS openblas)
if (NOT OpenBLAS)
  message("OpenBLAS not installed")
endif ()
link_libraries(${OpenBLAS})

# boost util
## Boost
find_package(Boost COMPONENTS system REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
link_libraries(${Boost_LIBRARIES})

file(GLOB_RECURSE NerdNet_SOURCE NerdNet/**.cc NerdNet/**.h *.cpp *.h)

message("--> 'nnet' will be build.")
add_library(NerdNet ${NerdNet_SOURCE})

set(LIBS ${ARMADILLO_LIBS})
