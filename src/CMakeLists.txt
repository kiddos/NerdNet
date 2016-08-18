# sources
file(GLOB NNET_SOURCE *.cpp)
file(GLOB NNET_HEADER *.h)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O2 -g -std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDEBUG")

if (LIBS_FOUND)
  message("--> 'nnet' will be build.")
  # add_library(nnet ${SOURCE} ${HEADER})
  add_library(nnet ${NNET_SOURCE} ${NNET_HEADER})

  set(LIBS ${ARMADILLO_LIBS})
  target_link_libraries(nnet armadillo)
endif (LIBS_FOUND)
