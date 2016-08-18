set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O2 -g -std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDEBUG")
set(CMAKE_CXX_COMPILER g++)

if (EXAMPLE_LIBS_FOUND)
  message("--> 'example' will be build.")

  include_directories(${CMAKE_SOURCE_DIR}/src)
  link_directories(/usr/lib ${CMAKE_BINARY_DIR}/src)

  add_executable(spiral spiral.cpp)
  add_executable(xor xor_problem.cpp)

  set(EXAMPLE_LIBS nnet armadillo ${OPENCV_LIBS} ${MATHGL_LIBS})

  target_link_libraries(spiral ${EXAMPLE_LIBS})
  target_link_libraries(xor ${EXAMPLE_LIBS})
endif (EXAMPLE_LIBS_FOUND)
