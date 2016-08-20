set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g -std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDEBUG")

if (LIBS_FOUND)
  message("--> 'test' will be build.")
  include_directories(${CMAKE_SOURCE_DIR}/src)
  link_directories(/usr/lib ${CMAKE_BINARY_DIR}/src)

  set(TEST_LIBS nnet armadillo ${OPENCV_LIBS} ${MATHGL_LIBS} ${BOOST_LIBS})

  add_executable(normlayer_test normlayer_test.cpp)
  add_executable(act_test act_test.cpp)

  target_link_libraries(normlayer_test ${TEST_LIBS})
  target_link_libraries(act_test ${TEST_LIBS})

  add_test(NormLayerTest ./normlayer_test)
  add_test(ActTest ./act_test)
endif (LIBS_FOUND)

