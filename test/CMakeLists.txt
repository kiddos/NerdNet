set (SOURCE ../src/nnet.cpp ../src/layer.cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O2 -g -std=c++11 -DDEBUG")

add_executable(test1 test1.cpp ${SOURCE})
target_link_libraries(test1 armadillo opencv_core opencv_highgui opencv_imgproc)

