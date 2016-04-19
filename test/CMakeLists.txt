set (SOURCE ../src/nnet.cpp ../src/layer.cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O2 -g -std=c++11 -DDEBUG")

add_executable(test1 test1.cpp ${SOURCE})
add_executable(test2 test2.cpp ${SOURCE})
add_executable(test3 test3.cpp ${SOURCE})
add_executable(test4 test4.cpp ${SOURCE})

target_link_libraries(test1 armadillo opencv_core opencv_highgui opencv_imgproc)
target_link_libraries(test2 armadillo opencv_core opencv_highgui opencv_imgproc)
target_link_libraries(test3 armadillo opencv_core opencv_highgui opencv_imgproc)
target_link_libraries(test4 armadillo opencv_core opencv_highgui opencv_imgproc)

add_test(test1 ./test1)
add_test(test2 ./test2)
add_test(test3 ./test3)
add_test(test4 ./test4)

