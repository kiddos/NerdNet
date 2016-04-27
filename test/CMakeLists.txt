set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -g -std=c++11 -DDEBUG")

link_directories(/usr/lib ${CMAKE_BINARY_DIR}/src)

add_executable(test1 test1.cpp)
add_executable(test2 test2.cpp)
add_executable(test3 test3.cpp)
add_executable(test4 test4.cpp)
add_executable(test5 test5.cpp)
add_executable(test6 test6.cpp)
add_executable(test7 test7.cpp)

set(LIBS nnet armadillo opencv_core opencv_highgui opencv_imgproc)

target_link_libraries(test1 ${LIBS})
target_link_libraries(test2 ${LIBS})
target_link_libraries(test3 ${LIBS})
target_link_libraries(test4 ${LIBS})
target_link_libraries(test5 ${LIBS})
target_link_libraries(test6 ${LIBS})
target_link_libraries(test7 ${LIBS})

add_test(test1 ./test1)
add_test(test2 ./test2)
add_test(test3 ./test3)
add_test(test4 ./test4)
add_test(test5 ./test5)
add_test(test6 ./test6)
add_test(test7 ./test6)

