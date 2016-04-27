set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -g -std=c++11 -DDEBUG")

link_directories(/usr/lib ${CMAKE_BINARY_DIR}/src)

add_executable(XORProblem test1.cpp)
add_executable(MNIST test2.cpp)
add_executable(Dataset1 test3.cpp)
add_executable(Dataset2 test4.cpp)
add_executable(iris test5.cpp)
add_executable(diabetes test6.cpp)
add_executable(pigsslaughter test7.cpp)

set(LIBS nnet armadillo opencv_core opencv_highgui opencv_imgproc)

target_link_libraries(XORProblem ${LIBS})
target_link_libraries(MNIST ${LIBS})
target_link_libraries(Dataset1 ${LIBS})
target_link_libraries(Dataset2 ${LIBS})
target_link_libraries(iris ${LIBS})
target_link_libraries(diabetes ${LIBS})
target_link_libraries(pigsslaughter ${LIBS})

add_test(XORProblem ./test1)
add_test(MNIST ./test2)
add_test(Dataset1 ./test3)
add_test(Dataset2 ./test4)
add_test(iris ./test5)
add_test(diabetes ./test6)
add_test(pigsslaughter ./test6)

