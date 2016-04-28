include (CheckIncludeFiles)
check_include_files(armadillo HAVE_ARMADILLO)
check_include_files(mat.h HAVE_MAT)

set(SOURCE layer.cpp nnet.cpp act.cpp)
set(HEADER layer.h nnet.h type.h act.h)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O2 -g -std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDEBUG ")

add_library(nnet ${SOURCE} ${HEADER})
target_link_libraries(nnet armadillo)

