include (CheckIncludeFiles)
check_include_files(armadillo HAVE_ARMADILLO)
check_include_files(mat.h HAVE_MAT)

set(LAYER_SOURCE layer.cpp inputlayer.cpp outputlayer.cpp)
set(LAYER_SOURCE ${LAYER_SOURCE} convlayer.cpp)
set(BASE_SOURCE act.cpp matop.cpp)
set(SOURCE ${LAYER_SOURCE} ${BASE_SOURCE} neuralnet.cpp)

set(LAYER_HEADER layer.h inputlayer.h outputlayer.h convlayer.h softmaxlayer.h)
set(LAYER_HEADER ${LAYER_HEADER} softmaxlayer.h convlayer.h)
set(BASE_HEADER act.h type.h matop.h debug.h)
set(HEADER ${LAYER_HEADER} ${BASE_HEADER} neuralnet.h nnet.h)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3 -g -std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDEBUG ")

add_library(nnet ${SOURCE} ${HEADER})
target_link_libraries(nnet armadillo)

