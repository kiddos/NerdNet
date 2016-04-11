include (CheckIncludeFiles)
check_include_files(armadillo HAVE_ARMADILLO)
check_include_files(mat.h HAVE_MAT)

set(SOURCE layer.cpp nnet.cpp)
set(HEADER layer.h nnet.h type.h)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O2 -g")

add_library(nnet ${SOURCE} ${HEADER})

