include (CheckIncludeFiles)
check_include_files(armadillo HAVE_ARMADILLO)
check_include_files(mat.h HAVE_MAT)
check_include_files(omp.h HAVE_OPENMP)

set(BASE_SOURCE act.cpp matop.cpp)

set(LAYER_SOURCE layer.cpp inputlayer.cpp convlayer.cpp)
set(OUTPUT_LAYER_SOURCE outputlayer.cpp)
set(OUTPUT_LAYER_SOURCE ${OUTPUT_LAYER_SOURCE} softmaxoutput.cpp)
set(OUTPUT_LAYER_SOURCE ${OUTPUT_LAYER_SOURCE} crossentropyoutput.cpp)
set(OUTPUT_LAYER_SOURCE ${OUTPUT_LAYER_SOURCE} quadraticoutput.cpp)
set(OUTPUT_LAYER_SOURCE ${OUTPUT_LAYER_SOURCE} kullbackleibleroutput.cpp)
set(LAYER_SOURCE ${LAYER_SOURCE} ${OUTPUT_LAYER_SOURCE})

set(TRAINER_SOURCE trainer.cpp)
set(TRAINER_SOURCE ${TRAINER_SOURCE} momentumtrainer.cpp)
set(TRAINER_SOURCE ${TRAINER_SOURCE} sgdtrainer.cpp)
set(TRAINER_SOURCE ${TRAINER_SOURCE} batchtrainer.cpp)
set(SOURCE ${BASE_SOURCE} ${LAYER_SOURCE} neuralnet.cpp ${TRAINER_SOURCE})


set(BASE_HEADER act.h type.h matop.h debug.h config.h)
set(LAYER_HEADER layer.h inputlayer.h convlayer.h softmaxlayer.h)
set(OUTPUT_LAYER_HEADER outputlayer.h)
set(OUTPUT_LAYER_HEADER ${OUTPUT_LAYER_HEADER} softmaxoutput.h)
set(OUTPUT_LAYER_HEADER ${OUTPUT_LAYER_HEADER} crossentropyoutput.h)
set(OUTPUT_LAYER_HEADER ${OUTPUT_LAYER_HEADER} quadraticoutput.h)
set(OUTPUT_LAYER_HEADER ${OUTPUT_LAYER_HEADER} kullbackleibleroutput.h)
set(LAYER_HEADER ${LAYER_HEADER} ${OUTPUT_LAYER_HEADER})

set(TRAINER_HEADER trainer.h)
set(TRAINER_HEADER ${TRAINER_HEADER} momentumtrainer.h)
set(TRAINER_HEADER ${TRAINER_HEADER} sgdtrainer.h)
set(TRAINER_HEADER ${TRAINER_HEADER} batchtrainer.h)

set(HEADER ${LAYER_HEADER} ${BASE_HEADER} neuralnet.h nnet.h ${TRAINER_HEADER})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O2 -g -std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDEBUG")
if (HAVE_OPENMP)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_OPENMP -fopenmp")
  message("Using Openmp")
endif (HAVE_OPENMP)

add_library(nnet ${SOURCE} ${HEADER})
target_link_libraries(nnet armadillo)

