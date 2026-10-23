add_library(tensor
  tensor/tensor_shape.cpp
  tensor/tensor.cpp
  tensor/ops/add.cpp
  tensor/ops/argmax.cpp
  tensor/ops/argmin.cpp
  tensor/ops/div.cpp
  tensor/ops/equal.cpp
  tensor/ops/exp.cpp
  tensor/ops/log.cpp
  tensor/ops/matmul.cpp
  tensor/ops/mul.cpp
  tensor/ops/negate.cpp
  tensor/ops/power.cpp
  tensor/ops/reduce_mean.cpp
  tensor/ops/reduce_sum.cpp
  tensor/ops/sqrt.cpp
  tensor/ops/sub.cpp
)

target_include_directories(tensor
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/tensor>
)

if(OpenMP_CXX_FOUND)
  target_link_libraries(tensor PUBLIC OpenMP::OpenMP_CXX)
endif()

add_library(nnet
  act.cpp
  matop.cpp
  neuralnet.cpp
  layer/conv2dlayer.cpp
  layer/inputlayer.cpp
  layer/layer.cpp
  layer/maxpoollayer.cpp
  layer/meanpoollayer.cpp
  layer/normlayer.cpp
  layer/outputlayer.cpp
  layer/softmaxlayer.cpp
  output/crossentropyoutput.cpp
  output/kullbackleibleroutput.cpp
  output/quadraticoutput.cpp
  output/softmaxoutput.cpp
  trainer/batchtrainer.cpp
  trainer/momentumtrainer.cpp
  trainer/sgdtrainer.cpp
  trainer/trainer.cpp
)

target_include_directories(nnet
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/layer>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/output>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/trainer>
)

target_link_libraries(nnet
  PUBLIC
    armadillo
    tensor
)

if(OpenMP_CXX_FOUND)
  target_link_libraries(nnet PUBLIC OpenMP::OpenMP_CXX)
endif()
