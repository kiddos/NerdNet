include_directories(${CMAKE_SOURCE_DIR}/src)
link_libraries(NerdNet)

find_package(GTest)
if (GTEST_FOUND)
  include_directories(${GTEST_INCLUDE_DIRS})
  link_libraries(${GTEST_BOTH_LIBRARIES} pthread)

  add_executable(test_input_layer unit/layer/test_input_layer.cc)
  add_executable(test_fc_layer unit/layer/test_fc_layer.cc)
  add_executable(test_sigmoid_layer unit/layer/test_sigmoid_layer.cc)
  add_executable(test_tanh_layer unit/layer/test_tanh_layer.cc)
  add_executable(test_arctan_layer unit/layer/test_arctan_layer.cc)
  # relu
  add_executable(test_relu_layer unit/layer/test_relu_layer.cc)
  add_executable(test_leaky_relu_layer unit/layer/test_leaky_relu_layer.cc)
  add_executable(test_smooth_relu_layer unit/layer/test_smooth_relu_layer.cc)
  # cost functions
  add_executable(test_mean_square_error unit/layer/test_mean_square_error.cc)
  add_executable(test_softmax_cross_entropy
    unit/layer/test_softmax_cross_entropy.cc)
  add_executable(test_sigmoid_cross_entropy
    unit/layer/test_sigmoid_cross_entropy.cc)
  add_executable(test_kullback_leibler_divergence
    unit/layer/test_kullback_leibler_divergence.cc)
  # neural network
  add_executable(test_nerdnet unit/test_nerdnet.cc)
  # trainers
  add_executable(test_gradient_descent_trainer
    unit/trainer/test_gradient_descent_trainer.cc)

  add_test(InputLayerTest test_input_layer )
  add_test(FCLayerTest test_fc_layer)
  add_test(SigmoidLayerTest test_sigmoid_layer)
  add_test(TanhLayerTest test_tanh_layer)
  add_test(ArctanLayerTest test_arctan_layer)

  add_test(ReluLayerTest test_relu_layer)
  add_test(LeakyReluLayerTest test_leaky_relu_layer)
  add_test(SmoothReluLayerTest test_smooth_relu_layer)

  add_test(MeanSquareErrorTest test_mean_square_error)
  add_test(SoftmaxCrossEntropyTest test_softmax_cross_entropy)
  add_test(SigmoidCrossEntropyTest test_sigmoid_cross_entropy)
  add_test(KullbackLeiblerDivergenceTest test_kullback_leibler_divergence)

  add_test(NerdNetTest test_nerdnet)
  add_test(GradientDescentTrainerTest test_gradient_descent_trainer)
endif ()

find_library(BENCHMARK_LIBRARIES benchmark)
if (BENCHMARK_LIBRARIES)
  message(STATUS "Using Google Benchmark ${BENCHMARK_LIBRARIES}")
  link_libraries(${BENCHMARK_LIBRARIES} pthread)

  add_executable(benchmark_convert bench/benchmark_convert.cc)
  add_test(ConvertBenchmark benchmark_convert)
endif ()

find_package(Qt5Widgets)
find_package(Qt5PrintSupport)
find_package(QCustomPlot)
if (Qt5Widgets_FOUND AND Qt5PrintSupport_FOUND AND QCustomPlot_FOUND)
  add_executable(test_plot_window util/qt/test_plot_window.cc)
endif ()
