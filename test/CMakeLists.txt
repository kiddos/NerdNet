# GTest Framework
find_package(GTest REQUIRED)

# Helper function to add unit tests
function(add_nnet_test name source)
  add_executable(${name} ${source})
  target_link_libraries(${name} PRIVATE tensor nnet GTest::gtest GTest::gtest_main)
  add_test(NAME ${name} COMMAND ${name})
endfunction()

# Tensor tests
add_nnet_test(test_tensor_shape tensor/test_tensor_shape.cpp)
add_nnet_test(test_tensor tensor/test_tensor.cpp)
add_nnet_test(test_tensor_basic_ops tensor/ops/test_tensor_basic_ops.cpp)
add_nnet_test(test_tensor_arg tensor/ops/test_tensor_arg.cpp)
add_nnet_test(test_tensor_equal tensor/ops/test_tensor_equal.cpp)
add_nnet_test(test_tensor_mathops tensor/ops/test_tensor_mathops.cpp)
add_nnet_test(test_tensor_matmul tensor/ops/test_tensor_matmul.cpp)
add_nnet_test(test_tensor_negate tensor/ops/test_tensor_negate.cpp)
add_nnet_test(test_tensor_reduce tensor/ops/test_tensor_reduce.cpp)

# Boost Unit Test Framework (Optional for legacy Boost tests)
find_package(Boost COMPONENTS unit_test_framework QUIET)
if(Boost_FOUND OR Boost_UNIT_TEST_FRAMEWORK_FOUND)
  add_executable(act_test act_test.cpp)
  target_link_libraries(act_test PRIVATE nnet Boost::unit_test_framework)
  add_test(NAME act_test COMMAND act_test)

  add_executable(layer_unit_test layer_unit_test.cpp)
  target_link_libraries(layer_unit_test PRIVATE nnet Boost::unit_test_framework)
  add_test(NAME layer_unit_test COMMAND layer_unit_test)
else()
  message(STATUS "Boost unit test framework not found. Skipping act_test and layer_unit_test.")
endif()

# OpenCV-dependent tests
find_package(OpenCV QUIET)
if(OpenCV_FOUND)
  add_executable(convlayer_test convlayer_test.cpp)
  target_link_libraries(convlayer_test PRIVATE nnet ${OpenCV_LIBS})
  target_include_directories(convlayer_test PRIVATE ${OpenCV_INCLUDE_DIRS})

  add_executable(normlayer_test normlayer_test.cpp gradient_checker.cpp)
  target_link_libraries(normlayer_test PRIVATE nnet ${OpenCV_LIBS})
  target_include_directories(normlayer_test PRIVATE ${OpenCV_INCLUDE_DIRS})

  add_executable(model_test model_test.cpp gradient_checker.cpp)
  target_link_libraries(model_test PRIVATE nnet ${OpenCV_LIBS} GTest::gtest GTest::gtest_main)
  target_include_directories(model_test PRIVATE ${OpenCV_INCLUDE_DIRS})
  add_test(NAME model_test COMMAND model_test)
else()
  message(STATUS "OpenCV not found. Skipping OpenCV-dependent tests (convlayer_test, normlayer_test, model_test).")
endif()

# Google Benchmark Framework (Optional)
find_package(benchmark QUIET)
if(benchmark_FOUND OR BENCHMARK_FOUND)
  if(TARGET benchmark::benchmark)
    set(BENCHMARK_TARGET benchmark::benchmark)
  else()
    set(BENCHMARK_TARGET ${BENCHMARK_LIBRARIES})
  endif()

  function(add_nnet_benchmark name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE tensor ${BENCHMARK_TARGET})
  endfunction()

  add_nnet_benchmark(benchmark_basic_ops tensor/ops/benchmark_tensor_basic_ops.cpp)
  add_nnet_benchmark(benchmark_math_ops tensor/ops/benchmark_tensor_mathops.cpp)
  add_nnet_benchmark(benchmark_matmul tensor/ops/benchmark_tensor_matmul.cpp)
  add_nnet_benchmark(benchmark_reduction tensor/ops/benchmark_tensor_reduction.cpp)
  add_nnet_benchmark(benchmark_arg tensor/ops/benchmark_tensor_arg.cpp)
endif()
