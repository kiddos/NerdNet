enable_testing()
# google test framework
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/src)
set(CMAKE_INCLUDE_CURRENT_DIR on)

if (TARGET tensor)
  set(TEST_LIBS ${GTEST_BOTH_LIBRARIES} pthread tensor)
  add_executable(test_tensor_shape tensor/test_tensor_shape.cpp)
  target_link_libraries(test_tensor_shape ${TEST_LIBS})
  add_test(TestTensorShapeObject test_tensor_shape)

  add_executable(test_tensor tensor/test_tensor.cpp)
  target_link_libraries(test_tensor ${TEST_LIBS})
  add_test(TestTensorObject test_tensor)

  # basic operations
  add_executable(test_tensor_basic_ops tensor/ops/test_tensor_basic_ops.cpp)
  target_link_libraries(test_tensor_basic_ops ${TEST_LIBS})
  add_test(TestTensorBasicOps test_tensor_basic_ops)

  # argmax
  add_executable(test_tensor_argmax tensor/ops/test_tensor_argmax.cpp)
  target_link_libraries(test_tensor_argmax ${TEST_LIBS})
  add_test(TestTensorArgmax test_tensor_argmax)

  # equal
  add_executable(test_tensor_equal tensor/ops/test_tensor_equal.cpp)
  target_link_libraries(test_tensor_equal ${TEST_LIBS})
  add_test(TestTensorEqual test_tensor_equal)

  # basic math
  add_executable(test_tensor_mathops tensor/ops/test_tensor_mathops.cpp)
  target_link_libraries(test_tensor_mathops ${TEST_LIBS})
  add_test(TestTensorMathOps test_tensor_mathops)

  # matrix multiplication
  add_executable(test_tensor_matmul tensor/ops/test_tensor_matmul.cpp)
  target_link_libraries(test_tensor_matmul ${TEST_LIBS})
  add_test(TestTensorMatMul test_tensor_matmul)

  # negation
  add_executable(test_tensor_negate tensor/ops/test_tensor_negate.cpp)
  target_link_libraries(test_tensor_negate ${TEST_LIBS})
  add_test(TestTensorNegate test_tensor_negate)

  # reduction
  add_executable(test_tensor_reduce tensor/ops/test_tensor_reduce.cpp)
  target_link_libraries(test_tensor_reduce ${TEST_LIBS})
  add_test(TestTensorReduce test_tensor_reduce)
endif ()

find_library(BENCHMARK benchmark)
if (BENCHMARK)
  set(BENCHMARK_LIBS ${BENCHMARK} pthread tensor)
  add_executable(benchmark_basic_ops tensor/ops/benchmark_tensor_basic_ops.cpp)
  target_link_libraries(benchmark_basic_ops ${BENCHMARK_LIBS})
endif ()
