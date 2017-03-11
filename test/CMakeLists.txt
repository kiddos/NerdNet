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

  # operations
  set(TEST_TENSOR_BASIC_OPS tensor/ops/test_tensor_basic_ops.cpp)
  add_executable(test_tensor_basic_ops ${TEST_TENSOR_BASIC_OPS})
  target_link_libraries(test_tensor_basic_ops ${TEST_LIBS})
  add_test(TestTensorBasicOps test_tensor_basic_ops)
endif ()

find_library(BENCHMARK benchmark)
if (BENCHMARK)
  set(BENCHMARK_LIBS ${BENCHMARK} pthread tensor)
  add_executable(benchmark_basic_ops tensor/ops/benchmark_tensor_basic_ops.cpp)
  target_link_libraries(benchmark_basic_ops ${BENCHMARK_LIBS})
endif ()
