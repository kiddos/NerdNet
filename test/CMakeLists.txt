enable_testing()
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
  add_executable(test_tensor_add tensor/ops/test_tensor_add.cpp)
  target_link_libraries(test_tensor_add ${TEST_LIBS})
  add_test(TestTensorOpsAdd test_tensor_add)

  add_executable(test_tensor_sub tensor/ops/test_tensor_sub.cpp)
  target_link_libraries(test_tensor_sub ${TEST_LIBS})
  add_test(TestTensorOpsSub test_tensor_sub)
endif ()
