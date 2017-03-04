#include <gtest/gtest.h>
#include "tensor/ops/argmax.h"
#include "tensor/ops/mul.h"


template <typename DType>
void TestArgmax(int size) {
  using namespace nn::tensor;

  Tensor<DType> t = Tensor<DType>::Eyes(size) * static_cast<DType>(10);

  Tensor<DType> argmax = nn::tensor::argmax(t, 1);
  for (int i = 0 ; i < size ; ++i) {
    EXPECT_EQ(argmax[i], 10);
  }

  argmax = nn::tensor::argmax(t, 0);
  for (int i = 0 ; i < size ; ++i) {
    EXPECT_EQ(argmax[i], 10);
  }
}

TEST(TestTensorOps, ArgmaxSmallScale) {
  TestArgmax<float>(10);
  TestArgmax<double>(10);

  TestArgmax<float>(100);
  TestArgmax<double>(100);
}

TEST(TestTensorOps, ArgmaxLargeScale) {
  TestArgmax<float>(1000);
  TestArgmax<double>(1000);

  TestArgmax<float>(10000);
  TestArgmax<double>(10000);
}
