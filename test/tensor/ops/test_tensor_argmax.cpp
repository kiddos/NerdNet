#include <gtest/gtest.h>
#include "tensor/ops/argmax.h"
#include "tensor/ops/mul.h"

template <typename DType>
void TestArgmax(int size) {
  using namespace nn::tensor;

  Tensor<DType> t = Tensor<DType>::Eyes(size) * static_cast<DType>(10);

  Tensor<DType> argmax = nn::tensor::argmax(t, 1);
  for (int i = 0; i < size; ++i) {
    EXPECT_EQ(argmax[i], 10);
  }

  argmax = nn::tensor::argmax(t, 0);
  for (int i = 0; i < size; ++i) {
    EXPECT_EQ(argmax[i], 10);
  }
}

TEST(TestTensorArgmax, SmallScale) {
  for (int i = 1; i < 64; ++i) {
    TestArgmax<float>(i);
    TestArgmax<double>(i);
  }
}

TEST(TestTensorArgmax, LargeScale) {
  for (int i = 64; i <= 1024; ++i) {
    TestArgmax<float>(i);
    TestArgmax<double>(i);
  }
}
