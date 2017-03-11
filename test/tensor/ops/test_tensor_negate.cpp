#include <gtest/gtest.h>
#include <vector>
#include "tensor/ops/negate.h"

template <typename DType>
void TestTensorNegate(std::vector<int> shape) {
  using namespace nn::tensor;

  Tensor<DType> t = Tensor<DType>::Gaussian(shape, 0, 1.0);
  Tensor<DType> n = -t;

  int chunk = t.chunk(0);
  for (int i = 0 ; i < chunk ; ++i) {
    EXPECT_EQ(t[i], -n[i]);
  }
}

TEST(TestTensorNegate, SmallScale) {
  for (int i = 1 ; i < 64 ; ++i) {
    TestTensorNegate<float>({i});
    TestTensorNegate<double>({i});

    TestTensorNegate<float>({i, i});
    TestTensorNegate<double>({i, i});
  }
}

TEST(TestTensorNegate, MediumScale) {
  for (int i = 1 ; i < 64 ; ++i) {
    TestTensorNegate<float>({i, i, i});
    TestTensorNegate<double>({i, i, i});
  }
}

TEST(TestTensorNegate, LargeScale) {
  TestTensorNegate<float>({1024, 1024});
  TestTensorNegate<double>({1024, 1024});
}
