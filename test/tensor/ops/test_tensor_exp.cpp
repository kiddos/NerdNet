#include <gtest/gtest.h>
#include <cmath>
#include "tensor/ops/exp.h"

template <typename DType>
void TestExponential(int d1, int d2, int d3) {
  using namespace nn::tensor;

  Tensor<DType> t = Tensor<DType>::Gaussian({d1, d2, d3}, 0, 1.0);
  Tensor<DType> e = exp(t);

  int chunk = e.shape().chunk(0);
  for (int i = 0 ; i < chunk ; ++i) {
    EXPECT_EQ(e[i], std::exp(t[i]));
  }
}

TEST(TestTensorOps, ExponentialSmallScale) {
  for (int i = 1 ; i < 64 ; ++i) {
    TestExponential<float>(i, i, 1);
    TestExponential<double>(i, i, 1);
  }
}

TEST(TestTensorOps, ExponentialLargeScale) {
  for (int i = 1 ; i < 64 ; ++i) {
    TestExponential<float>(i, i, i);
    TestExponential<double>(i, i, i);
  }
}
