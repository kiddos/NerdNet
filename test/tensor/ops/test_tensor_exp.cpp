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
  TestExponential<float>(10, 10, 1);
  TestExponential<double>(10, 10, 1);

  TestExponential<float>(100, 10, 10);
  TestExponential<double>(100, 10, 10);
}

TEST(TestTensorOps, ExponentialLargeScale) {
  TestExponential<float>(100, 100, 100);
  TestExponential<double>(100, 100, 100);

  TestExponential<float>(600, 200, 10);
  TestExponential<double>(600, 200, 10);
}
