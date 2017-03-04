#include <gtest/gtest.h>
#include <cmath>
#include "tensor/ops/log.h"
#include "tensor/ops/add.h"

template <typename DType>
void TestLog(int d1, int d2, int d3) {
  using namespace nn::tensor;

  Tensor<DType> t = Tensor<DType>::Gaussian({d1, d2, d3}, 0, 1.0) +
      static_cast<DType>(10);
  Tensor<DType> e = log(t);

  int chunk = e.shape().chunk(0);
  for (int i = 0 ; i < chunk ; ++i) {
    EXPECT_EQ(e[i], std::log(t[i]));
  }
}

TEST(TestTensorOps, LogSmallScale) {
  TestLog<float>(10, 10, 1);
  TestLog<double>(10, 10, 1);

  TestLog<float>(100, 10, 10);
  TestLog<double>(100, 10, 10);
}

TEST(TestTensorOps, LogLargeScale) {
  TestLog<float>(100, 100, 100);
  TestLog<double>(100, 100, 100);

  TestLog<float>(600, 200, 10);
  TestLog<double>(600, 200, 10);
}
