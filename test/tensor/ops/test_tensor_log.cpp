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
  for (int i = 1 ; i < 64 ; ++i) {
    TestLog<float>(i, i, 1);
    TestLog<double>(i, i, 1);
  }
}

TEST(TestTensorOps, LogLargeScale) {
  for (int i = 1 ; i < 64 ; ++i) {
    TestLog<float>(i, i, i);
    TestLog<double>(i, i, i);
  }
}
