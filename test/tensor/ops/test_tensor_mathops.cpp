#include <gtest/gtest.h>
#include <cmath>
#include "tensor/ops/basic_ops.h"
#include "tensor/ops/exp.h"
#include "tensor/ops/log.h"
#include "tensor/ops/sqrt.h"

using nn::tensor::Tensor;
using nn::tensor::TensorShape;

template <typename DType>
void TestMathOps(int d1, int d2, int d3) {
  Tensor<DType> t =
      Tensor<DType>::Gaussian({d1, d2, d3}, 0, 1.0) + static_cast<DType>(10);
  Tensor<DType> l = log(t);
  Tensor<DType> e = exp(t);
  Tensor<DType> s = sqrt(t);

  int chunk = l.chunk(0);
  EXPECT_EQ(chunk, t.chunk(0));
  for (int i = 0; i < chunk; ++i) {
    EXPECT_EQ(l[i], std::log(t[i]));
  }

  chunk = e.chunk(0);
  EXPECT_EQ(chunk, t.chunk(0));
  for (int i = 0; i < chunk; ++i) {
    EXPECT_EQ(e[i], std::exp(t[i]));
  }

  chunk = s.chunk(0);
  EXPECT_EQ(chunk, t.chunk(0));
  for (int i = 0; i < chunk; ++i) {
    EXPECT_EQ(s[i], std::sqrt(t[i]));
  }
}

TEST(TestTensorMathOps, SmallScale) {
  for (int i = 64; i > 1; i /= 2) {
    TestMathOps<float>(i, i, 1);
    TestMathOps<double>(i, i, 1);
  }
}

TEST(TestTensorMathOps, LargeScale) {
  for (int i = 64; i >= 1; i /= 2) {
    TestMathOps<float>(i, i, i);
    TestMathOps<double>(i, i, i);
  }
}
