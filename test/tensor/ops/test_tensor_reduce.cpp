#include <gtest/gtest.h>
#include <cmath>
#include "tensor/ops/reduce_mean.h"
#include "tensor/ops/reduce_sum.h"
#include "tensor/ops/basic_ops.h"

using nn::tensor::Tensor;
using nn::tensor::TensorShape;

template <typename DType>
void TestReduceSum(int scale, int val) {
  Tensor<DType> t1 = Tensor<DType>::Eyes(scale) * static_cast<DType>(val);
  Tensor<DType> result = reduce_sum(t1, 0);

  int shape = result.shape(0);
  for (int i = 0 ; i < shape ; ++i) {
    EXPECT_EQ(result[i], val);
  }

  result = reduce_sum(t1, 1);
  shape = result.shape(0);
  for (int i = 0 ; i < shape ; ++i) {
    EXPECT_EQ(result[i], val);
  }

  DType r = reduce_sum(t1);
  EXPECT_EQ(r, val * scale);
}

template <typename DType>
void TestReduceMean(int scale, int val) {
  Tensor<DType> t1 = Tensor<DType>::Eyes(scale) * static_cast<DType>(val);

  Tensor<DType> result = reduce_mean(t1, 0);
  int shape = result.shape(0);
  for (int i = 0 ; i < shape ; ++i) {
    DType expect = static_cast<DType>(val) / scale;
    EXPECT_LT(std::abs(result[i] - expect), 1e-6);
  }

  result = reduce_mean(t1, 1);
  shape = result.shape(0);
  for (int i = 0 ; i < shape ; ++i) {
    DType expect = static_cast<DType>(val) / scale;
    EXPECT_LT(std::abs(result[i] - expect), 1e-6);
  }

  DType r = reduce_mean(t1);
  EXPECT_LT(std::abs(r - val / scale), 1e-6);
}

TEST(TestReduceSum, SmallScale) {
  for (int i = 64 ; i >= 1 ; i /= 2) {
    TestReduceSum<float>(i, i);
    TestReduceSum<double>(i, i);
  }
}

TEST(TestReduceSum, LargeScale) {
  for (int i = 4096 ; i >= 256 ; i /= 2) {
    TestReduceSum<float>(i, i);
    TestReduceSum<double>(i, i);
  }
}

TEST(TestReduceMean, SmallScale) {
  for (int i = 64 ; i >= 1 ; i /= 2) {
    TestReduceMean<float>(i, i);
    TestReduceMean<double>(i, i);
  }
}

TEST(TestReduceMean, LargeScale) {
  for (int i = 4096 ; i >= 256 ; i /= 2) {
    TestReduceMean<float>(i, i);
    TestReduceMean<double>(i, i);
  }
}
