#include <gtest/gtest.h>
#include "tensor/ops/sub.h"

using nn::tensor::Tensor;
using nn::tensor::TensorShape;

template <typename DType>
void TestSubtractTensor(TensorShape shape, DType mean1, DType mean2,
                        DType stddev1, DType stddev2) {
  Tensor<DType> t1 = Tensor<DType>::Gaussian(shape, mean1, stddev1);
  Tensor<DType> t2 = Tensor<DType>::Gaussian(shape, mean2, stddev2);
  Tensor<DType> t3 = t1 - t2;

  for (int i = 0; i < shape.chunk(0); ++i) {
    EXPECT_EQ(t3[i], t1[i] - t2[i]);
  }
}

template <typename DType>
void TestBroadcastSubtractTensor(TensorShape shape1, TensorShape shape2,
                                 DType mean1, DType mean2, DType stddev1,
                                 DType stddev2) {
  Tensor<DType> t1 = Tensor<DType>::Gaussian(shape1, mean1, stddev1);
  Tensor<DType> t2 = Tensor<DType>::Gaussian(shape2, mean2, stddev2);
  Tensor<DType> t3 = t1 - t2;

  EXPECT_GT(shape1.size(), shape2.size());
  int offset = shape1.size() - shape2.size();
  int shape = shape1.shape(offset - 1);
  int chunk = shape1.chunk(offset);

  for (int i = 0; i < shape; ++i) {
    for (int j = 0; j < chunk; ++j) {
      EXPECT_EQ(t3[i * chunk + j], t1[i * chunk + j] - t2[j]);
    }
  }
}

TEST(TestTensorSubtact, SubtractSmallTensor) {
  TestSubtractTensor(TensorShape({30, 6}), 0.0f, 0.0f, 10.0f, 20.0f);
  TestSubtractTensor(TensorShape({60, 12}), 0.0, 0.0, 30.0, 60.0);
}

TEST(TestTensorSubtractBroadcast, SubtractBroadcastSmallTensor) {
  TestBroadcastSubtractTensor(TensorShape({30, 12}), TensorShape({12}), 0.0f,
                              0.0f, 30.0f, 10.0f);
  TestBroadcastSubtractTensor(TensorShape({30, 12, 1}), TensorShape({12, 1}),
                              0.0f, 0.0f, 10.0f, 0.5f);
  TestBroadcastSubtractTensor(TensorShape({90, 30, 3}), TensorShape({30, 3}),
                              0.0, 0.0, 1.0, 0.6);
}

TEST(TestTensorSubtract, SubtractMediumTensor) {
  TestSubtractTensor(TensorShape({90, 30, 10}), 0.0f, 0.0f, 10.0f, 20.0f);
  TestSubtractTensor(TensorShape({60, 90, 20}), 0.0, 0.0, 30.0, 60.0);
}

TEST(TestTensorSubtractBroadcast, SubtractBroadcastMediumTensor) {
  TestBroadcastSubtractTensor(TensorShape({100, 60, 30}), TensorShape({60, 30}),
                              0.0f, 0.0f, 30.0f, 20.0f);
  TestBroadcastSubtractTensor(TensorShape({300, 30, 30}), TensorShape({30}),
                              0.0, 0.0, 100.0, 40.0);
}

TEST(TestTensorSubtract, SubtractLargeTensor) {
  TestSubtractTensor(TensorShape({100, 60, 60, 3}), 0.0f, 0.0f, 60.0f, 10.0f);
  TestSubtractTensor(TensorShape({100, 30, 30, 6}), 0.0, 0.0, 80.0, 20.0);
}

TEST(TestTensorSubtractBroadcast, SubtractBroadcastLargeTensor) {
  TestBroadcastSubtractTensor(TensorShape({100, 32, 32, 64}), TensorShape({64}),
                              0.0f, 0.0f, 30.0f, 1.0f);
  TestBroadcastSubtractTensor(TensorShape({100, 28, 28, 128}),
                              TensorShape({128}), 0.0, 0.0, 100.0, 3.0);
}
