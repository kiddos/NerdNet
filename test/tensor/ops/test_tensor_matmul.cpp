#include <gtest/gtest.h>
#include "tensor/ops/matmul.h"

template <typename DType>
void TestTensorMatMul(int m, int n, int k) {
  using namespace nn::tensor;

  Tensor<DType> a = Tensor<DType>::Gaussian({m, k}, 0, 1.0);
  Tensor<DType> b = Tensor<DType>::Gaussian({k, n}, 0, 1.0);
  Tensor<DType> c = matmul(a, b);

  EXPECT_EQ(c.shape().shape(0), m);
  EXPECT_EQ(c.shape().shape(1), n);

  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      DType sum = 0;
      for (int l = 0 ; l < k ; ++l) {
        sum += a[i * k + l] * b[l * n + j];
      }
      EXPECT_EQ(c[i * n + j], sum);
    }
  }
}

TEST(TestTensorMatMul, SmallScale) {
  for (int i = 1; i < 64; ++i) {
    TestTensorMatMul<float>(1, 1, i);
    TestTensorMatMul<double>(1, 1, i);

    TestTensorMatMul<float>(i, 1, i);
    TestTensorMatMul<double>(i, 1, i);

    TestTensorMatMul<float>(1, i, i);
    TestTensorMatMul<double>(1, i, i);
  }
}

TEST(TestTensorMatMul, MediumScale) {
  for (int i = 64; i <= 256; i *= 2) {
    TestTensorMatMul<float>(1, 1, i);
    TestTensorMatMul<double>(1, 1, i);

    TestTensorMatMul<float>(1, i, i);
    TestTensorMatMul<double>(1, i, i);

    TestTensorMatMul<float>(i, 1, i);
    TestTensorMatMul<double>(i, 1, i);

    TestTensorMatMul<float>(i, i, i);
    TestTensorMatMul<double>(i, i, i);
  }
}

TEST(TestTensorMatMul, LargeScale) {
  for (int i = 16 ; i >= 1 ; i /= 2) {
    TestTensorMatMul<float>(i * 1024, i * 1024, i * 1024);
    TestTensorMatMul<double>(i * 1024, i * 1024, i * 1024);
  }
}
