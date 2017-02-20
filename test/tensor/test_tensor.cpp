#include <gtest/gtest.h>
#include "tensor/tensor.h"

using nn::tensor::Tensor;

TEST(Tensor, EmptyContructor) {
  Tensor<float> ftensor;
  EXPECT_EQ(ftensor.ptr<float>(), nullptr);

  Tensor<double> dtensor;
  EXPECT_EQ(dtensor.ptr<double>(), nullptr);
}

TEST(Tensor, ConstructorWithShape) {
  Tensor<float> ftensor({100, 200});
  EXPECT_NE(ftensor.ptr<float>(), nullptr);
  EXPECT_EQ(ftensor.ptr<float>()[0], 0);
  EXPECT_EQ(ftensor.ptr<float>()[100*200-1], 0);

  Tensor<double> dtensor({100, 200});
  EXPECT_NE(dtensor.ptr<double>(), nullptr);
  EXPECT_EQ(dtensor.ptr<double>()[0], 0);
  EXPECT_EQ(dtensor.ptr<double>()[100*200-1], 0);
}

TEST(Tensor, CreateZerosTensor) {
  Tensor<float> ft = Tensor<float>::Zeros({300, 200});
  for (int i = 0 ; i < ft.shape().chunk(0) ; ++i) {
    EXPECT_EQ(ft.data(i), 0);
  }
}

TEST(Tensor, CreateOnesTensor) {
  Tensor<float> ft = Tensor<float>::Ones({600, 200});
  for (int i = 0 ; i < ft.shape().chunk(0) ; ++i) {
    EXPECT_EQ(ft.data(i), 1);
  }
}

TEST(Tensor, CreateIdentityMatrixTensor) {
  Tensor<float> ft = Tensor<float>::Eyes(300);
  for (int i = 0 ; i < ft.shape().shape(0) ; ++i) {
    for (int j = 0 ; j < ft.shape().shape(1) ; ++j) {
      if (i == j) {
        EXPECT_EQ(ft.data(i * ft.shape().shape(1) + j), 1);
      } else {
        EXPECT_EQ(ft.data(i * ft.shape().shape(1) + j), 0);
      }
    }
  }
}

TEST(Tensor, ReshapeTensor) {
  Tensor<float> ft = Tensor<float>::Eyes(300);
  ft.Reshape({300*300});
  EXPECT_EQ(ft.shape().chunk(0), 300*300);
  EXPECT_EQ(ft.shape().shape(0), 300*300);
}
