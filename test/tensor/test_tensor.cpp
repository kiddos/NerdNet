#include <gtest/gtest.h>
#include <chrono>
#include <random>
#include <cmath>
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

TEST(Tensor, ConstructorWithSingleValue) {
  Tensor<float> ftensor(1.0f);
  EXPECT_EQ(ftensor.data(0), 1.0f);

  Tensor<double> dtensor(9.0f);
  EXPECT_EQ(dtensor.data(0), 9.0f);
}

int randrange(int begin, int end) {
  std::mt19937 gen(std::chrono::system_clock::now().time_since_epoch().count());
  std::uniform_int_distribution<int> dist(begin, end);
  return dist(gen) + 1;
}

template <typename DType>
void TestCreateZeroTensor(int d1, int d2, int d3) {
  Tensor<DType> ft = Tensor<DType>::Zeros({d1, d2, d3});
  for (int i = 0 ; i < ft.shape().chunk(0) ; ++i) {
    EXPECT_EQ(ft.data(i), 0);
  }
}

TEST(Tensor, CreateZerosTensor) {
  int d1 = randrange(100, 200);
  int d2 = randrange(100, 200);
  int d3 = randrange(10, 20);
  TestCreateZeroTensor<float>(d1, d2, d3);
  TestCreateZeroTensor<double>(d1, d2, d3);
}

template <typename DType>
void TestCreateOneTensor(int d1, int d2, int d3) {
  Tensor<DType> ft = Tensor<DType>::Ones({d1, d2, d3});
  for (int i = 0 ; i < ft.shape().chunk(0) ; ++i) {
    EXPECT_EQ(ft.data(i), 1);
  }
}

TEST(Tensor, CreateOnesTensor) {
  int d1 = randrange(100, 200);
  int d2 = randrange(100, 200);
  int d3 = randrange(10, 20);
  TestCreateOneTensor<float>(d1, d2, d3);
  TestCreateOneTensor<double>(d1, d2, d3);
}

template <typename DType>
void TestCreateIdentityMatixTensor(int size) {
  Tensor<DType> ft = Tensor<DType>::Eyes(size);
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

TEST(Tensor, CreateIdentityMatrixTensor) {
  int d = randrange(100, 600);
  TestCreateIdentityMatixTensor<float>(d);
  TestCreateIdentityMatixTensor<double>(d);
}

template <typename DType>
void TestCreateGaussianTensor(int d1, int d2, int d3) {
  Tensor<DType> t = Tensor<DType>::Gaussian({d1, d2, d3}, 0, 1.0);
  int chunk = t.shape().chunk(0);
  DType sum = 0;
  for (int i = 0 ; i < chunk ; ++i) {
    sum += t.data(i);
  }
  DType mean = sum / chunk;

  DType sq = 0;
  for (int i = 0 ; i < chunk ; ++i) {
    sq += std::pow(mean - t.data(i), 2);
  }
  EXPECT_LT(std::abs(mean), 1e-2);
  DType stddev = std::sqrt(sq / chunk);
  EXPECT_LT(std::abs(stddev - 1.0), 1e-2);
}

TEST(TestTensor, CreateGaussianTensor) {
  int d1 = randrange(100, 200);
  int d2 = randrange(100, 200);
  int d3 = randrange(10, 20);
  TestCreateGaussianTensor<float>(d1, d2, d3);
  TestCreateGaussianTensor<double>(d1, d2, d3);
}

template <typename DType>
void TestReshapeTensor(int size) {
  Tensor<DType> ft = Tensor<DType>::Eyes(size);
  ft.Reshape({size*size});
  EXPECT_EQ(ft.shape().chunk(0), size * size);
  EXPECT_EQ(ft.shape().shape(0), size * size);
}

TEST(Tensor, ReshapeTensor) {
  int d = randrange(300, 600);
  TestReshapeTensor<float>(d);
  TestReshapeTensor<double>(d);
}

template <typename DType>
void TestCopyTensor(int d1, int d2) {
  Tensor<DType> t1 = Tensor<DType>::Gaussian({d1, d2}, 0.0f, 1.0f);
  Tensor<DType> t2 = Tensor<DType>::Gaussian({d1, d2}, 0.0f, 1.0f);
  DType* original_ptr = &t1[0];
  t1 = t2;
  EXPECT_EQ(original_ptr, &t1[0]);
}

TEST(Tensor, CopyTensor) {
  TestCopyTensor<float>(300, 300);
  TestCopyTensor<double>(300, 300);
}
