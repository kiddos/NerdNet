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

TEST(Tensor, ConstructorWithSingleValue) {
  Tensor<float> ftensor(1.0f);
  EXPECT_EQ(ftensor.data(0), 1.0f);

  Tensor<double> dtensor(9.0f);
  EXPECT_EQ(dtensor.data(0), 9.0f);
}

template <typename DType>
testing::AssertionResult TestCreateZeroTensor(int d1, int d2, int d3) {
  Tensor<DType> ft = Tensor<DType>::Zeros({d1, d2, d3});
  for (int i = 0 ; i < ft.shape().chunk(0) ; ++i) {
    if (ft.data(i) != 0) {
      return testing::AssertionFailure();
    }
  }
  return testing::AssertionSuccess();
}

TEST(Tensor, CreateZerosTensor) {
  EXPECT_TRUE(TestCreateZeroTensor<float>(200, 200, 10));
  EXPECT_TRUE(TestCreateZeroTensor<double>(200, 200, 20));
}

template <typename DType>
testing::AssertionResult TestCreateOneTensor(int d1, int d2, int d3) {
  Tensor<DType> ft = Tensor<DType>::Ones({d1, d2, d3});
  for (int i = 0 ; i < ft.shape().chunk(0) ; ++i) {
    if (ft.data(i) != 1) {
      return testing::AssertionFailure();
    }
  }
  return testing::AssertionSuccess();
}

TEST(Tensor, CreateOnesTensor) {
  EXPECT_TRUE(TestCreateOneTensor<float>(300, 200, 10));
  EXPECT_TRUE(TestCreateOneTensor<double>(100, 200, 20));
}

template <typename DType>
testing::AssertionResult TestCreateIdentityMatixTensor(int size) {
  Tensor<DType> ft = Tensor<DType>::Eyes(size);
  for (int i = 0 ; i < ft.shape().shape(0) ; ++i) {
    for (int j = 0 ; j < ft.shape().shape(1) ; ++j) {
      if (i == j) {
        if (ft.data(i * ft.shape().shape(1) + j) != 1) {
          return testing::AssertionFailure();
        }
      } else {
        if (ft.data(i * ft.shape().shape(1) + j) != 0) {
          return testing::AssertionFailure();
        }
      }
    }
  }
  return testing::AssertionSuccess();
}

TEST(Tensor, CreateIdentityMatrixTensor) {
  EXPECT_TRUE(TestCreateIdentityMatixTensor<float>(300));
  EXPECT_TRUE(TestCreateIdentityMatixTensor<float>(600));
}

template <typename DType>
testing::AssertionResult TestCreateGaussianTensor(int d1, int d2, int d3) {
  Tensor<DType> ft = Tensor<DType>::Gaussian({d1, d2, d3}, 0, 1.0);
  for (int i = 0 ; i < ft.shape().shape(0) ; ++i) {
    for (int j = 0 ; j < ft.shape().shape(1) ; ++j) {
      if (i == j) {
        if (ft.data(i * ft.shape().shape(1) + j) != 1) {
          testing::AssertionFailure();
        }
      } else {
        if (ft.data(i * ft.shape().shape(1) + j) != 0) {
          testing::AssertionFailure();
        }
      }
    }
  }
  return testing::AssertionSuccess();
}

TEST(TestTensor, CreateGaussianTensor) {
  EXPECT_TRUE(TestCreateGaussianTensor<float>(100, 200, 10));
  EXPECT_TRUE(TestCreateGaussianTensor<double>(200, 100, 20));
}

template <typename DType>
testing::AssertionResult TestReshapeTensor() {
  Tensor<DType> ft = Tensor<DType>::Eyes(300);
  ft.Reshape({300*300});
  if (ft.shape().chunk(0) == 300 * 300 && ft.shape().shape(0)) {
    return testing::AssertionSuccess();
  } else {
    return testing::AssertionFailure();
  }
}

TEST(Tensor, ReshapeTensor) {
  EXPECT_TRUE(TestReshapeTensor<float>());
  EXPECT_TRUE(TestReshapeTensor<double>());
}

template <typename DType>
testing::AssertionResult TestCopyTensor(int d1, int d2) {
  Tensor<DType> t1 = Tensor<DType>::Gaussian({d1, d2}, 0.0f, 1.0f);
  Tensor<DType> t2 = Tensor<DType>::Gaussian({d1, d2}, 0.0f, 1.0f);
  DType* original_ptr = &t1[0];
  t1 = t2;
  if (original_ptr == &t1[0]) {
    return testing::AssertionSuccess() << "No reallocation of memeory";
  } else {
    return testing::AssertionFailure()
        << "Tensor data pointer changed, Memory reallocation occur";
  }
}

TEST(Tensor, CopyTensor) {
  EXPECT_TRUE(TestCopyTensor<float>(300, 300));
  EXPECT_TRUE(TestCopyTensor<double>(300, 300));
}
