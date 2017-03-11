#include <gtest/gtest.h>
#include <cmath>
#include "tensor/ops/basic_ops.h"

using nn::tensor::Tensor;
using nn::tensor::TensorShape;

template <typename DType>
void TestBasicOps(TensorShape shape, DType mean, DType stddev, DType val) {
  Tensor<DType> t = Tensor<DType>::Gaussian(shape, mean, stddev);

  Tensor<DType> t2 = t + val;
  for (int i = 0; i < shape.chunk(0); ++i) {
    EXPECT_EQ(t2[i], t[i] + val);
  }

  t2 = t - val;
  for (int i = 0; i < shape.chunk(0); ++i) {
    EXPECT_EQ(t2[i], t[i] - val);
  }

  t2 = t * val;
  for (int i = 0; i < shape.chunk(0); ++i) {
    EXPECT_EQ(t2[i], t[i] * val);
  }

  t2 = t / val;
  for (int i = 0; i < shape.chunk(0); ++i) {
    EXPECT_EQ(t2[i], t[i] / val);
  }

  t2 = t ^ val;
  for (int i = 0; i < shape.chunk(0); ++i) {
    EXPECT_EQ(t2[i], std::pow(t[i], val));
  }
}

template <typename DType>
void TestBasicOps(TensorShape shape, DType mean1, DType mean2, DType stddev1,
                  DType stddev2) {
  Tensor<DType> t1 = Tensor<DType>::Gaussian(shape, mean1, stddev1);
  Tensor<DType> t2 = Tensor<DType>::Gaussian(shape, mean2, stddev2);

  Tensor<DType> t3 = t1 + t2;
  for (int i = 0; i < shape.chunk(0); ++i) {
    EXPECT_EQ(t3[i], t1[i] + t2[i]);
  }

  t3 = t1 - t2;
  for (int i = 0; i < shape.chunk(0); ++i) {
    EXPECT_EQ(t3[i], t1[i] - t2[i]);
  }

  t3 = t1 * t2;
  for (int i = 0; i < shape.chunk(0); ++i) {
    EXPECT_EQ(t3[i], t1[i] * t2[i]);
  }

  t3 = t1 / t2;
  for (int i = 0; i < shape.chunk(0); ++i) {
    EXPECT_EQ(t3[i], t1[i] / t2[i]);
  }

  t3 = t1 ^ t2;
  for (int i = 0; i < shape.chunk(0); ++i) {
    if (std::isnan(t3[i])) {
      EXPECT_TRUE(std::isnan(t3[i]) == std::isnan(std::pow(t1[i], t2[i])));
    } else {
      EXPECT_EQ(t3[i], std::pow(t1[i], t2[i]));
    }
  }
}

template <typename DType>
void TestBroadcastOpsTensor(TensorShape shape1, TensorShape shape2, DType mean1,
                            DType mean2, DType stddev1, DType stddev2) {
  EXPECT_GT(shape1.size(), shape2.size());
  int offset = shape1.size() - shape2.size();
  int shape = shape1.shape(offset - 1);
  int chunk = shape1.chunk(offset);

  Tensor<DType> t1 = Tensor<DType>::Gaussian(shape1, mean1, stddev1);
  Tensor<DType> t2 = Tensor<DType>::Gaussian(shape2, mean2, stddev2);

  Tensor<DType> t3 = t1 + t2;
  for (int i = 0; i < shape; ++i) {
    for (int j = 0; j < chunk; ++j) {
      EXPECT_EQ(t3[i * chunk + j], t1[i * chunk + j] + t2[j]);
    }
  }

  t3 = t1 - t2;
  for (int i = 0; i < shape; ++i) {
    for (int j = 0; j < chunk; ++j) {
      EXPECT_EQ(t3[i * chunk + j], t1[i * chunk + j] - t2[j]);
    }
  }

  t3 = t1 * t2;
  for (int i = 0; i < shape; ++i) {
    for (int j = 0; j < chunk; ++j) {
      EXPECT_EQ(t3[i * chunk + j], t1[i * chunk + j] * t2[j]);
    }
  }

  t3 = t1 / t2;
  for (int i = 0; i < shape; ++i) {
    for (int j = 0; j < chunk; ++j) {
      EXPECT_EQ(t3[i * chunk + j], t1[i * chunk + j] / t2[j]);
    }
  }
}

TEST(TestTensorBasicOps, ZeroShape) {
  TestBasicOps(TensorShape({0}), 0.0f, 0.0f, 1.0f, 1.0f);
  TestBasicOps(TensorShape({0}), 0.0, 0.0, 1.0, 1.0);
}

TEST(TestTensorBasicOps, SmallScale) {
  for (int i = 64; i >= 1; i /= 2) {
    TestBasicOps(TensorShape({1, i}), 0.0f, 0.0f, 1.0f, 1.0f);
    TestBasicOps(TensorShape({1, i}), 0.0, 0.0, 1.0, 1.0);

    TestBasicOps(TensorShape({i, 1}), 0.0f, 0.0f, 1.0f, 1.0f);
    TestBasicOps(TensorShape({i, 1}), 0.0, 0.0, 1.0, 1.0);

    TestBasicOps(TensorShape({i, i}), 0.0f, 0.0f, 1.0f, 1.0f);
    TestBasicOps(TensorShape({i, i}), 0.0, 0.0, 1.0, 1.0);
  }
}

TEST(TestTensorBroadcastBasicOps, SmallScale) {
  for (int i = 64; i >= 1; i /= 2) {
    TestBroadcastOpsTensor(TensorShape({i, i}), TensorShape({i}), 0.0f, 0.0f,
                           1.0f, 1.0f);
    TestBroadcastOpsTensor(TensorShape({i, i}), TensorShape({i}), 0.0, 0.0, 1.0,
                           1.0);
  }
}

TEST(TestTensorBasicOps, MediumScale) {
  for (int i = 64; i >= 1; i /= 2) {
    TestBasicOps(TensorShape({1, i, 1}), 0.0f, 0.0f, 1.0f, 1.0f);
    TestBasicOps(TensorShape({1, i, 1}), 0.0, 0.0, 1.0, 1.0);

    TestBasicOps(TensorShape({i, 1}), 0.0f, 0.0f, 1.0f, 1.0f);
    TestBasicOps(TensorShape({i, 1}), 0.0, 0.0, 1.0, 1.0);

    TestBasicOps(TensorShape({i, i, 1}), 0.0f, 0.0f, 1.0f, 1.0f);
    TestBasicOps(TensorShape({i, i, 1}), 0.0, 0.0, 1.0, 1.0);
  }
}

TEST(TestTensorBroadcastBasicOps, MediumScale) {
  for (int i = 64; i >= 1; i /= 2) {
    TestBroadcastOpsTensor(TensorShape({i, 1, i}), TensorShape({i}), 0.0f,
                           0.0f, 1.0f, 1.0f);
    TestBroadcastOpsTensor(TensorShape({i, 1, i}), TensorShape({i}), 0.0,
                           0.0, 1.0, 1.0);

    TestBroadcastOpsTensor(TensorShape({i, i, i}), TensorShape({i}), 0.0f,
                           0.0f, 1.0f, 1.0f);
    TestBroadcastOpsTensor(TensorShape({i, i, i}), TensorShape({i}), 0.0,
                           0.0, 1.0, 1.0);
  }
}

TEST(TestTensorBasicOps, LargeScale) {
  TestBasicOps<float>(TensorShape({1024, 1024}), 0.0f, 0.0f, 1.0f, 1.0f);
  TestBasicOps<double>(TensorShape({1024, 1024}), 0.0f, 0.0f, 1.0, 1.0);

  for (int i = 16; i >= 1; i /= 2) {
    TestBasicOps<float>(TensorShape({1, i, 1, i}), 0.0f, 0.0f, 1.0f, 1.0f);
    TestBasicOps<double>(TensorShape({1, i, 1, i}), 0.0f, 0.0f, 1.0, 1.0);

    TestBasicOps<float>(TensorShape({i, i, i, 1}), 0.0f, 0.0f, 1.0f, 1.0f);
    TestBasicOps<double>(TensorShape({i, i, i, 1}), 0.0f, 0.0f, 1.0, 1.0);

    TestBasicOps<float>(TensorShape({i, i, i, i}), 0.0f, 0.0f, 1.0f, 1.0f);
    TestBasicOps<double>(TensorShape({i, i, i, i}), 0.0f, 0.0f, 1.0, 1.0);
  }
}

TEST(TestTensorBroadcastBasicOps, LargeScale) {
  for (int i = 16; i >= 1; i /= 2) {
    TestBroadcastOpsTensor(TensorShape({i, i, i}), TensorShape({i, i}), 0.0f,
                           0.0f, 1.0f, 1.0f);
    TestBroadcastOpsTensor(TensorShape({i, i, i}), TensorShape({i, i}), 0.0,
                           0.0, 1.0, 1.0);
  }
}
