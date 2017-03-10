#include <gtest/gtest.h>
#include <cmath>
#include "tensor/tensor_shape.h"

using nn::tensor::TensorShape;

TEST(TensorShapeConstructor, Empty) {
  TensorShape shape;
  EXPECT_EQ(shape.size(), 0);
}

TEST(TensorShapeConstructor, Zero) {
  TensorShape shape({0});
  EXPECT_EQ(shape.shape(0), 0);
  EXPECT_EQ(shape.chunk(0), 1);
}

TEST(TensorShapeConstructor, SmallDimension) {
  // using max size of 1024 because tensor should usually be construct with 1024
  // or smaller chunk
  for (int i = 1 ; i <= 1024 ; ++i) {
    TensorShape shape({i});
    EXPECT_EQ(shape.chunk(0), i);
    EXPECT_EQ(shape.shape(0), i);

    shape = TensorShape({i, i});
    EXPECT_EQ(shape.chunk(0), std::pow(i, 2));
    EXPECT_EQ(shape.shape(0), i);
    EXPECT_EQ(shape.shape(1), i);
  }
}

TEST(TensorShapeConstructor, MediumDimension) {
  for (int i = 1 ; i < 1024 ; ++i) {
    TensorShape shape({i, i, i});
    for (int j = 0 ; j < 3 ; ++j) {
      EXPECT_EQ(shape.chunk(j), std::pow(i, 3 - j));
    }
    for (int j = 0 ; j < 3 ; ++j) {
      EXPECT_EQ(shape.shape(j), i);
    }
  }
}

int factorial(int n) {
  if (n <= 1) return 1;
  else return n * factorial(n - 1);
}

TEST(TensorShapeConstructor, LargeDimension) {
  std::vector<int> v;
  int max_shape = 12;
  for (int i = 1 ; i <= max_shape ; ++i) v.push_back(i);
  TensorShape temp(v);
  EXPECT_EQ(temp.chunk(0), factorial(max_shape));
}

TEST(TensorShape, CopyConstructor) {
  TensorShape temp({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  TensorShape shape(temp);

  for (int i = 0 ; i < 10; ++i) {
    EXPECT_EQ(shape.chunk(i), temp.chunk(i));
  }
}

TEST(TensorShape, Copy) {
  TensorShape temp({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  TensorShape shape = temp;

  for (int i = 0 ; i < 10; ++i) {
    EXPECT_EQ(shape.chunk(i), temp.chunk(i));
  }
}
