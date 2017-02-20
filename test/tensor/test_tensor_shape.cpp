#include <gtest/gtest.h>
#include <tensor/tensor_shape.h>

using nn::tensor::TensorShape;

TEST(TensorShapeConstructor, Empty) {
  TensorShape shape;
  EXPECT_EQ(shape.size(), 0);
}

TEST(TensorShapeConstructor, Small) {
  TensorShape shape({10});
  EXPECT_EQ(shape.chunk(0), 10);
}

TEST(TensorShapeConstructor, Medium) {
  TensorShape shape({10, 30});
  EXPECT_EQ(shape.chunk(0), 300);
  EXPECT_EQ(shape.chunk(1), 30);
}

TEST(TensorShapeConstructor, Large) {
  TensorShape shape({10, 30, 10, 20});
  EXPECT_EQ(shape.chunk(0), 60000);
  EXPECT_EQ(shape.chunk(1), 6000);
  EXPECT_EQ(shape.chunk(2), 200);
  EXPECT_EQ(shape.chunk(3), 20);
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
