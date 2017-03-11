#include <gtest/gtest.h>
#include "tensor/ops/equal.h"
#include "tensor/ops/sub.h"

template <typename DType>
void TestEqual(int size) {
  using namespace nn::tensor;

  Tensor<DType> t1 = Tensor<DType>::Eyes(size);
  Tensor<DType> t2 = Tensor<DType>::Eyes(size) - static_cast<DType>(1);

  Tensor<DType> e = t1 == t1;
  for (int i = 0; i < size * size; ++i) {
    EXPECT_EQ(e[i], 1);
  }

  e = t1 == t2;
  for (int i = 0; i < size * size; ++i) {
    EXPECT_EQ(e[i], 0);
  }

  e = t1 == static_cast<DType>(0);
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      if (i == j) {
        EXPECT_EQ(e[i * size + j], 0);
      } else {
        EXPECT_EQ(e[i * size + j], 1);
      }
    }
  }
}

TEST(TestTensorOps, EqualSmallScale) {
  for (int i = 1; i < 64; ++i) {
    TestEqual<float>(i);
    TestEqual<double>(i);
  }
}

TEST(TestTensorOps, EqualLargeScale) {
  for (int i = 64; i < 1024; i *= 4) {
    TestEqual<float>(i);
    TestEqual<double>(i);
  }
}
