#include <gtest/gtest.h>
#include "tensor/ops/equal.h"
#include "tensor/ops/sub.h"


template <typename DType>
void TestEqual(int size) {
  using namespace nn::tensor;

  Tensor<DType> t1 = Tensor<DType>::Eyes(size);
  Tensor<DType> t2 = Tensor<DType>::Eyes(size) - static_cast<DType>(1);

  Tensor<DType> e = t1 == t1;
  for (int i = 0 ; i < size * size ; ++i) {
    EXPECT_EQ(e[i], 1);
  }

  e = t1 == t2;
  for (int i = 0 ; i < size * size ; ++i) {
    EXPECT_EQ(e[i], 0);
  }

  e = t1 == static_cast<DType>(0);
  for (int i = 0 ; i < size ; ++i) {
    for (int j = 0 ; j < size ; ++j) {
      if (i == j) {
        EXPECT_EQ(e[i * size + j], 0);
      } else {
        EXPECT_EQ(e[i * size + j], 1);
      }
    }
  }
}

TEST(TestTensorOps, EqualSmallScale) {
  TestEqual<float>(10);
  TestEqual<double>(10);

  TestEqual<float>(100);
  TestEqual<double>(100);
}

TEST(TestTensorOps, EqualLargeScale) {
  TestEqual<float>(1000);
  TestEqual<double>(1000);

  TestEqual<float>(6000);
  TestEqual<double>(6000);
}
