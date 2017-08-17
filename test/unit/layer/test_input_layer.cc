#include <gtest/gtest.h>

#include "NerdNet/convert.h"
#include "NerdNet/layer/input_layer.h"

using nerd::nn::InputLayer;
using nerd::nn::Tensor;
using arma::Mat;
using arma::Cube;
using arma::field;

class InputLayerTest : public ::testing::Test {
 public:
  enum { DATA_SIZE = 1024, INPUT_SIZE = 2 };

 protected:
  void SetUp() override {
    arma::arma_rng::set_seed_random();
  }

  InputLayer input_layer_;
};

TEST_F(InputLayerTest, TensorInput) {
  Mat<float> test_input(DATA_SIZE, INPUT_SIZE);
  test_input.randn();

  Tensor<float> test_input_tensor;
  Matrix2Tensor(test_input, test_input_tensor);
  input_layer_.SetInput(test_input_tensor);

  Tensor<float> target = input_layer_.ForwardProp();
  EXPECT_EQ(target.shape().size(), 2);
  EXPECT_EQ(target.shape()[0], DATA_SIZE);
  EXPECT_EQ(target.shape()[1], INPUT_SIZE);

  for (int i = 0; i < target.shape()[0]; ++i) {
    for (int j = 0; j < target.shape()[1]; ++j) {
      EXPECT_EQ(target.data()[j * target.shape()[0] + i],
                test_input_tensor.data()[j * target.shape()[0] + i]);
    }
  }
}

TEST_F(InputLayerTest, MatrixInput) {
  Mat<float> test_input(DATA_SIZE, INPUT_SIZE);
  test_input.randn();
  input_layer_.SetInput(test_input);

  Tensor<float> target = input_layer_.ForwardProp();
  EXPECT_EQ(target.shape().size(), 2);
  EXPECT_EQ(target.shape()[0], DATA_SIZE);
  EXPECT_EQ(target.shape()[1], INPUT_SIZE);

  const float* data = target.data();
  for (int i = 0; i < target.shape()[0]; ++i) {
    for (int j = 0; j < target.shape()[1]; ++j) {
      EXPECT_EQ(data[j * target.shape()[0] + i], test_input(i, j));
    }
  }
}

TEST_F(InputLayerTest, CubeInput) {
  Cube<float> test_input(DATA_SIZE, INPUT_SIZE, INPUT_SIZE);
  test_input.randn();
  input_layer_.SetInput(test_input);

  Tensor<float> target_tensor = input_layer_.ForwardProp();
  EXPECT_EQ(target_tensor.shape().size(), 3);
  EXPECT_EQ(target_tensor.shape()[0], DATA_SIZE);
  EXPECT_EQ(target_tensor.shape()[1], INPUT_SIZE);
  EXPECT_EQ(target_tensor.shape()[2], INPUT_SIZE);

  Cube<float> target;
  Tensor2Cube(target_tensor, target);
  int r = target.n_rows;
  int c = target.n_cols;
  int s = target.n_slices;
  for (int i = 0; i < r; ++i) {
    for (int j = 0; j < c; ++j) {
      for (int k = 0; k < s; ++k) {
        EXPECT_EQ(target(i, j, k), test_input(i, j, k));
      }
    }
  }
}

TEST_F(InputLayerTest, MatricesInput) {
  field<Mat<float>> test_input(DATA_SIZE);
  for (int i = 0; i < DATA_SIZE; ++i) {
    Mat<float> m(INPUT_SIZE, INPUT_SIZE);
    m.randn();
    test_input[i] = m;
  }
  input_layer_.SetInput(test_input);

  Tensor<float> target_tensor = input_layer_.ForwardProp();
  EXPECT_EQ(target_tensor.shape().size(), 3);
  EXPECT_EQ(target_tensor.shape()[0], DATA_SIZE);
  EXPECT_EQ(target_tensor.shape()[1], INPUT_SIZE);
  EXPECT_EQ(target_tensor.shape()[2], INPUT_SIZE);

  field<Mat<float>> target;
  Tensor2Matrices(target_tensor, target);
  int e = target.n_elem;
  for (int i = 0; i < e; ++i) {
    Mat<float> t = target[i];
    Mat<float> in = test_input[i];
    int r = t.n_rows;
    int c = t.n_cols;
    for (int j = 0; j < r; ++j) {
      for (int k = 0; k < c; ++k) {
        EXPECT_EQ(t(j, k), in(j, k));
      }
    }
  }
}


TEST_F(InputLayerTest, CubesInput) {
  field<Cube<float>> test_input(DATA_SIZE);
  for (int i = 0; i < DATA_SIZE; ++i) {
    Cube<float> c(INPUT_SIZE, INPUT_SIZE, INPUT_SIZE);
    c.randn();
    // test_input << c;
    test_input[i] = c;
  }
  input_layer_.SetInput(test_input);

  Tensor<float> target_tensor = input_layer_.ForwardProp();
  EXPECT_EQ(target_tensor.shape().size(), 4);
  EXPECT_EQ(target_tensor.shape()[0], DATA_SIZE);
  EXPECT_EQ(target_tensor.shape()[1], INPUT_SIZE);
  EXPECT_EQ(target_tensor.shape()[2], INPUT_SIZE);

  field<Cube<float>> target;
  Tensor2Cubes(target_tensor, target);
  int e = target.n_elem;
  for (int i = 0; i < e; ++i) {
    Cube<float> t = target[i];
    Cube<float> in = test_input[i];
    int r = t.n_rows;
    int c = t.n_cols;
    int s = t.n_slices;
    for (int j = 0; j < r; ++j) {
      for (int k = 0; k < c; ++k) {
        for (int l = 0; l < s; ++l) {
          EXPECT_EQ(t(j, k, l), in(j, k, l));
        }
      }
    }
  }
}
