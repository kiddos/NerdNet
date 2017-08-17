#include <gtest/gtest.h>

#include "NerdNet/convert.h"
#include "NerdNet/except/nullptr_exception.h"
#include "NerdNet/layer/fc_layer.h"
#include "NerdNet/layer/input_layer.h"
#include "NerdNet/layer/mean_square_error.h"

using nerd::nn::BaseLayer;
using nerd::nn::InputLayer;
using nerd::nn::FCLayer;
using nerd::nn::MeanSquareError;
using nerd::nn::Tensor;
using arma::Mat;
using arma::Row;

class FCLayerTest : public ::testing::Test {
 public:
  enum { INPUT_SIZE = 2, HIDDEN_SIZE = 4, OUTPUT_SIZE = 6, DATA_SIZE = 4 };
  FCLayerTest()
      : fc_layer_(nullptr, {INPUT_SIZE, HIDDEN_SIZE}),
        error_layer_(nullptr, {HIDDEN_SIZE, OUTPUT_SIZE}) {}

 protected:
  void SetUp() override {
    // prepare test data
    test_data_ = Mat<float>(DATA_SIZE, INPUT_SIZE);
    test_data_.randn();

    test_label_ = Mat<float>(DATA_SIZE, OUTPUT_SIZE);
    test_label_.randn();
  }

  InputLayer input_layer_;
  FCLayer fc_layer_;
  MeanSquareError error_layer_;

  arma::Mat<float> test_data_;
  arma::Mat<float> test_label_;
};

TEST_F(FCLayerTest, Initialization) {
  bool exception_occur = false;
  try {
    error_layer_.Init();
  } catch (nerd::nn::except::NullPtrException& e) {
    exception_occur = true;
  }
  EXPECT_TRUE(exception_occur);

  fc_layer_ = FCLayer(&input_layer_, {INPUT_SIZE, HIDDEN_SIZE});
  error_layer_ = MeanSquareError(&fc_layer_, {HIDDEN_SIZE, OUTPUT_SIZE});
  EXPECT_TRUE(error_layer_.Init());
}

TEST_F(FCLayerTest, DerivativeTest) {
  fc_layer_ = FCLayer(&input_layer_, {INPUT_SIZE, HIDDEN_SIZE});
  error_layer_ = MeanSquareError(&fc_layer_, {HIDDEN_SIZE, OUTPUT_SIZE});
  error_layer_.Init();

  input_layer_.SetInput(test_data_);
  error_layer_.SetLabel(test_label_);

  error_layer_.ComputeCost();
  Tensor<float> deriv_tensor = error_layer_.ComputeDerivative();
  Mat<float> deriv;
  Tensor2Matrix(deriv_tensor, deriv);

  Mat<float> test_input = test_data_;
  Mat<float> test_deriv(test_input.n_rows, test_input.n_cols);
  constexpr float delta = 1e-4;
  constexpr float tol = 1e-1;
  for (int i = 0; i < DATA_SIZE; ++i) {
    for (int j = 0; j < INPUT_SIZE; ++j) {
      test_input(i, j) += delta;
      input_layer_.SetInput(test_input);
      float e1 = error_layer_.ComputeCost();

      test_input(i, j) -= 2 * delta;
      input_layer_.SetInput(test_input);
      float e2 = error_layer_.ComputeCost();

      test_input(i, j) += delta;

      test_deriv(i, j) = (e1 - e2) / (2 * delta);
      EXPECT_NEAR(test_deriv(i, j), deriv(i, j), tol);
    }
  }
}

TEST_F(FCLayerTest, GradientChecking) {
  fc_layer_ = FCLayer(&input_layer_, {INPUT_SIZE, HIDDEN_SIZE});
  error_layer_ = MeanSquareError(&fc_layer_, {HIDDEN_SIZE, OUTPUT_SIZE});
  error_layer_.Init();

  input_layer_.SetInput(test_data_);
  error_layer_.SetLabel(test_label_);

  error_layer_.ComputeCost();
  error_layer_.ComputeDerivative();

  Mat<float> wgrad = fc_layer_.weight_gradient();
  Mat<float> test_w = fc_layer_.weight();
  Mat<float> numeric_wgrad(test_w.n_rows, test_w.n_cols);

  Row<float> bgrad = fc_layer_.bias_gradient();
  Row<float> test_b = fc_layer_.bias();
  Row<float> numeric_bgrad(test_b.n_cols);

  constexpr float delta = 1e-4;
  constexpr float tol = 3e-1;
  int r = test_w.n_rows;
  int c = test_w.n_cols;
  for (int i = 0; i < r; ++i) {
    for (int j = 0; j < c; ++j) {
      test_w(i, j) += delta;
      fc_layer_.set_weight(test_w);
      float e1 = error_layer_.ComputeCost();

      test_w(i, j) -= 2 * delta;
      fc_layer_.set_weight(test_w);
      float e2 = error_layer_.ComputeCost();

      test_w(i, j) += delta;

      numeric_wgrad(i, j) = (e1 - e2) / (2 * delta);
      EXPECT_NEAR(wgrad(i, j), numeric_wgrad(i, j), tol);
    }
  }

  for (int i = 0; i < c; ++i) {
    test_b(i) += delta;
    fc_layer_.set_bias(test_b);
    float e1 = error_layer_.ComputeCost();

    test_b(i) -= 2 * delta;
    fc_layer_.set_bias(test_b);
    float e2 = error_layer_.ComputeCost();

    test_b(i) += delta;

    numeric_bgrad(i) = (e1 - e2) / (2 * delta);
    EXPECT_NEAR(bgrad(i), numeric_bgrad(i), tol);
  }
}

TEST_F(FCLayerTest, Convergence) {
  fc_layer_ = FCLayer(&input_layer_, {INPUT_SIZE, HIDDEN_SIZE});
  error_layer_ = MeanSquareError(&fc_layer_, {HIDDEN_SIZE, OUTPUT_SIZE});
  error_layer_.Init();

  input_layer_.SetInput(test_data_);
  error_layer_.SetLabel(test_label_);

  constexpr float learning_rate = 1e-3;
  float cost = error_layer_.ComputeCost();
  for (int i = 0; i < 50000; ++i) {
    error_layer_.ComputeCost();
    error_layer_.ComputeDerivative();
    fc_layer_.Update(learning_rate);
    error_layer_.Update(learning_rate);
  }
  float cost2 = error_layer_.ComputeCost();
  EXPECT_LT(cost2, cost);
}
