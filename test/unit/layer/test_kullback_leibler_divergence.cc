#include <gtest/gtest.h>

#include "NerdNet/convert.h"
#include "NerdNet/except/nullptr_exception.h"
#include "NerdNet/layers.h"

#include <chrono>
#include <random>

using nerd::nn::BaseLayer;
using nerd::nn::InputLayer;
using nerd::nn::KullbackLeiblerDivergence;
using nerd::nn::Tensor;
using arma::Mat;
using arma::Row;

class KullbackLeiblerDivergenceTest : public ::testing::Test {
 public:
  enum { INPUT_SIZE = 2, OUTPUT_SIZE = 6, DATA_SIZE = 4 };

  KullbackLeiblerDivergenceTest()
      : error_layer_(nullptr, {INPUT_SIZE, OUTPUT_SIZE}) {}

 protected:
  void SetUp() override {
    // prepare test data
    test_data_ = Mat<float>(DATA_SIZE, INPUT_SIZE);
    test_data_.randn();
    test_data_ += 10;

    test_label_ = Mat<float>(DATA_SIZE, OUTPUT_SIZE);
    test_label_.randu();
  }

  InputLayer input_layer_;
  KullbackLeiblerDivergence error_layer_;

  arma::Mat<float> test_data_;
  arma::Mat<float> test_label_;
};

TEST_F(KullbackLeiblerDivergenceTest, Initialization) {
  bool exception_occur = false;
  try {
    error_layer_.Init();
  } catch (nerd::nn::except::NullPtrException& e) {
    exception_occur = true;
  }
  EXPECT_TRUE(exception_occur);

  error_layer_ =
      KullbackLeiblerDivergence(&input_layer_, {INPUT_SIZE, OUTPUT_SIZE});
  EXPECT_TRUE(error_layer_.Init());
}

TEST_F(KullbackLeiblerDivergenceTest, DerivativeTest) {
  error_layer_ =
      KullbackLeiblerDivergence(&input_layer_, {INPUT_SIZE, OUTPUT_SIZE});
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
  constexpr float tol = 3e-1;
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

TEST_F(KullbackLeiblerDivergenceTest, GradientChecking) {
  error_layer_ =
      KullbackLeiblerDivergence(&input_layer_, {INPUT_SIZE, OUTPUT_SIZE});
  error_layer_.Init();

  input_layer_.SetInput(test_data_);
  error_layer_.SetLabel(test_label_);

  error_layer_.ComputeCost();
  error_layer_.ComputeDerivative();

  Mat<float> wgrad = error_layer_.weight_gradient();
  Mat<float> test_w = error_layer_.weight();
  Mat<float> numeric_wgrad(test_w.n_rows, test_w.n_cols);

  Row<float> bgrad = error_layer_.bias_gradient();
  Row<float> test_b = error_layer_.bias();
  Row<float> numeric_bgrad(test_b.n_cols);

  constexpr float delta = 1e-4;
  constexpr float tol = 3e-1;
  int r = test_w.n_rows;
  int c = test_w.n_cols;
  for (int i = 0; i < r; ++i) {
    for (int j = 0; j < c; ++j) {
      test_w(i, j) += delta;
      error_layer_.set_weight(test_w);
      float e1 = error_layer_.ComputeCost();

      test_w(i, j) -= 2 * delta;
      error_layer_.set_weight(test_w);
      float e2 = error_layer_.ComputeCost();

      test_w(i, j) += delta;

      numeric_wgrad(i, j) = (e1 - e2) / (2 * delta);
      EXPECT_NEAR(wgrad(i, j), numeric_wgrad(i, j), tol);
    }
  }

  for (int i = 0; i < c; ++i) {
    test_b(i) += delta;
    error_layer_.set_bias(test_b);
    float e1 = error_layer_.ComputeCost();

    test_b(i) -= 2 * delta;
    error_layer_.set_bias(test_b);
    float e2 = error_layer_.ComputeCost();

    test_b(i) += delta;

    numeric_bgrad(i) = (e1 - e2) / (2 * delta);
    EXPECT_NEAR(bgrad(i), numeric_bgrad(i), tol);
  }
}

TEST_F(KullbackLeiblerDivergenceTest, Convergence) {
  error_layer_ =
      KullbackLeiblerDivergence(&input_layer_, {INPUT_SIZE, OUTPUT_SIZE});
  error_layer_.Init();

  input_layer_.SetInput(test_data_);
  error_layer_.SetLabel(test_label_);

  constexpr float learning_rate = 1e-3;
  float cost = error_layer_.ComputeCost();
  for (int i = 0; i < 50000; ++i) {
    error_layer_.ComputeCost();
    error_layer_.ComputeDerivative();
    error_layer_.Update(learning_rate);
  }
  float cost2 = error_layer_.ComputeCost();
  EXPECT_LT(cost2, cost);
}
