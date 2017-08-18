#include <gtest/gtest.h>
#include <memory>

#include "NerdNet/convert.h"
#include "NerdNet/layer/fc_layer.h"
#include "NerdNet/layer/mean_square_error.h"
#include "NerdNet/layer/relu_layer.h"
#include "NerdNet/nerd_net.h"
#include "NerdNet/trainer/gradient_descent_trainer.h"

using nerd::nn::NerdNet;
using nerd::nn::FCLayer;
using nerd::nn::ReluLayer;
using nerd::nn::MeanSquareError;
using nerd::nn::VariableShape;
using nerd::nn::Tensor;
using nerd::nn::trainer::GradientDescentTrainer;
using arma::Mat;

class GradientDescentTrainerTest : public ::testing::Test {
 public:
  enum {
    DATA_SIZE = 64,
    INPUT_SIZE = 4,
    HIDDEN_SIZE = 16,
    OUTPUT_SIZE = 8,
    ITERATIONS = 10000
  };

  GradientDescentTrainerTest() : trainer_(1e-4) {
    trainer_.set_verbose(true);
  }

 protected:
  void SetUp() override {
    nerdnet_ = std::make_shared<NerdNet>();
    nerdnet_->AddLayer<FCLayer>(VariableShape{INPUT_SIZE, HIDDEN_SIZE});
    nerdnet_->AddLayer<ReluLayer>();
    nerdnet_->AddLayer<FCLayer>(VariableShape{HIDDEN_SIZE, HIDDEN_SIZE});
    nerdnet_->AddLayer<ReluLayer>();
    nerdnet_->AddLayer<MeanSquareError>(
        VariableShape{HIDDEN_SIZE, OUTPUT_SIZE});

    trainer_.set_nerdnet(nerdnet_);

    Mat<float> test_data(DATA_SIZE, INPUT_SIZE),
        test_label(DATA_SIZE, OUTPUT_SIZE);
    test_data.randn();
    test_label.randn();
    Matrix2Tensor(test_data, test_data_tensor_);
    Matrix2Tensor(test_label, test_label_tensor_);
  }

  Tensor<float> test_data_tensor_, test_label_tensor_;
  std::shared_ptr<NerdNet> nerdnet_;
  GradientDescentTrainer trainer_;
};

TEST_F(GradientDescentTrainerTest, Convergence) {
  float start_cost = trainer_.Train(test_data_tensor_, test_label_tensor_);

  for (int i = 0; i < ITERATIONS; ++i) {
    trainer_.Train(test_data_tensor_, test_label_tensor_);
  }
  float end_cost = trainer_.Train(test_data_tensor_, test_label_tensor_);
  EXPECT_LT(end_cost, start_cost);
}
