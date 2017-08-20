#include <gtest/gtest.h>
#include <memory>

#include "NerdNet/convert.h"
#include "NerdNet/layer/layers.h"
#include "NerdNet/nerd_net.h"

using nerd::nn::NerdNet;
using nerd::nn::FCLayer;
using nerd::nn::ReluLayer;
using nerd::nn::MeanSquareError;
using nerd::nn::VariableShape;
using nerd::nn::Tensor;
using arma::Mat;

class NerdNetTest : public ::testing::Test {
 public:
  enum { DATA_SIZE = 1024, INPUT_SIZE = 2, HIDDEN_SIZE = 16, OUTPUT_SIZE = 2 };

 protected:
  void SetUp() override {
    nerdnet_ = std::make_shared<NerdNet>();

    EXPECT_EQ(nerdnet_->layer_count(), 1);

    nerdnet_->AddLayer<FCLayer>(VariableShape{INPUT_SIZE, HIDDEN_SIZE});
    nerdnet_->AddLayer<ReluLayer>();
    nerdnet_->AddLayer<FCLayer>(VariableShape{HIDDEN_SIZE, HIDDEN_SIZE});
    nerdnet_->AddLayer<ReluLayer>();
    nerdnet_->AddLayer<MeanSquareError>(
        VariableShape{HIDDEN_SIZE, OUTPUT_SIZE});
    EXPECT_EQ(nerdnet_->layer_count(), 6);
  }

  std::shared_ptr<NerdNet> nerdnet_;
};

TEST_F(NerdNetTest, FeedTensor) {
  Mat<float> test_data(DATA_SIZE, INPUT_SIZE);
  test_data.randn();
  Tensor<float> test_data_tensor;
  Matrix2Tensor(test_data, test_data_tensor);

  // unable to work before init
  bool error_occur = false;
  try {
    nerdnet_->Feed(test_data_tensor);
  } catch (std::exception& e) {
    error_occur = true;
  }
  EXPECT_TRUE(error_occur);

  EXPECT_TRUE(nerdnet_->Init());

  error_occur = false;
  try {
    nerdnet_->Feed(test_data_tensor);
  } catch (std::logic_error& e) {
    error_occur = true;
  }
  EXPECT_FALSE(error_occur);
}

TEST_F(NerdNetTest, FeedMatrix) {
  Mat<float> test_data(DATA_SIZE, INPUT_SIZE);
  test_data.randn();

  // unable to work before init
  bool error_occur = false;
  try {
    nerdnet_->Feed(test_data);
  } catch (std::exception& e) {
    error_occur = true;
  }
  EXPECT_TRUE(error_occur);

  EXPECT_TRUE(nerdnet_->Init());

  error_occur = false;
  try {
    nerdnet_->Feed(test_data);
  } catch (std::logic_error& e) {
    error_occur = true;
  }
  EXPECT_FALSE(error_occur);
}
