#include "NerdNet/trainer/momentum_trainer.h"

#include "NerdNet/except/trainer_exception.h"

namespace nerd {
namespace nn {
namespace trainer {

MomentumTrainer::MomentumTrainer(float learning_rate, float momentum)
    : Trainer(learning_rate), momentum_(momentum) {
  if (momentum_ < 0 || momentum_ > 1.0f) {
    throw except::TrainerException(
        "MomentumTrainer momentum"
        " needs to be in [0, 1.0]");
  }
}

MomentumTrainer::MomentumTrainer(float learning_rate, float momentum,
                                 std::shared_ptr<NerdNet> nerdnet)
    : Trainer(learning_rate, nerdnet), momentum_(momentum) {}

float MomentumTrainer::Train(const Tensor<float> data,
                             const Tensor<float> label) {
  // feed data
  nerdnet_->input_layer()->SetInput(data);
  nerdnet_->cost_function()->SetLabel(label);
  // run
  float loss = nerdnet_->cost_function()->ComputeCost();
  nerdnet_->cost_function()->ComputeDerivative();
  // update weights
  if (wgrads_.size() == 0) {
    for (int i = 1; i < nerdnet_->layer_count(); ++i) {
      if (nerdnet_->has_variable(i)) {
        nerdnet_->layer(i)->Update(learning_rate_);
      }
    }

    for (int i = 0; i < nerdnet_->layer_count(); ++i) {
      if (nerdnet_->has_variable(i)) {
        FCLayer* fc_layer = dynamic_cast<FCLayer*>(nerdnet_->layer(i));
        wgrads_.push_back(fc_layer->weight_gradient());
        bgrads_.push_back(fc_layer->bias_gradient());
      }
    }
  } else {
    for (int i = 0, j = 0; i < nerdnet_->layer_count(); ++i) {
      if (nerdnet_->has_variable(i)) {
        FCLayer* fc_layer = dynamic_cast<FCLayer*>(nerdnet_->layer(i));
        arma::Mat<float> weight_grad = wgrads_[j];
        arma::Mat<float> bias_grad = bgrads_[j];

        arma::Mat<float> new_weight_grad =
            momentum_ * weight_grad +
            (1.0f - momentum_) * fc_layer->weight_gradient();
        arma::Mat<float> new_bias_grad =
            momentum_ * bias_grad +
            (1.0f - momentum_) * fc_layer->bias_gradient();
        fc_layer->set_weight_gradient(new_weight_grad);
        fc_layer->set_bias_gradient(new_bias_grad);
        wgrads_[j] = new_weight_grad;
        bgrads_[j] = new_bias_grad;

        ++j;
      }
    }

    for (int i = 1; i < nerdnet_->layer_count(); ++i) {
      if (nerdnet_->has_variable(i)) {
        nerdnet_->layer(i)->Update(learning_rate_);
      }
    }
  }

  if (verbose_) {
    std::stringstream ss;
    ss << "loss: " << loss / data[0];
    Log(ss.str());
  }
  return loss;
}

} /* end of trainer namespace */
} /* end of nn namespace */
} /* end of nerd namespace */
